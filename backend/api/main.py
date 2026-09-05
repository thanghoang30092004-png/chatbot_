import asyncio
import time
import uuid
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler        
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langchain_core.messages import HumanMessage, AIMessage
from backend.core.load_llm import load_llm
from backend.retrievers.load_retrievers import load_retrievers
from backend.agents.agent import build_agent_executor, run_agent
from backend.api.model import ChatRequest, ChatResponse, SessionInfo, ToolCall

# ── Config ────────────────────────────────────────────────────────────
MAX_HISTORY      = 10 # chỉ nhớ 10 tin nhắn gần nhất 
SESSION_TTL      = 3600   # session tự xoá sau 1 tiếng
AGENT_TIMEOUT    = 60     # s    fix: missing timeout
CLEANUP_INTERVAL = 300    # cứ 5 phút chạy 1 lần để xoá session zombie (không active nhưng chưa xóa) — fix: zombie session
RATE_LIMIT       = "20/minute"


# ── Session ───────────────────────────────────────────────────────────
@dataclass
class Session:
    history: list        = field(default_factory=list)
    lock: asyncio.Lock   = field(default_factory=asyncio.Lock)
    last_accessed: float = field(default_factory=time.time)

    def touch(self):
        self.last_accessed = time.time()

    @property
    def expired(self) -> bool:
        return (time.time() - self.last_accessed) > SESSION_TTL

    @property
    def message_count(self) -> int:
        return sum(isinstance(m, HumanMessage) for m in self.history)


# ── Background GC — fix: zombie session ──────────────────────────────
async def _gc_loop(app: FastAPI):
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        async with app.state.registry_lock:
            dead = [sid for sid, s in app.state.sessions.items() if s.expired]
            for sid in dead:
                del app.state.sessions[sid]
        if dead:
            print(f"[GC] Removed {len(dead)} zombie sessions")


# ── Rate limiter — fix: no rate limit ────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── Lifespan ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading LLM + retrievers...")
    llm        = load_llm()
    retrievers = load_retrievers(k=5)

    # fix: shared executor — thêm max_iterations + max_execution_time
    # AgentExecutor stateless vì history truyền vào mỗi call,
    # nhưng dùng semaphore để tránh concurrent Qwen calls trên 1 GPU
    app.state.executor   = build_agent_executor(llm, retrievers)
    app.state.llm_sem    = asyncio.Semaphore(1)   # fix: shared executor (1 GPU)

    app.state.sessions = {}   # dict[str, Session]
    app.state.registry_lock = asyncio.Lock()

    _gc = asyncio.create_task(_gc_loop(app))
    print("Ready.")
    yield
    _gc.cancel()
    app.state.sessions.clear()


# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(title="Recruitment Chatbot", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Helper ────────────────────────────────────────────────────────────
async def _get_or_create(session_id: str) -> Session:
    """fix: global_lock bottleneck — double-checked locking."""
    if session_id in app.state.sessions:        # fast path, no lock
        return app.state.sessions[session_id]
    async with app.state.registry_lock:         # slow path, chỉ khi tạo mới
        if session_id not in app.state.sessions:
            app.state.sessions[session_id] = Session()
    return app.state.sessions[session_id]


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sessions", response_model=list[SessionInfo])
def list_sessions():
    return [
        SessionInfo(session_id=sid, message_count=s.message_count)
        for sid, s in app.state.sessions.items()
        if not s.expired
    ]


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    async with app.state.registry_lock:
        if session_id not in app.state.sessions:
            raise HTTPException(404, "Session not found")
        # fix: lock leak — chờ session lock trước khi xóa khỏi registry
        async with app.state.sessions[session_id].lock:
            del app.state.sessions[session_id]
    return {"message": "Cleared"}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(RATE_LIMIT)                      # fix: no rate limit
async def chat(request: Request, req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Empty message")

    session_id = req.session_id or str(uuid.uuid4())
    session    = await _get_or_create(session_id)

    async with session.lock:                    # per-session lock, không block global
        session.touch()                         # fix: zombie session — reset TTL
        snapshot = list(session.history)        # snapshot tránh mutation khi chạy agent

        start = time.time()
        try:
            # fix: missing timeout + không block event loop
            # fix: shared executor — Semaphore giới hạn 1 Qwen call cùng lúc
            async with app.state.llm_sem:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_agent,
                        question=req.message,
                        chat_history=snapshot,
                        executor=app.state.executor,
                    ),
                    timeout=AGENT_TIMEOUT,
                )
        except asyncio.TimeoutError:
            raise HTTPException(504, f"Timeout sau {AGENT_TIMEOUT}s")
        except Exception:
            traceback.print_exc()
            raise HTTPException(500, "Agent error — xem server log")

        # Chỉ update history sau khi thành công hoàn toàn
        session.history.append(HumanMessage(content=req.message))
        session.history.append(AIMessage(content=result["answer"]))
        session.history[:] = session.history[-MAX_HISTORY:]

    return ChatResponse(
        session_id=session_id,
        answer=result["answer"],
        tools_used=[
            ToolCall(tool=t["tool"], input=str(t["input"]))
            for t in result["tools_used"]
        ],
        latency_ms=(time.time() - start) * 1000,
        success=True,
    )