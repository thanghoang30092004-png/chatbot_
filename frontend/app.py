import os
import re
import uuid
import time
import requests
import streamlit as st
from dataclasses import dataclass, field

# ── Config class ─────────────────────────────────────────────────────────────
# fix #23: config class thay vì hardcode, hỗ trợ env
@dataclass
class Config:
    api_url: str = field(
        default_factory=lambda: os.getenv("API_URL", "http://localhost:6000").rstrip("/")
    )
    max_turns: int = 20       # fix #21: trim theo turns (user+assistant pair)
    max_display: int = 20     # số messages tối đa hiển thị
    max_retries: int = 2
    timeout_s: int = 90

cfg = Config()

st.set_page_config(
    page_title="Recruitment AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached static data ────────────────────────────────────────────────────────
# fix #28: cache example questions, không recompute mỗi rerun
@st.cache_data
def get_examples() -> list[str]:
    return [
        "Nam có phù hợp AI Engineer không?",
        "So sánh CV Nam với JD AI Engineer",
        "Ứng viên nào match_score cao nhất?",
        "Quy trình phỏng vấn có mấy vòng?",
        "Chính sách phúc lợi có hybrid không?",
    ]


# fix #27: cache health check — tránh spam backend mỗi rerun
@st.cache_data(ttl=30)
def cached_health(api_url: str) -> bool:
    try:
        return requests.get(f"{api_url}/health", timeout=5).status_code == 200
    except Exception:
        return False


# ── State init ────────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "session_id": None,
        "messages":   [],
        "is_loading": False,
        "pending":    None,   # fix #3: explicit field thay vì pop
        "last_error": None,
        "last_question": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── API service ───────────────────────────────────────────────────────────────
# fix #6: phân loại lỗi HTTP rõ ràng — không retry lỗi logic

_NO_RETRY_CODES = {401, 403, 422, 429}

def api_chat(message: str, session_id: str | None) -> dict:
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id

    last_err = None
    for attempt in range(cfg.max_retries + 1):
        try:
            resp = requests.post(
                f"{cfg.api_url}/chat", json=payload, timeout=cfg.timeout_s
            )

            # fix #6: phân loại HTTP status
            if resp.status_code == 429:
                raise RuntimeError("Rate limit: quá 20 request/phút. Chờ 1 phút rồi thử lại.")
            if resp.status_code in (401, 403):
                raise RuntimeError("Không có quyền truy cập API.")
            if resp.status_code == 504:
                raise RuntimeError(f"Gateway timeout: inference quá {cfg.timeout_s}s.")
            if resp.status_code == 500:
                raise RuntimeError("Lỗi nội bộ server. Xem log uvicorn để debug.")
            resp.raise_for_status()

            # fix #6: handle malformed JSON
            try:
                return resp.json()
            except ValueError:
                raise RuntimeError(
                    f"Backend trả về dữ liệu không hợp lệ:\n{resp.text[:200]}"
                )

        except RuntimeError:
            raise  # lỗi logic → không retry

        except requests.Timeout:
            last_err = f"Timeout sau {cfg.timeout_s}s. Backend inference đang bận."
        except requests.ConnectionError:
            last_err = "Không kết nối được backend. Kiểm tra uvicorn đang chạy chưa."
        except requests.HTTPError as e:
            last_err = f"HTTP {e.response.status_code}: {e.response.text[:150]}"
            if e.response.status_code in _NO_RETRY_CODES:
                break

        if attempt < cfg.max_retries:
            time.sleep(1.5 ** attempt)  # exponential backoff: 1.5s, 2.25s

    raise RuntimeError(last_err)


def api_clear(session_id: str):
    try:
        requests.delete(f"{cfg.api_url}/sessions/{session_id}", timeout=10)
    except Exception:
        pass


# ── Markdown sanitization ─────────────────────────────────────────────────────
# fix #16: strip HTML injection từ LLM output, giới hạn độ dài render
_DANGEROUS_TAGS = re.compile(
    r"<(script|iframe|object|embed|form|input|link|meta|style)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_ANY_TAG = re.compile(r"<[^>]{0,300}>")

def sanitize(text: str, max_len: int = 8000) -> str:
    text = _DANGEROUS_TAGS.sub("", text)
    text = _ANY_TAG.sub("", text)
    if len(text) > max_len:
        text = text[:max_len] + "\n\n*(Nội dung bị cắt do quá dài)*"
    return text


# ── Turn-based trim ───────────────────────────────────────────────────────────
# fix #21: trim theo turns (user+assistant pair), không cắt số lẻ → mất context
def trim_by_turns(messages: list, max_turns: int) -> list:
    # Tách user message cuối nếu chưa có cặp
    tail: list = []
    msgs = list(messages)
    if msgs and msgs[-1]["role"] == "user":
        tail = [msgs.pop()]

    # Group thành turns
    turns: list = []
    i = 0
    while i + 1 < len(msgs):
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            turns.append((msgs[i], msgs[i + 1]))
            i += 2
        else:
            i += 1

    turns = turns[-max_turns:]
    result: list = []
    for u, a in turns:
        result.extend([u, a])
    result.extend(tail)
    return result


# ── Auto-scroll ───────────────────────────────────────────────────────────────
# fix #11: JS để scroll xuống cuối sau khi render xong
def inject_autoscroll():
    st.components.v1.html(
        """<script>
        (function() {
            var body = window.parent.document.body;
            body.scrollTop = body.scrollHeight;
            var els = window.parent.document.querySelectorAll('section.main');
            if (els.length) els[els.length-1].scrollTop = els[els.length-1].scrollHeight;
        })();
        </script>""",
        height=0,
    )


# ── Reusable components ───────────────────────────────────────────────────────
TOOL_ICONS = {
    "search_jd":     ("📋", "JD"),
    "search_cv":     ("👤", "CV"),
    "search_policy": ("📜", "Policy"),
    "sql_query":     ("🗄️", "SQL"),
}


def render_tool_trace(tools_used: list, msg_id: str):
    """fix #17, #18, #19, #26: tool trace với source viewer + collapsed + truncated."""
    if not tools_used:
        return

    # fix #18: citations với tên nguồn rõ ràng
    parts = []
    for t in tools_used:
        icon, label = TOOL_ICONS.get(t["tool"], ("🔧", t["tool"]))
        parts.append(f"{icon} {label}")
    st.caption("Nguồn: " + " · ".join(parts))

    # fix #26: collapsed mặc định
    with st.expander(f"🔧 Tool trace ({len(tools_used)} bước)", expanded=False):
        for i, t in enumerate(tools_used, 1):
            tool_name = t.get("tool", "unknown")
            icon, label = TOOL_ICONS.get(tool_name, ("🔧", tool_name))
            raw_input = str(t.get("input", ""))

            st.markdown(f"**{i}. {icon} {label}**")

            # fix #26: truncated preview (300 ký tự)
            preview = raw_input[:300] + ("…" if len(raw_input) > 300 else "")
            st.code(preview, language="text")

            # fix #19: source viewer — nested expander cho full content
            if len(raw_input) > 300:
                with st.expander("📄 Xem evidence đầy đủ", expanded=False):
                    st.code(raw_input[:3000], language="text")


def render_feedback(msg_id: str):
    """fix #15: thumbs feedback."""
    c1, c2, _ = st.columns([1, 1, 16])
    with c1:
        if st.button("👍", key=f"up_{msg_id}", help="Câu trả lời tốt"):
            st.toast("Cảm ơn!", icon="👍")
    with c2:
        if st.button("👎", key=f"dn_{msg_id}", help="Chưa tốt"):
            st.toast("Đã ghi nhận!", icon="📝")


def render_message(msg: dict):
    """Render 1 tin nhắn từ history."""
    with st.chat_message(msg["role"]):
        # fix #16: sanitize trước khi render
        st.markdown(sanitize(msg["content"]))
        if msg["role"] == "assistant":
            render_tool_trace(msg.get("tools_used", []), msg["id"])
            if msg.get("latency_ms"):
                st.caption(f"⏱ {msg['latency_ms']:.0f} ms")
            render_feedback(msg["id"])


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💼 Recruitment AI")

    # fix #27: cached — không spam backend mỗi rerun
    online = cached_health(cfg.api_url)
    if online:
        st.success("Backend online", icon="✅")
    else:
        st.error("Backend offline", icon="🔴")
        st.caption(f"URL: `{cfg.api_url}`")

    st.divider()

    if st.session_state.session_id:
        st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ New", use_container_width=True,
                     disabled=st.session_state.is_loading):
            st.session_state.update(
                session_id=None, messages=[], last_error=None, pending=None
            )
            st.rerun()
    with c2:
        if st.button("🗑 Clear", use_container_width=True,
                     disabled=st.session_state.is_loading
                               or not st.session_state.session_id):
            api_clear(st.session_state.session_id)
            st.session_state.update(messages=[], last_error=None)
            st.rerun()

    st.divider()
    st.markdown("**Câu hỏi mẫu:**")

    # fix #28: dùng cached list
    for q in get_examples():
        if st.button(q, use_container_width=True, key=f"ex_{q[:24]}",
                     disabled=st.session_state.is_loading):
            # fix #3: set pending field, không dùng pop để tránh race condition
            st.session_state.pending = q
            st.rerun()


# ── Chat area ─────────────────────────────────────────────────────────────────
st.title("Recruitment AI Assistant")

all_msgs = st.session_state.messages
if len(all_msgs) > cfg.max_display:
    st.info(f"Hiển thị {cfg.max_display} / {len(all_msgs)} tin nhắn gần nhất.")

for msg in all_msgs[-cfg.max_display:]:
    render_message(msg)

# Lỗi cuối + retry
if st.session_state.last_error:
    st.error(st.session_state.last_error)
    if st.session_state.last_question:
        if st.button("🔄 Thử lại", type="primary"):
            st.session_state.pending = st.session_state.last_question
            st.session_state.last_error = None
            st.rerun()


# ── Input ─────────────────────────────────────────────────────────────────────
# fix #3: chỉ consume pending khi KHÔNG đang loading — tránh race condition
question = None

if not st.session_state.is_loading:
    if st.session_state.pending:
        question = st.session_state.pending
        st.session_state.pending = None     # atomic clear sau khi lấy
    elif prompt := st.chat_input("Hỏi về tuyển dụng..."):
        question = prompt
else:
    st.info("⏳ Đang xử lý, vui lòng chờ…")


# ── Send ──────────────────────────────────────────────────────────────────────
if question:
    st.session_state.last_question = question
    st.session_state.last_error = None
    st.session_state.is_loading = True

    user_msg = {"role": "user", "content": question, "id": str(uuid.uuid4())}
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.status("Đang xử lý…", expanded=True)
        data = None
        try:
            with status:
                st.write("🔍 Đang phân tích câu hỏi…")
                data = api_chat(question, st.session_state.session_id)
                st.write("✅ Nhận được câu trả lời!")
            status.update(label="Hoàn thành!", state="complete", expanded=False)

        except RuntimeError as e:
            status.update(label="Lỗi!", state="error", expanded=False)
            st.session_state.last_error = str(e)
            # Xóa user message chưa có cặp để tránh orphan
            st.session_state.messages = [
                m for m in st.session_state.messages if m["id"] != user_msg["id"]
            ]

        finally:
            st.session_state.is_loading = False

        if data:
            st.session_state.session_id = data["session_id"]

            answer      = data["answer"]
            tools_used  = data.get("tools_used", [])
            latency_ms  = data.get("latency_ms", 0)
            msg_id      = str(uuid.uuid4())

            st.markdown(sanitize(answer))      # fix #16
            render_tool_trace(tools_used, msg_id)
            st.caption(f"⏱ {latency_ms:.0f} ms")
            render_feedback(msg_id)

            st.session_state.messages.append({
                "role":       "assistant",
                "content":    answer,
                "tools_used": tools_used,
                "latency_ms": latency_ms,
                "id":         msg_id,
            })

            # fix #21: trim theo turns sau khi có đủ cặp user+assistant
            st.session_state.messages = trim_by_turns(
                st.session_state.messages, cfg.max_turns
            )

    # fix #11: auto-scroll sau khi render xong
    inject_autoscroll()
#streamlit run frontend/app.py --server.fileWatcherType none