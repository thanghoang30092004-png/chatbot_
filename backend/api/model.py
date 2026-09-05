
from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    session_id: Optional[str] = None

    message: str

    source: Optional[str] = "all"

    debug: bool = False


class ToolCall(BaseModel):
    tool: str

    input: str

    output: Optional[str] = None

    success: bool = True

    error: Optional[str] = None


class RetrievedDoc(BaseModel):
    source: Optional[str] = None

    page: Optional[int] = None

    score: Optional[float] = None

    preview: Optional[str] = None


class TokenUsage(BaseModel):
    prompt_tokens: int = 0

    completion_tokens: int = 0

    total_tokens: int = 0


class ChatResponse(BaseModel):
    session_id: str

    answer: str

    tools_used: list[ToolCall] = []

    retrieved_docs: list[RetrievedDoc] = []

    retrieval_strategy: Optional[str] = None

    docs_count: Optional[int] = None

    token_usage: Optional[TokenUsage] = None

    latency_ms: Optional[float] = None

    success: bool = True

    error: Optional[str] = None


class SessionInfo(BaseModel):
    session_id: str

    message_count: int

    created_at: Optional[str] = None

    updated_at: Optional[str] = None
