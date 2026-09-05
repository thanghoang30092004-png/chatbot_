from typing import Any, TypedDict 
from typing_extensions import Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ClassState(TypedDict, total = False):
    """
    State là dữ liệu đi qua các node của LangGraph.
    Checkpointer sẽ lưu state này theo thread_id/session_id.
    """
    #lịch sử hội thoại 
    messages: Annotated[list[BaseMessage], add_messages]
    
    question: str

    stanalone_question: str 
    blocked: bool 
    block_reason: str 
    subquestion : list[dict[str, Any]]
    sub_results: list[dict[str, Any]]
    envidence: dict[str, Any]
    answer : str
    debug : dict[str, Any]
    

