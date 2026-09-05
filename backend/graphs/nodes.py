from typing import Any 
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.core.load_llm import load_llm
from backend.retrievers.load_retrievers import load_retrievers

from backend.tools.rag_tool import run_rag_tool
from backend.tools.sql_tool import run_sql_tool

from backend.routers.router import route_sources_llm
from backend.retrieval_stategies.decomposition import decompose_question
