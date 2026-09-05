from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from backend.tools.rag_tool import run_rag_tool
from backend.tools.sql_tool import run_sql_tool
from backend.core.load_llm import load_llm
from backend.retrievers.load_retrievers import load_retrievers

SYSTEM_PROMPT = """
Bạn là trợ lý tuyển dụng AI của AIPT.
Bạn có các tools:
1. search_jd
- Dùng để tìm kiếm Job Description:
  + mô tả công việc
  + kỹ năng yêu cầu
  + mức lương
  + kinh nghiệm
  + yêu cầu tuyển dụng
2. search_cv
- Dùng để tìm kiếm CV ứng viên:
  + kỹ năng
  + kinh nghiệm
  + học vấn
  + dự án
  + thông tin ứng viên
3. search_policy
- Dùng để tìm kiếm:
  + chính sách công ty
  + quy trình tuyển dụng
  + phúc lợi
  + quy định nội bộ
4. sql_query
- Dùng để truy vấn database:
  + match_score
  + trạng thái ứng tuyển
  + lịch phỏng vấn
  + thống kê tuyển dụng
  + danh sách ứng viên
  + dữ liệu có cấu trúc
Quy tắc:
- Nếu cần dữ liệu thực tế => phải dùng tool.
- Không tự bịa dữ liệu.
- Chỉ dùng tool phù hợp nhất.
- Không gọi nhiều tool nếu không cần thiết.
- Trả lời bằng tiếng Việt.
- Trả lời rõ ràng, ngắn gọn.
- Nếu không tìm thấy dữ liệu, hãy nói rõ.
- Dùng SQL cho số liệu và dữ liệu cấu trúc.
- Dùng RAG tools cho mô tả và văn bản.
- Nếu câu hỏi chứa cả số liệu và mô tả => dùng nhiều tools.
- Không tự suy diễn dữ liệu.
""".strip()
# =========================
# TOOL INPUT SCHEMAS
# =========================

class SearchInput(BaseModel):
    question: str = Field(
        ...,
        description="Câu hỏi tìm kiếm tài liệu tuyển dụng"
    )
class SQLInput(BaseModel):
    question: str = Field(
        ...,
        description="Câu hỏi truy vấn database tuyển dụng"
    )


# =========================
# BUILD TOOLS
# =========================

def build_tools(llm, retrievers: dict):
    @tool(args_schema=SearchInput)
    def search_jd(question: str) -> str:
        """
        Tìm kiếm Job Description (JD).

        Dùng khi cần:
        - mô tả công việc
        - kỹ năng yêu cầu
        - mức lương
        - kinh nghiệm yêu cầu
        - thông tin tuyển dụng
        """
        try:
            result = run_rag_tool(
                question=question,
                source="jd",
                retriever=retrievers["jd"],
                llm=llm,
                retrievers=retrievers
            )
            evidence = result.get("evidence")
            if not evidence:
                return "Không tìm thấy JD phù hợp."
            return evidence
        except Exception as e:
            return f"Lỗi search_jd: {str(e)}"


    @tool(args_schema=SearchInput)
    def search_cv(question: str) -> str:
        """
        Tìm kiếm CV ứng viên.

        Dùng khi cần:
        - kỹ năng ứng viên
        - kinh nghiệm làm việc
        - học vấn
        - dự án
        - thông tin ứng viên
        """
        try:
            result = run_rag_tool(
                question=question,
                source="cv",
                retriever=retrievers["cv"],
                llm=llm,
                retrievers=retrievers
            )
            evidence = result.get("evidence")
            if not evidence:
                return "Không tìm thấy CV phù hợp."
            return evidence
        except Exception as e:
            return f"Lỗi search_cv: {str(e)}"


    @tool(args_schema=SearchInput)
    def search_policy(question: str) -> str:
        """
        Tìm kiếm chính sách tuyển dụng và quy định công ty.

        Dùng khi cần:
        - quy trình tuyển dụng
        - phúc lợi
        - chính sách công ty
        - quy định nội bộ
        """

        try:
            result = run_rag_tool(
                question=question,
                source="policy",
                retriever=retrievers["policy"],
                llm=llm,
                retrievers=retrievers
            )
            evidence = result.get("evidence")
            if not evidence:
                return "Không tìm thấy policy phù hợp."
            return evidence
        except Exception as e:
            return f"Lỗi search_policy: {str(e)}"


    @tool(args_schema=SQLInput)
    def sql_query(question: str) -> str:
        """
        Truy vấn database tuyển dụng.

        Dùng khi cần:
        - match_score
        - trạng thái ứng tuyển
        - lịch phỏng vấn
        - thống kê
        - dữ liệu có cấu trúc
        - danh sách ứng viên
        """
        try:
            result = run_sql_tool(
                question=question,
                llm=llm
            )
            sql = result.get("sql", "Không sinh được SQL")
            query_result = result.get("result", "Không có kết quả")

            return f"""
SQL:
{sql}
Kết quả:
{query_result}
""".strip()
        except Exception as e:
            return f"Lỗi sql_query: {str(e)}"
    return [
        search_jd,
        search_cv,
        search_policy,
        sql_query
    ]
# =========================
# BUILD AGENT
# =========================
def build_agent_executor(llm, retrievers: dict) -> AgentExecutor:
    tools = build_tools(llm, retrievers)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=5,
        max_execution_time=55,
        early_stopping_method="generate",
    )
    return executor
# =========================
# RUN AGENT
# =========================
def run_agent(
    question: str,
    chat_history: list,
    executor: AgentExecutor
) -> dict:
    try:
        result = executor.invoke({
            "input": question,
            "chat_history": chat_history
        })
        tools_used = []
        for step in result.get("intermediate_steps", []):
            try:
                action = step[0]
                tools_used.append({
                    "tool": action.tool,
                    "input": action.tool_input
                })
            except Exception:
                pass
        return {
            "answer": result.get("output", "Không có phản hồi."),
            "tools_used": tools_used
        }
    except Exception as e:
        return {
            "answer": f"Lỗi agent: {str(e)}",
            "tools_used": []
        }
    

def main():
    # Load LLM
    llm = load_llm()
    # Load retrievers
    retrievers = load_retrievers(k=5)
    # Build agent
    executor = build_agent_executor(
        llm=llm,
        retrievers=retrievers
    )
    # Test questions
    questions = ["với tư cách là 1 nhà tuyển dụng mày có thấy bạn trần thị diệu linh có xinh gái hay không?"
    ]
    # Empty history
    chat_history = []
    for question in questions:
        print("\n" + "=" * 100)
        print(f"QUESTION: {question}")
        result = run_agent(
            question=question,
            chat_history=chat_history,
            executor=executor
        )
        print("\nANSWER:")
        print(result["answer"])
        print("\nTOOLS USED:")
        for tool in result["tools_used"]:
            print(tool)
        print("=" * 100)

if __name__ == "__main__":
    main()