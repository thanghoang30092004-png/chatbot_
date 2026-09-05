import sqlite3
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

DB_PATH = Path("/mnt/data/build_chatbot/data/recruitment_data/sql/recruitment.db")

SCHEMA = """
Tables:
- jobs(job_id, code, title, department, level, location,
        salary_min_million, salary_max_million, status, jd_file)
- candidates(candidate_id, full_name, email, phone, current_title,
             years_experience, location, cv_file, expected_salary_million, notice_period_days)
- applications(application_id, candidate_id, job_id, source,
               status, match_score, recruiter_note, applied_at)
  * status: screening | technical_interview | manager_interview |
            offer | rejected_screening | rejected_technical | rejected_manager | hired
- interviews(interview_id, application_id, round_name, scheduled_at,
             interviewer, technical_score, communication_score, culture_score, result, feedback)
"""
def build_nl2sql_chain(llm):
    prompt = PromptTemplate.from_template("""
Bạn là SQL expert cho hệ thống tuyển dụng (SQLite).
Chuyển câu hỏi tự nhiên thành SQL hợp lệ.
{schema}
Quy tắc:
- Chỉ trả về SQL thuần, không markdown, không giải thích.
- Dùng JOIN khi cần nhiều bảng.
- Dùng LIKE '%name%' để tìm tên.
- Nếu không liên quan DB: SELECT 'Không có dữ liệu' AS message

Câu hỏi: {question}
SQL:""".strip())
    return prompt | llm | StrOutputParser()


def execute_sql(sql: str) -> str:
    sql = sql.strip().strip("```sql").strip("```").strip()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return "Không có kết quả."
        cols = list(rows[0].keys())
        lines = [" | ".join(cols), "-" * 60]
        for row in rows:
            lines.append(" | ".join("NULL" if v is None else str(v) for v in row))
        return "\n".join(lines)
    except Exception as e:
        return f"Lỗi SQL: {e}"


def run_sql_tool(question: str, llm) -> dict:
    sql = build_nl2sql_chain(llm).invoke({"schema": SCHEMA, "question": question})
    return {"sql": sql, "result": execute_sql(sql)}