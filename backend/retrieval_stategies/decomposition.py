from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


VALID_ROUTES = {"jd", "cv", "policy"}


def parse_sub_questions(text: str):
    sub_questions = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = line.lstrip("-* ").strip()
        if len(line) > 2 and line[0].isdigit() and line[1] in [".", ")"]:
            line = line[2:].strip()
        if line:
            sub_questions.append(line)

    unique = []
    seen = set()

    for q in sub_questions:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            unique.append(q)

    return unique[:5]
def parse_route(text: str):
    route = text.strip().lower()
    if "jd" in route or "job" in route or "job_description" in route:
        return "jd"
    if "cv" in route or "candidate" in route or "ứng viên" in route:
        return "cv"
    if "policy" in route or "chính sách" in route:
        return "policy"

    if route in VALID_ROUTES:
        return route

    return "jd"


def build_decomposition_chain(llm):
    prompt = PromptTemplate.from_template("""
Bạn là trợ lý phân tích câu hỏi cho hệ thống RAG tuyển dụng nội bộ.
Nhiệm vụ của bạn là tách câu hỏi phức tạp của người dùng thành các câu hỏi nhỏ hơn
để truy xuất tài liệu chính xác hơn trong vector database.
Các nguồn tài liệu:
- JD: mô tả công việc, yêu cầu vị trí, kỹ năng cần có, địa điểm, cấp bậc, lương, nhiệm vụ.
- CV: thông tin ứng viên, kỹ năng, kinh nghiệm, học vấn, dự án, candidate_id.
- Policy: quy trình phỏng vấn, chính sách phúc lợi, chế độ, quyền lợi, quy định tuyển dụng.
Yêu cầu:
- Chỉ tạo câu hỏi con khi câu hỏi gốc có nhiều ý.
- Nếu câu hỏi gốc đơn giản, chỉ trả về chính câu hỏi gốc.
- Mỗi câu hỏi con nằm trên một dòng riêng.
- Không giải thích.
- Không đánh số nếu không cần.
- Giữ nguyên các thực thể quan trọng như tên vị trí, tên ứng viên, mã ứng viên, loại chính sách.
- Câu hỏi con phải rõ ràng, độc lập, có thể dùng trực tiếp để retrieval.
Ví dụ:
Câu hỏi gốc:
So sánh CV của Nam với JD AI Engineer và cho biết Nam còn thiếu kỹ năng gì?
Câu hỏi con:
JD AI Engineer yêu cầu những kỹ năng gì?
CV của Nam có những kỹ năng và kinh nghiệm nào?
Nam còn thiếu kỹ năng nào so với JD AI Engineer?

Câu hỏi gốc:
{question}

Câu hỏi con:
""".strip())
    return prompt | llm | StrOutputParser() | parse_sub_questions


def build_router_chain(llm):
    router_prompt = PromptTemplate.from_template("""
Bạn là bộ định tuyến truy vấn cho hệ thống RAG tuyển dụng nội bộ.
Nhiệm vụ của bạn là chọn đúng một nguồn tài liệu phù hợp nhất cho câu hỏi bên dưới.
Các nguồn có thể chọn:
- jd: dùng cho câu hỏi về mô tả công việc, yêu cầu vị trí, kỹ năng cần có, nhiệm vụ, địa điểm, cấp bậc, lương của vị trí tuyển dụng.
- cv: dùng cho câu hỏi về ứng viên, CV, kỹ năng cá nhân, kinh nghiệm, học vấn, dự án, candidate_id, tên ứng viên như Nam hoặc Hoa.
- policy: dùng cho câu hỏi về quy trình phỏng vấn, chính sách phúc lợi, chế độ, quyền lợi, quy định tuyển dụng.
Yêu cầu bắt buộc:
- Chỉ trả về một trong ba nhãn: jd, cv, policy.
- Không giải thích.
- Không thêm dấu câu.
- Không viết thêm nội dung khác.
Câu hỏi:
{question}
Nhãn:
""".strip())
    return router_prompt | llm | StrOutputParser() | parse_route

def deduplicate_docs(docs):
    unique_docs = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        content_key = doc.page_content.strip()[:300]
        key = f"{source}|{page}|{content_key}"
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    return unique_docs

def decomposition_retrieve(question: str, retrievers: dict, llm, top_k: int = 8):
    """
    Decomposition retrieval với LLM router:
    question -> sub-questions -> LLM route -> retrieve -> merge docs
    retrievers gồm:
    {
        "jd": retriever_jd,
        "cv": retriever_cv,
        "policy": retriever_policy
    }
    """
    decomposition_chain = build_decomposition_chain(llm)
    router_chain = build_router_chain(llm)

    sub_questions = decomposition_chain.invoke({
        "question": question
    })
    if not sub_questions:
        sub_questions = [question]

    all_docs = []

    for sub_question in sub_questions:
        route = router_chain.invoke({
            "question": sub_question
        })

        retriever = retrievers.get(route)

        if retriever is None:
            retriever = retrievers.get("jd")
        if retriever is None:
            continue

        docs = retriever.invoke(sub_question)
        all_docs.extend(docs)
    unique_docs = deduplicate_docs(all_docs)
    return unique_docs[:top_k]
if __name__ == "__main__":
    from backend.core.load_llm import load_llm
    from backend.retrievers.load_retrievers import load_retrievers

    llm = load_llm()
    retrievers = load_retrievers(k=5)
    docs = decomposition_retrieve(
        question="So sánh CV của Nam với JD AI Engineer và cho biết Nam còn thiếu kỹ năng gì?",
        retrievers=retrievers,
        llm=llm,
        top_k=8
    )
    for i, doc in enumerate(docs, start=1):
        print("=" * 80)
        print(f"DOC {i}")
        print("METADATA:")
        print(doc.metadata)
        print("CONTENT:")
        print(doc.page_content[:700])
