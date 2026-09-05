from typing import List, Tuple, Optional, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def normalize_text(text: str) -> str:
    return text.strip().lower()
def is_yes(text: str) -> bool:
    """
    Parse kết quả grader.
    Chấp nhận:
    - yes
    - yes.
    - có
    - relevant
    """
    value = normalize_text(text)
    return (
        value.startswith("yes")
        or value.startswith("có")
        or value.startswith("relevant")
    )

def build_doc_grader(llm):
    """
    Grader đánh giá document có liên quan tới question không.
    Output bắt buộc chỉ yes/no.
    """
    prompt = PromptTemplate.from_template("""
Bạn là bộ đánh giá độ liên quan của tài liệu cho hệ thống RAG tuyển dụng nội bộ.
Nhiệm vụ:
Xác định đoạn tài liệu có giúp trả lời câu hỏi của người dùng hay không.
Quy tắc đánh giá:
- Trả lời "yes" nếu tài liệu chứa thông tin trực tiếp hoặc gần trực tiếp để trả lời câu hỏi.
- Trả lời "yes" nếu tài liệu chứa thông tin nền cần thiết để suy luận câu trả lời.
- Trả lời "no" nếu tài liệu chỉ trùng vài từ khóa nhưng không giúp trả lời câu hỏi.
- Trả lời "no" nếu tài liệu thuộc sai nguồn, sai ứng viên, sai vị trí, hoặc sai loại chính sách.
- Không giải thích.
Chỉ trả về đúng một trong hai từ:
yes
no
Câu hỏi:
{question}
Metadata tài liệu:
{metadata}
Nội dung tài liệu:
{document}
Kết quả:
""".strip())
    return prompt | llm | StrOutputParser()


def build_query_rewriter(llm):
    """
    Rewriter viết lại question thành search query tốt hơn.
    """
    prompt = PromptTemplate.from_template("""
Bạn là trợ lý viết lại truy vấn cho hệ thống tìm kiếm tài liệu tuyển dụng nội bộ.
Nhiệm vụ:
Viết lại câu hỏi của người dùng thành một truy vấn tìm kiếm tốt hơn cho vector database.
Yêu cầu:
- Chỉ trả về truy vấn đã viết lại.
- Không giải thích.
- Giữ nguyên tên vị trí, tên ứng viên, candidate_id, policy_type nếu có.
- Thêm các từ khóa quan trọng liên quan đến JD, CV hoặc Policy nếu phù hợp.
- Truy vấn nên ngắn gọn, rõ ý.
Ví dụ:
Câu hỏi: AI Engineer cần kỹ năng gì?
Truy vấn tốt hơn: yêu cầu kỹ năng bắt buộc vị trí AI Engineer Python machine learning computer vision FastAPI
Câu hỏi: Nam có phù hợp JD AI Engineer không?
Truy vấn tốt hơn: so sánh CV Nam candidate CAND001 với JD AI Engineer kỹ năng kinh nghiệm còn thiếu
Câu hỏi:
{question}
Truy vấn tốt hơn:
""".strip())
    return prompt | llm | StrOutputParser()


def doc_to_text(doc) -> str:
    return getattr(doc, "page_content", "") or ""


def doc_metadata_to_text(doc) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    if not metadata:
        return "Không có metadata."

    return "\n".join(
        f"{key}: {value}"
        for key, value in metadata.items()
    )


def grade_docs(
    question: str,
    docs: List[Any],
    grader,
    debug: bool = False,
) -> Tuple[List[Any], List[Any]]:
    """
    Chấm relevance cho danh sách docs.
    Return:
    - relevant_docs
    - rejected_docs
    """

    relevant_docs = []
    rejected_docs = []

    for idx, doc in enumerate(docs, start=1):
        document_text = doc_to_text(doc)
        metadata_text = doc_metadata_to_text(doc)

        if not document_text.strip():
            rejected_docs.append(doc)
            continue

        grade = grader.invoke({
            "question": question,
            "metadata": metadata_text,
            "document": document_text,
        })

        if debug:
            print("=" * 80)
            print(f"GRADE DOC {idx}: {grade}")
            print("METADATA:")
            print(getattr(doc, "metadata", {}))
            print("CONTENT:")
            print(document_text[:500])

        if is_yes(grade):
            relevant_docs.append(doc)
        else:
            rejected_docs.append(doc)

    return relevant_docs, rejected_docs


def rewrite_query(
    question: str,
    rewriter,
    debug: bool = False,
) -> str:
    rewritten_query = rewriter.invoke({
        "question": question
    }).strip()

    if not rewritten_query:
        rewritten_query = question

    if debug:
        print("=" * 80)
        print("ORIGINAL QUESTION:")
        print(question)
        print("REWRITTEN QUERY:")
        print(rewritten_query)

    return rewritten_query


def deduplicate_docs(docs: List[Any]) -> List[Any]:
    unique_docs = []
    seen = set()

    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}

        source = metadata.get("source", "")
        page = metadata.get("page", "")
        content_key = doc_to_text(doc).strip()[:300]

        key = f"{source}|{page}|{content_key}"

        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs


def crag_filter_or_retry(
    question: str,
    docs: List[Any],
    retriever,
    llm,
    top_k: int = 5,
    max_retries: int = 1,
    debug: bool = False,
) -> List[Any]:
    """
    CRAG:
    1. Grade docs ban đầu.
    2. Nếu có docs relevant -> trả về docs relevant.
    3. Nếu không có docs relevant -> rewrite query.
    4. Retrieve lại bằng query mới.
    5. Grade lại docs mới.
    6. Trả về docs relevant sau retry.

    Input:
    - question: câu hỏi gốc của user
    - docs: docs retrieve ban đầu
    - retriever: LangChain retriever
    - llm: LangChain Runnable LLM, ví dụ ChatOpenAI
    - top_k: số docs trả về tối đa
    - max_retries: số lần rewrite + retrieve lại
    - debug: in log debug
    """

    grader = build_doc_grader(llm)
    rewriter = build_query_rewriter(llm)

    relevant_docs, rejected_docs = grade_docs(
        question=question,
        docs=docs,
        grader=grader,
        debug=debug,
    )

    if relevant_docs:
        return deduplicate_docs(relevant_docs)[:top_k]

    current_query = question
    all_retry_docs = []

    for retry_idx in range(max_retries):
        rewritten_query = rewrite_query(
            question=current_query,
            rewriter=rewriter,
            debug=debug,
        )

        new_docs = retriever.invoke(rewritten_query)

        all_retry_docs.extend(new_docs)

        relevant_docs, rejected_docs = grade_docs(
            question=question,
            docs=new_docs,
            grader=grader,
            debug=debug,
        )

        if relevant_docs:
            return deduplicate_docs(relevant_docs)[:top_k]

        current_query = rewritten_query

    return deduplicate_docs(all_retry_docs)[:top_k]

