# backend/tools/rag_tool.py

from typing import Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.retrieval_stategies.basic import basic_retrieve
from backend.retrieval_stategies.hyde import hyde_retrieve
from backend.retrieval_stategies.rag_fution import rag_fusion_retrieve
from backend.retrieval_stategies.decomposition import decomposition_retrieve
from backend.retrieval_stategies.crag import crag_filter_or_retry

VALID_STRATEGIES = {
    "basic",
    "hyde",
    "rag_fusion",
    "decomposition"
}


def parse_strategy(text: str) -> str:
    value = text.strip().lower()
    if "decomposition" in value or "decompose" in value or "phân rã" in value or "tách" in value:
        return "decomposition"
    if "rag_fusion" in value or "rag fusion" in value or "fusion" in value:
        return "rag_fusion"
    if "hyde" in value:
        return "hyde"
    if "basic" in value:
        return "basic"
    if value in VALID_STRATEGIES:
        return value

    return "basic"


def build_strategy_router_chain(llm):
    prompt = PromptTemplate.from_template("""
Bạn là bộ chọn retrieval strategy cho hệ thống RAG tuyển dụng nội bộ.
Nhiệm vụ:
Chọn đúng một retrieval strategy phù hợp nhất cho câu hỏi của người dùng.
Các strategy hợp lệ:
1. basic
Dùng khi câu hỏi rõ ràng, trực tiếp, chỉ cần tìm tài liệu bằng câu hỏi gốc.
Ví dụ:
- AI Engineer cần kỹ năng gì?
- Quy trình phỏng vấn gồm mấy vòng?
- CV của Nam có kinh nghiệm gì?
2. hyde
Dùng khi câu hỏi quá ngắn, mơ hồ, thiếu từ khóa, hoặc cần mở rộng thành một đoạn tài liệu giả định để tìm kiếm tốt hơn.
Ví dụ:
- Nam thế nào?
- Chính sách?
- Backend yêu cầu gì?
- Có phù hợp không?
3. rag_fusion
Dùng khi câu hỏi cần nhiều cách diễn đạt, nhiều góc tìm kiếm, so sánh hoặc đánh giá.
Ví dụ:
- Nam có phù hợp với JD AI Engineer không?
- So sánh CV Nam với JD AI Engineer.
- Hoa và Nam ai phù hợp Backend hơn?
- JD AI Engineer khác Backend Engineer ở điểm nào?
4. decomposition
Dùng khi câu hỏi có nhiều ý hoặc cần lấy dữ liệu từ nhiều nguồn JD/CV/Policy.
Ví dụ:
- So sánh CV của Nam với JD AI Engineer và cho biết Nam còn thiếu kỹ năng gì?
- Quy trình phỏng vấn AI Engineer gồm mấy vòng và phúc lợi thế nào?
- Nam có phù hợp AI Engineer không và cần cải thiện kỹ năng gì?
Thông tin source:
- jd: tài liệu mô tả công việc, yêu cầu vị trí, kỹ năng, nhiệm vụ, cấp bậc, lương.
- cv: tài liệu ứng viên, kỹ năng, kinh nghiệm, học vấn, dự án.
- policy: tài liệu quy trình phỏng vấn, phúc lợi, chính sách tuyển dụng.
- all: câu hỏi có thể cần nhiều nguồn JD/CV/Policy.
Yêu cầu bắt buộc:
- Chỉ trả về một trong bốn nhãn: basic, hyde, rag_fusion, decomposition.
- Không giải thích.
- Không thêm dấu câu.
- Không viết thêm nội dung khác.
Source:
{source}
Câu hỏi:
{question}
Strategy:
""".strip())
    return prompt | llm | StrOutputParser() | parse_strategy


def choose_retrieval_strategy(question: str, source: str, llm) -> str:
    """
    Dùng LLM để chọn retrieval strategy.
    """
    router_chain = build_strategy_router_chain(llm)
    strategy = router_chain.invoke({
        "question": question,
        "source": source,
    })
    return strategy


def format_docs(docs) -> str:
    """
    Format docs thành evidence text.
    Có kèm metadata để prompt trả lời cuối biết nguồn.
    """

    formatted = []

    for i, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        source_file = metadata.get("source", metadata.get("file_name", "unknown"))
        page = metadata.get("page", "unknown")
        doc_type = metadata.get("doc_type", "unknown")
        job_title = metadata.get("job_title")
        candidate_name = metadata.get("candidate_name")
        candidate_id = metadata.get("candidate_id")
        policy_type = metadata.get("policy_type")
        meta_parts = [
            f"source={source_file}",
            f"page={page}",
            f"doc_type={doc_type}",
        ]
        if job_title:
            meta_parts.append(f"job_title={job_title}")
        if candidate_name:
            meta_parts.append(f"candidate_name={candidate_name}")
        if candidate_id:
            meta_parts.append(f"candidate_id={candidate_id}")
        if policy_type:
            meta_parts.append(f"policy_type={policy_type}")
        formatted.append(
            f"[DOC {i} | {' | '.join(meta_parts)}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


def run_rag_tool(
    question: str,
    source: str,
    retriever,
    llm,
    retrievers: Optional[dict] = None,
    debug: bool = False,
):
    """
    RAG tool tổng quát cho JD/CV/Policy.
    Luồng:
    question
    -> LLM chọn Basic/HyDE/RAG-Fusion/Decomposition
    -> retrieve docs
    -> return evidence text
    Không dùng CRAG.
    """

    strategy = choose_retrieval_strategy(
        question=question,
        source=source,
        llm=llm,
    )

    if debug:
        print("=" * 80)
        print("QUESTION:", question)
        print("SOURCE:", source)
        print("STRATEGY:", strategy)

    if strategy == "decomposition":
        if retrievers is not None:
            docs = decomposition_retrieve(
                question=question,
                retrievers=retrievers,
                llm=llm,
                top_k=8,
            )
        else:
            strategy = "basic"
            docs = basic_retrieve(
                question=question,
                retriever=retriever,
            )

    elif strategy == "hyde":
        docs = hyde_retrieve(
            question=question,
            retriever=retriever,
            llm=llm,
        )

    elif strategy == "rag_fusion":
        docs = rag_fusion_retrieve(
            question=question,
            retriever=retriever,
            llm=llm,
            top_k=8,
        )

    else:
        docs = basic_retrieve(
            question=question,
            retriever=retriever,
        )

    if strategy != "decomposition":
        docs = crag_filter_or_retry(
            question=question,
            docs=docs,
            retriever=retriever,
            llm=llm,
            top_k=8,
            max_retries=1,
            debug=debug,
        )
    evidence = format_docs(docs)
    
    return {
        "source": source,
        "strategy": strategy,
        "docs_count": len(docs),
        "docs": docs,
        "evidence": evidence,
    }



