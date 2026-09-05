# backend/retrieval_strategies/rag_fusion.py

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.load import dumps, loads


def parse_queries(text: str):
    queries = []

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        line = line.lstrip("-* ").strip()

        if len(line) > 2 and line[0].isdigit() and line[1] in [".", ")"]:
            line = line[2:].strip()

        if line:
            queries.append(line)

    unique = []
    seen = set()

    for q in queries:
        key = q.lower()

        if key not in seen:
            seen.add(key)
            unique.append(q)

    return unique[:4]


def reciprocal_rank_fusion(results: list[list], k: int = 30):
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)

            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0

            fused_scores[doc_str] += 1 / (k + rank + 1)

    fused_docs = [
        loads(doc_str)
        for doc_str, score in sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
    ]

    return fused_docs


def build_query_generation_chain(llm):
    prompt = PromptTemplate.from_template("""
Bạn là trợ lý tìm kiếm tài liệu tuyển dụng nội bộ.

Nhiệm vụ của bạn là viết lại câu hỏi của người dùng thành 4 câu truy vấn tìm kiếm khác nhau
để truy xuất tài liệu tốt hơn trong vector database.

Yêu cầu:
- Mỗi truy vấn nằm trên một dòng riêng.
- Không giải thích.
- Không đánh số nếu không cần.
- Giữ nguyên các thực thể quan trọng như tên vị trí, tên ứng viên, mã ứng viên, loại chính sách.
- Có thể dùng cả từ đồng nghĩa hoặc cách diễn đạt khác.
- Truy vấn nên ngắn gọn, rõ ý, phù hợp để tìm trong JD, CV hoặc policy.
Ví dụ:
Câu hỏi: AI Engineer cần kỹ năng gì?
Truy vấn:
kỹ năng bắt buộc của vị trí AI Engineer
yêu cầu công việc AI Engineer về Python machine learning computer vision
AI Engineer cần kinh nghiệm và công nghệ nào
mô tả công việc AI Engineer phần yêu cầu kỹ năng

Câu hỏi:
{question}

Truy vấn:
""".strip())

    return prompt | llm | StrOutputParser() | parse_queries


def rag_fusion_retrieve(question: str, retriever, llm, top_k: int = 5):
    """
    RAG-Fusion:
    question -> multiple Vietnamese queries -> retrieve each -> RRF -> top docs
    """
    generate_queries = build_query_generation_chain(llm)

    queries = generate_queries.invoke({
        "question": question
    })

    if not queries:
        queries = [question]

    all_results = []

    for q in queries:
        docs = retriever.invoke(q)
        all_results.append(docs)

    fused_docs = reciprocal_rank_fusion(all_results)

    return fused_docs[:top_k]
if __name__ == "__main__":
    from backend.core.load_llm import load_llm
    from backend.retrievers.load_retrievers import load_retrievers

    llm = load_llm()
    retrievers = load_retrievers(k=5)

    docs = rag_fusion_retrieve(
        question="AI Engineer cần kỹ năng gì?",
        retriever=retrievers["jd"],
        llm=llm,
        top_k=5
    )

    for i, doc in enumerate(docs, start=1):
        print("=" * 80)
        print(f"DOC {i}")
        print("METADATA:")
        print(doc.metadata)
        print("CONTENT:")
        print(doc.page_content[:700])