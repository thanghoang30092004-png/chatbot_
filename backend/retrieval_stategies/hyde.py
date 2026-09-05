from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def build_hyde_chain(llm):
    hyde_prompt = PromptTemplate.from_template("""
Bạn là trợ lý tuyển dụng chuyên nghiệp.

Dựa trên câu hỏi của người dùng, hãy viết một đoạn tài liệu ngắn
có khả năng chứa câu trả lời cho câu hỏi đó.

Không nói đây là tài liệu giả định.
Chỉ viết một đoạn văn nội dung.

Câu hỏi:
{question}

Đoạn tài liệu:
""")

    return hyde_prompt | llm | StrOutputParser()


def hyde_retrieve(question: str, retriever, llm):
    """
    HyDE:
    question -> hypothetical document -> retriever -> docs thật
    """
    hyde_chain = build_hyde_chain(llm)

    hypothetical_doc = hyde_chain.invoke({
        "question": question
    })

    docs = retriever.invoke(hypothetical_doc)

    return docs
"""
if __name__ == "__main__":
    from backend.core.load_llm import load_llm
    from backend.retrievers.load_retrievers import load_retrievers

    llm = load_llm()
    retrievers = load_retrievers(k=5)

    docs = hyde_retrieve(
        question="AI Engineer cần kỹ năng gì?",
        retriever=retrievers["jd"],
        llm=llm
    )

    for i, doc in enumerate(docs, start=1):
        print("=" * 80)
        print(f"DOC {i}")
        print(doc.metadata)
        print(doc.page_content[:700])
"""