
def basic_retrieve(question: str, retriever):
    """
    Retrieval cơ bản:
    question -> retriever -> docs
    """
    return retriever.invoke(question)