from pathlib import Path 
from typing import Optional, Dict, Any
from langchain_chroma import Chroma
from backend.core.load_embedding import load_embeddings


BASE_DIR = Path("/mnt/data/build_chatbot")
VECTOR_DIR = BASE_DIR / "vectorstores"
def load_vectorstore(persist_dir: Path, collection_name: str, embedding_model: None = None) -> Optional[Chroma]:

    persist_path = VECTOR_DIR / persist_dir

    if not persist_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy vectorstore: {persist_path}. "
            f"Hãy chạy script build_indexes.py trước."
        )

    if embedding_model is None:
        embedding_model = load_embeddings()

    return Chroma(
        persist_directory=str(persist_path),
        collection_name=collection_name,
        embedding_function=embedding_model,
    )

def load_retrievers(
    k: int = 5,
    filters: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """
    Load retrievers cho JD, CV, Policy.

    filters ví dụ:
    {
        "jd": {"status": "open"},
        "cv": {"candidate_id": "CAND001"},
        "policy": {"policy_type": "interview_process"}
    }
    """

    filters = filters or {}

    embedding_model = load_embeddings()

    jd_vectorstore = load_vectorstore(
        persist_dir="jd_chroma",
        collection_name="jd_docs",
        embedding_model=embedding_model,
    )

    cv_vectorstore = load_vectorstore(
        persist_dir="cv_chroma",
        collection_name="cv_docs",
        embedding_model=embedding_model,
    )

    policy_vectorstore = load_vectorstore(
        persist_dir="policy_chroma",
        collection_name="policy_docs",
        embedding_model=embedding_model,
    )

    return {
        "jd": jd_vectorstore.as_retriever(
            search_kwargs={
                "k": k,
                **({"filter": filters["jd"]} if "jd" in filters else {}),
            }
        ),
        "cv": cv_vectorstore.as_retriever(
            search_kwargs={
                "k": k,
                **({"filter": filters["cv"]} if "cv" in filters else {}),
            }
        ),
        "policy": policy_vectorstore.as_retriever(
            search_kwargs={
                "k": k,
                **({"filter": filters["policy"]} if "policy" in filters else {}),
            }
        ),
    }


if __name__ == "__main__":
    retrievers = load_retrievers(k=5)
    docs = retrievers["jd"].invoke("AI Engineer cần kỹ năng gì?")
    for doc in docs:
        print("=" * 80)
        #print(doc.metadata)
        print(doc.page_content[:500])