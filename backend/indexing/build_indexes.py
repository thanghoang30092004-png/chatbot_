from pathlib import Path 
import json 
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path("/mnt/data/build_chatbot")
DATA_DIR = BASE_DIR / "data"/"recruitment_data"
VECTOR_DIR = BASE_DIR / "vectorstores"

embedding_model = HuggingFaceEmbeddings(
     model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def load_metadata():
    metadata_path = DATA_DIR/ "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        item["file_name"]: item for item in metadata
    }
def load_pdf_folder(folder_path: Path , doc_type: str, metadata_map: dict):
    docs = []

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found at {folder_path}")
    for file_path in folder_path.glob("*.pdf"):
        file_metadata = metadata_map.get(file_path.name)

        if file_metadata is None:
            raise ValueError(f"Thiếu metadata cho file: {file_path.name}")

        loader = PyMuPDFLoader(str(file_path))
        loaded_docs = loader.load()
        for page_idx, doc in enumerate(loaded_docs):
            doc.metadata.update(file_metadata)

            doc.metadata["source"] = file_path.name
            doc.metadata["file_path"] = str(file_path)
            doc.metadata["doc_type"] = doc_type
            doc.metadata["page"] = page_idx + 1

        docs.extend(loaded_docs)

    return docs
def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return text_splitter.split_documents(docs)

def build_vector_store(docs, persist_dir: Path, collection_name: str):
    persist_dir.mkdir(parents=True, exist_ok=True)

    if not docs:
        print(f"Không có docs cho collection: {collection_name}")
        return None

    return Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=str(persist_dir)
    )



def main():
    metadata_map = load_metadata()
    jd_docs = load_pdf_folder(
        DATA_DIR / "jd",
        "job_description",
        metadata_map
    )

    cv_docs = load_pdf_folder(
        DATA_DIR / "cv",
        "candidate_cv",
        metadata_map
    )

    policy_docs = load_pdf_folder(
        DATA_DIR / "policy",
        "policy",
        metadata_map
    )
    jd_chunks = split_docs(jd_docs)
    cv_chunks = split_docs(cv_docs)
    policy_chunks = split_docs(policy_docs)

    build_vector_store(jd_chunks, VECTOR_DIR / "jd_chroma", "jd_docs")
    build_vector_store(cv_chunks, VECTOR_DIR / "cv_chroma", "cv_docs")
    build_vector_store(policy_chunks, VECTOR_DIR / "policy_chroma", "policy_docs")

    print("Done building vectorstores.")
    print(f"JD chunks: {len(jd_chunks)}")
    print(f"CV chunks: {len(cv_chunks)}")
    print(f"Policy chunks: {len(policy_chunks)}")


if __name__ == "__main__":
    main()