from langchain_community.vectorstores import FAISS

from core.models import get_embedder



def build_vector_store(documents, device: str = "cpu"):
    embedder = get_embedder(device=device)
    return FAISS.from_documents(documents, embedder)