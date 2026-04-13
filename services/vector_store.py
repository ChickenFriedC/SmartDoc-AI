from langchain_community.vectorstores import FAISS

from core.models import get_embedder
from config import RETRIEVER_K, RETRIEVER_FETCH_K

def build_vector_store(documents, device: str = "cpu"):
    embedder = get_embedder(device=device)
    return FAISS.from_documents(documents, embedder)

def build_retriever(vector_store):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_FETCH_K},
    )