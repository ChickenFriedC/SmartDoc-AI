from langchain_community.vectorstores import FAISS
from core.models import get_embedder

def build_vector_store(documents, device: str = None):
    embedder = get_embedder(device=device)
    vector = FAISS.from_documents(documents, embedder)
    return vector

def get_retriever(vector):
    return vector.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
