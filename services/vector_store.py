from langchain_community.vectorstores import FAISS
from core.models import get_embedder

def build_vector_store(documents, device: str = None):
    # Theo Listing 4: FAISS Vector Store (Trang 11)
    embedder = get_embedder(device=device)
    vector = FAISS.from_documents(documents, embedder)
    return vector

def get_retriever(vector):
    # Theo Listing 4: Create retriever (Trang 11)
    return vector.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
