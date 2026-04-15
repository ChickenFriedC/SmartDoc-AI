from langchain_community.vectorstores import FAISS

from core.models import get_embedder
from config import RETRIEVER_K, RETRIEVER_FETCH_K


_EMBEDDER_CACHE = {}


def get_cached_embedder(device: str = "cuda"):
    if device not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[device] = get_embedder(device=device)
    return _EMBEDDER_CACHE[device]


def build_vector_store(documents, device: str = "cuda"):
    embedder = get_cached_embedder(device=device)
    return FAISS.from_documents(documents, embedder)


def load_vector_store(cache_dir: str, device: str = "cuda"):
    embedder = get_cached_embedder(device=device)
    return FAISS.load_local(
        folder_path=cache_dir,
        embeddings=embedder,
        allow_dangerous_deserialization=True,
    )


def save_vector_store(vector_store, cache_dir: str):
    vector_store.save_local(cache_dir)


def build_retriever(vector_store):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": RETRIEVER_K,
            "fetch_k": RETRIEVER_FETCH_K,
        },
    )