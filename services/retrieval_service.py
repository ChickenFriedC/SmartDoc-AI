import torch
from typing import List

try:
    from langchain.retrievers import EnsembleRetriever
except (ImportError, ModuleNotFoundError):
    try:
        from langchain_community.retrievers import EnsembleRetriever
    except (ImportError, ModuleNotFoundError):
        from langchain_classic.retrievers import EnsembleRetriever

from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

from config import (
    ENSEMBLE_BM25_WEIGHT,
    ENSEMBLE_VECTOR_WEIGHT,
    RERANK_TOP_N,
    RETRIEVER_FETCH_K,
    RETRIEVER_K,
    RERANK_MODEL,
)

_cross_encoder = None


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _cross_encoder = CrossEncoder(RERANK_MODEL, device=device)
    return _cross_encoder



def build_base_retrievers(vector_store, documents):
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_FETCH_K},
    )

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = RETRIEVER_FETCH_K
    return vector_retriever, bm25_retriever

def build_hybrid_retriever(vector_store, documents):
    vector_retriever, bm25_retriever = build_base_retrievers(vector_store, documents)
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[ENSEMBLE_VECTOR_WEIGHT, ENSEMBLE_BM25_WEIGHT],
    )



def filter_docs_by_metadata(docs: List, selected_files: List[str]):
    if not selected_files:
        return docs
    return [doc for doc in docs if doc.metadata.get("filename") in selected_files]

def rerank_documents(question: str, docs: List, top_n: int = RERANK_TOP_N, selected_files: List[str] = None):
    if not docs:
        return []
    
    # Lọc tài liệu theo metadata trước khi rerank
    filtered_docs = filter_docs_by_metadata(docs, selected_files)
    if not filtered_docs:
        return []

    cross_encoder = get_cross_encoder()
    pairs = [(question, doc.page_content) for doc in filtered_docs]
    scores = cross_encoder.predict(pairs)

    scored_docs = []
    for doc, score in zip(docs, scores):
        doc.metadata["rerank_score"] = float(score)
        scored_docs.append(doc)

    scored_docs.sort(key=lambda d: d.metadata.get("rerank_score", 0.0), reverse=True)
    return scored_docs[:top_n]