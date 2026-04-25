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
        _cross_encoder = CrossEncoder(RERANK_MODEL)
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



def filter_docs_by_source(docs: List, selected_source: str):
    if selected_source == "Tất cả":
        return docs
    return [doc for doc in docs if doc.metadata.get("filename") == selected_source]



def rerank_documents(question: str, docs: List, top_n: int = RERANK_TOP_N):
    if not docs:
        return []

    cross_encoder = get_cross_encoder()
    pairs = [(question, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)

    scored_docs = []
    for doc, score in zip(docs, scores):
        doc.metadata["rerank_score"] = float(score)
        scored_docs.append(doc)

    scored_docs.sort(key=lambda d: d.metadata.get("rerank_score", 0.0), reverse=True)
    return scored_docs[:top_n]