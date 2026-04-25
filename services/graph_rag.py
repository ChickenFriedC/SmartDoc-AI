from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, List, Set, Tuple

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import GRAPH_ENTITY_MAX_PER_CHUNK, GRAPH_NEIGHBOR_EXPANSION, GRAPH_RETRIEVER_K


_WORD_RE = re.compile(r"[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9_\-]{2,}", re.UNICODE)


def _tokenize_entities(text: str) -> List[str]:
    if not text:
        return []
    # Very lightweight "entity" extraction: keep word-like tokens (incl. Vietnamese).
    raw = _WORD_RE.findall(text)
    if not raw:
        return []

    stop = {
        # EN
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "from",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "will",
        "would",
        "should",
        "can",
        "could",
        "may",
        "might",
        "not",
        "your",
        "you",
        "our",
        "their",
        "then",
        "than",
        "into",
        "over",
        "under",
        "between",
        # VI
        "và",
        "của",
        "cho",
        "với",
        "này",
        "đó",
        "từ",
        "là",
        "được",
        "các",
        "một",
        "những",
        "trong",
        "khi",
        "nếu",
        "thì",
        "như",
        "để",
        "không",
        "vậy",
    }

    tokens: List[str] = []
    for t in raw:
        t_norm = t.strip().lower()
        if t_norm in stop:
            continue
        # avoid very numeric-like tokens
        if t_norm.isdigit():
            continue
        tokens.append(t_norm)
    return tokens


@dataclass(frozen=True)
class GraphIndex:
    entity_to_docs: Dict[str, Set[int]]
    neighbors: Dict[str, Dict[str, int]]


def build_graph_index(documents: List[Document]) -> GraphIndex:
    entity_to_docs: DefaultDict[str, Set[int]] = defaultdict(set)
    neighbors: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))

    for doc_id, doc in enumerate(documents):
        entities = _tokenize_entities(doc.page_content)
        if not entities:
            continue

        # Limit per chunk to keep graph build O(n^2) bounded.
        unique: List[str] = []
        seen: Set[str] = set()
        for e in entities:
            if e in seen:
                continue
            unique.append(e)
            seen.add(e)
            if len(unique) >= GRAPH_ENTITY_MAX_PER_CHUNK:
                break

        for e in unique:
            entity_to_docs[e].add(doc_id)

        # Co-occurrence edges within a chunk.
        for i in range(len(unique)):
            a = unique[i]
            for j in range(i + 1, len(unique)):
                b = unique[j]
                neighbors[a][b] += 1
                neighbors[b][a] += 1

    return GraphIndex(entity_to_docs=dict(entity_to_docs), neighbors={k: dict(v) for k, v in neighbors.items()})


class GraphRetriever(BaseRetriever):
    """Lightweight graph-based retriever.

    It extracts word-like entities, finds chunks containing them, optionally expands
    to neighboring entities (co-occurrence graph), and ranks chunks by a simple
    weighted vote.
    """

    documents: List[Document]
    index: GraphIndex
    k: int = GRAPH_RETRIEVER_K
    neighbor_expansion: int = GRAPH_NEIGHBOR_EXPANSION

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        q_entities = _tokenize_entities(query)
        if not q_entities:
            return []

        doc_scores: DefaultDict[int, float] = defaultdict(float)

        # Direct matches get higher weight.
        for e in q_entities:
            for doc_id in self.index.entity_to_docs.get(e, set()):
                doc_scores[doc_id] += 2.0

        # Neighbor expansion (graph hop): bring in related entities.
        if self.neighbor_expansion > 0:
            for e in q_entities:
                neigh = self.index.neighbors.get(e)
                if not neigh:
                    continue
                # Take top-N neighbors by co-occurrence count.
                top_neighbors = sorted(neigh.items(), key=lambda kv: kv[1], reverse=True)[: self.neighbor_expansion]
                for nb, weight in top_neighbors:
                    for doc_id in self.index.entity_to_docs.get(nb, set()):
                        doc_scores[doc_id] += 1.0 + min(1.0, weight / 3.0)

        if not doc_scores:
            return []

        ranked = sorted(doc_scores.items(), key=lambda kv: (-kv[1], kv[0]))[: self.k]
        results: List[Document] = []
        for doc_id, score in ranked:
            doc = self.documents[doc_id]
            doc.metadata = dict(doc.metadata)
            doc.metadata["graph_score"] = float(score)
            results.append(doc)
        return results


def build_graph_retriever(documents: List[Document]) -> GraphRetriever:
    index = build_graph_index(documents)
    return GraphRetriever(documents=documents, index=index)
