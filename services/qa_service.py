import json

from core.models import get_llm
from core.prompts import (
    build_prompt,
    build_query_rewrite_prompt,
    build_self_rag_prompt,
    build_multi_hop_prompt,
)
from core.utils import build_source_label
from services.retrieval_service import rerank_documents



def _format_chat_history(chat_history: list[dict], max_turns: int = 4) -> str:
    lines = []
    for turn in chat_history[-max_turns:]:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)



def _format_context(docs) -> str:
    blocks = []
    for idx, doc in enumerate(docs, start=1):
        label = build_source_label(doc.metadata)
        blocks.append(f"[{idx}] {label}\n{doc.page_content}")
    return "\n\n".join(blocks)



def _format_sources(docs):
    sources = []
    for doc in docs:
        meta = doc.metadata
        sources.append(
            {
                "label": build_source_label(meta),
                "filename": meta.get("filename"),
                "page": meta.get("page"),
                "paragraph_index": meta.get("paragraph_index"),
                "chunk_id": meta.get("chunk_id"),
                "start_index": meta.get("start_index"),
                "end_index": meta.get("end_index"),
                "rerank_score": meta.get("rerank_score"),
                "content": doc.page_content,
            }
        )
    return sources



def rewrite_query_if_needed(question: str, chat_history: list[dict], enabled: bool) -> str:
    if not enabled or not chat_history:
        return question
    llm = get_llm()
    prompt = build_query_rewrite_prompt(question, _format_chat_history(chat_history))
    rewritten = llm.invoke(prompt).strip()
    return rewritten or question



def validate_answer_if_needed(answer: str, docs, enabled: bool):
    if not enabled:
        return {"supported": True, "confidence": None, "reason": "disabled"}

    llm = get_llm()
    prompt = build_self_rag_prompt(answer, _format_context(docs))
    raw = llm.invoke(prompt)
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"supported": True, "confidence": None, "reason": raw}
    return parsed



def answer_question(
    retriever,
    question: str,
    chat_history: list[dict] | None = None,
    query_rewrite: bool = True,
    self_rag: bool = True,
    multi_hop: bool = False
):
    history = chat_history or []
    effective_query = rewrite_query_if_needed(question, history, query_rewrite)

    # Lần truy xuất thứ 1
    retrieved_docs = retriever.invoke(effective_query)
    
    # Logic Multi-hop: Truy xuất lần 2 nếu được kích hoạt
    if multi_hop and retrieved_docs:
        llm = get_llm()
        first_context = _format_context(retrieved_docs[:2]) # Lấy 2 đoạn đầu làm gợi ý
        sub_query_prompt = build_multi_hop_prompt(effective_query, first_context)
        sub_query = llm.invoke(sub_query_prompt).strip()
        
        if sub_query and sub_query != effective_query:
            second_docs = retriever.invoke(sub_query)
            # Gộp và loại trùng dựa trên content
            seen_content = {d.page_content for d in retrieved_docs}
            for d in second_docs:
                if d.page_content not in seen_content:
                    retrieved_docs.append(d)

    if not retrieved_docs:
        return {
            "answer": "Không tìm thấy đoạn văn bản liên quan trong tài liệu.",
            "query": effective_query,
            "sources": [],
            "validation": {"supported": False, "confidence": 0.0, "reason": "no_context"},
        }

    reranked_docs = rerank_documents(question, retrieved_docs)
    context = _format_context(reranked_docs)
    prompt = build_prompt(context, question, _format_chat_history(history))

    llm = get_llm()
    answer = llm.invoke(prompt)
    validation = validate_answer_if_needed(answer, reranked_docs, self_rag)

    return {
        "answer": answer,
        "query": effective_query,
        "sources": _format_sources(reranked_docs),
        "validation": validation,
    }