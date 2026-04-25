import json
from core.models import get_llm
from core.prompts import (
    build_prompt,
    build_query_rewrite_prompt,
    build_self_rag_prompt,
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
        sources.append({"label": build_source_label(meta), "filename": meta.get("filename"), "content": doc.page_content})
    return sources

def rewrite_query_if_needed(question: str, chat_history: list[dict], enabled: bool) -> str:
    if not enabled or not chat_history: return question
    llm = get_llm()
    prompt = build_query_rewrite_prompt(question, _format_chat_history(chat_history))
    return llm.invoke(prompt).strip() or question

def validate_answer_if_needed(answer: str, docs, enabled: bool):
    if not enabled: return {"supported": True, "confidence": None, "reason": "disabled"}
    llm = get_llm()
    prompt = build_self_rag_prompt(answer, _format_context(docs))
    try: return json.loads(llm.invoke(prompt))
    except: return {"supported": True}

def answer_question(retriever, question, chat_history=None, query_rewrite=True, self_rag=True, multi_hop=False, knowledge_graph=None):
    history = chat_history or []
    
    # 1. Query Rewrite (Co the dung model nho hon neu muon nhanh)
    effective_query = rewrite_query_if_needed(question, history, query_rewrite)
    
    # 2. Retrieval
    retrieved_docs = retriever.invoke(effective_query)
    
    # 3. Reranking (Gioi han so luong de nhanh hon)
    reranked_docs = rerank_documents(question, retrieved_docs[:10]) 
    
    graph_context = ""
    if knowledge_graph is not None:
        from services.graph_service import get_relevant_entities
        entities = effective_query.split()
        relevant_nodes = get_relevant_entities(knowledge_graph, entities)
        if relevant_nodes: graph_context = "\nGraphRAG: " + ", ".join(relevant_nodes)

    context = _format_context(reranked_docs)
    if graph_context: context = graph_context + "\n\n" + context
    
    prompt = build_prompt(context, question, _format_chat_history(history))
    
    # 4. Trả về kết quả kèm theo LLM object để stream ở app.py
    return {
        "prompt": prompt, 
        "sources": _format_sources(reranked_docs), 
        "docs": reranked_docs
    }
