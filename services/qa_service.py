import json
import time
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
        label = build_source_label(meta)
        if "page" in meta:
            label += f" (Trang {meta['page']})"
        elif "paragraph_index" in meta:
            label += f" (Đoạn {meta['paragraph_index']})"
            
        sources.append({
            "label": label, 
            "filename": meta.get("filename"), 
            "content": doc.page_content,
            "page": meta.get("page"),
            "paragraph": meta.get("paragraph_index")
        })
    return sources

def rewrite_query_if_needed(question: str, chat_history: list[dict], enabled: bool) -> str:
    if not enabled or not chat_history: return question
    llm = get_llm()
    prompt = build_query_rewrite_prompt(question, _format_chat_history(chat_history))
    return llm.invoke(prompt).strip() or question

def validate_answer_if_needed(answer: str, docs, enabled: bool):
    if not enabled: return {"supported": True, "confidence": 1.0, "reason": "disabled"}
    llm = get_llm()
    prompt = build_self_rag_prompt(answer, _format_context(docs))
    try:
        import json_repair
        raw_res = llm.invoke(prompt)
        # Sửa lỗi JSON và parse
        data = json_repair.loads(raw_res)
        if isinstance(data, dict):
            return {
                "supported": data.get("supported", True),
                "confidence": data.get("confidence", 0.0)
            }
    except:
        pass
    return {"supported": True, "confidence": 0.5}

def answer_question(retriever, question, chat_history=None, query_rewrite=True, self_rag=True, multi_hop=False, selected_files=None, rerank=True):
    history = chat_history or []
    
    rewrite_t0 = time.time()
    effective_query = rewrite_query_if_needed(question, history, query_rewrite)
    rewrite_time = round(time.time() - rewrite_t0, 2)
    
    retrieval_t0 = time.time()
    retrieved_docs = retriever.invoke(effective_query)
    
    if multi_hop:
        temp_context = _format_context(retrieved_docs[:3])
        from core.prompts import build_multi_hop_prompt
        mh_prompt = build_multi_hop_prompt(question, temp_context)
        llm = get_llm()
        sub_query = llm.invoke(mh_prompt).strip()
        
        if sub_query and "NONE" not in sub_query.upper():
            extra_docs = retriever.invoke(sub_query)
            retrieved_docs.extend(extra_docs)
            unique_docs = []
            seen_content = set()
            for d in retrieved_docs:
                if d.page_content not in seen_content:
                    unique_docs.append(d)
                    seen_content.add(d.page_content)
            retrieved_docs = unique_docs

    if rerank:
        processed_docs = rerank_documents(question, retrieved_docs, selected_files=selected_files)
    else:
        from services.retrieval_service import filter_docs_by_metadata
        processed_docs = filter_docs_by_metadata(retrieved_docs, selected_files)[:5]
        
    pure_query_time = round(time.time() - retrieval_t0, 2)
    
    context = _format_context(processed_docs)
    prompt = build_prompt(context, question, _format_chat_history(history))
    
    # Bổ sung ràng buộc cuối cùng để tránh nhảy sang tiếng Trung
    if "á" in prompt or "đ" in prompt or "ơ" in prompt:
        prompt += "\nLƯU Ý: TRẢ LỜI HOÀN TOÀN BẰNG TIẾNG VIỆT. KHÔNG DÙNG TIẾNG TRUNG."
    
    return {
        "prompt": prompt, 
        "sources": _format_sources(processed_docs), 
        "docs": processed_docs,
        "pure_query_time": pure_query_time,
        "rewrite_time": rewrite_time
    }
