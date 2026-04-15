from core.models import get_llm
from core.prompts import build_prompt


def answer_question(vector_store, question: str, top_k: int = 5, score_threshold: float = 0.8):
    results = vector_store.similarity_search_with_score(question, k=top_k)

    if not results:
        return None, []

    filtered_docs = []
    for doc, score in results:
        if score <= score_threshold:
            doc.metadata["score"] = float(score)
            filtered_docs.append(doc)

    if not filtered_docs:
        fallback_results = sorted(results, key=lambda x: x[1])[:3]
        filtered_docs = []
        for doc, score in fallback_results:
            doc.metadata["score"] = float(score)
            filtered_docs.append(doc)

    context = "\n\n".join(doc.page_content for doc in filtered_docs)
    prompt = build_prompt(context, question)

    llm = get_llm()
    response = llm.invoke(prompt)

    return response, filtered_docs