from core.models import get_llm
from core.prompts import build_prompt

def answer_question(retriever, question: str):
    relevant_docs = retriever.invoke(question)
    if not relevant_docs:
        return None, []

    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    prompt = build_prompt(context, question)
    llm = get_llm()
    response = llm.invoke(prompt)

    return response, relevant_docs