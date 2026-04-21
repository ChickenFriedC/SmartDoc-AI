from core.utils import detect_vietnamese


def build_prompt(context: str, question: str, chat_history: str = "") -> str:
    if detect_vietnamese(question):
        return f"""
Bạn là trợ lý hỏi đáp tài liệu.
Chỉ trả lời dựa trên ngữ cảnh được cung cấp.
Nếu tài liệu không đủ thông tin, hãy nói rõ là không tìm thấy trong tài liệu.
Ưu tiên trả lời bằng tiếng Việt, ngắn gọn nhưng chính xác.
Nếu có lịch sử hội thoại, hãy dùng nó để hiểu câu hỏi follow-up, nhưng không bịa thêm dữ kiện.

Lịch sử hội thoại:
{chat_history}

Ngữ cảnh truy xuất:
{context}

Câu hỏi:
{question}

Trả lời:
""".strip()

    return f"""
You are a document QA assistant.
Answer only from the retrieved context.
If the document does not contain enough information, say so clearly.
Use chat history only to resolve follow-up questions.
Keep the answer concise and accurate.

Chat history:
{chat_history}

Retrieved context:
{context}

Question:
{question}

Answer:
""".strip()


def build_query_rewrite_prompt(question: str, chat_history: str = "") -> str:
    if detect_vietnamese(question):
        return f"""
Hãy viết lại câu hỏi tìm kiếm cho hệ thống RAG.
Mục tiêu: làm câu hỏi rõ nghĩa hơn, đầy đủ chủ ngữ hơn, tự chứa ngữ cảnh nếu đây là follow-up.
Chỉ trả về 1 câu truy vấn đã viết lại, không giải thích.

Lịch sử hội thoại:
{chat_history}

Câu hỏi hiện tại:
{question}
""".strip()

    return f"""
Rewrite the user question for retrieval.
Make it self-contained if it depends on prior turns.
Return only one rewritten query.

Chat history:
{chat_history}

Current question:
{question}
""".strip()


def build_self_rag_prompt(answer: str, context: str) -> str:
    return f"""
You are validating an answer generated from retrieved context.
Decide whether the answer is fully supported by the context.
Return JSON with keys: supported (true/false), confidence (0 to 1), reason.

Context:
{context}

Answer:
{answer}
""".strip()


def build_multi_hop_prompt(question: str, first_context: str) -> str:
    return f"""
Dựa trên câu hỏi và thông tin tìm được ở bước 1, hãy tạo ra 1 câu hỏi phụ để tìm kiếm thêm thông tin bị thiếu nhằm trả lời trọn vẹn câu hỏi gốc.
Chỉ trả về 1 câu hỏi phụ, không giải thích. Nếu thấy thông tin bước 1 đã đủ, hãy trả về chính câu hỏi gốc.

Câu hỏi gốc: {question}
Thông tin bước 1: {first_context}

Câu hỏi phụ:""".strip()