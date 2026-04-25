def build_prompt(context: str, question: str, history: str = "") -> str:
    return f"""Bạn là một trợ lý AI thông minh chuyên trả lời câu hỏi dựa trên tài liệu được cung cấp.
Dưới đây là nội dung tài liệu liên quan:
{context}

{f"Lịch sử hội thoại:\n{history}" if history else ""}

Câu hỏi: {question}

Hãy trả lời câu hỏi một cách chi tiết, chính xác dựa trên tài liệu. Nếu tài liệu không chứa thông tin để trả lời, hãy nói rằng bạn không biết, đừng tự bịa ra câu trả lời."""

def build_query_rewrite_prompt(question: str, history: str) -> str:
    return f"""Dựa trên lịch sử hội thoại và câu hỏi mới nhất, hãy viết lại câu hỏi để nó đầy đủ ngữ cảnh và có thể dùng để tìm kiếm độc lập.
Lịch sử hội thoại:
{history}

Câu hỏi mới: {question}

Chỉ trả về 1 câu truy vấn đã viết lại, không giải thích."""

def build_self_rag_prompt(answer: str, context: str) -> str:
    return f"""Thẩm định câu trả lời dựa trên ngữ cảnh được cung cấp.
Ngữ cảnh:
{context}

Câu trả lời: {answer}

Hãy kiểm tra xem câu trả lời có được hỗ trợ bởi ngữ cảnh hay không.
Trả về kết quả dưới dạng JSON với các khóa:
- supported: true nếu câu trả lời được hỗ trợ, false nếu không.
- confidence: mức độ tin cậy từ 0 đến 1.
- reason: giải thích ngắn gọn lý do.

Chỉ trả về JSON."""
