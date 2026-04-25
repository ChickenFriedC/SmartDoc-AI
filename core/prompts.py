def build_prompt(context: str, question: str, history: str = "") -> str:
    vn_chars = "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"
    is_vietnamese = any(char in question.lower() for char in vn_chars)
    
    if is_vietnamese:
        return f"""Bạn là một trợ lý ảo chuyên nghiệp. Sử dụng duy nhất TIẾNG VIỆT để trả lời.
Nhiệm vụ: Dựa vào ngữ cảnh dưới đây để trả lời câu hỏi một cách chính xác.
Yêu cầu:
- Nếu thông tin không có trong ngữ cảnh, hãy trả lời 'Tôi không biết'.
- Trả lời ngắn gọn, súc tích (3-4 câu).
- Tuyệt đối không sử dụng ngôn ngữ khác ngoài Tiếng Việt.

Ngữ cảnh: {context}
Câu hỏi: {question}
Trả lời: """
    else:
        return f"""You are a professional AI assistant. Use ONLY ENGLISH to answer.
Task: Answer the question based on the provided context.
Requirements:
- If the answer is not in the context, say 'I don't know'.
- Keep it concise (3-4 sentences).
- Do not use any language other than English.

Context: {context}
Question: {question}
Answer:"""

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

Trả về kết quả dưới dạng JSON với các khóa: supported (true/false), confidence (0-1), reason."""

def build_multi_hop_prompt(question: str, context: str) -> str:
    return f"""Dựa trên câu hỏi và thông tin đã tìm thấy, hãy xác định xem có cần thêm thông tin gì để trả lời đầy đủ không. 
Nếu cần, hãy viết một câu truy vấn mới để tìm kiếm phần thông tin còn thiếu đó.
Thông tin hiện tại:
{context}

Câu hỏi gốc: {question}

Chỉ trả về câu truy vấn mới nếu thực sự cần thiết, nếu không hãy trả về 'NONE'."""
