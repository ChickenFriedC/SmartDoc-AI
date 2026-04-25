def build_prompt(context: str, question: str, history: str = "") -> str:
    vietnamese_chars = 'aaaaeeeiooouuuuyyyyd'
    is_vietnamese = any(char in question.lower() for char in vietnamese_chars)
    
    if is_vietnamese:
        return f"""Su dung ngu canh sau day de tra loi cau hoi.
Neu ban khong biet, chi can noi la ban khong biet.
Tra loi ngan gon (3-4 cau) BAT BUOC bang tieng Viet.

Ngu canh: {context}

Cau hoi: {question}

Tra loi: """
    else:
        return f"""Use the following context to answer the question.
If you don't know the answer, just say you don't know.
Keep answer concise (3-4 sentences).

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
