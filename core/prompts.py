from core.utils import detect_vietnamese

def build_prompt(context: str, question: str) -> str:
    if detect_vietnamese(question):
        return f"""Bạn là trợ lý hỏi đáp tài liệu.
Chỉ trả lời dựa trên ngữ cảnh được cung cấp.
Nếu thông tin không đủ, hãy nói rõ là không tìm thấy trong tài liệu.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh:
{context}

Câu hỏi:
{question}

Trả lời:"""

    return f"""You are a document QA assistant.
Answer only based on the given context.
If the answer is not in the document, say so clearly.
Keep the answer concise and accurate.

Context:
{context}

Question:
{question}

Answer:"""


def compare_chunk_strategy_guidance(chunk_size: int, chunk_overlap: int) -> str:
    if chunk_size == 500:
        size_note = "Chunk nhỏ: truy xuất chính xác chi tiết tốt hơn, nhưng dễ thiếu ngữ cảnh tổng thể."
    elif chunk_size == 1000:
        size_note = "Chunk cân bằng: thường là mức ổn cho cả độ chính xác và tốc độ."
    elif chunk_size == 1500:
        size_note = "Chunk lớn hơn: giữ được nhiều ngữ cảnh hơn, nhưng đôi khi truy xuất kém tập trung."
    else:
        size_note = "Chunk rất lớn: ngữ cảnh rộng, nhưng có thể kéo theo thông tin thừa."

    if chunk_overlap == 50:
        overlap_note = "Overlap thấp: nhanh hơn, nhưng dễ mất ý ở ranh giới chunk."
    elif chunk_overlap == 100:
        overlap_note = "Overlap vừa phải: thường là lựa chọn an toàn."
    else:
        overlap_note = "Overlap cao: giữ mạch nội dung tốt hơn, nhưng tăng trùng lặp."

    return f"{size_note} {overlap_note}"