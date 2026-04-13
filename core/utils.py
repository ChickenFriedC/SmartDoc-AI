import hashlib
import re

def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_vietnamese(question: str) -> bool:
    vietnamese_pattern = r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]"
    return bool(re.search(vietnamese_pattern, question.lower()))


def highlight_snippet(text: str, query: str, max_terms: int = 6) -> str:
    safe_text = text
    words = re.findall(r"\w+", query.lower())
    words = [w for w in words if len(w) >= 3][:max_terms]

    for w in words:
        pattern = re.compile(rf"({re.escape(w)})", re.IGNORECASE)
        safe_text = pattern.sub(r"<mark>\1</mark>", safe_text)

    return safe_text


def get_file_hash(file_path: str) -> str:
    buf_size = 65536
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()