import re
import hashlib

def normalize_text(text: str) -> str:
    if not text:
        return ""
    # Loại bỏ các ký tự điều khiển và khoảng trắng thừa
    text = text.replace("\x00", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_source_label(meta: dict) -> str:
    filename = meta.get("filename", "Không rõ")
    page = meta.get("page")
    para = meta.get("paragraph_index")
    
    if page:
        return f"{filename} (Trang {page})"
    if para:
        return f"{filename} (Đoạn {para})"
    return filename

def generate_id(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()
