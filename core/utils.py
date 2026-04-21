import hashlib
import re
from typing import Iterable


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_vietnamese(text: str) -> bool:
    vietnamese_pattern = (
        r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
        r"íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]"
    )
    return bool(re.search(vietnamese_pattern, text.lower()))


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_source_label(metadata: dict) -> str:
    source = metadata.get("filename", metadata.get("source", "Unknown"))
    source_type = metadata.get("source_type", "FILE")
    if source_type == "PDF":
        page = metadata.get("page", "N/A")
        return f"{source} - trang {page}"
    if source_type == "DOCX":
        para = metadata.get("paragraph_index", "N/A")
        return f"{source} - đoạn {para}"
    return str(source)


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result