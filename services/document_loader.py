import io
import os
import tempfile
import hashlib
from typing import List, Tuple

from docx import Document as DocxDocument
from langchain.docstore.document import Document
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.utils import normalize_text


def file_bytes_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def load_pdf_from_path(file_path: str, source_name: str) -> List[Document]:
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    cleaned_docs: List[Document] = []
    for d in docs:
        text = normalize_text(d.page_content)
        if not text:
            continue

        page_num = d.metadata.get("page")
        if page_num is not None:
            page_num += 1

        cleaned_docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": source_name,
                    "source_type": "PDF",
                    "page": page_num if page_num is not None else "N/A",
                },
            )
        )

    return cleaned_docs


def load_pdf_from_bytes(file_bytes: bytes, source_name: str) -> List[Document]:
    # PDFPlumberLoader cần path, nên chỉ tạo file tạm khi thật sự cần
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        return load_pdf_from_path(temp_path, source_name)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def load_docx_from_bytes(file_bytes: bytes, source_name: str) -> List[Document]:
    # python-docx hỗ trợ file-like object
    docx_file = DocxDocument(io.BytesIO(file_bytes))
    docs: List[Document] = []

    # Gom paragraph hợp lệ trước để giảm số object nhỏ lẻ
    paragraphs = []
    for para in docx_file.paragraphs:
        text = normalize_text(para.text)
        if text:
            paragraphs.append(text)

    if not paragraphs:
        return []

    # Gộp thành 1 document lớn, splitter sẽ xử lý tiếp
    combined_text = "\n".join(paragraphs)
    docs.append(
        Document(
            page_content=combined_text,
            metadata={
                "source": source_name,
                "source_type": "DOCX",
            },
        )
    )
    return docs


def split_documents_with_metadata(
    docs: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    split_docs = splitter.split_documents(docs)

    for i, d in enumerate(split_docs, start=1):
        start_idx = d.metadata.get("start_index", 0)
        d.metadata["chunk_id"] = i
        d.metadata["start_index"] = start_idx
        d.metadata["end_index"] = start_idx + len(d.page_content)

    return split_docs


def process_uploaded_file(
    uploaded_file,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[List[Document], List[Document], str]:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    file_bytes = uploaded_file.getvalue()
    file_hash = file_bytes_sha256(file_bytes)

    if suffix == ".pdf":
        raw_docs = load_pdf_from_bytes(file_bytes, uploaded_file.name)
    elif suffix == ".docx":
        raw_docs = load_docx_from_bytes(file_bytes, uploaded_file.name)
    else:
        raise ValueError("Chỉ hỗ trợ PDF và DOCX.")

    if not raw_docs:
        raise ValueError("Không trích xuất được nội dung từ tài liệu.")

    split_docs = split_documents_with_metadata(raw_docs, chunk_size, chunk_overlap)
    return raw_docs, split_docs, file_hash