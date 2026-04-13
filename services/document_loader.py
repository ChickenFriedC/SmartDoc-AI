import os
import tempfile
from typing import List, Tuple

from docx import Document as DocxDocument
from langchain.docstore.document import Document
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.utils import normalize_text


def load_pdf(file_path: str) -> List[Document]:
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    cleaned_docs = []
    for d in docs:
        page_num = d.metadata.get("page", None)
        if page_num is not None:
            page_num += 1

        cleaned_docs.append(
            Document(
                page_content=normalize_text(d.page_content),
                metadata={
                    "source": file_path,
                    "source_type": "PDF",
                    "page": page_num if page_num is not None else "N/A",
                },
            )
        )
    return cleaned_docs


def load_docx(file_path: str) -> List[Document]:
    docx_file = DocxDocument(file_path)
    docs: List[Document] = []

    for idx, para in enumerate(docx_file.paragraphs, start=1):
        text = normalize_text(para.text)
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "source_type": "DOCX",
                        "paragraph_index": idx,
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
        d.metadata["chunk_id"] = i
        start_idx = d.metadata.get("start_index", 0)
        d.metadata["start_index"] = start_idx
        d.metadata["end_index"] = start_idx + len(d.page_content)

    return split_docs


def process_uploaded_file(uploaded_file, chunk_size: int, chunk_overlap: int) -> Tuple[List[Document], List[Document]]:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name

    try:
        if suffix == ".pdf":
            raw_docs = load_pdf(temp_path)
        elif suffix == ".docx":
            raw_docs = load_docx(temp_path)
        else:
            raise ValueError("Chỉ hỗ trợ PDF và DOCX.")

        if not raw_docs:
            raise ValueError("Không trích xuất được nội dung từ tài liệu.")

        split_docs = split_documents_with_metadata(raw_docs, chunk_size, chunk_overlap)
        return raw_docs, split_docs
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)