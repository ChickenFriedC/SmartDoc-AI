import os
import tempfile
from datetime import datetime
from typing import List

from docx import Document as DocxDocument
from langchain.docstore.document import Document
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.utils import normalize_text



def load_pdf(file_path: str, filename: str) -> List[Document]:
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    cleaned_docs: List[Document] = []

    for page_idx, doc in enumerate(docs, start=1):
        text = normalize_text(doc.page_content)
        if not text:
            continue
        cleaned_docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "filename": filename,
                    "source_type": "PDF",
                    "page": page_idx,
                    "uploaded_at": datetime.now().isoformat(),
                },
            )
        )

    return cleaned_docs



def load_docx(file_path: str, filename: str) -> List[Document]:
    docx_file = DocxDocument(file_path)
    docs: List[Document] = []

    for idx, para in enumerate(docx_file.paragraphs, start=1):
        text = normalize_text(para.text)
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "filename": filename,
                    "source_type": "DOCX",
                    "paragraph_index": idx,
                    "uploaded_at": datetime.now().isoformat(),
                },
            )
        )

    return docs



def split_documents_with_metadata(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    split_docs = splitter.split_documents(docs)

    for i, doc in enumerate(split_docs, start=1):
        start_idx = doc.metadata.get("start_index", 0)
        doc.metadata["chunk_id"] = i
        doc.metadata["start_index"] = start_idx
        doc.metadata["end_index"] = start_idx + len(doc.page_content)

    return split_docs

def process_uploaded_files(uploaded_files, chunk_size: int, chunk_overlap: int):
    all_raw_docs: List[Document] = []
    file_summaries = []

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        try:
            if suffix == ".pdf":
                raw_docs = load_pdf(temp_path, uploaded_file.name)
            elif suffix == ".docx":
                raw_docs = load_docx(temp_path, uploaded_file.name)
            else:
                raise ValueError(f"Không hỗ trợ file: {uploaded_file.name}")

            if not raw_docs:
                raise ValueError(f"Không trích xuất được nội dung từ {uploaded_file.name}")

            all_raw_docs.extend(raw_docs)
            file_summaries.append(
                {
                    "filename": uploaded_file.name,
                    "type": suffix.replace('.', '').upper(),
                    "raw_units": len(raw_docs),
                }
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    split_docs = split_documents_with_metadata(all_raw_docs, chunk_size, chunk_overlap)
    return all_raw_docs, split_docs, file_summaries