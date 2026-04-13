import os
import tempfile
import streamlit as st

from config import PAGE_TITLE, LAYOUT, MAX_FILE_SIZE_MB, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from core.session import init_session_state
from services.document_loader import process_uploaded_file
from services.vector_store import build_vector_store, build_retriever
from services.qa_service import answer_question
from ui.sidebar import render_sidebar
from ui.main_view import render_page_header, render_answer, render_sources

st.set_page_config(page_title=PAGE_TITLE, layout=LAYOUT)

init_session_state()
render_sidebar()
render_page_header()

uploaded_file = st.file_uploader(
    "Tải lên tài liệu để bắt đầu",
    type=["pdf", "docx"]
)

if uploaded_file:
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"❌ File quá lớn, vui lòng chọn file < {MAX_FILE_SIZE_MB}MB")
        st.stop()

    try:
        raw_docs, split_docs = process_uploaded_file(
            uploaded_file,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        )

        vector_store = build_vector_store(split_docs, device="cpu")
        retriever = build_retriever(vector_store)

        user_question = st.text_input("💬 Đặt câu hỏi về nội dung tài liệu của bạn:")

        if user_question:
            with st.spinner("Đang suy nghĩ..."):
                response, relevant_docs = answer_question(retriever, user_question)

                if not relevant_docs:
                    st.warning("⚠️ Không tìm thấy đoạn văn bản nào liên quan trong tài liệu.")
                else:
                    render_answer(response)
                    render_sources(relevant_docs)

    except Exception as e:
        st.error(f"⚠️ Lỗi hệ thống: {str(e)}")
else:
    st.info("Vui lòng tải lên một file PDF hoặc DOCX để bắt đầu.")