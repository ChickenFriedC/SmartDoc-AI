import streamlit as st

from config import MAX_FILE_SIZE_MB
from core.session import add_chat_turn, init_session_state
from services.document_loader import process_uploaded_files
from services.qa_service import answer_question
from services.retrieval_service import build_hybrid_retriever, build_base_retrievers
from services.vector_store import build_vector_store
from ui.main_view import render_answer, render_page_header, render_sources
from ui.sidebar import render_sidebar


st.set_page_config(page_title="SmartDoc AI+", layout="wide")
init_session_state()
render_sidebar()
render_page_header()

uploaded_files = st.file_uploader(
    "Tải lên tài liệu để bắt đầu",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)

needs_rebuild = False
if uploaded_files:
    total_size = sum(f.size for f in uploaded_files)
    if total_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"❌ Tổng dung lượng file quá lớn, vui lòng chọn file < {MAX_FILE_SIZE_MB}MB")
        st.stop()

    current_names = [f.name for f in uploaded_files]
    current_chunk_cfg = (st.session_state.chunk_size, st.session_state.chunk_overlap)
    if (
        st.session_state.uploaded_file_names != current_names
        or st.session_state.last_chunk_config != current_chunk_cfg
        or st.session_state.vector_store is None
    ):
        needs_rebuild = True

    if needs_rebuild:
        try:
            with st.spinner("Đang xử lý tài liệu..."):
                raw_docs, split_docs, file_summaries = process_uploaded_files(
                    uploaded_files,
                    chunk_size=st.session_state.chunk_size,
                    chunk_overlap=st.session_state.chunk_overlap,
                )
                vector_store = build_vector_store(split_docs, device="cpu")

                if st.session_state.retriever_mode == "hybrid":
                    retriever = build_hybrid_retriever(vector_store, split_docs)
                else:
                    retriever, _ = build_base_retrievers(vector_store, split_docs)

                st.session_state.raw_docs = raw_docs
                st.session_state.processed_docs = split_docs
                st.session_state.vector_store = vector_store
                st.session_state.retriever = retriever
                st.session_state.uploaded_files_meta = file_summaries
                st.session_state.uploaded_file_names = current_names
                st.session_state.last_chunk_config = current_chunk_cfg
        except Exception as e:
            st.error(f"⚠️ Lỗi hệ thống khi xử lý tài liệu: {e}")
            st.stop()

if st.session_state.uploaded_files_meta:
    st.subheader("Tài liệu đã nạp")
    for item in st.session_state.uploaded_files_meta:
        st.write(f"- {item['filename']} ({item['type']}) | units={item['raw_units']}")

if st.session_state.processed_docs:
    source_options = ["Tất cả"] + sorted({doc.metadata.get("filename") for doc in st.session_state.processed_docs})
    st.session_state.selected_source = st.selectbox(
        "Lọc theo tài liệu",
        source_options,
        index=source_options.index(st.session_state.selected_source) if st.session_state.selected_source in source_options else 0,
    )

    user_question = st.text_input("Đặt câu hỏi về nội dung tài liệu của bạn")

    if user_question:
        with st.spinner("Đang suy nghĩ..."):
            result = answer_question(
                retriever=st.session_state.retriever,
                question=user_question,
                chat_history=st.session_state.chat_history,
                query_rewrite=st.session_state.query_rewrite,
                self_rag=st.session_state.self_rag,
                multi_hop=st.session_state.multi_hop,
            )

        render_answer(result["answer"], result.get("validation"))
        render_sources(result["sources"])
        add_chat_turn(user_question, result["answer"], result["sources"])
else:
    st.info("Vui lòng tải lên một hoặc nhiều file PDF/DOCX để bắt đầu.")