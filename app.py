import hashlib
import time
import streamlit as st

from config import (
    PAGE_TITLE,
    LAYOUT,
    MAX_FILE_SIZE_MB,
    EMBEDDING_MODEL,
)
from core.session import (
    init_session_state,
    add_to_chat_history,
    add_experiment_result,
)
from services.document_loader import process_uploaded_file
from services.qa_service import answer_question
from ui.sidebar import render_sidebar
from ui.main_view import (
    render_page_header,
    render_answer,
    render_sources,
    render_timing_summary,
)

from services.cache_service import (
    build_cache_key,
    get_cache_dir,
    cache_exists,
    save_cache_metadata,
)
from services.vector_store import (
    build_vector_store,
    load_vector_store,
    save_vector_store,
    build_retriever,
)

st.set_page_config(page_title=PAGE_TITLE, layout=LAYOUT)

init_session_state()
render_sidebar()
render_page_header()


def get_uploaded_file_hash(uploaded_file) -> str:
    file_bytes = uploaded_file.getvalue()
    return hashlib.sha256(file_bytes).hexdigest()


uploaded_file = st.file_uploader(
    "Tải lên tài liệu để bắt đầu",
    type=["pdf", "docx"]
)

progress_bar = st.progress(0)
status_box = st.empty()
log_box = st.empty()

process_logs = []


def add_log(message: str):
    process_logs.append(message)
    log_box.markdown("\n".join([f"- {msg}" for msg in process_logs]))


if uploaded_file:
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File quá lớn, vui lòng chọn file < {MAX_FILE_SIZE_MB}MB")
        st.stop()

    try:
        upload_timings = {}

        progress_bar.progress(5)
        status_box.info("Đang kiểm tra file upload...")
        add_log(f"Đã nhận file: `{uploaded_file.name}` ({uploaded_file.size / 1024 / 1024:.2f} MB)")

        t0 = time.perf_counter()
        uploaded_file_hash = get_uploaded_file_hash(uploaded_file)
        upload_timings["Tính hash file"] = time.perf_counter() - t0
        add_log("Đã tính hash file để kiểm tra cache.")
        progress_bar.progress(15)

        current_chunk_config = (
            st.session_state.chunk_size,
            st.session_state.chunk_overlap
        )

        need_reprocess = (
            st.session_state.current_file_hash != uploaded_file_hash
            or st.session_state.last_chunk_config != current_chunk_config
        )

        if need_reprocess:
            status_box.info("Đang đọc và xử lý tài liệu...")
            t1 = time.perf_counter()
            raw_docs, split_docs, file_hash = process_uploaded_file(
                uploaded_file,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap,
            )
            upload_timings["Đọc file + chia chunk"] = time.perf_counter() - t1
            add_log(
                f"Đã xử lý tài liệu với chunk_size={st.session_state.chunk_size}, "
                f"chunk_overlap={st.session_state.chunk_overlap} thành {len(split_docs)} chunks."
            )
            progress_bar.progress(40)

            t2 = time.perf_counter()
            cache_key = build_cache_key(
                file_hash=file_hash,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap,
                embedding_model=EMBEDDING_MODEL,
            )
            cache_dir = get_cache_dir(cache_key)
            cache_hit = cache_exists(cache_dir)
            upload_timings["Kiểm tra cache"] = time.perf_counter() - t2
            add_log(f"Kiểm tra cache: {'Tìm thấy cache' if cache_hit else 'Chưa có cache'}")
            progress_bar.progress(55)

            if cache_hit:
                status_box.info("Đang load vector store từ cache...")
                t3 = time.perf_counter()
                vector_store = load_vector_store(cache_dir, device="cpu")
                upload_timings["Load vector store từ cache"] = time.perf_counter() - t3
                add_log("Đã load vector store từ cache.")
                progress_bar.progress(80)
            else:
                status_box.info("Đang build vector store...")
                t3 = time.perf_counter()
                vector_store = build_vector_store(split_docs, device="cpu")
                upload_timings["Build vector store"] = time.perf_counter() - t3
                add_log("Đã build vector store.")

                t4 = time.perf_counter()
                save_vector_store(vector_store, cache_dir)
                save_cache_metadata(
                    cache_dir,
                    {
                        "file_hash": file_hash,
                        "chunk_size": st.session_state.chunk_size,
                        "chunk_overlap": st.session_state.chunk_overlap,
                        "embedding_model": EMBEDDING_MODEL,
                        "num_chunks": len(split_docs),
                    },
                )
                upload_timings["Lưu cache vector store"] = time.perf_counter() - t4
                add_log("Đã lưu cache vector store.")
                progress_bar.progress(80)

            t5 = time.perf_counter()
            retriever = build_retriever(vector_store)
            upload_timings["Khởi tạo retriever"] = time.perf_counter() - t5
            add_log("Đã khởi tạo retriever.")
            progress_bar.progress(95)

            st.session_state.current_file_hash = file_hash
            st.session_state.current_cache_key = cache_key
            st.session_state.raw_docs = raw_docs
            st.session_state.split_docs = split_docs
            st.session_state.vector_store = vector_store
            st.session_state.retriever = retriever
            st.session_state.file_ready = True
            st.session_state.upload_timings = upload_timings
            st.session_state.last_chunk_config = current_chunk_config
        else:
            add_log("File và chunk config không thay đổi, dùng dữ liệu đã có trong session.")
            progress_bar.progress(95)

        progress_bar.progress(100)
        status_box.success("Hoàn tất xử lý tài liệu.")
        st.success(
            f"Đã xử lý xong tài liệu. "
            f"Số chunk: {len(st.session_state.split_docs)} | "
            f"chunk_size={st.session_state.chunk_size}, "
            f"chunk_overlap={st.session_state.chunk_overlap}"
        )
        render_timing_summary(st.session_state.upload_timings, "⏱ Thời gian xử lý tài liệu")

        user_question = st.text_input(
            "Đặt câu hỏi về nội dung tài liệu của bạn:",
            value=st.session_state.selected_question,
            key="question_input"
        )

        if user_question:
            status_box.info("Đang truy xuất tài liệu liên quan...")
            q0 = time.perf_counter()
            response, relevant_docs = answer_question(
                st.session_state.vector_store,
                user_question
            )
            answer_time = time.perf_counter() - q0

            st.session_state.qa_timings = {
                "Retrieval + LLM inference": answer_time
            }

            add_to_chat_history(user_question, response)
            add_experiment_result(
                {
                    "question": user_question,
                    "chunk_size": st.session_state.chunk_size,
                    "chunk_overlap": st.session_state.chunk_overlap,
                    "num_chunks": len(st.session_state.split_docs),
                    "num_retrieved_docs": len(relevant_docs) if relevant_docs else 0,
                    "processing_time": sum(st.session_state.upload_timings.values()),
                    "answer_time": answer_time,
                }
            )

            status_box.success("Đã nhận được câu trả lời.")
            render_timing_summary(st.session_state.qa_timings, "⏱ Thời gian trả lời câu hỏi")

            if not relevant_docs:
                st.warning("⚠️ Không tìm thấy đoạn văn bản nào liên quan trong tài liệu.")
            else:
                render_answer(response)
                render_sources(relevant_docs)

            st.session_state.selected_question = ""

    except Exception as e:
        status_box.empty()
        progress_bar.empty()
        st.error(f"⚠️ Lỗi hệ thống: {str(e)}")
else:
    progress_bar.empty()
    status_box.empty()
    log_box.empty()
    st.info("Vui lòng tải lên một file PDF hoặc DOCX để bắt đầu.")