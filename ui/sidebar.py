import streamlit as st

from config import CHUNK_SIZE_OPTIONS, CHUNK_OVERLAP_OPTIONS
from core.session import (
    clear_history,
    clear_vector_store,
    clear_experiment_results,
    reset_clear_flags,
)
from services.cache_service import clear_cache_dir


def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Tùy chọn")

        st.subheader("🧩 Chunk Strategy")

        chunk_size = st.selectbox(
            "Chunk Size",
            CHUNK_SIZE_OPTIONS,
            index=CHUNK_SIZE_OPTIONS.index(st.session_state.chunk_size)
            if st.session_state.chunk_size in CHUNK_SIZE_OPTIONS else 1
        )

        chunk_overlap = st.selectbox(
            "Chunk Overlap",
            CHUNK_OVERLAP_OPTIONS,
            index=CHUNK_OVERLAP_OPTIONS.index(st.session_state.chunk_overlap)
            if st.session_state.chunk_overlap in CHUNK_OVERLAP_OPTIONS else 1
        )

        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap

        st.caption(
            f"Cấu hình hiện tại: chunk_size={st.session_state.chunk_size}, "
            f"chunk_overlap={st.session_state.chunk_overlap}"
        )

        st.markdown("---")
        st.subheader("🕘 Lịch sử hội thoại")

        history = st.session_state.get("chat_history", [])

        if not history:
            st.info("Chưa có lịch sử hội thoại.")
        else:
            for i, item in enumerate(reversed(history), start=1):
                real_index = len(history) - i
                question = item.get("question", "")
                answer = item.get("answer", "")

                title = question[:40] + "..." if len(question) > 40 else question
                with st.expander(f"Câu hỏi {len(history) - i + 1}: {title}"):
                    st.markdown(f"**Câu hỏi:** {question}")
                    st.markdown(f"**Trả lời:** {answer}")

                    if st.button("Xem lại", key=f"history_btn_{real_index}"):
                        st.session_state.selected_question = question
                        st.session_state.question_input = question

        st.markdown("---")
        st.subheader("🗑 Quản lý dữ liệu")

        if not st.session_state.confirm_clear_history:
            if st.button("Clear History", use_container_width=True):
                reset_clear_flags()
                st.session_state.confirm_clear_history = True
        else:
            st.warning("Bạn có chắc muốn xóa toàn bộ lịch sử chat?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Xác nhận", key="confirm_clear_history_btn", use_container_width=True):
                    clear_history()
                    st.success("Đã xóa toàn bộ lịch sử chat.")
                    st.rerun()
            with col2:
                if st.button("Hủy", key="cancel_clear_history_btn", use_container_width=True):
                    st.session_state.confirm_clear_history = False
                    st.rerun()

        if not st.session_state.confirm_clear_vector:
            if st.button("Clear Vector Store", use_container_width=True):
                reset_clear_flags()
                st.session_state.confirm_clear_vector = True
        else:
            st.warning("Bạn có chắc muốn xóa tài liệu đã upload và vector store hiện tại?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Xác nhận xóa vector", key="confirm_clear_vector_btn", use_container_width=True):
                    clear_vector_store()
                    st.success("Đã xóa vector store.")
                    st.rerun()
            with col2:
                if st.button("Hủy", key="cancel_clear_vector_btn", use_container_width=True):
                    st.session_state.confirm_clear_vector = False
                    st.rerun()

        if not st.session_state.confirm_clear_cache:
            if st.button("Clear Cache", use_container_width=True):
                reset_clear_flags()
                st.session_state.confirm_clear_cache = True
        else:
            st.warning("Bạn có chắc muốn xóa toàn bộ cache?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Xác nhận xóa cache", key="confirm_clear_cache_btn", use_container_width=True):
                    clear_cache_dir()
                    st.success("Đã xóa toàn bộ cache.")
                    st.rerun()
            with col2:
                if st.button("Hủy", key="cancel_clear_cache_btn", use_container_width=True):
                    st.session_state.confirm_clear_cache = False
                    st.rerun()

        st.markdown("---")
        st.subheader("🧪 Kết quả thử nghiệm")

        if st.session_state.experiment_results:
            for i, result in enumerate(reversed(st.session_state.experiment_results), start=1):
                with st.expander(
                    f"Lần {len(st.session_state.experiment_results) - i + 1}: "
                    f"size={result['chunk_size']}, overlap={result['chunk_overlap']}"
                ):
                    st.write(f"**Câu hỏi:** {result['question']}")
                    st.write(f"**Số chunks:** {result['num_chunks']}")
                    st.write(f"**Số đoạn retrieve:** {result['num_retrieved_docs']}")
                    st.write(f"**Thời gian xử lý:** {result['processing_time']:.3f}s")
                    st.write(f"**Thời gian trả lời:** {result['answer_time']:.3f}s")
        else:
            st.info("Chưa có kết quả thử nghiệm.")

        if st.button("Clear Experiment Results", use_container_width=True):
            clear_experiment_results()
            st.success("Đã xóa kết quả thử nghiệm.")
            st.rerun()