import streamlit as st

from config import (
    CHUNK_OVERLAP_OPTIONS,
    CHUNK_SIZE_OPTIONS,
    DEFAULT_MULTI_HOP,
    DEFAULT_QUERY_REWRITE,
    DEFAULT_RETRIEVER_MODE,
    DEFAULT_SELF_RAG,
)
from core.session import clear_history, clear_vector_store



def render_sidebar():
    st.sidebar.title("⚙️ Cấu hình hệ thống")
    st.sidebar.markdown("---")
    st.sidebar.write("**Model:** Qwen2.5:7b")
    st.sidebar.write("**Embeddings:** Multilingual MPNet")
    st.sidebar.write("**Status:** Local Runtime (Ollama)")

    st.sidebar.markdown("### Chunk strategy")
    st.session_state.chunk_size = st.sidebar.selectbox(
        "Chunk size",
        CHUNK_SIZE_OPTIONS,
        index=CHUNK_SIZE_OPTIONS.index(st.session_state.chunk_size) if st.session_state.chunk_size in CHUNK_SIZE_OPTIONS else 1,
    )
    st.session_state.chunk_overlap = st.sidebar.selectbox(
        "Chunk overlap",
        CHUNK_OVERLAP_OPTIONS,
        index=CHUNK_OVERLAP_OPTIONS.index(st.session_state.chunk_overlap) if st.session_state.chunk_overlap in CHUNK_OVERLAP_OPTIONS else 1,
    )

    st.sidebar.markdown("### Retrieval")
    st.session_state.retriever_mode = st.sidebar.selectbox(
        "Retriever mode",
        ["vector", "hybrid"],
        index=1 if st.session_state.retriever_mode == "hybrid" else 0,
    )

    st.session_state.query_rewrite = st.sidebar.checkbox("Query rewrite", value=st.session_state.get("query_rewrite", DEFAULT_QUERY_REWRITE))
    st.session_state.self_rag = st.sidebar.checkbox("Self-RAG validation", value=st.session_state.get("self_rag", DEFAULT_SELF_RAG))
    st.session_state.multi_hop = st.sidebar.checkbox("Multi-hop reasoning", value=st.session_state.get("multi_hop", DEFAULT_MULTI_HOP))

    st.sidebar.markdown("### Quản lý dữ liệu")
    if st.sidebar.button("Clear History"):
        st.session_state.confirm_clear_history = True
    if st.session_state.confirm_clear_history:
        st.sidebar.warning("Bạn có chắc muốn xóa toàn bộ lịch sử chat?")
        c1, c2 = st.sidebar.columns(2)
        if c1.button("Xác nhận xóa lịch sử", use_container_width=True):
            clear_history()
            st.rerun()
        if c2.button("Hủy xóa lịch sử", use_container_width=True):
            st.session_state.confirm_clear_history = False
            st.rerun()

    if st.sidebar.button("Clear Vector Store", use_container_width=True):
        st.session_state.confirm_clear_vector = True
    if st.session_state.confirm_clear_vector:
        st.sidebar.warning("Bạn có chắc muốn xóa toàn bộ tài liệu đã xử lý?")
        c1, c2 = st.sidebar.columns(2)
        if c1.button("Xác nhận xóa vector", use_container_width=True):
            clear_vector_store()
            st.rerun()
        if c2.button("Hủy xóa vector", use_container_width=True):
            st.session_state.confirm_clear_vector = False
            st.rerun()

    st.sidebar.markdown("### Lịch sử hội thoại")
    if not st.session_state.chat_history:
        st.sidebar.caption("Chưa có câu hỏi nào.")
    else:
        for idx, turn in enumerate(reversed(st.session_state.chat_history), start=1):
            with st.sidebar.expander(f"#{idx} {turn['question'][:40]}"):
                st.write("**Q:**", turn["question"])
                st.write("**A:**", turn["answer"])