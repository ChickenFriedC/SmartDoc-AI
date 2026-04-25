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
    st.sidebar.title("System Configuration")
    st.sidebar.markdown("---")
    st.sidebar.write("Model: Qwen2.5:7b")
    st.sidebar.write("Embeddings: Multilingual MPNet")
    st.sidebar.write("Status: Local Runtime (Ollama)")
    
    st.sidebar.write("") # Spacer

    st.sidebar.markdown("### Chunking strategy")
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

    st.sidebar.write("") # Spacer
    st.sidebar.markdown("### Retrieval configuration")
    st.session_state.retriever_mode = st.sidebar.selectbox(
        "Retriever mode",
        ["Vector", "Hybrid"],
        index=0,
    ).lower()

    st.session_state.query_rewrite = st.sidebar.checkbox("Query rewrite", value=False)
    st.session_state.rerank = st.sidebar.checkbox("Re-ranking (Cross-Encoder)", value=False)
    st.session_state.self_rag = st.sidebar.checkbox("Self-RAG validation", value=False)
    st.session_state.multi_hop = st.sidebar.checkbox("Multi-hop reasoning", value=False)

    st.sidebar.write("") # Spacer
    st.sidebar.markdown("### Document Filtering")
    all_files = st.session_state.get("uploaded_file_names", [])
    if all_files:
        st.session_state.selected_files = st.sidebar.multiselect(
            "Select documents to search",
            options=all_files,
            default=all_files,
            help="Filter search results to specific documents."
        )
    else:
        st.sidebar.caption("No documents uploaded yet.")

    st.sidebar.write("") # Spacer
    st.sidebar.markdown("### Data management")
    if st.sidebar.button("Clear History", use_container_width=True):
        st.session_state.confirm_clear_history = True
    if st.session_state.confirm_clear_history:
        st.sidebar.warning("Are you sure you want to clear chat history?")
        c1, c2 = st.sidebar.columns(2)
        if c1.button("Confirm", key="confirm_hist", use_container_width=True):
            clear_history()
            st.rerun()
        if c2.button("Cancel", key="cancel_hist", use_container_width=True):
            st.session_state.confirm_clear_history = False
            st.rerun()

    st.sidebar.write("")
    if st.sidebar.button("Clear Vector Store", use_container_width=True):
        st.session_state.confirm_clear_vector = True
    if st.session_state.confirm_clear_vector:
        st.sidebar.warning("Are you sure you want to clear all processed documents?")
        c1, c2 = st.sidebar.columns(2)
        if c1.button("Confirm", key="confirm_vec", use_container_width=True):
            clear_vector_store()
            st.rerun()
        if c2.button("Cancel", key="cancel_vec", use_container_width=True):
            st.session_state.confirm_clear_vector = False
            st.rerun()

    st.sidebar.write("") # Spacer
    st.sidebar.markdown("### Chat History")
    if not st.session_state.chat_history:
        st.sidebar.caption("No questions asked yet.")
    else:
        for idx, turn in enumerate(reversed(st.session_state.chat_history), start=1):
            with st.sidebar.expander(f"Q{len(st.session_state.chat_history)-idx+1}: {turn['question'][:30]}..."):
                st.write(f"**Question:** {turn['question']}")
                st.write(f"**Answer:** {turn['answer'][:100]}...")
