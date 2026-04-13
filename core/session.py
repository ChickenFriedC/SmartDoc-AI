import streamlit as st

def init_session_state() -> None:
    defaults = {
        "vector_store": None,
        "retriever": None,
        "processed_docs": None,
        "raw_docs": None,
        "chat_history": [],
        "uploaded_filename": None,
        "uploaded_filetype": None,
        "confirm_clear_history": False,
        "confirm_clear_vector": False,
        "last_chunk_config": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_history() -> None:
    st.session_state.chat_history = []
    st.session_state.confirm_clear_history = False


def clear_vector_store() -> None:
    st.session_state.vector_store = None
    st.session_state.retriever = None
    st.session_state.processed_docs = None
    st.session_state.raw_docs = None
    st.session_state.uploaded_filename = None
    st.session_state.uploaded_filetype = None
    st.session_state.last_chunk_config = None
    st.session_state.confirm_clear_vector = False