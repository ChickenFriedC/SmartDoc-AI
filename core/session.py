import streamlit as st


def init_session_state() -> None:
    defaults = {
        "vector_store": None,
        "retriever": None,
        "processed_docs": [],
        "raw_docs": [],
        "chat_history": [],
        "uploaded_files_meta": [],
        "uploaded_file_names": [],
        "confirm_clear_history": False,
        "confirm_clear_vector": False,
        "last_chunk_config": None,
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "retriever_mode": "hybrid",
        "selected_source": "Tất cả",
        "query_rewrite": True,
        "self_rag": True,
        "multi_hop": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_chat_turn(question: str, answer: str, sources: list[dict]) -> None:
    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": answer,
            "sources": sources,
        }
    )


def clear_history() -> None:
    st.session_state.chat_history = []
    st.session_state.confirm_clear_history = False


def clear_vector_store() -> None:
    st.session_state.vector_store = None
    st.session_state.retriever = None
    st.session_state.processed_docs = []
    st.session_state.raw_docs = []
    st.session_state.uploaded_files_meta = []
    st.session_state.uploaded_file_names = []
    st.session_state.last_chunk_config = None
    st.session_state.selected_source = "Tất cả"
    st.session_state.confirm_clear_vector = False