import streamlit as st


def init_session_state() -> None:
    defaults = {
        "vector_store": None,
        "retriever": None,
        "processed_docs": None,
        "raw_docs": None,
        "split_docs": None,
        "chat_history": [],
        "uploaded_filename": None,
        "uploaded_filetype": None,
        "confirm_clear_history": False,
        "confirm_clear_vector": False,
        "confirm_clear_cache": False,
        "last_chunk_config": None,
        "current_file_hash": None,
        "current_cache_key": None,
        "file_ready": False,
        "upload_timings": {},
        "qa_timings": {},
        "selected_question": "",
        "question_input": "",

        # thêm cho chunk strategy
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "experiment_results": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_to_chat_history(question: str, answer: str) -> None:
    if not question or not answer:
        return

    history = st.session_state.chat_history
    if history:
        last = history[-1]
        if last["question"] == question and last["answer"] == answer:
            return

    history.append({
        "question": question,
        "answer": answer,
    })

    if len(history) > 20:
        st.session_state.chat_history = history[-20:]

def clear_history() -> None:
    st.session_state.chat_history = []
    st.session_state.selected_question = ""
    st.session_state.question_input = ""
    st.session_state.confirm_clear_history = False


def clear_vector_store() -> None:
    st.session_state.vector_store = None
    st.session_state.retriever = None
    st.session_state.processed_docs = None
    st.session_state.raw_docs = None
    st.session_state.split_docs = None
    st.session_state.uploaded_filename = None
    st.session_state.uploaded_filetype = None
    st.session_state.last_chunk_config = None
    st.session_state.current_file_hash = None
    st.session_state.current_cache_key = None
    st.session_state.file_ready = False
    st.session_state.upload_timings = {}
    st.session_state.qa_timings = {}
    st.session_state.selected_question = ""
    st.session_state.question_input = ""
    st.session_state.confirm_clear_vector = False


def clear_experiment_results() -> None:
    st.session_state.experiment_results = []


def reset_clear_flags() -> None:
    st.session_state.confirm_clear_history = False
    st.session_state.confirm_clear_vector = False
    st.session_state.confirm_clear_cache = False

def add_experiment_result(result: dict) -> None:
    """
    Lưu kết quả thử nghiệm chunk strategy
    """
    if "experiment_results" not in st.session_state:
        st.session_state.experiment_results = []

    st.session_state.experiment_results.append(result)

    # Giữ tối đa 20 kết quả gần nhất
    if len(st.session_state.experiment_results) > 20:
        st.session_state.experiment_results = st.session_state.experiment_results[-20:]