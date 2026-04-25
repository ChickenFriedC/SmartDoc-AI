import streamlit as st
import warnings
import os
import time

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Accessing `__path__`.*")
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*instantiate class '__path__._path'.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_WATCH_LEVEL"] = "error"

from config import MAX_FILE_SIZE_MB
from core.session import add_chat_turn, init_session_state
from ui.main_view import render_answer, render_page_header, render_sources
from ui.sidebar import render_sidebar

from services.document_loader import process_uploaded_files
from services.vector_store import build_vector_store
from services.qa_service import answer_question
from services.retrieval_service import build_graph_hybrid_retriever, build_hybrid_retriever, build_base_retrievers
from core.reporter import export_performance_report_async

st.set_page_config(page_title="SmartDoc AI", layout="wide")

init_session_state()
render_page_header()

uploaded_files = st.file_uploader("Browse files or drag-and-drop PDF/DOCX", type=["pdf", "docx"], accept_multiple_files=True)
needs_rebuild = False

if uploaded_files:
    total_size = sum(f.size for f in uploaded_files)
    if total_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"Tổng dung lượng file quá lớn < {MAX_FILE_SIZE_MB}MB")
        st.stop()
    
    current_names = [f.name for f in uploaded_files]
    current_chunk_cfg = (st.session_state.chunk_size, st.session_state.chunk_overlap)
    
    if st.session_state.uploaded_file_names != current_names or st.session_state.last_chunk_config != current_chunk_cfg or st.session_state.vector_store is None:
        needs_rebuild = True

    if needs_rebuild:
        try:
            st.success("PDF uploaded successfully!")
            
            t0 = time.time()
            with st.spinner("Splitting..."):
                raw_docs, split_docs, file_summaries = process_uploaded_files(uploaded_files, st.session_state.chunk_size, st.session_state.chunk_overlap)
            t1 = time.time()
            
            with st.spinner("Creating embeddings..."):
                vector_store = build_vector_store(split_docs, device=None)
            t2 = time.time()

            if st.session_state.retriever_mode == "hybrid":
                retriever = build_hybrid_retriever(vector_store, split_docs)
            elif st.session_state.retriever_mode == "graph_hybrid":
                retriever = build_graph_hybrid_retriever(vector_store, split_docs)
            else:
                retriever, _ = build_base_retrievers(vector_store, split_docs)

            load_time = round(t1 - t0, 2)
            embed_time = round(t2 - t1, 2)

            st.session_state.update({
                "raw_docs": raw_docs, 
                "processed_docs": split_docs, 
                "vector_store": vector_store, 
                "retriever": retriever, 
                "uploaded_files_meta": file_summaries, 
                "uploaded_file_names": current_names, 
                "last_chunk_config": current_chunk_cfg,
                "perf_load_time": load_time,
                "perf_embed_time": embed_time
            })
            
            export_performance_report_async({
                "load_time": load_time,
                "embed_time": embed_time,
                "chunk_size": st.session_state.chunk_size,
                "chunk_overlap": st.session_state.chunk_overlap,
                "retriever_mode": st.session_state.retriever_mode
            }, st.session_state.quality_metrics)
        except Exception as e:
            st.error(f"Lỗi: {e}")
            st.stop()

render_sidebar()

if st.session_state.processed_docs:
    st.markdown("---")
    st.subheader("Ask a Question")
    user_question = st.text_input("Nhập câu hỏi vào đây và nhấn Enter", key="query_input")
    if user_question:
        st.subheader("Response")
        with st.container():
            with st.spinner("Processing your query..."):
                result_data = answer_question(
                    st.session_state.retriever, 
                    user_question, 
                    st.session_state.chat_history, 
                    st.session_state.query_rewrite, 
                    st.session_state.self_rag, 
                    st.session_state.multi_hop,
                    selected_files=st.session_state.get("selected_files"),
                    rerank=st.session_state.get("rerank", True)
                )

            answer_placeholder = st.empty()
            full_answer = ""
            
            from core.models import get_llm
            llm = get_llm()
            
            gen_t0 = time.time()
            chunk_count = 0
            for chunk in llm.stream(result_data["prompt"]):
                full_answer += chunk
                chunk_count += 1
                if chunk_count % 5 == 0:
                    answer_placeholder.markdown(full_answer)
            answer_placeholder.markdown(full_answer)
            gen_t1 = time.time()

            supported, confidence = "N/A", "N/A"
            validation_time = 0
            if st.session_state.self_rag:
                with st.spinner("Thẩm định..."):
                    val_t0 = time.time()
                    from services.qa_service import validate_answer_if_needed
                    validation = validate_answer_if_needed(full_answer, result_data["docs"], True)
                    supported = validation.get("supported", "N/A")
                    confidence = validation.get("confidence", "N/A")
                    
                    st.session_state.quality_metrics.append({
                        "supported": supported,
                        "confidence": confidence
                    })
                    
                    validation_time = round(time.time() - val_t0, 2)
                st.caption(f"Self-RAG | Supported: {supported} | Confidence: {confidence}")

            query_time = result_data["pure_query_time"]
            gen_time = round(gen_t1 - gen_t0, 2)

            export_performance_report_async({
                "load_time": st.session_state.get("perf_load_time", "N/A"),
                "embed_time": st.session_state.get("perf_embed_time", "N/A"),
                "query_time": query_time,
                "gen_time": gen_time,
                "rewrite_time": result_data["rewrite_time"],
                "validation_time": validation_time
            }, st.session_state.quality_metrics)

            render_sources(result_data["sources"])
            add_chat_turn(user_question, full_answer, result_data["sources"])
else:
    st.info("Vui lòng tải lên một hoặc nhiều file PDF/DOCX để bắt đầu.")
