import streamlit as st
import warnings
import os

# Ẩn các cảnh báo từ thư viện để log sạch hơn
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Accessing `__path__`.*")
warnings.filterwarnings("ignore", message=".*torch.classes.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Tránh cảnh báo đa luồng của HuggingFace
os.environ["STREAMLIT_WATCH_LEVEL"] = "error" # Giảm nhiễu từ file watcher

from config import MAX_FILE_SIZE_MB
from core.session import add_chat_turn, init_session_state
from ui.main_view import render_answer, render_page_header, render_sources, render_graph_rag
from ui.sidebar import render_sidebar

# Import các hàm gốc từ services
from services.document_loader import process_uploaded_files
from services.vector_store import build_vector_store
from services.graph_service import build_knowledge_graph
from services.qa_service import answer_question
from services.retrieval_service import build_hybrid_retriever, build_base_retrievers

st.set_page_config(page_title="SmartDoc AI", layout="wide")

init_session_state()
render_sidebar()
render_page_header()

# Giao diện chính
uploaded_files = st.file_uploader("Tải lên tài liệu để bắt đầu", type=["pdf", "docx"], accept_multiple_files=True)
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
            with st.spinner("Đang xử lý tài liệu..."):
                raw_docs, split_docs, file_summaries = process_uploaded_files(uploaded_files, st.session_state.chunk_size, st.session_state.chunk_overlap)
                vector_store = build_vector_store(split_docs, device="cpu")
                # Luôn xây dựng graph nếu tài liệu thay đổi
                knowledge_graph = build_knowledge_graph(split_docs)

                if st.session_state.retriever_mode == "hybrid": retriever = build_hybrid_retriever(vector_store, split_docs)
                else: retriever, _ = build_base_retrievers(vector_store, split_docs)

                st.session_state.update({"raw_docs": raw_docs, "processed_docs": split_docs, "vector_store": vector_store, "knowledge_graph": knowledge_graph, "retriever": retriever, "uploaded_files_meta": file_summaries, "uploaded_file_names": current_names, "last_chunk_config": current_chunk_cfg})
                
                # Chạy ngầm việc xuất báo cáo hiệu năng
                from core.reporter import export_performance_report_async
                export_performance_report_async()
        except Exception as e:
            st.error(f"Lỗi: {e}")
            st.stop()

if st.session_state.processed_docs:
    tab_rag, tab_graph = st.tabs(["RAG Truyền thống", "GraphRAG"])

    with tab_rag:
        user_question = st.text_input("Đặt câu hỏi")
        if user_question:
            status_placeholder = st.empty()
            with status_placeholder.container():
                with st.status("Đang xử lý...", expanded=True) as status:
                    st.write("Đang phân tích câu hỏi...")
                    st.write("Đang truy xuất tài liệu...")
                    st.write("Đang tối ưu hóa nội dung...")
                    
                    result_data = answer_question(
                        st.session_state.retriever, 
                        user_question, 
                        st.session_state.chat_history, 
                        st.session_state.query_rewrite, 
                        st.session_state.self_rag, 
                        st.session_state.multi_hop, 
                        None # Không dùng graph ở tab truyền thống
                    )
                    status.update(label="Hoàn thành!", state="complete", expanded=False)
            
            # Hiển thị Streaming
            st.subheader("Trả lời")
            answer_placeholder = st.empty()
            full_answer = ""
            
            # Sử dụng stream method của LangChain Ollama
            from core.models import get_llm
            llm = get_llm()
            
            # Luồng stream chữ chạy
            for chunk in llm.stream(result_data["prompt"]):
                full_answer += chunk
                answer_placeholder.markdown(full_answer)
            answer_placeholder.markdown(full_answer)

            # Kiểm tra Self-RAG sau khi đã có câu trả lời
            if st.session_state.self_rag:
                with st.spinner("Đang thẩm định độ chính xác..."):
                    from services.qa_service import validate_answer_if_needed
                    validation = validate_answer_if_needed(full_answer, result_data["docs"], True)
                    confidence = validation.get("confidence")
                    supported = validation.get("supported")
                    st.caption(f"Self-RAG | Supported: {supported} | Confidence: {confidence}")

            render_sources(result_data["sources"])
            add_chat_turn(user_question, full_answer, result_data["sources"])

    with tab_graph:
        render_graph_rag()
        if st.session_state.knowledge_graph:
            from services.graph_service import visualize_graph
            st.subheader("Sơ đồ GraphRAG")
            fig = visualize_graph(st.session_state.knowledge_graph)
            if fig: st.pyplot(fig)
else:
    st.info("Vui lòng tải lên một hoặc nhiều file PDF/DOCX để bắt đầu.")
