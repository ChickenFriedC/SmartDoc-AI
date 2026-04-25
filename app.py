import streamlit as st
<<<<<<< Updated upstream
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# ==========================================
# 5. GIAO DIỆN NGƯỜI DÙNG (Mục 5.1)
# ==========================================
st.set_page_config(page_title="SmartDoc AI", layout="wide")

# Thiết kế UI/UX - Color Palette (Mục 5.1.1)
st.markdown("""
    <style>
    .main { background-color: #F8F9FA; }
    .stButton>button { background-color: #007BFF; color: white; border-radius: 5px; }
    .stFileUploader { border: 1px dashed #FFC107; padding: 10px; border-radius: 10px; }
    [data-testid="stSidebar"] { background-color: #2C2F33; color: #FFFFFF; }
    h1 { color: #212529; }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title("⚙️ Cấu hình hệ thống")
st.sidebar.markdown("---")
st.sidebar.write("**Model:** Qwen2.5:7b")
st.sidebar.write("**Embeddings:** Multilingual MPNet")
st.sidebar.write("**Status:** Local Runtime (Ollama)")

st.title("📄 SmartDoc AI - Intelligent Document Q&A System")
st.write("Hệ thống hỏi đáp tài liệu thông minh dựa trên kỹ thuật RAG.")

# ==========================================
# 3.2.1 DOCUMENT PROCESSING FLOW (Mục 3.2.1)
# ==========================================
uploaded_file = st.file_uploader("Tải lên tài liệu PDF để bắt đầu", type="pdf")

if uploaded_file:
    # Lưu file tạm để xử lý
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.status("Đang thực hiện quy trình RAG...", expanded=True) as status:
        # 3.3.1 Document Loader
        st.write("1. Đang tải tài liệu (PDFPlumber)...")
        loader = PDFPlumberLoader("temp.pdf")
        docs = loader.load()

        # 3.3.2 Text Splitter (Mục 4.4.2 & 3.3.2)
        st.write("2. Đang phân mảnh văn bản (RecursiveCharacterTextSplitter)...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,    # Theo mục 3.3.2
            chunk_overlap=100   # Theo mục 3.3.2
        )
        documents = text_splitter.split_documents(docs)

        # 3.3.3 & 4.4.1 Embedding Configuration
        st.write("3. Đang tạo vector embedding (Multilingual MPNet)...")
        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}, # Or 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': True}
        )

        # 3.3.4 & 4.4.2 Retriever Configuration
        st.write("4. Đang khởi tạo Vector Store (FAISS)...")
        vector = FAISS.from_documents(documents, embedder)
        
        retriever = vector.as_retriever(
            search_type="similarity", # Or "mmr" for diverse results
            search_kwargs={
                "k": 3, # Number of chunks to return
                "fetch_k": 20 # Fetch more then filter
            }
        )

        # 3.3.5 & 4.4.3 LLM Configuration
        st.write("5. Đang kết nối mô hình Qwen2.5 (Ollama)...")
        llm = Ollama(
            model="qwen2.5:7b",
            temperature=0.7, # Creativity level
            top_p=0.9, # Nucleus sampling
            repeat_penalty=1.1 # Avoid repetition
        )
        
        status.update(label="Hệ thống đã sẵn sàng!", state="complete", expanded=False)

    # ==========================================
    # 3.4 PROMPT ENGINEERING (Mục 3.4)
    # ==========================================
    user_question = st.text_input("💬 Đặt câu hỏi về nội dung tài liệu của bạn:")
    
    if user_question:
        with st.spinner("Đang suy nghĩ..."):
            # Truy xuất ngữ cảnh
            relevant_docs = retriever.invoke(user_question)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Auto-detect language (Mục 3.4)
            vietnamese_chars = 'aaaaeeeiooouuuuyyyyd'
            is_vietnamese = any(char in user_question.lower() for char in vietnamese_chars)
            
            if is_vietnamese:
                prompt_template = """Su dung ngu canh sau day de tra loi cau hoi.
Neu ban khong biet, chi can noi la ban khong biet.
Tra loi ngan gon (3-4 cau) BAT BUOC bang tieng Viet.

Ngu canh: {context}

Cau hoi: {question}

Tra loi:"""
            else:
                prompt_template = """Use the following context to answer the question.
If you don't know the answer, just say you don't know.
Keep answer concise (3-4 sentences).

Context: {context}

Question: {question}

Answer:"""
            
            full_prompt = prompt_template.format(context=context, question=user_question)
            
            # Gọi LLM sinh câu trả lời
            response = llm.invoke(full_prompt)
            
            # Hiển thị kết quả
            st.subheader("💡 Trả lời:")
            st.write(response)
            
            # Hiển thị nguồn trích dẫn (Câu hỏi 5 - Mục 8.2.5 chuẩn bị trước)
            with st.expander("🔍 Xem nguồn tài liệu được sử dụng"):
                for i, doc in enumerate(relevant_docs):
                    st.markdown(f"**Đoạn trích {i+1}:**")
                    st.info(doc.page_content)

    # Dọn dẹp file tạm
    if os.path.exists("temp.pdf"):
        os.remove("temp.pdf")
else:
    st.info("Vui lòng tải lên một file PDF để bắt đầu trò chuyện với tài liệu.")
=======
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
from ui.main_view import render_answer, render_page_header, render_sources
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

# Giao diện chính (Đã loại bỏ Tabs theo yêu cầu)
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
                knowledge_graph = build_knowledge_graph(split_docs) if st.session_state.use_graph_rag else None

                if st.session_state.retriever_mode == "hybrid": retriever = build_hybrid_retriever(vector_store, split_docs)
                else: retriever, _ = build_base_retrievers(vector_store, split_docs)

                st.session_state.update({"raw_docs": raw_docs, "processed_docs": split_docs, "vector_store": vector_store, "knowledge_graph": knowledge_graph, "retriever": retriever, "uploaded_files_meta": file_summaries, "uploaded_file_names": current_names, "last_chunk_config": current_chunk_cfg})
        except Exception as e:
            st.error(f"Lỗi: {e}")
            st.stop()

if st.session_state.processed_docs:
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
                    st.session_state.knowledge_graph if st.session_state.use_graph_rag else None
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
            answer_placeholder.markdown(full_answer + "▌")
        answer_placeholder.markdown(full_answer)

        # Kiểm tra Self-RAG sau khi đã có câu trả lời
        if st.session_state.self_rag:
            with st.spinner("⚖️ Đang thẩm định độ chính xác..."):
                from services.qa_service import validate_answer_if_needed
                validation = validate_answer_if_needed(full_answer, result_data["docs"], True)
                confidence = validation.get("confidence")
                supported = validation.get("supported")
                st.caption(f"Self-RAG | Supported: {supported} | Confidence: {confidence}")

        if st.session_state.use_graph_rag and st.session_state.knowledge_graph:
            from services.graph_service import visualize_graph
            st.subheader("Sơ đồ GraphRAG")
            fig = visualize_graph(st.session_state.knowledge_graph)
            if fig: st.pyplot(fig)
            
        render_sources(result_data["sources"])
        add_chat_turn(user_question, full_answer, result_data["sources"])
>>>>>>> Stashed changes
