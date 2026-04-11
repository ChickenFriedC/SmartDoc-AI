import streamlit as st
import os
import hashlib
import pickle
import faiss
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

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
# Hàm tiện ích: tạo hash cho file PDF
# ==========================================
def get_file_hash(file_path):
    BUF_SIZE = 65536
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

# ==========================================
# 3.2.1 DOCUMENT PROCESSING FLOW (Mục 3.2.1)
# ==========================================
uploaded_file = st.file_uploader("Tải lên tài liệu PDF để bắt đầu", type="pdf")

if uploaded_file:
    # Kiểm tra dung lượng file (File size validation)
    if uploaded_file.size > 200 * 1024 * 1024:  # 200MB
        st.error("❌ File quá lớn, vui lòng chọn file < 200MB")
    else:
        # Lưu file tạm để xử lý
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("✅ PDF uploaded successfully!")  # Success notification

        # Tạo hash để nhận diện file duy nhất
        file_hash = get_file_hash("temp.pdf")
        cache_dir = "src/data/cache"
        os.makedirs(cache_dir, exist_ok=True)
        faiss_path = os.path.join(cache_dir, f"{file_hash}.faiss")
        retriever_path = os.path.join(cache_dir, f"{file_hash}.pkl")

        try:
            if os.path.exists(faiss_path) and os.path.exists(retriever_path):
                st.info("⚡ Đang tải lại dữ liệu từ cache...")
                index = faiss.read_index(faiss_path)
                with open(retriever_path, "rb") as f:
                    vector = pickle.load(f)
                retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3, "fetch_k": 20})
            else:
                with st.status("Đang thực hiện quy trình RAG...", expanded=True) as status:
                    # 3.3.1 Document Loader
                    st.write("1. Đang tải tài liệu (PDFPlumber)...")
                    loader = PDFPlumberLoader("temp.pdf")
                    docs = loader.load()
                    if not docs:
                        st.error("❌ Không thể đọc nội dung từ file PDF.")
                        st.stop()

                    # 3.3.2 Text Splitter
                    st.write("2. Đang phân mảnh văn bản (RecursiveCharacterTextSplitter)...")
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    documents = text_splitter.split_documents(docs)
                    if not documents:
                        st.error("❌ Không có văn bản nào được trích xuất từ PDF.")
                        st.stop()

                    # 3.3.3 Embedding Configuration
                    st.write("3. Đang tạo vector embedding (Multilingual MPNet)...")
                    embedder = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                        model_kwargs={'device': 'cpu'},  # đổi sang 'cuda' nếu có GPU
                        encode_kwargs={'normalize_embeddings': True}
                    )

                    # 3.3.4 Retriever Configuration
                    st.write("4. Đang khởi tạo Vector Store (FAISS)...")
                    vector = FAISS.from_documents(documents, embedder)
                    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3, "fetch_k": 20})

                    # Lưu retriever và FAISS index vào cache
                    faiss.write_index(vector.index, faiss_path)
                    with open(retriever_path, "wb") as f:
                        pickle.dump(vector, f)

                    status.update(label="Hệ thống đã sẵn sàng!", state="complete", expanded=False)

            # 3.3.5 LLM Configuration
            llm = Ollama(model="qwen2.5:7b", temperature=0.7, top_p=0.9, repeat_penalty=1.1)

            # ==========================================
            # 3.4 PROMPT ENGINEERING (Mục 3.4)
            # ==========================================
            user_question = st.text_input("💬 Đặt câu hỏi về nội dung tài liệu của bạn:")
            
            if user_question:
                with st.spinner("Đang suy nghĩ..."):
                    try:
                        relevant_docs = retriever.invoke(user_question)
                        if not relevant_docs:
                            st.warning("⚠️ Không tìm thấy đoạn văn bản nào liên quan trong tài liệu.")
                        else:
                            context = "\n\n".join([doc.page_content for doc in relevant_docs])

                            # Auto-detect language
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

                            response = llm.invoke(full_prompt)

                            st.subheader("💡 Trả lời:")
                            st.write(response)

                            with st.expander("🔍 Xem nguồn tài liệu được sử dụng"):
                                for i, doc in enumerate(relevant_docs):
                                    st.markdown(f"**Đoạn trích {i+1}:**")
                                    st.info(doc.page_content)
                    except Exception as e:
                        st.error(f"⚠️ Lỗi xử lý câu hỏi: {str(e)}")

        except Exception as e:
            st.error(f"⚠️ Lỗi hệ thống: {str(e)}")

        # Dọn dẹp file tạm
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")
else:
    st.info("Vui lòng tải lên một file PDF để bắt đầu trò chuyện với tài liệu.")