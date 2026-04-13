import os
import re
import tempfile
from typing import List, Tuple, Dict, Any

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
# CẤU HÌNH TRANG
# ==========================================
st.set_page_config(page_title="SmartDoc AI+", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #F8F9FA; }
    .stButton>button { border-radius: 8px; }
    .stFileUploader { border: 1px dashed #FFC107; padding: 10px; border-radius: 10px; }
    [data-testid="stSidebar"] { background-color: #2C2F33; color: #FFFFFF; }
    h1 { color: #212529; }
    .source-box {
        background: #f6f8fa;
        border-left: 4px solid #007BFF;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        color: #111111;
    }
    mark {
        background-color: #fff59d;
        padding: 0.1em 0.2em;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# SESSION STATE
# ==========================================
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


init_session_state()


# ==========================================
# HÀM PHỤ TRỢ
# ==========================================
def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_vietnamese(question: str) -> bool:
    vietnamese_pattern = r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]"
    return bool(re.search(vietnamese_pattern, question.lower()))


def highlight_snippet(text: str, query: str, max_terms: int = 6) -> str:
    """
    Highlight các từ khóa trong câu hỏi xuất hiện trong source text.
    """
    safe_text = text
    words = re.findall(r"\w+", query.lower())
    words = [w for w in words if len(w) >= 3][:max_terms]

    for w in words:
        pattern = re.compile(rf"({re.escape(w)})", re.IGNORECASE)
        safe_text = pattern.sub(r"<mark>\1</mark>", safe_text)

    return safe_text


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


def build_source_label(doc: Document) -> str:
    source_type = doc.metadata.get("source_type", "unknown").upper()

    if source_type == "PDF":
        page = doc.metadata.get("page", "N/A")
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        start = doc.metadata.get("start_index", "N/A")
        end = doc.metadata.get("end_index", "N/A")
        return f"PDF | Trang {page} | Chunk {chunk_id} | Vị trí {start}-{end}"

    if source_type == "DOCX":
        para = doc.metadata.get("paragraph_index", "N/A")
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        start = doc.metadata.get("start_index", "N/A")
        end = doc.metadata.get("end_index", "N/A")
        return f"DOCX | Đoạn {para} | Chunk {chunk_id} | Vị trí {start}-{end}"

    return "Nguồn không xác định"


def load_pdf(file_path: str) -> List[Document]:
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    cleaned_docs = []
    for d in docs:
        page_num = d.metadata.get("page", None)
        if page_num is not None:
            page_num = page_num + 1  # PDFPlumber thường bắt đầu từ 0
        cleaned_docs.append(
            Document(
                page_content=normalize_text(d.page_content),
                metadata={
                    "source": file_path,
                    "source_type": "PDF",
                    "page": page_num if page_num is not None else "N/A",
                },
            )
        )
    return cleaned_docs


def load_docx(file_path: str) -> List[Document]:
    """
    Hỗ trợ DOCX bằng python-docx.
    Tách theo paragraph để giữ vị trí nguồn tốt hơn.
    """
    docx_file = DocxDocument(file_path)
    docs: List[Document] = []

    for idx, para in enumerate(docx_file.paragraphs, start=1):
        text = normalize_text(para.text)
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "source_type": "DOCX",
                        "paragraph_index": idx,
                    },
                )
            )

    return docs


def split_documents_with_metadata(
    docs: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    split_docs = splitter.split_documents(docs)

    for i, d in enumerate(split_docs, start=1):
        d.metadata["chunk_id"] = i
        start_idx = d.metadata.get("start_index", 0)
        d.metadata["start_index"] = start_idx
        d.metadata["end_index"] = start_idx + len(d.page_content)

    return split_docs


@st.cache_resource(show_spinner=False)
def get_embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def get_llm():
    return Ollama(
        model="qwen2.5:7b",
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
    )


def process_uploaded_file(
    uploaded_file,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[List[Document], List[Document], FAISS]:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name

    try:
        if suffix == ".pdf":
            raw_docs = load_pdf(temp_path)
        elif suffix == ".docx":
            raw_docs = load_docx(temp_path)
        else:
            raise ValueError("Chỉ hỗ trợ PDF và DOCX.")

        if not raw_docs:
            raise ValueError("Không trích xuất được nội dung từ tài liệu.")

        split_docs = split_documents_with_metadata(
            raw_docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        embedder = get_embedder()
        vector_store = FAISS.from_documents(split_docs, embedder)

        return raw_docs, split_docs, vector_store
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def build_prompt(context: str, question: str) -> str:
    if detect_vietnamese(question):
        return f"""Bạn là trợ lý hỏi đáp tài liệu.
Chỉ trả lời dựa trên ngữ cảnh được cung cấp.
Nếu thông tin không đủ, hãy nói rõ là không tìm thấy trong tài liệu.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh:
{context}

Câu hỏi:
{question}

Trả lời:"""

    return f"""You are a document QA assistant.
Answer only based on the given context.
If the answer is not in the document, say so clearly.
Keep the answer concise and accurate.

Context:
{context}

Question:
{question}

Answer:"""


def compare_chunk_strategy_guidance(chunk_size: int, chunk_overlap: int) -> str:
    """
    Gợi ý báo cáo định tính để phục vụ mục 8.2.4.
    Không bịa số liệu đo đạc.
    """
    if chunk_size == 500:
        size_note = "Chunk nhỏ: truy xuất chính xác chi tiết tốt hơn, nhưng dễ thiếu ngữ cảnh tổng thể."
    elif chunk_size == 1000:
        size_note = "Chunk cân bằng: thường là mức ổn cho cả độ chính xác và tốc độ."
    elif chunk_size == 1500:
        size_note = "Chunk lớn hơn: giữ được nhiều ngữ cảnh hơn, nhưng đôi khi truy xuất kém tập trung."
    else:
        size_note = "Chunk rất lớn: ngữ cảnh rộng, nhưng có thể kéo theo thông tin thừa."

    if chunk_overlap == 50:
        overlap_note = "Overlap thấp: nhanh hơn, nhưng dễ mất ý ở ranh giới chunk."
    elif chunk_overlap == 100:
        overlap_note = "Overlap vừa phải: thường là lựa chọn an toàn."
    else:
        overlap_note = "Overlap cao: giữ mạch nội dung tốt hơn, nhưng tăng trùng lặp."

    return f"{size_note} {overlap_note}"


# ==========================================
# SIDEBAR
# ==========================================
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
# GIAO DIỆN CHÍNH
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