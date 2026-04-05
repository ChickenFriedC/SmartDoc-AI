# SmartDoc AI - Intelligent Document Q&A System
Hệ thống hỏi đáp tài liệu thông minh (RAG - Retrieval-Augmented Generation) giúp người dùng tương tác và trích xuất thông tin từ tài liệu PDF một cách nhanh chóng và chính xác.
## 🛠 Công nghệ sử dụng
- **Frontend:** [Streamlit](https://streamlit.io/)
- **RAG Framework:** [LangChain](https://www.langchain.com/)
- **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
- **Embeddings:** `paraphrase-multilingual-mpnet-base-v2` (Sentence-Transformers)
- **LLM Engine:** [Ollama](https://ollama.com/) (Model: `qwen2.5:7b`)
- **Document Loader:** [PDFPlumber](https://github.com/jsvine/pdfplumber)
## Yêu cầu hệ thống
1.  **Python 3.10+**
2.  **Ollama runtime** 
3.  **pip package manager** 
## ⚙️ Cài đặt và Chạy ứng dụng

### 1. Clone repository (hoặc tải mã nguồn)
```bash
git clone <https://github.com/ChickenFriedC/SmartDoc-AI.git>
cd SmartDoc-AI
```

### 2. Thiết lập môi trường ảo (Khuyên dùng)
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Cài đặt thư viện cần thiết
```bash
pip install -r requirements.txt
```
### 4. Cài đặt Ollama
```bash
ollama pull qwen2.5:7b
```

### 5. Chạy ứng dụng
```bash
streamlit run app.py
```
## 📁 Cấu trúc thư mục
```text
SmartDoc-AI/
├── app.py                  # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt        # Danh sách các thư viện phụ thuộc
├── data/                   # Thư mục lưu trữ tài liệu (tùy chọn)
├── documentation/          # Báo cáo dự án và tài liệu liên quan
├── project_report_final.pdf # File báo cáo PDF hoàn chỉnh
└── README.md               # Hướng dẫn sử dụng dự án
```
## 💡 Cách hoạt động (Quy trình RAG)
1.  **Tải tài liệu:** Người dùng tải file PDF lên qua giao diện.
2.  **Phân đoạn (Chunking):** Tài liệu được chia thành các đoạn nhỏ (1000 ký tự) để xử lý hiệu quả.
3.  **Vector hóa (Embedding):** Mỗi đoạn văn được chuyển thành một vector số học bằng mô hình MPNet.
4.  **Lưu trữ:** Các vector này được lưu vào FAISS để tìm kiếm nhanh.
5.  **Hỏi đáp:** Khi có câu hỏi, hệ thống tìm các đoạn văn có nội dung gần nhất, gửi kèm vào Prompt để LLM (Qwen2.5) sinh câu trả lời chính xác dựa trên ngữ cảnh đó.

