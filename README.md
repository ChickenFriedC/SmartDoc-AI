# SmartDoc AI+

SmartDoc AI+ là một ứng dụng RAG (Retrieval-Augmented Generation) hiện đại, cho phép bạn tải lên các tài liệu (PDF, DOCX) và đặt câu hỏi dựa trên nội dung của chúng. Hệ thống sử dụng kết hợp tìm kiếm Hybrid (Vector + BM25) và các kỹ thuật nâng cao như Self-RAG và Query Rewriting để đảm bảo câu trả lời chính xác nhất.

## 🚀 Tính năng chính
- **Hỗ trợ đa định dạng:** PDF, DOCX.
- **Tìm kiếm Hybrid:** Kết hợp sức mạnh của FAISS (Dense) và BM25 (Sparse).
- **Mô hình ngôn ngữ lớn:** Sử dụng **Qwen2.5-7B** qua Ollama.
- **Xử lý tiếng Việt:** Tối ưu hóa với Embedding đa ngôn ngữ.
- **Giao diện trực quan:** Xây dựng bằng Streamlit.

---

## 🛠️ Hướng dẫn cài đặt

### 1. Yêu cầu hệ thống
- Python 3.9 trở lên.
- [Ollama](https://ollama.com/) (Để chạy mô hình LLM cục bộ).

### 2. Cài đặt Ollama và Model
1. Tải và cài đặt Ollama từ [ollama.com](https://ollama.com/).
2. Mở terminal/command prompt và tải mô hình Qwen2.5:
   ```bash
   ollama run qwen2.5:7b
   ```

### 3. Cài đặt Project
1. Clone hoặc tải mã nguồn về máy.
2. Di chuyển vào thư mục dự án:
   ```bash
   cd SmartDoc-AI
   ```
3. Khởi tạo môi trường ảo (Khuyến khích):
   ```bash
   python -m venv venv
   # Kích hoạt trên Windows:
   .\venv\Scripts\activate
   # Kích hoạt trên macOS/Linux:
   source venv/bin/activate
   ```
4. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Cách chạy ứng dụng

Sau khi hoàn tất cài đặt, bạn chạy ứng dụng bằng lệnh:

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8501`

---

## ⚙️ Cấu hình (config.py)
Bạn có thể tùy chỉnh các tham số trong file `config.py`:
- `MODEL_NAME`: Tên model Ollama sử dụng (mặc định: `qwen2.5:7b`).
- `MAX_FILE_SIZE_MB`: Giới hạn dung lượng file tải lên.
- `CHUNK_SIZE` & `CHUNK_OVERLAP`: Cấu hình cắt nhỏ văn bản.

## 📂 Cấu trúc thư mục
- `app.py`: File chạy chính của ứng dụng.
- `core/`: Chứa logic xử lý session, prompts và utils.
- `services/`: Các dịch vụ xử lý tài liệu, vector store và truy vấn.
- `ui/`: Các thành phần giao diện người dùng.
- `config.py`: Các thông số cấu hình hệ thống.
