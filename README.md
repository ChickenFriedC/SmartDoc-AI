# SmartDoc AI+ (Intelligent RAG & GraphRAG System)

Hệ thống hỏi đáp tài liệu thông minh kết hợp Vector Search truyền thống và Đồ thị tri thức (GraphRAG).

## Tính năng nổi bật
- **Hybrid Retrieval**: Kết hợp FAISS (tìm kiếm tương đồng) và BM25 (tìm kiếm từ khóa).
- **GraphRAG**: Tự động trích xuất thực thể và quan hệ để giải quyết các câu hỏi yêu cầu suy luận kết nối thông tin.
- **Reranking**: Sử dụng Cross-Encoder để tinh lọc top kết quả chính xác nhất.
- **Self-RAG**: Tự động thẩm định câu trả lời để tránh ảo giác (hallucination).
- **Advanced UI**: Hỗ trợ phản hồi dạng Streaming và hiển thị nguồn dẫn chi tiết.

---

## Hướng dẫn thiết lập chi tiết

### 1. Chuẩn bị môi trường
- **Python**: Phiên bản 3.10 trở lên.
- **Ollama**: Tải và cài đặt tại [ollama.com](https://ollama.com/).

### 2. Cấu hình Mô hình (Ollama)
Mở terminal và chạy các lệnh sau để tải các mô hình cần thiết:
```bash
ollama pull qwen2.5:7b
```

### 3. Cài đặt mã nguồn
```bash
cd SmartDoc-AI
```

### 4. Thiết lập Môi trường ảo
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 5. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 6. Khởi chạy ứng dụng
```bash
streamlit run app.py
```
## Cấu trúc thư mục
```text
SmartDoc-AI/
├── app.py                  # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt        # Danh sách các thư viện phụ thuộc
├── data/                   # Thư mục lưu trữ tài liệu (tùy chọn)
├── documentation/          # Báo cáo dự án và tài liệu liên quan
├── project_report_final.pdf # File báo cáo PDF hoàn chỉnh
└── README.md               # Hướng dẫn sử dụng dự án
```
## Cách hoạt động (Quy trình RAG)
1.  **Tải tài liệu:** Người dùng tải file PDF lên qua giao diện.
2.  **Phân đoạn (Chunking):** Tài liệu được chia thành các đoạn nhỏ (1000 ký tự) để xử lý hiệu quả.
3.  **Vector hóa (Embedding):** Mỗi đoạn văn được chuyển thành một vector số học bằng mô hình MPNet.
4.  **Lưu trữ:** Các vector này được lưu vào FAISS để tìm kiếm nhanh.
5.  **Hỏi đáp:** Khi có câu hỏi, hệ thống tìm các đoạn văn có nội dung gần nhất, gửi kèm vào Prompt để LLM (Qwen2.5) sinh câu trả lời chính xác dựa trên ngữ cảnh đó.

---

## Cấu trúc dự án
- `app.py`: Điểm khởi đầu của ứng dụng Streamlit.
- `services/`: Chứa logic lõi (Loader, Vector Store, RAG, Graph).
- `core/`: Chứa cấu hình Prompt và khởi tạo mô hình.
- `ui/`: Các thành phần giao diện người dùng.
- `data/`: Lưu trữ cache FAISS và các tệp tin tạm.
- `documentation/`: Chứa các tệp LaTeX cho báo cáo khoa học.
