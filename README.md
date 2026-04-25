# SmartDoc AI - Intelligent Document Q&A System

Hệ thống hỏi đáp tài liệu thông minh dựa trên kỹ thuật RAG (Retrieval-Augmented Generation), sử dụng mô hình ngôn ngữ lớn Qwen2.5 và cơ sở dữ liệu vector FAISS. Hệ thống được thiết kế để xử lý đa tài liệu với độ chính xác cao thông qua cơ chế Re-ranking và Self-RAG.

## 1. Yêu cầu hệ thống
* **Python**: 3.9 trở lên
* **Ollama**: Đã cài đặt và đang chạy local
* **RAM**: Tối thiểu 8GB (Khuyến nghị 16GB)
* **GPU**: Tùy chọn (Hệ thống tự động phát hiện và sử dụng CUDA để tăng tốc)

## 2. Hướng dẫn cài đặt

### Bước 1: Khởi tạo môi trường ảo
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### Bước 2: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 3: Tải mô hình Ollama
Đảm bảo Ollama đang chạy, sau đó thực hiện lệnh:
```bash
ollama pull qwen2.5:7b
```
*(Lưu ý: Nếu máy không có GPU, hãy sử dụng bản `qwen2.5:1.5b` để đạt tốc độ Answer Generation mục tiêu 3-8 giây)*

## 3. Vận hành ứng dụng

Khởi chạy giao diện người dùng bằng Streamlit:
```bash
streamlit run app.py
```
Sau đó truy cập vào địa chỉ: `http://localhost:8501`

## 4. Kiểm thử hệ thống (Automated Testing)

Hệ thống cung cấp bộ kiểm thử tự động tích hợp logic xác thực từ khóa (Keyword Validation) để đảm bảo độ chính xác của câu trả lời:

```bash
python tests/test_scenarios.py
```
**Các kịch bản kiểm thử:**
1. **Factual Question**: Kiểm tra khả năng trích xuất thông tin kỹ thuật chính xác.
2. **Complex Reasoning**: Kiểm tra khả năng tổng hợp dữ liệu và đưa ra nhận định.
3. **Out-of-context**: Kiểm tra khả năng từ chối trả lời (ngăn chặn ảo giác) khi thông tin không có trong tài liệu.

## 5. Các tính năng cốt lõi
* **Đa định dạng**: Hỗ trợ PDF (sử dụng PDFPlumber) và DOCX.
* **Hybrid Search**: Kết hợp Vector Search (ngữ nghĩa) và BM25 (từ khóa).
* **Re-ranking**: Sử dụng Cross-Encoder (`ms-marco-MiniLM`) để xếp hạng lại tài liệu.
* **Self-RAG**: Tự động thẩm định câu trả lời và cung cấp điểm số tin cậy (Confidence Score).
* **Multi-hop Reasoning**: Suy luận qua nhiều bước để giải quyết các câu hỏi phức tạp.
* **Metadata Filtering**: Cho phép người dùng lọc tìm kiếm theo từng tệp tin cụ thể.

## 6. Mục tiêu hiệu năng
Hệ thống được tối ưu hóa sâu (Throttling UI, CPU Threading, CUDA Auto-detect) để đạt:
* **Document Loading**: 2-5 giây.
* **Embedding Generation**: 5-10 giây / 100 chunks.
* **Query Processing**: 1-3 giây (bao gồm cả Reranking).
* **Answer Generation**: 3-8 giây (tốc độ streaming mượt mà).
