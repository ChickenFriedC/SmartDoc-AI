# SmartDoc AI - Intelligent Document Q&A System

Hệ thống hỏi đáp tài liệu thông minh dựa trên kỹ thuật RAG (Retrieval-Augmented Generation), sử dụng mô hình ngôn ngữ lớn Qwen2.5 và cơ sở dữ liệu vector FAISS.

## 1. Yêu cầu hệ thống
* **Python**: 3.9 trở lên
* **Ollama**: Đã cài đặt và đang chạy local
* **RAM**: Tối thiểu 8GB (Khuyến nghị 16GB)
* **GPU**: Tùy chọn (Hệ thống tự động phát hiện CUDA)

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
*(Lưu ý: Có thể sử dụng bản `1.5b` để đạt tốc độ cao hơn trên CPU)*

## 3. Vận hành ứng dụng

Chạy ứng dụng bằng Streamlit:
```bash
streamlit run app.py
```
Sau đó truy cập vào địa chỉ: `http://localhost:8501`

## 4. Chạy Kiểm thử (Test Cases)

Hệ thống đi kèm với bộ kiểm thử tự động cho 3 kịch bản chính (Factual, Reasoning, Out-of-context):

```bash
python tests/test_scenarios.py
```

## 5. Mục tiêu hiệu năng (Targets)
Hệ thống được tối ưu hóa để đạt các mốc thời gian:
* **PDF Loading**: 2-5 giây
* **Embedding Generation**: 5-10 giây cho 100 chunks
* **Query Processing**: 1-3 giây
* **Answer Generation**: 3-8 giây

## 6. Tính năng nâng cao
Hệ thống hỗ trợ các kỹ thuật Advanced RAG có thể bật/tắt trong Sidebar:
* **Hybrid Search**: Kết hợp Vector và BM25.
* **Re-ranking**: Sử dụng Cross-Encoder để tăng độ chính xác.
* **Self-RAG**: Tự động thẩm định câu trả lời và chấm điểm tin cậy.
* **Multi-hop Reasoning**: Suy luận đa bước cho câu hỏi phức tạp.
