# Tài liệu dự án: SmartDoc AI

Thư mục này chứa toàn bộ các văn bản, báo cáo và tài liệu kỹ thuật liên quan đến hệ thống **SmartDoc AI - Intelligent Document Q&A System**.

## 1. Danh sách tài liệu

| Tên tệp tin | Loại tài liệu | Mô tả nội dung |
|:---|:---|:---|
| `project_report_final.pdf` | Báo cáo chi tiết | Tài liệu chính thức trình bày về cơ sở lý thuyết, thiết kế hệ thống và kết quả thực nghiệm. |
| `SmartDoc AI...CANVA.pptx` | Slide trình chiếu | Bài thuyết trình tóm tắt các điểm nổi bật của dự án để trình bày trước hội đồng/lớp học. |
| `bare_conf.pdf` | Tài liệu kỹ thuật | Bản tóm tắt kỹ thuật theo định dạng bài báo khoa học. |

## 2. Tổng quan kiến trúc (trích từ Báo cáo)

Hệ thống được thiết kế theo mô hình 4 lớp (Multi-layer Architecture):
* **Presentation Layer**: Giao diện người dùng Streamlit.
* **Application Layer**: Logic xử lý RAG sử dụng LangChain.
* **Data Layer**: Cơ sở dữ liệu Vector FAISS và lưu trữ tài liệu thô.
* **Model Layer**: Mô hình ngôn ngữ Qwen2.5 chạy trên nền tảng Ollama.

## 3. Các chức năng trọng tâm đã triển khai

Dựa trên yêu cầu phát triển tại **Chương 8** của báo cáo:
1. **Hỗ trợ đa định dạng**: Xử lý hoàn hảo tệp PDF và DOCX.
2. **Conversational RAG**: Ghi nhớ ngữ cảnh hội thoại và tự động viết lại truy vấn (Query Rewriting).
3. **Hybrid Search**: Kết hợp sức mạnh của tìm kiếm ngữ nghĩa (FAISS) và từ khóa (BM25).
4. **Re-ranking**: Sử dụng Cross-Encoder để tối ưu hóa tính liên quan của kết quả.
5. **Self-RAG**: Hệ thống tự thẩm định câu trả lời, đảm bảo tính trung thực và minh bạch.
6. **Multi-hop Reasoning**: Khả năng suy luận qua nhiều bước để trả lời các câu hỏi phức tạp.

## 4. Mục tiêu hiệu năng đạt được

Hệ thống đã được tối ưu hóa để đáp ứng các tiêu chuẩn khắt khe:
* **Thời gian nạp tài liệu**: 2-5 giây/file.
* **Thời gian tạo Embedding**: 5-10 giây cho mỗi 100 đoạn văn bản.
* **Thời gian xử lý truy vấn**: 1-3 giây.
* **Thời gian sinh câu trả lời**: 3-8 giây (tốc độ streaming).

---
*Lưu ý: Để hiểu rõ hơn về các bước triển khai mã nguồn, vui lòng tham khảo tệp `README.md` tại thư mục gốc của dự án.*
