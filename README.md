# SmartDoc AI

Cấu trúc mẫu cho dự án dùng Angular + Django + Ollama + Docker.

## Chạy bằng Docker

```bash
docker compose up --build
```

Frontend: http://localhost:4200  
Backend API: http://localhost:8000/api/health/  
Ollama: http://localhost:11434

## Tải model vào Ollama

Sau khi container Ollama chạy:

```bash
docker exec -it smartdoc_ollama ollama pull qwen2.5:7b
```

## Chạy thủ công

### Backend
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### Frontend
```bash
cd frontend
npm install
npm start
```
