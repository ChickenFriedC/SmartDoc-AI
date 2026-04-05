const API_BASE_URL = window.location.hostname === 'localhost'
  ? 'http://localhost:8000/api'
  : '/api';

const askButton = document.getElementById('askButton');
const promptInput = document.getElementById('prompt');
const result = document.getElementById('result');

askButton.addEventListener('click', async () => {
  const prompt = promptInput.value.trim();
  if (!prompt) {
    result.textContent = 'Vui lòng nhập câu hỏi.';
    return;
  }

  result.textContent = 'Đang xử lý...';

  try {
    const response = await fetch(`${API_BASE_URL}/ask/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });

    const data = await response.json();
    result.textContent = response.ok ? data.response : (data.error || 'Có lỗi xảy ra.');
  } catch (error) {
    result.textContent = `Không gọi được backend: ${error.message}`;
  }
});
