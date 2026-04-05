import requests
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status


VIETNAMESE_CHARACTERS = set(
    'ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ'
    'íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ'
)


def should_answer_in_vietnamese(text: str) -> bool:
    normalized = text.lower().strip()
    if any(ch in VIETNAMESE_CHARACTERS for ch in normalized):
        return True

    common_vietnamese_words = {
        'là', 'gì', 'sao', 'không', 'có', 'được', 'cho', 'tôi', 'mình', 'bạn',
        'và', 'cần', 'này', 'đó', 'giúp', 'việt', 'tiếng'
    }
    tokens = set(normalized.replace('?', ' ').replace('.', ' ').replace(',', ' ').split())
    return len(tokens.intersection(common_vietnamese_words)) > 0


@api_view(['GET'])
def health_check(request):
    return Response({
        'status': 'ok',
        'service': 'django-backend',
        'ollama_model': settings.OLLAMA_MODEL,
    })


@api_view(['POST'])
def ask_ai(request):
    prompt = request.data.get('prompt', '').strip()
    if not prompt:
        return Response({'error': 'Trường prompt là bắt buộc.'}, status=status.HTTP_400_BAD_REQUEST)

    answer_in_vietnamese = should_answer_in_vietnamese(prompt)

    if answer_in_vietnamese:
        final_prompt = (
            'Bạn là trợ lý AI cho người dùng Việt Nam. '
            'Hãy luôn trả lời hoàn toàn bằng tiếng Việt, rõ ràng, đúng trọng tâm. '
            'Không dùng tiếng Trung hoặc ngôn ngữ khác trừ khi người dùng yêu cầu.\n\n'
            f'Câu hỏi của người dùng: {prompt}\n\n'
            'Trả lời bằng tiếng Việt:'
        )
    else:
        final_prompt = (
            'You are a helpful AI assistant. '
            'Answer in the same language as the user unless they ask otherwise.\n\n'
            f'User question: {prompt}\n\n'
            'Answer:'
        )

    payload = {
        'model': settings.OLLAMA_MODEL,
        'prompt': final_prompt,
        'stream': False,
        'options': {
            'temperature': 0.2,
        },
    }

    try:
        ollama_response = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,
        )
        ollama_response.raise_for_status()
        data = ollama_response.json()
        return Response({
            'prompt': prompt,
            'response': data.get('response', '').strip(),
            'model': settings.OLLAMA_MODEL,
            'language': 'vi' if answer_in_vietnamese else 'auto',
        })
    except requests.RequestException as exc:
        return Response(
            {
                'error': 'Không thể kết nối tới Ollama.',
                'details': str(exc),
            },
            status=status.HTTP_502_BAD_GATEWAY,
        )