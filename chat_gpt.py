import openai
from dotenv import load_dotenv
import os

load_dotenv()

with open('prompt.txt', 'r', encoding='utf-8') as file:
    prompt1 = file.read()

import openai

async def chat_gpt(data, prompt=prompt1):
    """
    Асинхронная функция для отправки данных в модель ChatGPT OpenAI и получения результата.
    """
    openai.api_key = os.getenv('OPEN_AI')  # ваш реальный ключ

    try:
        response = await openai.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": str(data)},
            ],
            stream=False
        )
        result = response.choices[0].message.content
        print(f"Получен ответ от ChatGPT: {result[:50]}...")
        return result

    except Exception as e:
        print(f"Ошибка при запросе к OpenAI API: {e}")
        return f"Ошибка: {str(e)}"