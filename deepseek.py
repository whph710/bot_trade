import asyncio
from openai import AsyncOpenAI

# Загружаем промпт из файла
with open('prompt.txt', 'r', encoding='utf-8') as file:
    prompt1 = file.read()
with open('prompt_check.txt', 'r', encoding='utf-8') as file:
    prompt2 = file.read()


async def deep_seek(data, prompt=prompt1):
    """
    Асинхронная функция для отправки данных в модель DeepSeek и получения результата.

    Args:
        data: Данные для отправки модели (будут преобразованы в строку)
        prompt: Системный промпт для модели (по умолчанию из файла prompt.txt)

    Returns:
        str: Содержимое ответа от модели
    """
    # Здесь должен быть ваш API ключ
    api_key = "sk-3b36df2fa6d843a1877fc8dd69a3956a"  # Замените на ваш реальный API ключ

    try:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

        # Отправляем запрос к API
        response = await client.chat.completions.create(
            model="deepseek-chat",#deepseek-reasoner
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": str(data)},
            ],
            stream=False
        )

        # Получаем и возвращаем результат
        result = response.choices[0].message.content
        print(f"Получен ответ от DeepSeek: {result[:50]}...")  # Выводим первые 50 символов для отладки
        return result

    except Exception as e:
        print(f"Ошибка при запросе к DeepSeek API: {e}")
        return f"Ошибка: {str(e)}"




async def deep_seek_check(data, prompt=prompt2):

    # Здесь должен быть ваш API ключ
    api_key = "sk-3b36df2fa6d843a1877fc8dd69a3956a"  # Замените на ваш реальный API ключ

    try:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

        # Отправляем запрос к API
        response = await client.chat.completions.create(
            model="deepseek-reasoner",# deepseek-reasoner/deepseek-chat
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": str(data)},
            ],
            stream=False
        )

        # Получаем и возвращаем результат
        result = response.choices[0].message.content
        #print(f"Получен ответ от DeepSeek: {result[:50]}...")  # Выводим первые 50 символов для отладки
        return result

    except Exception as e:
        print(f"Ошибка при запросе к DeepSeek API: {e}")
        return f"Ошибка: {str(e)}"