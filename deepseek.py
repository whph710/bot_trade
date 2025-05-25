import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Загружаем промпт из файла
with open('prompt.txt', 'r', encoding='utf-8') as file:
    prompt1 = file.read()

import asyncio
import os
import httpx
from openai import AsyncOpenAI


async def deep_seek(data, prompt=prompt1, timeout=60, max_tokens=4000, max_retries=3):
    """
    Асинхронная функция для отправки данных в модель DeepSeek с обработкой connection error.

    Args:
        data: Данные для отправки модели (будут преобразованы в строку)
        prompt: Системный промпт для модели (по умолчанию из файла prompt.txt)
        timeout: Таймаут запроса в секундах (по умолчанию 60)
        max_tokens: Максимальное количество токенов в ответе
        max_retries: Количество попыток при ошибке соединения

    Returns:
        str: Содержимое ответа от модели
    """

    api_key = os.getenv('DEEPSEEK')

    if not api_key:
        return "Ошибка: не найден API ключ DEEPSEEK"

    # Настройки для более стабильного соединения
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        verify=True  # Проверка SSL сертификатов
    )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        http_client=http_client
    )

    for attempt in range(max_retries):
        try:
            print(f"Попытка {attempt + 1}/{max_retries}: отправляем запрос к DeepSeek...")
            print(f"Длина данных: {len(str(data))} символов")

            response = await client.chat.completions.create(
                model="deepseek-chat",  # Более стабильная модель
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": str(data)},
                ],
                stream=False,
                max_tokens=max_tokens,
                temperature=0.7
            )

            result = response.choices[0].message.content
            print(f"Успешно получен ответ: {result[:50]}...")

            # Закрываем клиент
            await http_client.aclose()
            return result

        except Exception as e:
            error_msg = str(e)
            print(f"Попытка {attempt + 1} неудачна: {error_msg}")

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Экспоненциальная задержка: 1, 2, 4 сек
                print(f"Ожидание {wait_time} сек перед следующей попыткой...")
                await asyncio.sleep(wait_time)
            else:
                await http_client.aclose()
                return f"Ошибка после {max_retries} попыток: {error_msg}"

    await http_client.aclose()
    return "Неизвестная ошибка"


# Функция для проверки соединения с API
async def test_deepseek_connection():
    """
    Тестирует соединение с DeepSeek API
    """
    api_key = os.getenv('DEEPSEEK')

    if not api_key:
        print("❌ API ключ DEEPSEEK не найден")
        return False

    print("🔍 Проверяем соединение с DeepSeek API...")

    try:
        # Простой HTTP клиент для проверки доступности
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("https://api.deepseek.com")
            print(f"✅ Сервер доступен, статус: {response.status_code}")
    except Exception as e:
        print(f"❌ Сервер недоступен: {e}")
        return False

    # Тестовый запрос к API
    try:
        http_client = httpx.AsyncClient(timeout=30.0)
        api_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            http_client=http_client
        )

        response = await api_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Привет"}],
            max_tokens=10
        )

        await http_client.aclose()
        print("✅ API работает корректно")
        return True

    except Exception as e:
        print(f"❌ Ошибка API: {e}")
        return False


# Альтернативная версия с стримингом для больших ответов
async def deep_seek_streaming(data, trade1, prompt=prompt1, timeout=120):
    """
    Версия со стримингом - показывает прогресс генерации ответа
    """

    api_key = os.getenv('DEEPSEEK')

    if not api_key:
        return "Ошибка: не найден API ключ DEEPSEEK"

    try:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=timeout
        )

        print("Начинаем генерацию ответа...")

        stream = await client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": trade1 + prompt},
                {"role": "user", "content": data},
            ],
            stream=True,
            max_tokens=4000
        )

        result = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                result += content
                print(content, end='', flush=True)  # Показываем прогресс в реальном времени

        print("\n--- Генерация завершена ---")
        return result

    except asyncio.TimeoutError:
        error_msg = f"Таймаут запроса ({timeout} сек)"
        print(error_msg)
        return f"Ошибка: {error_msg}"

    except Exception as e:
        error_msg = f"Ошибка при запросе к DeepSeek API: {e}"
        print(error_msg)
        return f"Ошибка: {str(e)}"

# a =""" {'long': ['AUCTIONUSDT', 'AXSUSDT', 'BCHUSDT', 'BSVUSDT', 'DASHUSDT', 'DEXEUSDT', 'ETCUSDT', 'ILVUSDT', 'LTCUSDT', 'OMNIUSDT', 'TAOUSDT'], 'short': ['BCHUSDT', 'FXSUSDT', 'NXPCUSDT']}"""
# b = asyncio.run(deep_seek_streaming(a))

# print(b)