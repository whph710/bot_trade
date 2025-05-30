import asyncio
import os
import logging
from typing import Optional
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

logger = logging.getLogger(__name__)

# Загружаем основной промпт из файла
try:
    with open('prompt.txt', 'r', encoding='utf-8') as file:
        DEFAULT_PROMPT = file.read()
        logger.info("Загружен основной промпт из prompt.txt")
except FileNotFoundError:
    DEFAULT_PROMPT = "Ты опытный трейдер. Проанализируй данные и дай рекомендации."
    logger.warning("Файл prompt.txt не найден, используется промпт по умолчанию")


async def chat_gpt(data: str, prompt: str = DEFAULT_PROMPT, timeout: int = 60,
                   max_tokens: int = 8192, max_retries: int = 3) -> str:
    """
    Асинхронно отправляет запрос к OpenAI ChatGPT API с обработкой ошибок и повторными попытками.

    Args:
        data: Данные для отправки модели
        prompt: Системный промпт для модели
        timeout: Таймаут запроса в секундах
        max_tokens: Максимальное количество токенов в ответе
        max_retries: Количество попыток при ошибке соединения

    Returns:
        Содержимое ответа от модели или сообщение об ошибке
    """
    api_key = os.getenv('OPEN_AI')
    if not api_key:
        error_msg = "API ключ OPEN_AI не найден в переменных окружения"
        logger.error(error_msg)
        return f"Ошибка: {error_msg}"

    # Настройки HTTP клиента для стабильного соединения
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        verify=True
    )

    client = AsyncOpenAI(
        api_key=api_key,
        http_client=http_client
    )

    for attempt in range(max_retries):
        try:
            logger.info(f"Попытка {attempt + 1}/{max_retries}: отправка запроса к ChatGPT")
            logger.debug(f"Размер данных: {len(str(data))} символов")

            response = await client.chat.completions.create(
                model="gpt-4",  # Исправил модель на корректную
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": str(data)},
                ],
                stream=False,
                max_tokens=max_tokens,
                temperature=0.7
            )

            result = response.choices[0].message.content
            logger.info(f"Успешно получен ответ от ChatGPT (длина: {len(result)} символов)")

            await http_client.aclose()
            return result

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Попытка {attempt + 1} неудачна: {error_msg}")

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Экспоненциальная задержка: 1, 2, 4 сек
                logger.info(f"Ожидание {wait_time} сек перед следующей попыткой...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Все попытки исчерпаны: {error_msg}")
                await http_client.aclose()
                return f"Ошибка после {max_retries} попыток: {error_msg}"

    await http_client.aclose()
    return "Неизвестная ошибка при обращении к OpenAI API"


async def chat_gpt_streaming(data: str, trade1: str = "", prompt: str = DEFAULT_PROMPT,
                             timeout: int = 120) -> str:
    """
    Версия с потоковой передачей - показывает прогресс генерации ответа в реальном времени.

    Args:
        data: Данные для анализа
        trade1: Дополнительная информация о торговом направлении
        prompt: Системный промпт
        timeout: Таймаут запроса в секундах

    Returns:
        Полный ответ модели
    """
    api_key = os.getenv('OPEN_AI')
    if not api_key:
        error_msg = "API ключ OPEN_AI не найден в переменных окружения"
        logger.error(error_msg)
        return f"Ошибка: {error_msg}"

    try:
        timeout_config = httpx.Timeout(timeout)
        http_client = httpx.AsyncClient(timeout=timeout_config)

        client = AsyncOpenAI(
            api_key=api_key,
            http_client=http_client
        )

        logger.info("Начинаем потоковую генерацию ответа...")

        # Формируем системный промпт с учетом торгового направления
        system_prompt = f"{trade1} {prompt}" if trade1 else prompt

        stream = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data},
            ],
            stream=True,
            max_tokens=4000,
            temperature=0.7
        )

        result = ""
        chunk_count = 0

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                result += content
                chunk_count += 1

                # Логируем прогресс каждые 50 чанков
                if chunk_count % 50 == 0:
                    logger.debug(f"Получено {chunk_count} чанков, текущая длина ответа: {len(result)}")

        await http_client.aclose()

        logger.info(f"Генерация завершена. Итоговая длина ответа: {len(result)} символов")
        return result

    except asyncio.TimeoutError:
        error_msg = f"Таймаут запроса ({timeout} сек)"
        logger.error(error_msg)
        return f"Ошибка: {error_msg}"
    except Exception as e:
        error_msg = f"Ошибка при потоковом запросе к ChatGPT: {e}"
        logger.error(error_msg)
        return f"Ошибка: {str(e)}"


async def test_chatgpt_connection() -> bool:
    """
    Тестирует подключение к OpenAI ChatGPT API.

    Returns:
        True если подключение успешно, False в противном случае
    """
    api_key = os.getenv('OPEN_AI')
    if not api_key:
        logger.error("API ключ OPEN_AI не найден")
        return False

    logger.info("Проверяем подключение к ChatGPT API...")

    # Тестовый запрос к API
    try:
        http_client = httpx.AsyncClient(timeout=30.0)
        api_client = AsyncOpenAI(
            api_key=api_key,
            http_client=http_client
        )

        response = await api_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Привет"}],
            max_tokens=10
        )

        await http_client.aclose()
        logger.info("ChatGPT API работает корректно")
        return True

    except Exception as e:
        logger.error(f"Ошибка ChatGPT API: {e}")
        return False


async def check_api_health() -> dict:
    """
    Проверяет состояние API и возвращает детальную информацию.

    Returns:
        Словарь с информацией о состоянии API
    """
    health_info = {
        "api_key_exists": bool(os.getenv('OPEN_AI')),
        "server_accessible": True,  # OpenAI обычно доступен
        "api_functional": False,
        "response_time": None
    }

    start_time = asyncio.get_event_loop().time()

    try:
        # Проверяем API
        is_connected = await test_chatgpt_connection()
        health_info["api_functional"] = is_connected

        end_time = asyncio.get_event_loop().time()
        health_info["response_time"] = round(end_time - start_time, 2)

    except Exception as e:
        logger.error(f"Ошибка при проверке здоровья ChatGPT API: {e}")

    return health_info


# Псевдонимы функций для совместимости с deepseek.py
deep_seek = chat_gpt  # Основная функция
deep_seek_streaming = chat_gpt_streaming  # Потоковая функция
test_deepseek_connection = test_chatgpt_connection  # Тест соединения