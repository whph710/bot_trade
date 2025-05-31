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


async def deep_seek(data: str, prompt: str = DEFAULT_PROMPT, timeout: int = 60,
                    max_tokens: int = 4000, max_retries: int = 3) -> str:
    """
    Асинхронно отправляет запрос к DeepSeek API с обработкой ошибок и повторными попытками.

    Args:
        data: Данные для отправки модели
        prompt: Системный промпт для модели
        timeout: Таймаут запроса в секундах
        max_tokens: Максимальное количество токенов в ответе
        max_retries: Количество попыток при ошибке соединения

    Returns:
        Содержимое ответа от модели или сообщение об ошибке
    """
    api_key = os.getenv('DEEPSEEK')
    if not api_key:
        error_msg = "API ключ DEEPSEEK не найден в переменных окружения"
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
        base_url="https://api.deepseek.com",
        http_client=http_client
    )

    for attempt in range(max_retries):
        try:
            logger.info(f"Попытка {attempt + 1}/{max_retries}: отправка запроса к DeepSeek")
            logger.debug(f"Размер данных: {len(str(data))} символов")

            response = await client.chat.completions.create(
                model="deepseek-chat",  # Используем обычную модель вместо reasoner
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": str(data)},
                ],
                stream=False,
                max_tokens=max_tokens,
                temperature=0.7
            )

            result = response.choices[0].message.content
            logger.info(f"Успешно получен ответ от DeepSeek (длина: {len(result)} символов)")

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
    return "Неизвестная ошибка при обращении к DeepSeek API"


async def deep_seek_streaming(data: str, trade1: str = "", prompt: str = DEFAULT_PROMPT,
                              timeout: int = 60) -> str:
    api_key = os.getenv('DEEPSEEK')
    if not api_key:
        return "Ошибка: API ключ DEEPSEEK не найден"

    data_str = str(data)
    if len(data_str) > 50000:
        data_str = data_str[:50000] + "... [данные обрезаны]"

    try:
        timeout_config = httpx.Timeout(timeout)
        http_client = httpx.AsyncClient(timeout=timeout_config)
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            http_client=http_client
        )

        system_prompt = f"{trade1} {prompt}" if trade1 else prompt

        stream = await client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data_str},
            ],
            stream=True,
            max_tokens=4000,
            temperature=0.1
        )

        result = ""
        async for chunk in stream:
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                result += content

        await http_client.aclose()

        if not result.strip():
            return await deep_seek(data_str, prompt, timeout=60)

        return result

    except asyncio.TimeoutError:
        return f"Ошибка: Таймаут запроса ({timeout} сек)"



async def test_deepseek_connection() -> bool:
    """
    Тестирует подключение к DeepSeek API.

    Returns:
        True если подключение успешно, False в противном случае
    """
    api_key = os.getenv('DEEPSEEK')
    if not api_key:
        logger.error("API ключ DEEPSEEK не найден")
        return False

    logger.info("Проверяем подключение к DeepSeek API...")

    # Проверяем доступность сервера
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("https://api.deepseek.com")
            logger.info(f"Сервер доступен, статус: {response.status_code}")
    except Exception as e:
        logger.error(f"Сервер недоступен: {e}")
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
        logger.info("API работает корректно")
        return True

    except Exception as e:
        logger.error(f"Ошибка API: {e}")
        return False


async def check_api_health() -> dict:
    """
    Проверяет состояние API и возвращает детальную информацию.

    Returns:
        Словарь с информацией о состоянии API
    """
    health_info = {
        "api_key_exists": bool(os.getenv('DEEPSEEK')),
        "server_accessible": False,
        "api_functional": False,
        "response_time": None
    }

    start_time = asyncio.get_event_loop().time()

    try:
        # Проверяем API
        is_connected = await test_deepseek_connection()
        health_info["api_functional"] = is_connected
        health_info["server_accessible"] = True

        end_time = asyncio.get_event_loop().time()
        health_info["response_time"] = round(end_time - start_time, 2)

    except Exception as e:
        logger.error(f"Ошибка при проверке здоровья API: {e}")

    return health_info