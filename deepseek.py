import asyncio
import os
import logging
from typing import Optional
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Кэшируем промпт
_cached_prompt = None


def get_default_prompt():
    """Кэшированная загрузка промпта."""
    global _cached_prompt
    if _cached_prompt is None:
        try:
            with open('prompt.txt', 'r', encoding='utf-8') as file:
                _cached_prompt = file.read()
                logger.info("Промпт загружен из prompt.txt")
        except FileNotFoundError:
            _cached_prompt = "Ты опытный трейдер. Проанализируй данные и дай рекомендации."
            logger.warning("Используется промпт по умолчанию")
    return _cached_prompt


async def deep_seek(data: str, prompt: str = None, timeout: int = 60,
                    max_tokens: int = 4000, max_retries: int = 3) -> str:
    """
    Оптимизированная функция для работы с DeepSeek API.
    """
    api_key = os.getenv('DEEPSEEK')
    if not api_key:
        error_msg = "API ключ DEEPSEEK не найден"
        logger.error(error_msg)
        return f"Ошибка: {error_msg}"

    if prompt is None:
        prompt = get_default_prompt()

    # Оптимизированные настройки HTTP клиента
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        verify=True,
        http2=True  # Включаем HTTP/2 для лучшей производительности
    )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        http_client=http_client
    )

    for attempt in range(max_retries):
        try:
            logger.info(f"DeepSeek запрос {attempt + 1}/{max_retries}")

            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": str(data)},
                ],
                stream=False,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,  # Добавили для лучшего качества
                frequency_penalty=0.1  # Уменьшаем повторения
            )

            result = response.choices[0].message.content
            logger.info(f"Получен ответ от DeepSeek ({len(result)} символов)")

            await http_client.aclose()
            return result

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Попытка {attempt + 1} неудачна: {error_msg}")

            if attempt < max_retries - 1:
                # Экспоненциальная задержка с jitter
                wait_time = (2 ** attempt) + (attempt * 0.5)
                logger.info(f"Ожидание {wait_time:.1f}с...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Все попытки исчерпаны: {error_msg}")
                await http_client.aclose()
                return f"Ошибка после {max_retries} попыток: {error_msg}"

    await http_client.aclose()
    return "Неизвестная ошибка DeepSeek API"


async def test_deepseek_connection() -> bool:
    """Быстрая проверка подключения к DeepSeek API."""
    api_key = os.getenv('DEEPSEEK')
    if not api_key:
        logger.error("API ключ не найден")
        return False

    try:
        # Упрощенная проверка - только тестовый запрос
        http_client = httpx.AsyncClient(timeout=15.0, http2=True)
        api_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            http_client=http_client
        )

        response = await api_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )

        await http_client.aclose()
        logger.info("DeepSeek API работает")
        return True

    except Exception as e:
        logger.error(f"Ошибка DeepSeek API: {e}")
        return False



async def check_api_health() -> dict:
    """Быстрая проверка состояния API."""
    start_time = asyncio.get_event_loop().time()

    health_info = {
        "api_key_exists": bool(os.getenv('DEEPSEEK')),
        "api_functional": False,
        "response_time": None
    }

    try:
        health_info["api_functional"] = await test_deepseek_connection()
        end_time = asyncio.get_event_loop().time()
        health_info["response_time"] = round(end_time - start_time, 2)
    except Exception as e:
        logger.error(f"Ошибка проверки API: {e}")

    return health_info