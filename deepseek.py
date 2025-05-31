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


async def deepseek_chat(prompt: str = DEFAULT_PROMPT, data: str = "") -> str:
    """
    Отправляет запрос к DeepSeek Chat модели.

    Args:
        prompt: Системный промпт
        data: Данные для анализа

    Returns:
        Ответ от модели
    """
    api_key = os.getenv('DEEPSEEK')
    if not api_key:
        error_msg = "API ключ DEEPSEEK не найден в переменных окружения"
        logger.error(error_msg)
        return f"Ошибка: {error_msg}"

    http_client = httpx.AsyncClient(timeout=180)  # Увеличил таймаут
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        http_client=http_client
    )

    try:
        logger.info("Отправка запроса к DeepSeek Chat")

        stream = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": data},
            ],
            stream=True,
            max_tokens=4000,
            temperature=0.7
        )

        result = ""
        async for chunk in stream:
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                result += content

        await http_client.aclose()
        logger.info(f"Успешно получен ответ от DeepSeek Chat (длина: {len(result)} символов)")
        return result

    except Exception as e:
        await http_client.aclose()
        error_msg = f"Ошибка DeepSeek Chat: {str(e)}"
        logger.error(error_msg)
        return error_msg


async def deepseek_reasoner(prompt: str = DEFAULT_PROMPT, data: str = "", max_retries: int = 3) -> str:
    """
    Отправляет запрос к DeepSeek Reasoner модели с повторными попытками.

    Args:
        prompt: Системный промпт
        data: Данные для анализа
        max_retries: Максимальное количество попыток

    Returns:
        Ответ от модели
    """
    api_key = os.getenv('DEEPSEEK')
    if not api_key:
        error_msg = "API ключ DEEPSEEK не найден в переменных окружения"
        logger.error(error_msg)
        return f"Ошибка: {error_msg}"

    # Обрезаем данные если они слишком длинные для Reasoner
    data_str = str(data)
    if len(data_str) > 30000:  # Уменьшил лимит для Reasoner
        data_str = data_str[:30000] + "... [данные обрезаны для анализа]"
        logger.info(f"Данные обрезаны до 30000 символов для Reasoner")

    for attempt in range(max_retries):
        http_client = httpx.AsyncClient(timeout=300)  # Увеличил таймаут для Reasoner
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            http_client=http_client
        )

        try:
            logger.info(f"Отправка запроса к DeepSeek Reasoner (попытка {attempt + 1}/{max_retries})")

            # Используем обычный режим вместо stream для Reasoner
            response = await client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": data_str},
                ],
                max_tokens=8000,  # Увеличил лимит токенов
                temperature=0.7,  # Уменьшил температуру для более стабильных результатов
                # Отключил стриминг для Reasoner
            )

            await http_client.aclose()

            result = response.choices[0].message.content
            if result and len(result.strip()) > 0:
                logger.info(f"Успешно получен ответ от DeepSeek Reasoner (длина: {len(result)} символов)")
                return result
            else:
                logger.warning(f"Получен пустой ответ от Reasoner на попытке {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)  # Ждем перед повторной попыткой
                    continue

        except asyncio.TimeoutError:
            await http_client.aclose()
            logger.error(f"Таймаут при запросе к Reasoner на попытке {attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(5)
                continue
        except Exception as e:
            await http_client.aclose()
            error_msg = f"Ошибка DeepSeek Reasoner на попытке {attempt + 1}: {str(e)}"
            logger.error(error_msg)
            if attempt < max_retries - 1:
                await asyncio.sleep(3)
                continue

    # Если все попытки неудачны, используем fallback на Chat модель
    logger.warning("Все попытки Reasoner неудачны, используем fallback на Chat модель")
    return await deepseek_chat_fallback(prompt, data_str)


async def deepseek_chat_fallback(prompt: str, data: str) -> str:
    """
    Fallback функция, использующая Chat модель вместо Reasoner.

    Args:
        prompt: Системный промпт
        data: Данные для анализа

    Returns:
        Ответ от Chat модели
    """
    logger.info("Использую Chat модель как fallback для детального анализа")

    # Модифицируем промпт для более детального анализа через Chat
    enhanced_prompt = f"""
{prompt}

ВАЖНО: Проведи максимально детальный и глубокий анализ предоставленных данных.
Включи в анализ:
1. Технический анализ каждой торговой пары
2. Конкретные точки входа и выхода
3. Уровни стоп-лосса и тейк-профита
4. Обоснование каждой рекомендации
5. Оценку рисков

Предоставь структурированный и подробный ответ.
"""

    return await deepseek_chat(enhanced_prompt, data)


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

    try:
        # Простой тест с минимальным запросом
        response = await deepseek_chat("Привет", "Тест")
        if "Ошибка" not in response:
            logger.info("Chat API работает корректно")

            # Тестируем также Reasoner
            reasoner_response = await deepseek_reasoner("Привет", "Тест короткое сообщение")
            if "Ошибка" not in reasoner_response and len(reasoner_response.strip()) > 0:
                logger.info("Reasoner API работает корректно")
                return True
            else:
                logger.warning("Reasoner API не отвечает, но Chat работает")
                return True  # Возвращаем True, так как есть fallback

        else:
            logger.error(f"Ошибка API: {response}")
            return False

    except Exception as e:
        logger.error(f"Ошибка при тестировании API: {e}")
        return False