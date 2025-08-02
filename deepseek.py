import asyncio
import os
import logging
from typing import Optional
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv
import time

load_dotenv()
logger = logging.getLogger(__name__)

# Кэшируем промпты для скорости
_cached_prompts = {}

# СКАЛЬПИНГОВЫЕ НАСТРОЙКИ (критично для скорости)
SCALPING_CONFIG = {
    'default_timeout': 25,  # Быстрый таймаут для скальпинга
    'selection_timeout': 20,  # Еще быстрее для первичного отбора
    'analysis_timeout': 35,  # Чуть больше для детального анализа
    'max_retries': 2,  # Меньше попыток для скорости
    'max_tokens_selection': 1000,  # Меньше токенов для отбора
    'max_tokens_analysis': 3000,  # Больше для анализа
}


def get_cached_prompt(filename: str = 'prompt.txt') -> str:
    """Кэшированная загрузка промптов для скорости."""
    global _cached_prompts

    if filename not in _cached_prompts:
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                _cached_prompts[filename] = file.read()
                logger.info(f"Промпт загружен из {filename}")
        except FileNotFoundError:
            default_prompt = "Ты опытный трейдер-скальпер. Анализируй быстро и конкретно."
            _cached_prompts[filename] = default_prompt
            logger.warning(f"Используется промпт по умолчанию для {filename}")

    return _cached_prompts[filename]


# Глобальный HTTP клиент для переиспользования соединений (экономия времени)
_global_http_client = None


async def get_http_client(timeout: int = 25) -> httpx.AsyncClient:
    """Переиспользуемый HTTP клиент для скорости."""
    global _global_http_client

    if _global_http_client is None or _global_http_client.is_closed:
        _global_http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=10,  # Больше соединений для батчей
                max_keepalive_connections=5
            ),
            verify=True,
            http2=True,
            # Настройки для скорости
            headers={
                'Connection': 'keep-alive',
                'Keep-Alive': 'timeout=30, max=100'
            }
        )

    return _global_http_client


async def deep_seek(data: str,
                    prompt: str = None,
                    request_type: str = 'analysis',  # 'selection' или 'analysis'
                    timeout: int = None,
                    max_tokens: int = None,
                    max_retries: int = None) -> str:
    """
    Оптимизированная функция для скальпинга с адаптивными настройками.

    Args:
        data: Данные для анализа
        prompt: Промпт (если None - загружается из файла)
        request_type: 'selection' для быстрого отбора, 'analysis' для детального анализа
        timeout: Таймаут (если None - берется из конфига по типу запроса)
        max_tokens: Максимум токенов (если None - берется из конфига)
        max_retries: Максимум попыток (если None - берется из конфига)
    """
    start_time = time.time()

    api_key = os.getenv('DEEPSEEK')
    if not api_key:
        error_msg = "API ключ DEEPSEEK не найден"
        logger.error(error_msg)
        return f"Ошибка: {error_msg}"

    # Адаптивные настройки в зависимости от типа запроса
    if request_type == 'selection':
        timeout = timeout or SCALPING_CONFIG['selection_timeout']
        max_tokens = max_tokens or SCALPING_CONFIG['max_tokens_selection']
        max_retries = max_retries or SCALPING_CONFIG['max_retries']
        prompt_file = 'prompt2.txt'  # Промпт для отбора
    else:  # analysis
        timeout = timeout or SCALPING_CONFIG['analysis_timeout']
        max_tokens = max_tokens or SCALPING_CONFIG['max_tokens_analysis']
        max_retries = max_retries or SCALPING_CONFIG['max_retries']
        prompt_file = 'prompt.txt'  # Промпт для анализа

    # Загружаем промпт
    if prompt is None:
        prompt = get_cached_prompt(prompt_file)

    # Получаем переиспользуемый HTTP клиент
    http_client = await get_http_client(timeout)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        http_client=http_client
    )

    for attempt in range(max_retries):
        try:
            logger.info(f"DeepSeek {request_type} запрос {attempt + 1}/{max_retries}")

            # Оптимизированные параметры для скальпинга
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": str(data)},
                ],
                stream=False,
                max_tokens=max_tokens,

                # Настройки для скорости и качества скальпинга
                temperature=0.3 if request_type == 'selection' else 0.7,  # Меньше креативности для отбора
                top_p=0.8 if request_type == 'selection' else 0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1 if request_type == 'selection' else 0.05
            )

            result = response.choices[0].message.content
            execution_time = time.time() - start_time

            logger.info(f"✅ DeepSeek {request_type} ответ получен: {len(result)} символов за {execution_time:.2f}сек")

            # Предупреждение о медленной работе для скальпинга
            if execution_time > (timeout * 0.8):
                logger.warning(f"⚠️ Медленный ответ ИИ: {execution_time:.2f}сек (лимит {timeout}сек)")
            print(result)
            return result

        except asyncio.TimeoutError:
            logger.error(f"❌ Таймаут ИИ на попытке {attempt + 1}: {timeout}сек")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)  # Короткая пауза для скальпинга

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"❌ Попытка {attempt + 1} неудачна: {error_msg}")

            if attempt < max_retries - 1:
                # Короткая задержка для скальпинга (не экспоненциальная)
                wait_time = 1 + (attempt * 0.5)  # Максимум 2 секунды ожидания
                logger.info(f"⏳ Ожидание {wait_time:.1f}с...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"💥 Все попытки исчерпаны: {error_msg}")
                return f"Ошибка после {max_retries} попыток: {error_msg}"

    execution_time = time.time() - start_time
    logger.error(f"💥 Полная неудача DeepSeek за {execution_time:.2f}сек")
    return f"Критическая ошибка DeepSeek API после {max_retries} попыток"


async def deep_seek_selection(data: str, prompt: str = None) -> str:
    """Быстрая функция для первичного отбора пар (оптимизирована для скорости)."""
    return await deep_seek(
        data=data,
        prompt=prompt,
        request_type='selection'
    )


async def deep_seek_analysis(data: str, prompt: str = None) -> str:
    """Функция для детального анализа (баланс скорости и качества)."""
    return await deep_seek(
        data=data,
        prompt=prompt,
        request_type='analysis'
    )


async def test_deepseek_connection() -> bool:
    """Быстрая проверка подключения к DeepSeek API."""
    api_key = os.getenv('DEEPSEEK')
    if not api_key:
        logger.error("API ключ не найден")
        return False

    try:
        http_client = await get_http_client(15)  # Быстрая проверка
        api_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            http_client=http_client
        )

        response = await api_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Test connection"}],
            max_tokens=5,
            temperature=0.1
        )

        logger.info("✅ DeepSeek API работает")
        return True

    except Exception as e:
        logger.error(f"❌ Ошибка DeepSeek API: {e}")
        return False


async def batch_deep_seek(requests: list, request_type: str = 'selection') -> list:
    """
    Батчевая обработка запросов к ИИ для ускорения.
    НЕ ИСПОЛЬЗУЕТСЯ в текущей версии, но готова для будущих оптимизаций.
    """
    results = []

    # Создаем задачи для параллельного выполнения
    tasks = []
    for req_data in requests:
        task = deep_seek(
            data=req_data.get('data', ''),
            prompt=req_data.get('prompt'),
            request_type=request_type
        )
        tasks.append(task)

    # Выполняем параллельно с ограничением
    semaphore = asyncio.Semaphore(3)  # Максимум 3 одновременных запроса

    async def bounded_request(task):
        async with semaphore:
            return await task

    bounded_tasks = [bounded_request(task) for task in tasks]
    results = await asyncio.gather(*bounded_tasks, return_exceptions=True)

    return results


async def check_api_health() -> dict:
    """Быстрая проверка состояния API для скальпинга."""
    start_time = time.time()

    health_info = {
        "api_key_exists": bool(os.getenv('DEEPSEEK')),
        "api_functional": False,
        "response_time": None,
        "suitable_for_scalping": False
    }

    try:
        health_info["api_functional"] = await test_deepseek_connection()
        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        health_info["response_time"] = response_time

        # Проверяем подходит ли для скальпинга (быстрые ответы)
        health_info["suitable_for_scalping"] = response_time < 10.0

        if response_time > 15.0:
            logger.warning(f"⚠️ Медленное API ({response_time}сек) - не подходит для скальпинга")
        else:
            logger.info(f"✅ API быстрое ({response_time}сек) - подходит для скальпинга")

    except Exception as e:
        logger.error(f"❌ Ошибка проверки API: {e}")

    return health_info


async def cleanup_http_client():
    """Очистка глобального HTTP клиента при завершении работы."""
    global _global_http_client
    if _global_http_client and not _global_http_client.is_closed:
        await _global_http_client.aclose()
        _global_http_client = None
        logger.info("🧹 HTTP клиент очищен")


# Функция для совместимости со старым кодом
async def deep_seek_legacy(data: str, prompt: str = None, timeout: int = 60,
                           max_tokens: int = 4000, max_retries: int = 3) -> str:
    """Обратная совместимость со старым интерфейсом."""
    return await deep_seek(
        data=data,
        prompt=prompt,
        request_type='analysis',
        timeout=timeout,
        max_tokens=max_tokens,
        max_retries=max_retries
    )