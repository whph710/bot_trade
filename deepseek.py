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

# ЭКСТРЕМАЛЬНЫЕ НАСТРОЙКИ ДЛЯ 5M СКАЛЬПИНГА (критично для максимального профита)
SCALPING_CONFIG = {
    'default_timeout': 20,  # Очень быстрый таймаут для 5M
    'selection_timeout': 15,  # Экстремально быстро для первичного отбора
    'analysis_timeout': 30,  # Быстрее для детального анализа
    'max_retries': 2,  # Минимум попыток для скорости
    'max_tokens_selection': 800,  # Ещё меньше токенов для отбора
    'max_tokens_analysis': 2500,  # Оптимизировано для анализа
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
            default_prompt = "Ты опытный 5M скальпер. Анализируй молниеносно и конкретно."
            _cached_prompts[filename] = default_prompt
            logger.warning(f"Используется промпт по умолчанию для {filename}")

    return _cached_prompts[filename]


# Глобальный HTTP клиент для переиспользования соединений (критично для 5M)
_global_http_client = None


async def get_http_client(timeout: int = 20) -> httpx.AsyncClient:
    """Переиспользуемый HTTP клиент для экстремальной скорости."""
    global _global_http_client

    if _global_http_client is None or _global_http_client.is_closed:
        _global_http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=15,  # Больше соединений для 5M скальпинга
                max_keepalive_connections=8
            ),
            verify=True,
            http2=True,
            # Настройки для экстремальной скорости 5M
            headers={
                'Connection': 'keep-alive',
                'Keep-Alive': 'timeout=25, max=150'
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
    ЭКСТРЕМАЛЬНО оптимизированная функция для 5M скальпинга.

    Args:
        data: Данные для анализа
        prompt: Промпт (если None - загружается из файла)
        request_type: 'selection' для молниеносного отбора, 'analysis' для быстрого анализа
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

    # Экстремально адаптивные настройки для 5M
    if request_type == 'selection':
        timeout = timeout or SCALPING_CONFIG['selection_timeout']
        max_tokens = max_tokens or SCALPING_CONFIG['max_tokens_selection']
        max_retries = max_retries or SCALPING_CONFIG['max_retries']
        prompt_file = 'prompt2.txt'  # Промпт для молниеносного отбора
    else:  # analysis
        timeout = timeout or SCALPING_CONFIG['analysis_timeout']
        max_tokens = max_tokens or SCALPING_CONFIG['max_tokens_analysis']
        max_retries = max_retries or SCALPING_CONFIG['max_retries']
        prompt_file = 'prompt.txt'  # Промпт для быстрого анализа

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
            logger.info(f"DeepSeek 5M {request_type} запрос {attempt + 1}/{max_retries}")

            # Экстремально оптимизированные параметры для 5M скальпинга
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": str(data)},
                ],
                stream=False,
                max_tokens=max_tokens,

                # Настройки для экстремальной скорости и точности 5M
                temperature=0.2 if request_type == 'selection' else 0.6,  # Меньше рандома для отбора
                top_p=0.7 if request_type == 'selection' else 0.85,
                frequency_penalty=0.15,  # Избегаем повторов
                presence_penalty=0.1 if request_type == 'selection' else 0.05
            )

            result = response.choices[0].message.content
            execution_time = time.time() - start_time

            logger.info(f"✅ DeepSeek 5M {request_type} ответ: {len(result)} символов за {execution_time:.2f}сек")

            # Критическое предупреждение о медленной работе для 5M
            if execution_time > (timeout * 0.7):
                logger.warning(f"⚠️ МЕДЛЕННЫЙ ответ ИИ для 5M: {execution_time:.2f}сек (лимит {timeout}сек)")
            print(result)
            return result

        except asyncio.TimeoutError:
            logger.error(f"❌ Таймаут ИИ на попытке {attempt + 1}: {timeout}сек (КРИТИЧНО для 5M)")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5)  # Очень короткая пауза для 5M

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"❌ 5M попытка {attempt + 1} неудачна: {error_msg}")

            if attempt < max_retries - 1:
                # Минимальная задержка для 5M скальпинга
                wait_time = 0.5 + (attempt * 0.3)  # Максимум 1.1 секунды ожидания
                logger.info(f"⏳ Быстрое ожидание {wait_time:.1f}с...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"💥 Все попытки исчерпаны для 5M: {error_msg}")
                return f"Ошибка после {max_retries} попыток: {error_msg}"

    execution_time = time.time() - start_time
    logger.error(f"💥 КРИТИЧЕСКИЙ сбой DeepSeek за {execution_time:.2f}сек (5M)")
    return f"Критическая ошибка DeepSeek API после {max_retries} попыток"


async def deep_seek_selection(data: str, prompt: str = None) -> str:
    """Молниеносная функция для первичного отбора пар (максимальная скорость для 5M)."""
    return await deep_seek(
        data=data,
        prompt=prompt,
        request_type='selection'
    )


async def deep_seek_analysis(data: str, prompt: str = None) -> str:
    """Функция для быстрого анализа (баланс скорости и качества для 5M)."""
    return await deep_seek(
        data=data,
        prompt=prompt,
        request_type='analysis'
    )


async def test_deepseek_connection() -> bool:
    """Молниеносная проверка подключения к DeepSeek API для 5M."""
    api_key = os.getenv('DEEPSEEK')
    if not api_key:
        logger.error("API ключ не найден")
        return False

    try:
        http_client = await get_http_client(10)  # Очень быстрая проверка для 5M
        api_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            http_client=http_client
        )

        response = await api_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "5M test"}],
            max_tokens=3,
            temperature=0.1
        )

        logger.info("✅ DeepSeek API работает для 5M скальпинга")
        return True

    except Exception as e:
        logger.error(f"❌ Ошибка DeepSeek API: {e}")
        return False


async def batch_deep_seek_5m(requests: list, request_type: str = 'selection') -> list:
    """
    Батчевая обработка запросов к ИИ для ускорения 5M скальпинга.
    ЭКСТРЕМАЛЬНО оптимизирована для 5M.
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

    # Выполняем параллельно с ограничением (больше для 5M)
    semaphore = asyncio.Semaphore(4)  # Максимум 4 одновременных запроса для 5M

    async def bounded_request(task):
        async with semaphore:
            return await task

    bounded_tasks = [bounded_request(task) for task in tasks]
    results = await asyncio.gather(*bounded_tasks, return_exceptions=True)

    return results


async def check_api_health_5m() -> dict:
    """Быстрая проверка состояния API для 5M скальпинга."""
    start_time = time.time()

    health_info = {
        "api_key_exists": bool(os.getenv('DEEPSEEK')),
        "api_functional": False,
        "response_time": None,
        "suitable_for_5m_scalping": False
    }

    try:
        health_info["api_functional"] = await test_deepseek_connection()
        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        health_info["response_time"] = response_time

        # Проверяем подходит ли для 5M скальпинга (очень быстрые ответы)
        health_info["suitable_for_5m_scalping"] = response_time < 8.0

        if response_time > 12.0:
            logger.warning(f"⚠️ МЕДЛЕННОЕ API ({response_time}сек) - НЕ подходит для 5M скальпинга")
        elif response_time > 8.0:
            logger.warning(f"⚠️ API ({response_time}сек) - граница для 5M скальпинга")
        else:
            logger.info(f"✅ API быстрое ({response_time}сек) - отлично для 5M скальпинга")

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
async def deep_seek_legacy(data: str, prompt: str = None, timeout: int = 45,
                           max_tokens: int = 3500, max_retries: int = 3) -> str:
    """Обратная совместимость со старым интерфейсом (адаптировано для 5M)."""
    return await deep_seek(
        data=data,
        prompt=prompt,
        request_type='analysis',
        timeout=min(timeout, 30),  # Ограничиваем для 5M
        max_tokens=min(max_tokens, 2500),  # Ограничиваем для 5M
        max_retries=min(max_retries, 2)  # Ограничиваем для 5M
    )


# Новая функция для экстремально быстрых запросов
async def deep_seek_ultra_fast(data: str, prompt: str = None) -> str:
    """
    Экстремально быстрая функция для критических моментов 5M скальпинга.
    Минимальные настройки качества ради скорости.
    """
    return await deep_seek(
        data=data,
        prompt=prompt,
        request_type='selection',
        timeout=12,  # Экстремально быстро
        max_tokens=500,  # Минимум токенов
        max_retries=1  # Только одна попытка
    )