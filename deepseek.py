import asyncio
import os
import logging
from typing import Optional
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv
import time

# ИСПРАВЛЕННЫЙ импорт конфигурации
try:
    from config import config
except ImportError:
    # Если config не импортируется, создаем базовую конфигурацию
    class BasicConfig:
        class System:
            ENCODING = 'utf-8'

        class AI:
            API_KEY_ENV = 'DEEPSEEK'
            API_BASE_URL = 'https://api.deepseek.com'
            API_MODEL = 'deepseek-chat'
            DEFAULT_TIMEOUT = 40
            SELECTION_TIMEOUT = 40
            ANALYSIS_TIMEOUT = 60
            HEALTH_CHECK_TIMEOUT = 15
            MAX_RETRIES = 2
            RETRY_DELAY = 1.0
            MAX_TOKENS_SELECTION = 1000
            MAX_TOKENS_ANALYSIS = 4000
            MAX_TOKENS_TEST = 5
            TEMPERATURE_SELECTION = 0.3
            TEMPERATURE_ANALYSIS = 0.7
            TOP_P_SELECTION = 0.8
            TOP_P_ANALYSIS = 0.9
            FREQUENCY_PENALTY = 0.1
            PRESENCE_PENALTY_SELECTION = 0.1
            PRESENCE_PENALTY_ANALYSIS = 0.05
            SELECTION_PROMPT_FILE = 'prompt2.txt'
            ANALYSIS_PROMPT_FILE = 'prompt.txt'

        class Exchange:
            MAX_CONNECTIONS = 10
            MAX_KEEPALIVE_CONNECTIONS = 5
            KEEPALIVE_TIMEOUT = 30
            KEEPALIVE_MAX = 100

        def __init__(self):
            self.system = self.System()
            self.ai = self.AI()
            self.exchange = self.Exchange()


    config = BasicConfig()
    logging.warning("Используется базовая конфигурация из-за ошибки импорта")

load_dotenv()
logger = logging.getLogger(__name__)

# Кэшируем промпты для скорости
_cached_prompts = {}

# Глобальный HTTP клиент для переиспользования соединений
_global_http_client = None


def get_cached_prompt(filename: str = 'prompt.txt') -> str:
    """Кэшированная загрузка промптов для скорости."""
    global _cached_prompts

    if filename not in _cached_prompts:
        try:
            with open(filename, 'r', encoding=config.system.ENCODING) as file:
                _cached_prompts[filename] = file.read()
                logger.info(f"Промпт загружен из {filename}")
        except FileNotFoundError:
            default_prompt = "Ты опытный трейдер-скальпер. Анализируй быстро и конкретно."
            _cached_prompts[filename] = default_prompt
            logger.warning(f"Используется промпт по умолчанию для {filename}")

    return _cached_prompts[filename]


async def get_http_client(timeout: int = None) -> httpx.AsyncClient:
    """Переиспользуемый HTTP клиент для скорости."""
    global _global_http_client

    if timeout is None:
        timeout = config.ai.DEFAULT_TIMEOUT

    if _global_http_client is None or _global_http_client.is_closed:
        _global_http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=config.exchange.MAX_CONNECTIONS,
                max_keepalive_connections=config.exchange.MAX_KEEPALIVE_CONNECTIONS
            ),
            verify=True,
            http2=True,
            # Настройки для скорости
            headers={
                'Connection': 'keep-alive',
                'Keep-Alive': f'timeout={config.exchange.KEEPALIVE_TIMEOUT}, max={config.exchange.KEEPALIVE_MAX}'
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

    api_key = os.getenv(config.ai.API_KEY_ENV)
    if not api_key:
        error_msg = f"API ключ {config.ai.API_KEY_ENV} не найден"
        logger.error(error_msg)
        return f"Ошибка: {error_msg}"

    # Адаптивные настройки в зависимости от типа запроса
    if request_type == 'selection':
        timeout = timeout or config.ai.SELECTION_TIMEOUT
        max_tokens = max_tokens or config.ai.MAX_TOKENS_SELECTION
        max_retries = max_retries or config.ai.MAX_RETRIES
        prompt_file = config.ai.SELECTION_PROMPT_FILE
        temperature = config.ai.TEMPERATURE_SELECTION
        top_p = config.ai.TOP_P_SELECTION
        presence_penalty = config.ai.PRESENCE_PENALTY_SELECTION
    else:  # analysis
        timeout = timeout or config.ai.ANALYSIS_TIMEOUT
        max_tokens = max_tokens or config.ai.MAX_TOKENS_ANALYSIS
        max_retries = max_retries or config.ai.MAX_RETRIES
        prompt_file = config.ai.ANALYSIS_PROMPT_FILE
        temperature = config.ai.TEMPERATURE_ANALYSIS
        top_p = config.ai.TOP_P_ANALYSIS
        presence_penalty = config.ai.PRESENCE_PENALTY_ANALYSIS

    # Загружаем промпт
    if prompt is None:
        prompt = get_cached_prompt(prompt_file)

    # Получаем переиспользуемый HTTP клиент
    http_client = await get_http_client(timeout)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=config.ai.API_BASE_URL,
        http_client=http_client
    )

    for attempt in range(max_retries):
        try:
            logger.info(f"DeepSeek {request_type} запрос {attempt + 1}/{max_retries}")

            # Оптимизированные параметры для скальпинга
            response = await client.chat.completions.create(
                model=config.ai.API_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": str(data)},
                ],
                stream=False,
                max_tokens=max_tokens,

                # Настройки для скорости и качества скальпинга
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=config.ai.FREQUENCY_PENALTY,
                presence_penalty=presence_penalty
            )

            result = response.choices[0].message.content
            execution_time = time.time() - start_time

            logger.info(f"DeepSeek {request_type} ответ получен: {len(result)} символов за {execution_time:.2f}сек")

            # Предупреждение о медленной работе для скальпинга
            if execution_time > (timeout * 0.8):
                logger.warning(f"Медленный ответ ИИ: {execution_time:.2f}сек (лимит {timeout}сек)")

            print(result)
            return result

        except asyncio.TimeoutError:
            logger.error(f"Таймаут ИИ на попытке {attempt + 1}: {timeout}сек")
            if attempt < max_retries - 1:
                await asyncio.sleep(config.ai.RETRY_DELAY)

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Попытка {attempt + 1} неудачна: {error_msg}")

            if attempt < max_retries - 1:
                # Задержка для скальпинга
                wait_time = config.ai.RETRY_DELAY + (attempt * 0.5)  # Максимум 2 секунды ожидания
                logger.info(f"Ожидание {wait_time:.1f}с...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Все попытки исчерпаны: {error_msg}")
                return f"Ошибка после {max_retries} попыток: {error_msg}"

    execution_time = time.time() - start_time
    logger.error(f"Полная неудача DeepSeek за {execution_time:.2f}сек")
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
    api_key = os.getenv(config.ai.API_KEY_ENV)
    if not api_key:
        logger.error("API ключ не найден")
        return False

    try:
        http_client = await get_http_client(config.ai.HEALTH_CHECK_TIMEOUT)
        api_client = AsyncOpenAI(
            api_key=api_key,
            base_url=config.ai.API_BASE_URL,
            http_client=http_client
        )

        response = await api_client.chat.completions.create(
            model=config.ai.API_MODEL,
            messages=[{"role": "user", "content": "Test connection"}],
            max_tokens=config.ai.MAX_TOKENS_TEST,
            temperature=0.1
        )

        logger.info("DeepSeek API работает")
        return True

    except Exception as e:
        logger.error(f"Ошибка DeepSeek API: {e}")
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
    semaphore = asyncio.Semaphore(3)  # Используем базовое значение

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
        "api_key_exists": bool(os.getenv(config.ai.API_KEY_ENV)),
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
            logger.warning(f"Медленное API ({response_time}сек) - не подходит для скальпинга")
        else:
            logger.info(f"API быстрое ({response_time}сек) - подходит для скальпинга")

    except Exception as e:
        logger.error(f"Ошибка проверки API: {e}")

    return health_info


async def cleanup_http_client():
    """Очистка глобального HTTP клиента при завершении работы."""
    global _global_http_client
    if _global_http_client and not _global_http_client.is_closed:
        await _global_http_client.aclose()
        _global_http_client = None
        logger.info("HTTP клиент очищен")


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


# Функция для быстрой диагностики
def diagnose_import_issues():
    """Диагностика проблем с импортом"""
    print("🔍 ДИАГНОСТИКА ИМПОРТА:")
    print(f"   Config тип: {type(config)}")
    print(f"   AI настройки доступны: {hasattr(config, 'ai')}")
    print(f"   API ключ в окружении: {bool(os.getenv('DEEPSEEK'))}")

    if hasattr(config, 'ai'):
        print(f"   Таймауты: отбор={config.ai.SELECTION_TIMEOUT}с, анализ={config.ai.ANALYSIS_TIMEOUT}с")
        print(f"   Токены: отбор={config.ai.MAX_TOKENS_SELECTION}, анализ={config.ai.MAX_TOKENS_ANALYSIS}")

    print("✅ Диагностика завершена")


if __name__ == "__main__":
    diagnose_import_issues()