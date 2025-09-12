import asyncio
import os
import logging
from typing import Optional
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv
import time
import json

# Импорт конфигурации
from config import config

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
            if filename == config.ai.SELECTION_PROMPT_FILE:
                default_prompt = """Ты элитный аналитик. Отбери ТОП-3 пары для скальпинга.
                Критерии: ликвидность, волатильность, четкие сигналы.
                Ответ в формате: {"pairs": ["SYMBOL1", "SYMBOL2", "SYMBOL3"]}"""
            else:
                default_prompt = """Ты опытный трейдер-скальпер. Анализируй быстро и конкретно.
                Дай точные рекомендации по входу в позицию с уровнями стоп-лосса и тейк-профита."""

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


def optimize_data_for_ai(data: str, max_size: int = 50000) -> str:
    """
    НОВОЕ: Оптимизация данных для ИИ при превышении лимитов
    Сжимает большие объемы данных, сохраняя ключевую информацию
    """
    if len(data) <= max_size:
        return data

    logger.warning(f"Данные превышают лимит ({len(data)} > {max_size}), применяется сжатие")

    try:
        # Парсим JSON данные
        json_data = json.loads(data)

        # Сжимаем массивы индикаторов (берем только последние значения)
        if 'technical_analysis' in json_data:
            for tf_key in ['indicators_5m', 'indicators_15m']:
                if tf_key in json_data['technical_analysis']:
                    indicators = json_data['technical_analysis'][tf_key]

                    # Сжимаем исторические данные индикаторов
                    for category in ['ema_system', 'momentum_indicators', 'volatility_indicators', 'volume_analysis']:
                        if category in indicators:
                            for indicator_name, values in indicators[category].items():
                                if isinstance(values, list) and len(values) > 50:
                                    # Берем только последние 50 значений
                                    indicators[category][indicator_name] = values[-50:]

        # Сжимаем данные свечей (берем только последние)
        if 'market_data' in json_data:
            for tf_key in ['timeframe_5m', 'timeframe_15m']:
                if tf_key in json_data['market_data'] and 'candles' in json_data['market_data'][tf_key]:
                    candles = json_data['market_data'][tf_key]['candles']
                    if len(candles) > 200:
                        json_data['market_data'][tf_key]['candles'] = candles[-200:]
                        json_data['market_data'][tf_key]['candles_count'] = len(candles[-200:])

        # Преобразуем обратно в JSON
        optimized_data = json.dumps(json_data, ensure_ascii=False, separators=(',', ':'))

        logger.info(f"Данные сжаты: {len(data)} -> {len(optimized_data)} символов")
        return optimized_data

    except json.JSONDecodeError:
        # Если не JSON, применяем простое обрезание
        logger.warning("Не удалось распарсить JSON, применяется простое обрезание")
        return data[:max_size] + "... [ДАННЫЕ ОБРЕЗАНЫ ДЛЯ ЭКОНОМИИ ТОКЕНОВ]"
    except Exception as e:
        logger.error(f"Ошибка оптимизации данных: {e}")
        return data[:max_size]


async def deep_seek(data: str,
                    prompt: str = None,
                    request_type: str = 'analysis',  # 'selection' или 'analysis'
                    timeout: int = None,
                    max_tokens: int = None,
                    max_retries: int = None,
                    optimize_large_data: bool = True) -> str:
    """
    Оптимизированная функция для скальпинга с адаптивными настройками.
    ОБНОВЛЕНО: поддержка больших объемов данных для финального анализа

    Args:
        data: Данные для анализа
        prompt: Промпт (если None - загружается из файла)
        request_type: 'selection' для быстрого отбора, 'analysis' для детального анализа
        timeout: Таймаут (если None - берется из конфига по типу запроса)
        max_tokens: Максимум токенов (если None - берется из конфига)
        max_retries: Максимум попыток (если None - берется из конфига)
        optimize_large_data: Оптимизировать большие данные для экономии токенов
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

    # Оптимизируем данные для больших объемов (финальный анализ)
    if optimize_large_data and len(data) > 30000:  # Если данные больше 30KB
        logger.info(f"Обрабатываются большие данные ({len(data)} символов), применяется оптимизация")

        # Для финального анализа используем более мягкую оптимизацию
        if request_type == 'analysis':
            data = optimize_data_for_ai(data, max_size=80000)  # Больший лимит для финального анализа
        else:
            data = optimize_data_for_ai(data, max_size=40000)  # Меньший лимит для отбора

    # Получаем переиспользуемый HTTP клиент
    http_client = await get_http_client(timeout)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=config.ai.API_BASE_URL,
        http_client=http_client
    )

    for attempt in range(max_retries):
        try:
            logger.info(f"DeepSeek {request_type} запрос {attempt + 1}/{max_retries} "
                        f"(данные: {len(data)} символов)")

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

            # Валидация ответа для критически важных операций
            if request_type == 'analysis' and not validate_ai_response(result):
                logger.warning(f"Ответ ИИ не прошел валидацию на попытке {attempt + 1}")
                if attempt < max_retries - 1:
                    continue

            print(result)
            return result

        except asyncio.TimeoutError:
            logger.error(f"Таймаут ИИ на попытке {attempt + 1}: {timeout}сек")
            if attempt < max_retries - 1:
                await asyncio.sleep(config.ai.RETRY_DELAY)

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Попытка {attempt + 1} неудачна: {error_msg}")

            # Специальная обработка для больших данных
            if "token" in error_msg.lower() or "context" in error_msg.lower():
                logger.warning("Возможна проблема с размером данных, применяется дополнительная оптимизация")
                if len(data) > 20000:
                    data = optimize_data_for_ai(data, max_size=20000)
                    logger.info(f"Данные дополнительно сжаты до {len(data)} символов")

            if attempt < max_retries - 1:
                # Прогрессивная задержка для скальпинга
                wait_time = config.ai.RETRY_DELAY * (1.5 ** attempt)  # Экспоненциальная задержка
                logger.info(f"Ожидание {wait_time:.1f}с...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Все попытки исчерпаны: {error_msg}")
                return f"Ошибка после {max_retries} попыток: {error_msg}"

    execution_time = time.time() - start_time
    logger.error(f"Полная неудача DeepSeek за {execution_time:.2f}сек")
    return f"Критическая ошибка DeepSeek API после {max_retries} попыток"


def validate_ai_response(response: str) -> bool:
    """
    НОВОЕ: Валидация ответа ИИ для критически важных операций
    """
    if not response or len(response) < 50:
        return False

    # Проверяем наличие ключевых элементов для торгового анализа
    required_elements = ['signal', 'confidence', 'analysis']
    found_elements = sum(1 for element in required_elements if element.lower() in response.lower())

    # Должно быть найдено минимум 2 из 3 элементов
    return found_elements >= 2


async def deep_seek_selection(data: str, prompt: str = None) -> str:
    """Быстрая функция для первичного отбора пар (оптимизирована для скорости)."""
    return await deep_seek(
        data=data,
        prompt=prompt,
        request_type='selection',
        optimize_large_data=True
    )


async def deep_seek_analysis(data: str, prompt: str = None) -> str:
    """
    Функция для детального анализа (баланс скорости и качества).
    ОБНОВЛЕНО: поддержка больших объемов данных для финального анализа
    """
    return await deep_seek(
        data=data,
        prompt=prompt,
        request_type='analysis',
        optimize_large_data=True  # Включаем оптимизацию для больших данных
    )


async def deep_seek_full_analysis(data: str, prompt: str = None,
                                  preserve_data: bool = False) -> str:
    """
    НОВОЕ: Специальная функция для финального анализа с максимальным сохранением данных

    Args:
        data: Полные данные для анализа
        prompt: Промпт
        preserve_data: Если True, минимальная оптимизация данных
    """
    logger.info(f"Запуск полного анализа с данными {len(data)} символов")

    # Используем увеличенные лимиты для финального анализа
    return await deep_seek(
        data=data,
        prompt=prompt,
        request_type='analysis',
        timeout=config.ai.ANALYSIS_TIMEOUT + 30,  # Дополнительное время
        max_tokens=config.ai.MAX_TOKENS_ANALYSIS,
        optimize_large_data=not preserve_data  # Отключаем оптимизацию если нужно сохранить данные
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


async def batch_deep_seek_with_optimization(requests: list,
                                            request_type: str = 'selection',
                                            max_concurrent: int = None) -> list:
    """
    НОВОЕ: Улучшенная батчевая обработка с оптимизацией для больших данных
    """
    if not requests:
        return []

    if max_concurrent is None:
        max_concurrent = config.processing.MAX_CONCURRENT_REQUESTS

    logger.info(f"Батчевая обработка {len(requests)} запросов (макс. параллельно: {max_concurrent})")

    results = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single_request(req_data):
        async with semaphore:
            try:
                # Оптимизируем данные для каждого запроса
                data = req_data.get('data', '')
                if len(data) > 40000:  # Большие данные
                    data = optimize_data_for_ai(data, max_size=35000)

                result = await deep_seek(
                    data=data,
                    prompt=req_data.get('prompt'),
                    request_type=request_type,
                    optimize_large_data=True
                )
                return result
            except Exception as e:
                logger.error(f"Ошибка в батчевой обработке: {e}")
                return f"Ошибка обработки: {e}"

    # Создаем задачи
    tasks = [process_single_request(req) for req in requests]

    # Обрабатываем батчами для контроля нагрузки
    batch_size = min(max_concurrent, len(tasks))

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        results.extend(batch_results)

        # Небольшая пауза между батчами
        if i + batch_size < len(tasks):
            await asyncio.sleep(0.5)

    logger.info(f"Батчевая обработка завершена: {len(results)} результатов")
    return results


async def check_api_health() -> dict:
    """Быстрая проверка состояния API для скальпинга."""
    start_time = time.time()

    health_info = {
        "api_key_exists": bool(os.getenv(config.ai.API_KEY_ENV)),
        "api_functional": False,
        "response_time": None,
        "suitable_for_scalping": False,
        "supports_large_data": False  # НОВОЕ: поддержка больших данных
    }

    try:
        health_info["api_functional"] = await test_deepseek_connection()
        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        health_info["response_time"] = response_time

        # Проверяем подходит ли для скальпинга (быстрые ответы)
        health_info["suitable_for_scalping"] = response_time < 10.0

        # Тест с большими данными
        if health_info["api_functional"]:
            health_info["supports_large_data"] = await test_large_data_support()

        if response_time > 15.0:
            logger.warning(f"Медленное API ({response_time}сек) - не подходит для скальпинга")
        else:
            logger.info(f"API быстрое ({response_time}сек) - подходит для скальпинга")

    except Exception as e:
        logger.error(f"Ошибка проверки API: {e}")

    return health_info


async def test_large_data_support() -> bool:
    """
    НОВОЕ: Тест поддержки больших объемов данных
    """
    try:
        # Создаем тестовые данные среднего размера
        test_data = {
            "test": "large_data_support",
            "data": ["test_value"] * 1000  # ~15KB данных
        }

        test_json = json.dumps(test_data)

        # Тестируем отправку
        result = await deep_seek(
            data=test_json,
            prompt="Кратко подтверди получение данных.",
            request_type='selection',
            timeout=15,
            max_tokens=50
        )

        return "test" in result.lower() or "данн" in result.lower()

    except Exception as e:
        logger.warning(f"Тест больших данных не прошел: {e}")
        return False


async def cleanup_http_client():
    """Очистка глобального HTTP клиента при завершении работы."""
    global _global_http_client
    if _global_http_client and not _global_http_client.is_closed:
        await _global_http_client.aclose()
        _global_http_client = None
        logger.info("HTTP клиент очищен")


def clear_prompt_cache():
    """
    НОВОЕ: Очистка кэша промптов (полезно при обновлении файлов промптов)
    """
    global _cached_prompts
    _cached_prompts.clear()
    logger.info("Кэш промптов очищен")


async def get_api_usage_stats() -> dict:
    """
    НОВОЕ: Получение статистики использования API (упрощенная версия)
    """
    return {
        "cached_prompts": len(_cached_prompts),
        "http_client_active": _global_http_client is not None and not _global_http_client.is_closed,
        "config_timeouts": {
            "selection": config.ai.SELECTION_TIMEOUT,
            "analysis": config.ai.ANALYSIS_TIMEOUT
        },
        "config_tokens": {
            "selection": config.ai.MAX_TOKENS_SELECTION,
            "analysis": config.ai.MAX_TOKENS_ANALYSIS
        }
    }


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
        max_retries=max_retries,
        optimize_large_data=True
    )