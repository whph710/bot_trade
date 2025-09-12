import aiohttp
import logging
from typing import List, Dict, Any, Optional
import asyncio
import time

# Импорт конфигурации
from config import config

logger = logging.getLogger(__name__)

# Глобальная сессия для переиспользования соединений
_global_session = None


async def get_global_session() -> aiohttp.ClientSession:
    """
    Получение глобальной сессии для переиспользования соединений
    НОВОЕ: оптимизация для частых запросов больших данных
    """
    global _global_session

    if _global_session is None or _global_session.closed:
        timeout = aiohttp.ClientTimeout(
            total=config.exchange.API_TIMEOUT + 10,  # Дополнительное время для больших данных
            connect=10,
            sock_read=30
        )

        connector = aiohttp.TCPConnector(
            limit=config.exchange.MAX_CONNECTIONS,
            limit_per_host=config.exchange.MAX_KEEPALIVE_CONNECTIONS,
            ttl_dns_cache=300,  # DNS кэш на 5 минут
            use_dns_cache=True,
            keepalive_timeout=config.exchange.KEEPALIVE_TIMEOUT,
            enable_cleanup_closed=True
        )

        _global_session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'User-Agent': 'ScalpingBot/1.0',
                'Connection': 'keep-alive'
            }
        )

        logger.info("Создана новая глобальная HTTP сессия")

    return _global_session


async def get_klines_async(symbol: str, interval: str = "15", limit: int = 200,
                           retries: int = 3) -> List[List[str]]:
    """
    Асинхронно получает данные свечей для торговой пары.
    ОБНОВЛЕНО: улучшенная обработка ошибок и поддержка больших лимитов

    Args:
        symbol: Торговая пара (например, BTCUSDT)
        interval: Интервал свечей (по умолчанию "15")
        limit: Количество свечей (по умолчанию 200, макс 1000)
        retries: Количество попыток при ошибке

    Returns:
        Список свечей в формате Bybit [timestamp, open, high, low, close, volume, turnover]
    """
    # Ограничиваем лимит согласно API Bybit
    limit = min(limit, 1000)  # Максимум для Bybit API

    url = config.exchange.KLINE_URL
    params = {
        "category": config.exchange.API_CATEGORY,
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    session = await get_global_session()

    for attempt in range(retries):
        try:
            start_time = time.time()

            async with session.get(url, params=params) as response:
                # Проверяем статус ответа
                if response.status != 200:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status
                    )

                data = await response.json()
                request_time = time.time() - start_time

                if data.get("retCode") != 0:
                    error_msg = data.get('retMsg', 'Unknown API error')
                    logger.error(f"API ошибка для {symbol}: {error_msg}")

                    # Специальная обработка ошибок лимита
                    if "rate limit" in error_msg.lower():
                        wait_time = 2 ** attempt  # Экспоненциальная задержка
                        logger.warning(f"Rate limit для {symbol}, ожидание {wait_time}с")
                        await asyncio.sleep(wait_time)
                        continue

                    raise Exception(f"Bybit API Error for {symbol}: {error_msg}")

                klines = data["result"]["list"]

                if not klines:
                    logger.warning(f"Пустой ответ для {symbol}")
                    return []

                # Проверяем порядок и при необходимости разворачиваем (от старых к новым)
                if len(klines) > 1:
                    if int(klines[0][0]) > int(klines[-1][0]):
                        klines.reverse()

                # Убираем последнюю незакрытую свечу
                if klines:
                    klines = klines[:-1]

                logger.debug(f"Получено {len(klines)} свечей для {symbol} "
                             f"({interval}m) за {request_time:.2f}сек")

                # Предупреждение о медленных запросах
                if request_time > 5.0:
                    logger.warning(f"Медленный запрос для {symbol}: {request_time:.2f}сек")

                return klines

        except asyncio.TimeoutError:
            logger.warning(f"Таймаут для {symbol} на попытке {attempt + 1}/{retries}")
            if attempt < retries - 1:
                await asyncio.sleep(1.0 * (attempt + 1))  # Прогрессивная задержка
            else:
                raise Exception(f"Timeout error for {symbol} after {retries} attempts")

        except aiohttp.ClientError as e:
            logger.warning(f"Сетевая ошибка для {symbol} на попытке {attempt + 1}/{retries}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(1.0 * (attempt + 1))
            else:
                raise Exception(f"Network error for {symbol}: {e}")

        except Exception as e:
            logger.error(f"Ошибка получения данных для {symbol} на попытке {attempt + 1}/{retries}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
            else:
                raise

    raise Exception(f"Failed to get data for {symbol} after {retries} attempts")


async def get_klines_bulk(symbols: List[str], interval: str = "15",
                          limit: int = 200, max_concurrent: int = 10) -> Dict[str, List[List[str]]]:
    """
    НОВОЕ: Массовое получение данных свечей с контролем параллелизма
    Оптимизировано для обработки больших объемов данных
    """
    if not symbols:
        return {}

    logger.info(f"Массовое получение данных для {len(symbols)} символов "
                f"({interval}m, limit={limit})")

    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}

    async def fetch_single_symbol(symbol: str):
        async with semaphore:
            try:
                klines = await get_klines_async(symbol, interval, limit)
                return symbol, klines
            except Exception as e:
                logger.error(f"Ошибка для {symbol}: {e}")
                return symbol, []

    # Создаем задачи
    tasks = [fetch_single_symbol(symbol) for symbol in symbols]

    # Обрабатываем батчами для контроля нагрузки
    batch_size = min(max_concurrent, len(tasks))
    completed = 0

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)

        for result in batch_results:
            if isinstance(result, tuple):
                symbol, klines = result
                results[symbol] = klines
            else:
                logger.error(f"Исключение в батче: {result}")

        completed += len(batch)
        logger.debug(f"Обработано {completed}/{len(symbols)} символов")

        # Небольшая пауза между батчами
        if i + batch_size < len(tasks):
            await asyncio.sleep(0.1)

    successful = len([k for k, v in results.items() if v])
    logger.info(f"Массовое получение завершено: {successful}/{len(symbols)} успешно")

    return results


async def get_usdt_trading_pairs(use_cache: bool = True,
                                 min_volume_24h: float = None) -> List[str]:
    """
    Получает список активных USDT торговых пар.
    ОБНОВЛЕНО: кэширование и фильтрация по объему

    Args:
        use_cache: Использовать кэширование результатов
        min_volume_24h: Минимальный объем за 24ч (USD)

    Returns:
        Список символов торговых пар
    """
    # Простое кэширование в памяти
    if use_cache and hasattr(get_usdt_trading_pairs, '_cache'):
        cache_time, cached_pairs = get_usdt_trading_pairs._cache
        if time.time() - cache_time < 300:  # Кэш на 5 минут
            logger.debug(f"Используется кэш торговых пар: {len(cached_pairs)} пар")
            return cached_pairs

    url = config.exchange.INSTRUMENTS_URL
    params = {"category": config.exchange.API_CATEGORY}

    session = await get_global_session()

    try:
        start_time = time.time()

        async with session.get(url, params=params) as response:
            if response.status != 200:
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status
                )

            data = await response.json()
            request_time = time.time() - start_time

            if data.get("retCode") != 0:
                error_msg = data.get('retMsg', 'Unknown API error')
                raise Exception(f"Bybit API Error: {error_msg}")

            # Фильтруем активные USDT пары
            symbols = []
            instruments = data["result"]["list"]

            logger.info(f"Обработка {len(instruments)} инструментов за {request_time:.2f}сек")

            for item in instruments:
                symbol = item["symbol"]
                status = item["status"]

                # Базовая фильтрация
                if not (symbol.endswith('USDT') and
                        status == config.exchange.INSTRUMENT_STATUS and
                        not symbol.startswith('USDT') and  # Исключаем обратные пары
                        "-" not in symbol and  # Исключаем пары с дефисом
                        not any(char.isdigit() for char in symbol)):  # Исключаем пары с цифрами
                    continue

                # Дополнительная фильтрация по объему (если указан)
                if min_volume_24h is not None:
                    try:
                        volume_24h = float(item.get('turnover24h', '0'))
                        if volume_24h < min_volume_24h:
                            continue
                    except (ValueError, TypeError):
                        continue

                symbols.append(symbol)

            # Сортируем по алфавиту для стабильности
            symbols.sort()

            logger.info(f"Найдено {len(symbols)} активных USDT пар")

            # Кэшируем результат
            if use_cache:
                get_usdt_trading_pairs._cache = (time.time(), symbols)

            return symbols

    except Exception as e:
        logger.error(f"Ошибка получения списка инструментов: {e}")

        # Возвращаем кэшированный результат при ошибке, если есть
        if use_cache and hasattr(get_usdt_trading_pairs, '_cache'):
            cache_time, cached_pairs = get_usdt_trading_pairs._cache
            logger.warning(f"Используется устаревший кэш ({time.time() - cache_time:.0f}сек назад)")
            return cached_pairs

        raise


async def get_market_data_for_pairs(pairs: List[str], timeframes: List[str] = None,
                                    limits: Dict[str, int] = None) -> Dict[str, Dict[str, List[List[str]]]]:
    """
    НОВОЕ: Получение рыночных данных для множества пар и таймфреймов
    Оптимизировано для финального анализа с большими объемами данных

    Args:
        pairs: Список торговых пар
        timeframes: Список таймфреймов (по умолчанию ["5", "15"])
        limits: Лимиты свечей для каждого таймфрейма

    Returns:
        Dict[pair][timeframe] = List[candles]
    """
    if timeframes is None:
        timeframes = [config.timeframe.ENTRY_TF, config.timeframe.CONTEXT_TF]

    if limits is None:
        limits = {
            config.timeframe.ENTRY_TF: config.timeframe.DETAILED_CANDLES_5M,
            config.timeframe.CONTEXT_TF: config.timeframe.DETAILED_CANDLES_15M
        }

    logger.info(f"Получение данных для {len(pairs)} пар, таймфреймы: {timeframes}")

    results = {}
    total_requests = len(pairs) * len(timeframes)
    completed = 0

    # Контролируем параллелизм для не перегрузки API
    semaphore = asyncio.Semaphore(8)  # Умеренный параллелизм

    async def fetch_pair_timeframe(pair: str, timeframe: str):
        nonlocal completed
        async with semaphore:
            try:
                limit = limits.get(timeframe, 200)
                klines = await get_klines_async(pair, timeframe, limit)

                completed += 1
                if completed % 20 == 0:  # Логируем прогресс
                    logger.info(f"Прогресс: {completed}/{total_requests}")

                return pair, timeframe, klines
            except Exception as e:
                logger.error(f"Ошибка для {pair} {timeframe}m: {e}")
                completed += 1
                return pair, timeframe, []

    # Создаем все задачи
    tasks = []
    for pair in pairs:
        for timeframe in timeframes:
            tasks.append(fetch_pair_timeframe(pair, timeframe))

    # Выполняем батчами
    batch_size = 40
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)

        # Обрабатываем результаты
        for result in batch_results:
            if isinstance(result, tuple):
                pair, timeframe, klines = result

                if pair not in results:
                    results[pair] = {}

                results[pair][timeframe] = klines
            else:
                logger.error(f"Исключение в батче данных: {result}")

        # Небольшая пауза между батчами
        if i + batch_size < len(tasks):
            await asyncio.sleep(0.2)

    successful_pairs = len([p for p, data in results.items()
                            if any(len(candles) > 0 for candles in data.values())])

    logger.info(f"Получены данные для {successful_pairs}/{len(pairs)} пар")
    return results


async def get_detailed_klines(symbol: str, interval: str, limit: int,
                              split_requests: bool = True) -> List[List[str]]:
    """
    НОВОЕ: Получение больших объемов исторических данных с разбивкой запросов
    Для случаев когда нужно больше 1000 свечей

    Args:
        symbol: Торговая пара
        interval: Интервал
        limit: Количество свечей (может быть > 1000)
        split_requests: Разбивать большие запросы на части
    """
    if limit <= 1000 or not split_requests:
        return await get_klines_async(symbol, interval, min(limit, 1000))

    logger.info(f"Получение {limit} свечей для {symbol} {interval}m с разбивкой запросов")

    all_klines = []
    current_limit = limit
    end_time = None  # Для Bybit API можно использовать параметр endTime

    while current_limit > 0:
        batch_limit = min(current_limit, 1000)

        try:
            # Параметры для запроса
            params = {
                "category": config.exchange.API_CATEGORY,
                "symbol": symbol,
                "interval": interval,
                "limit": batch_limit
            }

            # Добавляем endTime если не первый запрос
            if end_time:
                params["endTime"] = str(end_time)

            session = await get_global_session()
            async with session.get(config.exchange.KLINE_URL, params=params) as response:
                if response.status != 200:
                    logger.error(f"HTTP {response.status} для {symbol}")
                    break

                data = await response.json()

                if data.get("retCode") != 0:
                    logger.error(f"API ошибка для {symbol}: {data.get('retMsg')}")
                    break

                batch_klines = data["result"]["list"]
                if not batch_klines:
                    break

                # Сортируем по времени (старые первые)
                if len(batch_klines) > 1 and int(batch_klines[0][0]) > int(batch_klines[-1][0]):
                    batch_klines.reverse()

                # Избегаем дубликатов
                if all_klines:
                    # Проверяем пересечение по времени
                    last_time = int(all_klines[-1][0])
                    batch_klines = [k for k in batch_klines if int(k[0]) < last_time]

                all_klines = batch_klines + all_klines  # Добавляем в начало
                current_limit -= len(batch_klines)

                # Устанавливаем endTime для следующего запроса
                if batch_klines:
                    end_time = int(batch_klines[0][0]) - 1  # На 1мс раньше первой свечи

                logger.debug(f"Получено {len(batch_klines)} свечей, всего: {len(all_klines)}")

                # Пауза между запросами
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Ошибка в детальном получении данных для {symbol}: {e}")
            break

    # Убираем последнюю незакрытую свечу
    if all_klines:
        all_klines = all_klines[:-1]

    logger.info(f"Получено {len(all_klines)} свечей для {symbol} {interval}m")
    return all_klines


async def validate_market_data(symbol: str, timeframes: List[str] = None) -> Dict[str, Any]:
    """
    НОВОЕ: Валидация качества рыночных данных
    Проверяет доступность и качество данных перед анализом
    """
    if timeframes is None:
        timeframes = [config.timeframe.ENTRY_TF, config.timeframe.CONTEXT_TF]

    validation_results = {
        'symbol': symbol,
        'valid': False,
        'timeframes': {},
        'overall_quality': 0.0
    }

    quality_scores = []

    for tf in timeframes:
        tf_result = {
            'available': False,
            'count': 0,
            'quality_score': 0.0,
            'issues': []
        }

        try:
            # Получаем небольшое количество свечей для проверки
            test_klines = await get_klines_async(symbol, tf, 50)

            if test_klines:
                tf_result['available'] = True
                tf_result['count'] = len(test_klines)

                # Проверяем качество данных
                quality_score = 100.0

                # Проверка 1: Достаточное количество данных
                if len(test_klines) < 20:
                    quality_score -= 30
                    tf_result['issues'].append('insufficient_data')

                # Проверка 2: Отсутствие пропусков во времени
                time_intervals = []
                for i in range(1, min(len(test_klines), 10)):
                    interval = int(test_klines[i][0]) - int(test_klines[i - 1][0])
                    time_intervals.append(interval)

                if time_intervals:
                    expected_interval = int(tf) * 60 * 1000  # в миллисекундах
                    avg_interval = sum(time_intervals) / len(time_intervals)

                    if abs(avg_interval - expected_interval) > expected_interval * 0.1:
                        quality_score -= 20
                        tf_result['issues'].append('irregular_intervals')

                # Проверка 3: Нулевые значения
                for kline in test_klines[:5]:
                    if any(float(val) == 0 for val in kline[1:6]):  # OHLCV
                        quality_score -= 10
                        tf_result['issues'].append('zero_values')
                        break

                tf_result['quality_score'] = max(0, quality_score)
                quality_scores.append(quality_score)

        except Exception as e:
            tf_result['issues'].append(f'error: {str(e)}')
            logger.warning(f"Ошибка валидации {symbol} {tf}m: {e}")

        validation_results['timeframes'][tf] = tf_result

    # Общая оценка качества
    if quality_scores:
        validation_results['overall_quality'] = sum(quality_scores) / len(quality_scores)
        validation_results['valid'] = validation_results['overall_quality'] >= 70.0

    return validation_results


async def cleanup_global_session():
    """Очистка глобальной сессии при завершении работы."""
    global _global_session
    if _global_session and not _global_session.closed:
        await _global_session.close()
        _global_session = None
        logger.info("Глобальная HTTP сессия закрыта")


def clear_cache():
    """
    НОВОЕ: Очистка всех кэшей
    """
    if hasattr(get_usdt_trading_pairs, '_cache'):
        delattr(get_usdt_trading_pairs, '_cache')
        logger.info("Кэш торговых пар очищен")


async def get_connection_stats() -> Dict[str, Any]:
    """
    НОВОЕ: Получение статистики соединений
    """
    global _global_session

    stats = {
        'session_active': _global_session is not None and not _global_session.closed,
        'session_created': _global_session is not None,
        'cache_active': hasattr(get_usdt_trading_pairs, '_cache')
    }

    if stats['session_active']:
        connector = _global_session.connector
        if hasattr(connector, '_conns'):
            stats['active_connections'] = len(connector._conns)

        stats['timeout_settings'] = {
            'total': _global_session.timeout.total,
            'connect': _global_session.timeout.connect
        }

    if stats['cache_active']:
        cache_time, cached_pairs = get_usdt_trading_pairs._cache
        stats['cache_age'] = int(time.time() - cache_time)
        stats['cached_pairs_count'] = len(cached_pairs)

    return stats


# Вспомогательные функции для массовой обработки

async def batch_validate_symbols(symbols: List[str],
                                 max_concurrent: int = 5) -> Dict[str, bool]:
    """
    НОВОЕ: Массовая валидация символов
    """
    if not symbols:
        return {}

    logger.info(f"Валидация {len(symbols)} символов")

    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}

    async def validate_single(symbol: str):
        async with semaphore:
            try:
                validation = await validate_market_data(symbol)
                return symbol, validation['valid']
            except Exception as e:
                logger.error(f"Ошибка валидации {symbol}: {e}")
                return symbol, False

    tasks = [validate_single(symbol) for symbol in symbols]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in batch_results:
        if isinstance(result, tuple):
            symbol, is_valid = result
            results[symbol] = is_valid
        else:
            logger.error(f"Исключение в валидации: {result}")

    valid_count = sum(results.values())
    logger.info(f"Валидация завершена: {valid_count}/{len(symbols)} валидных символов")

    return results


async def get_market_overview(top_pairs: int = 50) -> Dict[str, Any]:
    """
    НОВОЕ: Получение обзора рынка для анализа
    """
    logger.info(f"Получение обзора рынка для топ-{top_pairs} пар")

    try:
        # Получаем список пар
        all_pairs = await get_usdt_trading_pairs(min_volume_24h=config.trading.MIN_LIQUIDITY_VOLUME)

        if not all_pairs:
            return {'error': 'No trading pairs found'}

        # Берем топ пары
        selected_pairs = all_pairs[:top_pairs]

        # Получаем базовые данные для анализа волатильности
        market_data = {}
        semaphore = asyncio.Semaphore(10)

        async def get_pair_overview(pair: str):
            async with semaphore:
                try:
                    # Получаем небольшое количество свечей для базового анализа
                    klines_5m = await get_klines_async(pair, "5", 50)
                    if klines_5m and len(klines_5m) >= 20:
                        # Простая оценка волатильности
                        closes = [float(k[4]) for k in klines_5m]
                        volatility = (max(closes) - min(closes)) / closes[-1] * 100

                        return pair, {
                            'current_price': closes[-1],
                            'volatility_50_periods': round(volatility, 2),
                            'data_quality': 'good' if len(klines_5m) >= 40 else 'fair'
                        }
                except Exception as e:
                    logger.debug(f"Ошибка обзора для {pair}: {e}")

                return pair, None

        # Получаем обзор
        tasks = [get_pair_overview(pair) for pair in selected_pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, tuple) and result[1] is not None:
                pair, data = result
                market_data[pair] = data

        # Статистика рынка
        if market_data:
            volatilities = [data['volatility_50_periods'] for data in market_data.values()]

            overview = {
                'timestamp': int(time.time()),
                'total_pairs_analyzed': len(market_data),
                'market_stats': {
                    'avg_volatility': round(sum(volatilities) / len(volatilities), 2),
                    'max_volatility': max(volatilities),
                    'min_volatility': min(volatilities),
                    'high_volatility_pairs': len([v for v in volatilities if v > 3.0]),
                    'low_volatility_pairs': len([v for v in volatilities if v < 1.0])
                },
                'pairs_data': market_data
            }

            logger.info(f"Обзор рынка готов: {len(market_data)} пар, "
                        f"средняя волатильность: {overview['market_stats']['avg_volatility']}%")

            return overview
        else:
            return {'error': 'No market data obtained'}

    except Exception as e:
        logger.error(f"Ошибка получения обзора рынка: {e}")
        return {'error': str(e)}