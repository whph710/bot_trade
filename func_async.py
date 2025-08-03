import aiohttp
import logging
from typing import List, Dict, Any, Optional
import asyncio
import time

logger = logging.getLogger(__name__)

# Глобальная сессия для переиспользования соединений
_session = None


async def get_session():
    """Получение глобальной HTTP сессии"""
    global _session
    if _session is None or _session.closed:
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        _session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    return _session


async def close_session():
    """Закрытие глобальной HTTP сессии"""
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None


async def get_klines_async(symbol: str, interval: str = "15", limit: int = 200) -> List[List[str]]:
    """
    Асинхронно получает данные свечей для торговой пары.

    Args:
        symbol: Торговая пара (например, BTCUSDT)
        interval: Интервал свечей (по умолчанию "15")
        limit: Количество свечей (по умолчанию 200)

    Returns:
        Список свечей в формате Bybit [timestamp, open, high, low, close, volume, turnover]
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    try:
        session = await get_session()
        async with session.get(url, params=params) as response:
            data = await response.json()

            if data.get("retCode") != 0:
                error_msg = data.get('retMsg', 'Unknown API error')
                logger.error(f"API ошибка для {symbol}: {error_msg}")
                raise Exception(f"Bybit API Error for {symbol}: {error_msg}")

            klines = data["result"]["list"]

            # Проверяем порядок и при необходимости разворачиваем (от старых к новым)
            if klines and len(klines) > 1:
                if int(klines[0][0]) > int(klines[-1][0]):
                    klines.reverse()

            logger.debug(f"Получено {len(klines)} свечей для {symbol}")
            return klines[:-1]  # Убираем последнюю незавершенную свечу

    except aiohttp.ClientError as e:
        logger.error(f"Сетевая ошибка для {symbol}: {e}")
        raise Exception(f"Network error for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Ошибка получения данных для {symbol}: {e}")
        raise


async def get_usdt_trading_pairs() -> List[str]:
    """
    Получает список активных USDT торговых пар.

    Returns:
        Список символов торговых пар
    """
    url = "https://api.bybit.com/v5/market/instruments-info"
    params = {"category": "linear"}

    try:
        session = await get_session()
        async with session.get(url, params=params) as response:
            data = await response.json()

            if data.get("retCode") != 0:
                error_msg = data.get('retMsg', 'Unknown API error')
                raise Exception(f"Bybit API Error: {error_msg}")

            # Фильтруем активные USDT пары
            symbols = []
            for item in data["result"]["list"]:
                symbol = item["symbol"]
                status = item["status"]

                # Берем только USDT пары и активные
                if (symbol.endswith('USDT') and
                        status == 'Trading' and
                        not symbol.startswith('USDT') and  # Исключаем обратные пары
                        "-" not in symbol and  # Исключаем пары с дефисом
                        not any(char.isdigit() for char in symbol)):  # Исключаем пары с цифрами
                    symbols.append(symbol)

            logger.info(f"Найдено {len(symbols)} активных USDT пар")
            return symbols

    except Exception as e:
        logger.error(f"Ошибка получения списка инструментов: {e}")
        raise


async def get_orderbook_async(symbol: str, limit: int = 25) -> Optional[Dict[str, Any]]:
    """
    Асинхронно получает стакан заявок для торговой пары.

    Args:
        symbol: Торговая пара (например, BTCUSDT)
        limit: Глубина стакана (по умолчанию 25)

    Returns:
        Словарь с bid и ask заявками
    """
    url = "https://api.bybit.com/v5/market/orderbook"
    params = {
        "category": "linear",
        "symbol": symbol,
        "limit": limit
    }

    try:
        session = await get_session()
        async with session.get(url, params=params) as response:
            data = await response.json()

            if data.get("retCode") != 0:
                error_msg = data.get('retMsg', 'Unknown API error')
                logger.error(f"Orderbook API ошибка для {symbol}: {error_msg}")
                return None

            result = data["result"]
            return {
                'bids': result.get('b', []),  # [[price, size], ...]
                'asks': result.get('a', []),  # [[price, size], ...]
                'timestamp': result.get('ts', 0)
            }

    except Exception as e:
        logger.error(f"Ошибка получения стакана для {symbol}: {e}")
        return None


async def get_24h_stats_async(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Асинхронно получает 24-часовую статистику для торговой пары.

    Args:
        symbol: Торговая пара (например, BTCUSDT)

    Returns:
        Словарь со статистикой
    """
    url = "https://api.bybit.com/v5/market/tickers"
    params = {
        "category": "linear",
        "symbol": symbol
    }

    try:
        session = await get_session()
        async with session.get(url, params=params) as response:
            data = await response.json()

            if data.get("retCode") != 0:
                error_msg = data.get('retMsg', 'Unknown API error')
                logger.error(f"24h stats API ошибка для {symbol}: {error_msg}")
                return None

            result = data["result"]["list"]
            if not result:
                return None

            ticker = result[0]
            return {
                'symbol': ticker.get('symbol'),
                'lastPrice': float(ticker.get('lastPrice', 0)),
                'volume': float(ticker.get('volume24h', 0)),
                'turnover': float(ticker.get('turnover24h', 0)),
                'price24hPcnt': float(ticker.get('price24hPcnt', 0)),
                'fundingRate': float(ticker.get('fundingRate', 0)),
                'openInterest': float(ticker.get('openInterest', 0)),
                'bid1Price': float(ticker.get('bid1Price', 0)),
                'ask1Price': float(ticker.get('ask1Price', 0))
            }

    except Exception as e:
        logger.error(f"Ошибка получения 24h статистики для {symbol}: {e}")
        return None


async def batch_get_24h_stats_async(symbols: List[str]) -> Dict[str, Dict]:
    """
    Получает 24h статистику для множества пар одним запросом.

    Args:
        symbols: Список торговых пар

    Returns:
        Словарь с статистикой по парам
    """
    url = "https://api.bybit.com/v5/market/tickers"
    params = {"category": "linear"}

    try:
        session = await get_session()
        async with session.get(url, params=params) as response:
            data = await response.json()

            if data.get("retCode") != 0:
                error_msg = data.get('retMsg', 'Unknown API error')
                logger.error(f"Batch 24h stats API ошибка: {error_msg}")
                return {}

            result = {}
            symbols_set = set(symbols)  # Для быстрого поиска

            for ticker in data["result"]["list"]:
                symbol = ticker.get('symbol')
                if symbol in symbols_set:
                    result[symbol] = {
                        'symbol': symbol,
                        'lastPrice': float(ticker.get('lastPrice', 0)),
                        'volume': float(ticker.get('volume24h', 0)),
                        'turnover': float(ticker.get('turnover24h', 0)),
                        'price24hPcnt': float(ticker.get('price24hPcnt', 0)),
                        'fundingRate': float(ticker.get('fundingRate', 0)),
                        'openInterest': float(ticker.get('openInterest', 0)),
                        'bid1Price': float(ticker.get('bid1Price', 0)),
                        'ask1Price': float(ticker.get('ask1Price', 0))
                    }

            logger.info(f"Получена статистика для {len(result)} пар из {len(symbols)} запрошенных")
            return result

    except Exception as e:
        logger.error(f"Ошибка пакетного получения статистики: {e}")
        return {}


async def get_funding_rate_async(symbol: str) -> float:
    """
    Асинхронно получает текущую ставку финансирования.

    Args:
        symbol: Торговая пара (например, BTCUSDT)

    Returns:
        Ставка финансирования в процентах
    """
    try:
        stats = await get_24h_stats_async(symbol)
        if stats:
            return stats.get('fundingRate', 0.0) * 100  # Конвертируем в проценты
        return 0.0
    except Exception as e:
        logger.error(f"Ошибка получения funding rate для {symbol}: {e}")
        return 0.0


async def get_open_interest_async(symbol: str) -> float:
    """
    Асинхронно получает открытый интерес.

    Args:
        symbol: Торговая пара (например, BTCUSDT)

    Returns:
        Открытый интерес
    """
    try:
        stats = await get_24h_stats_async(symbol)
        if stats:
            return stats.get('openInterest', 0.0)
        return 0.0
    except Exception as e:
        logger.error(f"Ошибка получения open interest для {symbol}: {e}")
        return 0.0


async def get_server_time_async() -> int:
    """
    Получает серверное время Bybit.

    Returns:
        Timestamp в миллисекундах
    """
    url = "https://api.bybit.com/v5/market/time"

    try:
        session = await get_session()
        async with session.get(url) as response:
            data = await response.json()

            if data.get("retCode") != 0:
                error_msg = data.get('retMsg', 'Unknown API error')
                logger.error(f"Server time API ошибка: {error_msg}")
                return 0

            return int(data["result"]["timeSecond"]) * 1000

    except Exception as e:
        logger.error(f"Ошибка получения серверного времени: {e}")
        return 0


async def check_api_connection() -> bool:
    """
    Проверяет соединение с API Bybit.

    Returns:
        True если соединение работает
    """
    try:
        server_time = await get_server_time_async()
        return server_time > 0
    except Exception as e:
        logger.error(f"Ошибка проверки соединения с API: {e}")
        return False


async def get_batch_orderbooks_async(symbols: List[str], limit: int = 10) -> Dict[str, Dict]:
    """
    Получает стаканы заявок для множества пар параллельно.

    Args:
        symbols: Список торговых пар
        limit: Глубина стакана

    Returns:
        Словарь с стаканами по парам
    """
    tasks = [get_orderbook_async(symbol, limit) for symbol in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    orderbooks = {}
    for symbol, result in zip(symbols, results):
        if isinstance(result, dict) and result is not None:
            orderbooks[symbol] = result
        elif isinstance(result, Exception):
            logger.error(f"Ошибка получения стакана для {symbol}: {result}")

    logger.info(f"Получены стаканы для {len(orderbooks)} пар из {len(symbols)} запрошенных")
    return orderbooks


async def get_batch_klines_async(symbols: List[str], interval: str = "15", limit: int = 100) -> Dict[str, List]:
    """
    Получает свечи для множества пар параллельно.

    Args:
        symbols: Список торговых пар
        interval: Интервал свечей
        limit: Количество свечей

    Returns:
        Словарь со свечами по парам
    """
    # Ограничиваем количество одновременных запросов для избежания rate limit
    semaphore = asyncio.Semaphore(20)

    async def get_klines_with_semaphore(symbol):
        async with semaphore:
            try:
                return await get_klines_async(symbol, interval, limit)
            except Exception as e:
                logger.error(f"Ошибка получения свечей для {symbol}: {e}")
                return None

    tasks = [get_klines_with_semaphore(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    klines_data = {}
    for symbol, result in zip(symbols, results):
        if isinstance(result, list) and result:
            klines_data[symbol] = result
        elif isinstance(result, Exception):
            logger.error(f"Ошибка получения свечей для {symbol}: {result}")

    logger.info(f"Получены свечи для {len(klines_data)} пар из {len(symbols)} запрошенных")
    return klines_data


async def get_market_summary_async() -> Dict[str, Any]:
    """
    Получает сводную информацию о рынке.

    Returns:
        Словарь со сводной информацией
    """
    try:
        # Получаем время сервера
        server_time = await get_server_time_async()

        # Получаем статистику по основным парам
        major_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
        stats = await batch_get_24h_stats_async(major_pairs)

        # Получаем общий список пар
        all_pairs = await get_usdt_trading_pairs()

        return {
            'server_time': server_time,
            'total_pairs': len(all_pairs),
            'major_pairs_stats': stats,
            'api_status': 'online' if server_time > 0 else 'offline'
        }

    except Exception as e:
        logger.error(f"Ошибка получения сводки рынка: {e}")
        return {
            'server_time': 0,
            'total_pairs': 0,
            'major_pairs_stats': {},
            'api_status': 'error'
        }


async def health_check_async() -> Dict[str, Any]:
    """
    Проверка здоровья API соединения.

    Returns:
        Словарь с результатами проверки
    """
    start_time = time.time()

    try:
        # Проверяем базовое соединение
        api_connected = await check_api_connection()

        # Проверяем получение данных
        test_klines = await get_klines_async('BTCUSDT', '15', 10)
        klines_ok = test_klines is not None and len(test_klines) > 0

        # Проверяем получение стакана
        test_orderbook = await get_orderbook_async('BTCUSDT', 5)
        orderbook_ok = (test_orderbook is not None and
                        'bids' in test_orderbook and
                        'asks' in test_orderbook)

        # Проверяем получение статистики
        test_stats = await get_24h_stats_async('BTCUSDT')
        stats_ok = test_stats is not None and 'volume' in test_stats

        response_time = time.time() - start_time

        return {
            'overall_status': 'healthy' if all([api_connected, klines_ok, orderbook_ok, stats_ok]) else 'degraded',
            'api_connected': api_connected,
            'klines_endpoint': 'ok' if klines_ok else 'error',
            'orderbook_endpoint': 'ok' if orderbook_ok else 'error',
            'stats_endpoint': 'ok' if stats_ok else 'error',
            'response_time_seconds': round(response_time, 2),
            'timestamp': int(time.time())
        }

    except Exception as e:
        logger.error(f"Ошибка проверки здоровья API: {e}")
        return {
            'overall_status': 'error',
            'api_connected': False,
            'klines_endpoint': 'error',
            'orderbook_endpoint': 'error',
            'stats_endpoint': 'error',
            'response_time_seconds': time.time() - start_time,
            'timestamp': int(time.time()),
            'error': str(e)
        }


# Функция для корректного завершения работы
async def cleanup_async_resources():
    """Очистка асинхронных ресурсов"""
    await close_session()
    logger.info("Асинхронные ресурсы очищены")


# Алиас для обратной совместимости
async def cleanup_http_client():
    """Алиас для cleanup_async_resources"""
    await cleanup_async_resources()