import aiohttp
import logging
import asyncio
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


async def get_klines_async(symbol: str, interval: str = "5", limit: int = 200) -> List[List[str]]:
    """
    Асинхронно получает данные свечей для торговой пары.
    ИЗМЕНЕНО: по умолчанию интервал "5" (5 минут) для скальпинга

    Args:
        symbol: Торговая пара (например, BTCUSDT)
        interval: Интервал свечей (по умолчанию "5" для 5M скальпинга)
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
        timeout = aiohttp.ClientTimeout(total=15)  # Уменьшен таймаут для скорости
        async with aiohttp.ClientSession(timeout=timeout) as session:
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

                logger.debug(f"Получено {len(klines)} 5M свечей для {symbol}")
                return klines[:-1]  # Исключаем последнюю незавершённую свечу

    except aiohttp.ClientError as e:
        logger.error(f"Сетевая ошибка для {symbol}: {e}")
        raise Exception(f"Network error for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Ошибка получения 5M данных для {symbol}: {e}")
        raise


async def get_usdt_trading_pairs() -> List[str]:
    """
    Получает список активных USDT торговых пар.
    ОПТИМИЗИРОВАНО: фильтрация для высоколиквидных пар

    Returns:
        Список символов торговых пар
    """
    url = "https://api.bybit.com/v5/market/instruments-info"
    params = {"category": "linear"}

    try:
        timeout = aiohttp.ClientTimeout(total=12)  # Быстрее для скальпинга
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                data = await response.json()

                if data.get("retCode") != 0:
                    error_msg = data.get('retMsg', 'Unknown API error')
                    raise Exception(f"Bybit API Error: {error_msg}")

                # Фильтруем активные USDT пары с дополнительными условиями для скальпинга
                symbols = []
                for item in data["result"]["list"]:
                    symbol = item["symbol"]
                    status = item["status"]

                    # Берем только USDT пары и активные
                    if (symbol.endswith('USDT') and
                            status == 'Trading' and
                            not symbol.startswith('USDT') and  # Исключаем обратные пары
                            "-" not in symbol and  # Исключаем пары с дефисом
                            not any(char.isdigit() for char in symbol) and  # Исключаем пары с цифрами
                            len(symbol) <= 10):  # Исключаем слишком длинные названия
                        symbols.append(symbol)

                logger.info(f"Найдено {len(symbols)} активных USDT пар для 5M скальпинга")
                return symbols

    except Exception as e:
        logger.error(f"Ошибка получения списка инструментов: {e}")
        raise


async def get_market_summary_async(symbol: str) -> Dict[str, Any]:
    """
    Получает краткую сводку по рынку для быстрого фильтра
    НОВАЯ ФУНКЦИЯ для предварительной фильтрации

    Args:
        symbol: Торговая пара

    Returns:
        Словарь с ключевыми метриками или пустой словарь при ошибке
    """
    url = "https://api.bybit.com/v5/market/tickers"
    params = {
        "category": "linear",
        "symbol": symbol
    }

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                data = await response.json()

                if data.get("retCode") != 0 or not data.get("result", {}).get("list"):
                    return {}

                ticker = data["result"]["list"][0]

                return {
                    'symbol': symbol,
                    'last_price': float(ticker.get('lastPrice', 0)),
                    'volume_24h': float(ticker.get('volume24h', 0)),
                    'turnover_24h': float(ticker.get('turnover24h', 0)),
                    'price_change_24h': float(ticker.get('price24hPcnt', 0)) * 100,  # В процентах
                    'high_24h': float(ticker.get('highPrice24h', 0)),
                    'low_24h': float(ticker.get('lowPrice24h', 0)),
                }

    except Exception as e:
        logger.debug(f"Ошибка получения сводки для {symbol}: {e}")
        return {}


async def filter_high_volume_pairs(pairs: List[str], min_volume_usdt: float = 50_000_000) -> List[str]:
    """
    Фильтрует пары по объему торгов
    НОВАЯ ФУНКЦИЯ для ускорения - анализируем только высоколиквидные пары

    Args:
        pairs: Список торговых пар
        min_volume_usdt: Минимальный объем в USDT за 24ч (50M по умолчанию)

    Returns:
        Отфильтрованный список высоколиквидных пар
    """
    logger.info(f"🔍 Фильтрация {len(pairs)} пар по объему (мин. ${min_volume_usdt:,.0f})")

    # Получаем сводки параллельно батчами
    batch_size = 20
    filtered_pairs = []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]

        # Создаем задачи для батча
        tasks = [get_market_summary_async(pair) for pair in batch]

        # Выполняем параллельно
        summaries = await asyncio.gather(*tasks, return_exceptions=True)

        # Фильтруем по объему
        for summary in summaries:
            if isinstance(summary, dict) and summary:
                turnover = summary.get('turnover_24h', 0)
                if turnover >= min_volume_usdt:
                    filtered_pairs.append(summary['symbol'])

        # Небольшая пауза между батчами
        if i + batch_size < len(pairs):
            await asyncio.sleep(0.1)

    logger.info(f"✅ Отобрано {len(filtered_pairs)} высоколиквидных пар")
    return filtered_pairs


# Добавляем импорт для asyncio
