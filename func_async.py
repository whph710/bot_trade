import aiohttp
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


async def get_klines_async(symbol: str, interval: int = 15, limit: int = 100) -> List[List[str]]:
    """
    Асинхронно получает данные свечей для торговой пары.

    Args:
        symbol: Символ торговой пары (например, 'BTCUSDT')
        interval: Интервал свечей в минутах (по умолчанию 15)
        limit: Количество свечей для получения (по умолчанию 120)

    Returns:
        Список свечных данных в формате:
        [['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'], ...]

    Raises:
        Exception: При ошибке API или сети
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": str(interval),
        "limit": limit
    }

    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                data = await response.json()

                if data.get("retCode") != 0:
                    error_msg = data.get('retMsg', 'Unknown API error')
                    logger.error(f"API ошибка для {symbol}: {error_msg}")
                    raise Exception(f"Bybit API Error for {symbol}: {error_msg}")

                klines = data["result"]["list"]

                # Исключаем последнюю (незакрытую) свечу
                if klines:
                    klines = klines[::-1]
                    klines = klines[:-1]

                logger.debug(f"Получено {len(klines)} свечей для {symbol}")
                return klines

    except aiohttp.ClientError as e:
        logger.error(f"Сетевая ошибка при получении данных для {symbol}: {e}")
        raise Exception(f"Network error for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Неожиданная ошибка при получении данных для {symbol}: {e}")
        raise


async def get_usdt_linear_symbols() -> List[str]:
    """
    Асинхронно получает список всех доступных USDT торговых пар.

    Returns:
        Список символов торговых пар (например, ['BTCUSDT', 'ETHUSDT', ...])

    Raises:
        Exception: При ошибке API или сети
    """
    url = "https://api.bybit.com/v5/market/instruments-info"
    params = {
        "category": "linear"  # USDT-маржинальные фьючерсы
    }

    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                data = await response.json()

                if data.get("retCode") != 0:
                    error_msg = data.get('retMsg', 'Unknown API error')
                    logger.error(f"API ошибка при получении списка инструментов: {error_msg}")
                    raise Exception(f"Bybit API Error: {error_msg}")

                # Фильтруем только USDT пары, исключаем индексы и особые инструменты
                symbols = []
                for item in data["result"]["list"]:
                    if (item.get("quoteCoin") == "USDT" and
                            "-" not in item["symbol"] and
                            "1" not in item["symbol"] and
                            item.get("status") == "Trading"):  # Только торгуемые инструменты
                        symbols.append(item["symbol"])

                logger.info(f"Найдено {len(symbols)} активных USDT торговых пар")
                return symbols

    except aiohttp.ClientError as e:
        logger.error(f"Сетевая ошибка при получении списка инструментов: {e}")
        raise Exception(f"Network error: {e}")
    except Exception as e:
        logger.error(f"Неожиданная ошибка при получении списка инструментов: {e}")
        raise


async def get_ticker_info(symbol: str) -> Dict[str, Any]:
    """
    Получает текущую информацию о цене и объеме для торговой пары.

    Args:
        symbol: Символ торговой пары

    Returns:
        Словарь с информацией о тикере
    """
    url = "https://api.bybit.com/v5/market/tickers"
    params = {
        "category": "linear",
        "symbol": symbol
    }

    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                data = await response.json()

                if data.get("retCode") != 0:
                    error_msg = data.get('retMsg', 'Unknown API error')
                    raise Exception(f"Bybit API Error for {symbol}: {error_msg}")

                ticker_list = data["result"]["list"]
                if not ticker_list:
                    raise Exception(f"No ticker data found for {symbol}")

                return ticker_list[0]

    except Exception as e:
        logger.error(f"Ошибка получения тикера для {symbol}: {e}")
        raise