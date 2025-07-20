import aiohttp
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


async def get_klines_async(symbol: str, interval: int = 15, limit: int = 100) -> List[List[str]]:
    """
    Асинхронно получает данные свечей для торговой пары.
    Оптимизированная версия без лишних операций.
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": str(interval),
        "limit": limit + 1  # Запрашиваем +1 для исключения незакрытой
    }

    try:
        timeout = aiohttp.ClientTimeout(total=20)  # Уменьшили таймаут
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                data = await response.json()

                if data.get("retCode") != 0:
                    error_msg = data.get('retMsg', 'Unknown API error')
                    logger.error(f"API ошибка для {symbol}: {error_msg}")
                    raise Exception(f"Bybit API Error for {symbol}: {error_msg}")

                klines = data["result"]["list"]

                # Оптимизированная обработка: сразу в правильном порядке
                if klines and len(klines) > 1:
                    # Проверяем порядок только один раз
                    if int(klines[0][0]) > int(klines[-1][0]):
                        klines.reverse()
                    # Исключаем последнюю незакрытую свечу
                    # klines = klines[:-1]

                logger.debug(f"Получено {len(klines)} свечей для {symbol}")
                return klines

    except aiohttp.ClientError as e:
        logger.error(f"Сетевая ошибка для {symbol}: {e}")
        raise Exception(f"Network error for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Ошибка получения данных для {symbol}: {e}")
        raise


async def get_usdt_linear_symbols() -> List[str]:
    """
    Оптимизированное получение списка USDT торговых пар.
    """
    url = "https://api.bybit.com/v5/market/instruments-info"
    params = {"category": "linear"}

    try:
        timeout = aiohttp.ClientTimeout(total=15)  # Уменьшили таймаут
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                data = await response.json()

                if data.get("retCode") != 0:
                    error_msg = data.get('retMsg', 'Unknown API error')
                    raise Exception(f"Bybit API Error: {error_msg}")

                # Оптимизированная фильтрация
                symbols = [
                    item["symbol"] for item in data["result"]["list"]
                    if (item.get("quoteCoin") == "USDT" and
                        item.get("status") == "Trading" and
                        "-" not in item["symbol"] and
                        not any(char.isdigit() for char in item["symbol"]))
                ]

                logger.info(f"Найдено {len(symbols)} активных USDT пар")
                return symbols

    except Exception as e:
        logger.error(f"Ошибка получения списка инструментов: {e}")
        raise


async def get_ticker_info(symbol: str) -> Dict[str, Any]:
    """Получает информацию о тикере с оптимизированным таймаутом."""
    url = "https://api.bybit.com/v5/market/tickers"
    params = {"category": "linear", "symbol": symbol}

    try:
        timeout = aiohttp.ClientTimeout(total=10)  # Уменьшили таймаут
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                data = await response.json()

                if data.get("retCode") != 0:
                    raise Exception(f"Bybit API Error for {symbol}: {data.get('retMsg', 'Unknown')}")

                ticker_list = data["result"]["list"]
                if not ticker_list:
                    raise Exception(f"No ticker data for {symbol}")

                return ticker_list[0]

    except Exception as e:
        logger.error(f"Ошибка получения тикера для {symbol}: {e}")
        raise