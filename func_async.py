import aiohttp
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


async def get_klines_async(symbol: str, interval: str = "15", limit: int = 200) -> List[List[str]]:
    """
    Асинхронно получает данные свечей для торговой пары.
    ИСПРАВЛЕНО: правильная обработка порядка свечей и отрезание незакрытой свечи

    Args:
        symbol: Торговая пара (например, BTCUSDT)
        interval: Интервал свечей (по умолчанию "15")
        limit: Количество свечей (по умолчанию 200)

    Returns:
        Список свечей в формате Bybit [timestamp, open, high, low, close, volume, turnover]
        ОТ СТАРЫХ К НОВЫМ, БЕЗ ПОСЛЕДНЕЙ НЕЗАКРЫТОЙ СВЕЧИ
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit + 1  # Запрашиваем +1 свечу чтобы гарантированно отрезать последнюю
    }

    try:
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                data = await response.json()

                if data.get("retCode") != 0:
                    error_msg = data.get('retMsg', 'Unknown API error')
                    logger.error(f"API ошибка для {symbol}: {error_msg}")
                    raise Exception(f"Bybit API Error for {symbol}: {error_msg}")

                klines = data["result"]["list"]

                if not klines or len(klines) < 2:
                    logger.error(f"Недостаточно данных для {symbol}")
                    raise Exception(f"Insufficient data for {symbol}")

                # Проверяем порядок свечей (Bybit возвращает от новых к старым)
                if len(klines) > 1:
                    if int(klines[0][0]) > int(klines[-1][0]):
                        # Данные от новых к старым - разворачиваем
                        klines.reverse()
                        logger.debug(f"Развернул порядок свечей для {symbol} (новые→старые)")
                    else:
                        logger.debug(f"Порядок свечей корректный для {symbol} (старые→новые)")

                # ВСЕГДА отрезаем последнюю незакрытую свечу
                closed_klines = klines[:-1]

                logger.debug(f"Получено {len(closed_klines)} закрытых свечей для {symbol} "
                             f"(от {closed_klines[0][0] if closed_klines else 'N/A'} "
                             f"до {closed_klines[-1][0] if closed_klines else 'N/A'})")

                return closed_klines

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
        timeout = aiohttp.ClientTimeout(total=45)
        async with aiohttp.ClientSession(timeout=timeout) as session:
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