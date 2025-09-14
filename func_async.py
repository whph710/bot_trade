"""
Упрощенный API клиент - только необходимые функции
Убрано дублирование, оставлен минимум для работы
"""

import aiohttp
import asyncio
import logging
from typing import List, Dict, Any
from config import config

logger = logging.getLogger(__name__)

# Глобальная сессия для переиспользования
_session = None


async def get_session():
    """Получить переиспользуемую сессию"""
    global _session
    if _session is None or _session.closed:
        timeout = aiohttp.ClientTimeout(total=15)
        connector = aiohttp.TCPConnector(
            limit=20,
            limit_per_host=10,
            keepalive_timeout=60
        )
        _session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    return _session


async def fetch_klines(symbol: str, interval: str, limit: int) -> List[List[str]]:
    """
    Получение свечей с Bybit
    """
    session = await get_session()

    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    try:
        async with session.get("https://api.bybit.com/v5/market/kline", params=params) as response:
            if response.status != 200:
                return []

            data = await response.json()
            if data.get("retCode") != 0:
                return []

            klines = data["result"]["list"]

            # Сортируем по времени (старые → новые)
            if klines and len(klines) > 1:
                if int(klines[0][0]) > int(klines[-1][0]):
                    klines.reverse()

            # Убираем последнюю незавершенную свечу
            return klines[:-1] if klines else []

    except Exception as e:
        logger.error(f"Ошибка получения свечей {symbol}: {e}")
        return []


async def get_trading_pairs() -> List[str]:
    """
    Получение списка торговых пар USDT
    """
    session = await get_session()

    params = {"category": "linear"}

    try:
        async with session.get("https://api.bybit.com/v5/market/instruments-info", params=params) as response:
            if response.status != 200:
                return []

            data = await response.json()
            if data.get("retCode") != 0:
                return []

            # Фильтруем только активные USDT пары
            symbols = []
            for item in data["result"]["list"]:
                symbol = item["symbol"]
                status = item["status"]

                if (status == 'Trading' and
                        symbol.endswith('USDT') and
                        not symbol.startswith('USDT') and
                        "-" not in symbol):
                    symbols.append(symbol)

            return symbols  # Ограничиваем для скорости

    except Exception as e:
        logger.error(f"Ошибка получения торговых пар: {e}")
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']  # Фаллбек


async def batch_fetch_klines(requests: List[Dict]) -> List[Dict]:
    """
    Массовое получение свечей
    """
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT)

    async def bounded_request(req):
        async with semaphore:
            klines = await fetch_klines(
                req['symbol'],
                req.get('interval', '5'),
                req.get('limit', 100)
            )
            return {
                'symbol': req['symbol'],
                'klines': klines,
                'success': len(klines) > 0
            }

    tasks = [bounded_request(req) for req in requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return [r for r in results if isinstance(r, dict) and r.get('success')]


async def cleanup():
    """Очистка ресурсов"""
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None