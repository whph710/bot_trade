"""
Упрощенный модуль для работы с Bybit API
Фокус на скорости и надежности
"""

import aiohttp
import asyncio
import logging
from typing import List, Dict, Optional
from config import config

logger = logging.getLogger(__name__)


class BybitClient:
    """Упрощенный клиент для Bybit"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Получение HTTP сессии"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=config.exchange.TIMEOUT)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def close(self):
        """Закрытие сессии"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def get_klines(self, symbol: str, interval: str, limit: int) -> List[List]:
        """
        Получение свечных данных
        Возвращает: List[timestamp, open, high, low, close, volume]
        Порядок: от старых к новым (БЕЗ последней незакрытой свечи)
        """
        session = await self._get_session()

        url = f"{config.exchange.BASE_URL}{config.exchange.KLINE_ENDPOINT}"
        params = {
            "category": config.exchange.CATEGORY,
            "symbol": symbol,
            "interval": interval,
            "limit": limit + 1  # +1 чтобы убрать последнюю незакрытую
        }

        try:
            async with session.get(url, params=params) as response:
                data = await response.json()

                if data.get("retCode") != 0:
                    raise Exception(f"API Error: {data.get('retMsg', 'Unknown')}")

                klines = data["result"]["list"]
                if not klines:
                    raise Exception(f"No data for {symbol}")

                # Bybit возвращает от новых к старым - переворачиваем
                klines.reverse()

                # Убираем последнюю незакрытую свечу
                closed_klines = klines[:-1]

                logger.debug(f"Got {len(closed_klines)} candles for {symbol}")
                return closed_klines

        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            raise

    async def get_active_pairs(self) -> List[str]:
        """Получение активных USDT пар"""
        session = await self._get_session()

        url = f"{config.exchange.BASE_URL}{config.exchange.INSTRUMENTS_ENDPOINT}"
        params = {"category": config.exchange.CATEGORY}

        try:
            async with session.get(url, params=params) as response:
                data = await response.json()

                if data.get("retCode") != 0:
                    raise Exception(f"API Error: {data.get('retMsg', 'Unknown')}")

                symbols = []
                for item in data["result"]["list"]:
                    symbol = item["symbol"]
                    status = item["status"]

                    # Только активные USDT пары
                    if (symbol.endswith('USDT') and
                            status == 'Trading' and
                            not symbol.startswith('USDT') and
                            "-" not in symbol):
                        symbols.append(symbol)

                logger.info(f"Found {len(symbols)} active USDT pairs")
                return symbols

        except Exception as e:
            logger.error(f"Error getting pairs: {e}")
            raise


# Глобальный клиент
bybit_client = BybitClient()


async def get_candles(symbol: str, timeframe: str, limit: int) -> List[List]:
    """Удобная функция для получения свечей"""
    return await bybit_client.get_klines(symbol, timeframe, limit)


async def get_usdt_pairs() -> List[str]:
    """Удобная функция для получения пар"""
    return await bybit_client.get_active_pairs()


async def cleanup():
    """Очистка ресурсов"""
    await bybit_client.close()