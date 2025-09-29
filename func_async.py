"""
Оптимизированный API клиент для работы с биржей Bybit
"""

import aiohttp
import asyncio
import logging
from typing import List, Dict
from config import config

logger = logging.getLogger(__name__)

# Глобальная сессия с оптимизированными настройками
_session = None
_semaphore = None

async def get_optimized_session():
    """Получить оптимизированную сессию с connection pooling"""
    global _session, _semaphore

    if _session is None or _session.closed:
        timeout = aiohttp.ClientTimeout(total=15, connect=5)
        connector = aiohttp.TCPConnector(
            limit=50,
            limit_per_host=25,
            keepalive_timeout=120,
            enable_cleanup_closed=True
        )
        _session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={'User-Agent': 'ScalpBot/2.1'}
        )
        _semaphore = asyncio.Semaphore(config.MAX_CONCURRENT)

    return _session

async def fetch_klines(symbol: str, interval: str, limit: int) -> List[List[str]]:
    """Получение свечных данных с Bybit"""
    session = await get_optimized_session()

    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    async with _semaphore:
        for attempt in range(2):
            try:
                async with session.get(
                    "https://api.bybit.com/v5/market/kline",
                    params=params
                ) as response:

                    if response.status != 200:
                        if attempt == 0:
                            await asyncio.sleep(0.1)
                            continue
                        logger.warning(f"HTTP {response.status} для {symbol}")
                        return []

                    data = await response.json()
                    if data.get("retCode") != 0:
                        if attempt == 0:
                            await asyncio.sleep(0.1)
                            continue
                        logger.warning(f"API ошибка для {symbol}: {data.get('retMsg', 'Unknown')}")
                        return []

                    klines = data["result"]["list"]

                    # Сортировка если нужно (старые свечи первыми)
                    if klines and len(klines) > 1 and int(klines[0][0]) > int(klines[-1][0]):
                        klines.reverse()

                    # Убираем последнюю незавершенную свечу
                    return klines if klines else []

            except asyncio.TimeoutError:
                if attempt == 0:
                    continue
                logger.warning(f"Таймаут получения данных {symbol}")
                return []
            except Exception as e:
                if attempt == 0:
                    continue
                logger.warning(f"Ошибка получения {symbol}: {e}")
                return []

    return []

async def get_trading_pairs() -> List[str]:
    """Получение списка торговых пар"""
    session = await get_optimized_session()
    params = {"category": "linear"}

    try:
        async with session.get(
            "https://api.bybit.com/v5/market/instruments-info",
            params=params
        ) as response:

            if response.status != 200:
                logger.error(f"Ошибка API торговых пар: {response.status}")
                return _get_fallback_pairs()

            data = await response.json()
            if data.get("retCode") != 0:
                logger.error("API вернул ошибку для торговых пар")
                return _get_fallback_pairs()

            # Фильтрация активных USDT пар
            symbols = [
                item["symbol"] for item in data["result"]["list"]
                if (item["status"] == 'Trading' and
                    item["symbol"].endswith('USDT') and
                    not item["symbol"].startswith('USDT') and
                    "-" not in item["symbol"])
            ]

            logger.info(f"Загружено {len(symbols)} торговых пар")
            return symbols

    except Exception as e:
        logger.error(f"Критическая ошибка получения пар: {e}")
        return _get_fallback_pairs()

def _get_fallback_pairs() -> List[str]:
    """Резервный список популярных пар"""
    pairs = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
        'XRPUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT',
        'LINKUSDT', 'LTCUSDT', 'UNIUSDT', 'ATOMUSDT', 'FILUSDT',
        'AAVEUSDT', 'SUSHIUSDT', 'COMPUSDT', 'YFIUSDT', 'SNXUSDT'
    ]
    logger.info(f"Используется резервный список пар: {len(pairs)} пар")
    return pairs

async def batch_fetch_klines(requests: List[Dict]) -> List[Dict]:
    """Массовое получение свечных данных"""
    if not requests:
        return []

    # Создаем задачи для параллельного выполнения
    tasks = []
    for req in requests:
        task = _fetch_single_request(req)
        tasks.append(task)

    # Выполняем все запросы параллельно
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Фильтруем успешные результаты
    successful = []
    errors = 0

    for result in results:
        if isinstance(result, dict) and result.get('success'):
            successful.append(result)
        else:
            errors += 1

    if errors > 0:
        logger.warning(f"Ошибок в батче: {errors}/{len(requests)}")

    return successful

async def _fetch_single_request(req: Dict) -> Dict:
    """Обработка одного запроса с обработкой ошибок"""
    try:
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
    except Exception as e:
        logger.debug(f"Ошибка запроса {req['symbol']}: {e}")
        return {
            'symbol': req['symbol'],
            'klines': [],
            'success': False
        }

async def cleanup():
    """Очистка ресурсов"""
    global _session, _semaphore

    if _session and not _session.closed:
        try:
            await _session.close()
            await asyncio.sleep(0.1)  # Даем время на закрытие соединений
        except Exception as e:
            logger.debug(f"Ошибка закрытия сессии: {e}")
        finally:
            _session = None
            _semaphore = None