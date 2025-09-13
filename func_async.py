# func_async_optimized.py - Максимально оптимизированная версия API функций

import aiohttp
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# Глобальные кэши для скорости
_session_cache = None
_instruments_cache = None
_instruments_cache_time = 0
_klines_cache = {}
_cache_ttl = 30  # Время жизни кэша в секундах


@dataclass
class CacheEntry:
    data: Any
    timestamp: float
    ttl: float = 30.0

    def is_valid(self) -> bool:
        return time.time() - self.timestamp < self.ttl


class OptimizedBybitAPI:
    """Оптимизированный класс для работы с Bybit API"""

    def __init__(self):
        self.session = None
        self.base_url = "https://api.bybit.com/v5"
        self.timeout = aiohttp.ClientTimeout(total=8)  # Агрессивный таймаут
        self.connector = None
        self.request_count = 0

    async def get_session(self) -> aiohttp.ClientSession:
        """Переиспользуемая сессия с пулом соединений"""
        if self.session is None or self.session.closed:
            # Оптимизированный коннектор
            self.connector = aiohttp.TCPConnector(
                limit=100,  # Увеличиваем лимит соединений
                limit_per_host=50,
                keepalive_timeout=60,
                enable_cleanup_closed=True,
                use_dns_cache=True,
                ttl_dns_cache=300,
                family=0  # Используем IPv4 и IPv6
            )

            # Оптимизированные заголовки
            headers = {
                'Connection': 'keep-alive',
                'Keep-Alive': 'timeout=60, max=1000',
                'User-Agent': 'TradingBot/1.0',
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate'
            }

            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=self.timeout,
                headers=headers,
                json_serialize=json.dumps  # Быстрый JSON сериализатор
            )

        return self.session

    async def make_request(self, endpoint: str, params: Dict) -> Dict:
        """Оптимизированный HTTP запрос"""
        session = await self.get_session()
        self.request_count += 1

        try:
            async with session.get(f"{self.base_url}{endpoint}", params=params) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")

                data = await response.json()

                if data.get("retCode") != 0:
                    raise Exception(f"API Error: {data.get('retMsg', 'Unknown error')}")

                return data

        except asyncio.TimeoutError:
            raise Exception(f"Timeout after 8s for {endpoint}")
        except Exception as e:
            logger.error(f"Request error {endpoint}: {e}")
            raise

    async def cleanup(self):
        """Очистка ресурсов"""
        if self.session:
            await self.session.close()


# Глобальный экземпляр API
_api = OptimizedBybitAPI()


async def get_klines_async_optimized(symbol: str, interval: str = "15", limit: int = 200) -> List[List[str]]:
    """
    Супер-оптимизированное получение свечей с кэшированием

    Args:
        symbol: Торговая пара
        interval: Интервал
        limit: Количество свечей
    """
    global _klines_cache

    # Ключ кэша
    cache_key = f"{symbol}_{interval}_{limit}"

    # Проверяем кэш
    if cache_key in _klines_cache:
        cache_entry = _klines_cache[cache_key]
        if cache_entry.is_valid():
            return cache_entry.data

    try:
        # Параметры запроса
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        # Делаем запрос
        data = await _api.make_request("/market/kline", params)
        klines = data["result"]["list"]

        # Проверяем порядок и разворачиваем если нужно
        if klines and len(klines) > 1:
            if int(klines[0][0]) > int(klines[-1][0]):
                klines.reverse()

        # Убираем последнюю незавершенную свечу
        result = klines[:-1] if klines else []

        # Кэшируем результат
        _klines_cache[cache_key] = CacheEntry(result, time.time(), ttl=_cache_ttl)

        # Ограничиваем размер кэша
        if len(_klines_cache) > 1000:
            # Удаляем старые записи
            current_time = time.time()
            expired_keys = [k for k, v in _klines_cache.items()
                            if not v.is_valid()]
            for k in expired_keys:
                del _klines_cache[k]

        logger.debug(f"Кэш свечей {symbol}: {len(result)} свечей")
        return result

    except Exception as e:
        logger.error(f"Ошибка получения свечей {symbol}: {e}")
        raise


async def get_usdt_trading_pairs_optimized() -> List[str]:
    """
    Оптимизированное получение торговых пар с агрессивным кэшированием
    """
    global _instruments_cache, _instruments_cache_time

    current_time = time.time()

    # Проверяем кэш (кэшируем на 5 минут)
    if (_instruments_cache is not None and
            current_time - _instruments_cache_time < 300):
        logger.debug(f"Используем кэш пар: {len(_instruments_cache)} пар")
        return _instruments_cache

    try:
        logger.warning("Обновляем список торговых пар...")

        params = {"category": "linear"}
        data = await _api.make_request("/market/instruments-info", params)

        # Быстрая фильтрация с множественными условиями
        symbols = []
        for item in data["result"]["list"]:
            symbol = item["symbol"]
            status = item["status"]

            # Агрессивная фильтрация в одном условии
            if (status == 'Trading' and
                    symbol.endswith('USDT') and
                    not symbol.startswith('USDT') and
                    "-" not in symbol and
                    not any(c.isdigit() for c in symbol) and
                    len(symbol) >= 6 and len(symbol) <= 12):  # Дополнительные фильтры длины
                symbols.append(symbol)

        # Дополнительная фильтрация популярных пар (опционально)
        # Можно добавить белый список самых ликвидных пар
        popular_pairs = {
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
            'XRPUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT',
            'LINKUSDT', 'LTCUSDT', 'UNIUSDT', 'ATOMUSDT', 'VETUSDT',
            'FILUSDT', 'TRXUSDT', 'ETCUSDT', 'XLMUSDT', 'BCHUSDT'
        }

        # Сортируем: сначала популярные, потом остальные
        popular = [s for s in symbols if s in popular_pairs]
        others = [s for s in symbols if s not in popular_pairs]

        # Ограничиваем общее количество для скорости (можно настроить)
        final_symbols = popular + others[:200]  # Максимум 220 пар

        # Обновляем кэш
        _instruments_cache = final_symbols
        _instruments_cache_time = current_time

        logger.warning(f"Получено {len(final_symbols)} торговых пар USDT")
        return final_symbols

    except Exception as e:
        logger.error(f"Ошибка получения торговых пар: {e}")
        # Возвращаем кэш если есть, иначе минимальный список
        if _instruments_cache:
            return _instruments_cache
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']  # Фаллбек


async def batch_get_klines(pairs_with_params: List[Dict]) -> List[Dict]:
    """
    Массовое получение свечей с максимальным параллелизмом

    Args:
        pairs_with_params: Список словарей с параметрами
        [{'symbol': 'BTCUSDT', 'interval': '5', 'limit': 100}, ...]
    """
    # Создаем семафор для ограничения параллельных запросов
    semaphore = asyncio.Semaphore(20)  # Увеличиваем лимит

    async def bounded_request(params):
        async with semaphore:
            try:
                klines = await get_klines_async_optimized(
                    params['symbol'],
                    params.get('interval', '5'),
                    params.get('limit', 100)
                )
                return {
                    'symbol': params['symbol'],
                    'klines': klines,
                    'success': True
                }
            except Exception as e:
                return {
                    'symbol': params['symbol'],
                    'error': str(e),
                    'success': False
                }

    # Запускаем все запросы параллельно
    tasks = [bounded_request(params) for params in pairs_with_params]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Фильтруем успешные результаты
    successful_results = []
    for result in results:
        if isinstance(result, dict) and result.get('success'):
            successful_results.append(result)

    return successful_results


async def quick_liquidity_scan(symbols: List[str]) -> List[str]:
    """
    Быстрое сканирование ликвидности для предварительной фильтрации

    Args:
        symbols: Список символов для проверки

    Returns:
        Отфильтрованный список ликвидных пар
    """
    # Параметры для быстрой проверки (минимум данных)
    params_list = [
        {'symbol': symbol, 'interval': '5', 'limit': 10}
        for symbol in symbols
    ]

    # Быстрое массовое получение данных
    results = await batch_get_klines(params_list)

    liquid_pairs = []
    min_volume_threshold = 1_000_000  # $1M минимальный объем за 10 свечей

    for result in results:
        if not result.get('success') or not result.get('klines'):
            continue

        # Быстрая оценка ликвидности
        klines = result['klines']
        if len(klines) < 5:
            continue

        # Считаем приблизительный объем в USD
        total_volume_usd = 0
        for candle in klines[-5:]:  # Последние 5 свечей
            volume = float(candle[5])
            price = float(candle[4])
            total_volume_usd += volume * price

        # Экстраполируем на 24 часа (5 свечей * 12 * 24)
        daily_volume_estimate = total_volume_usd * 288

        if daily_volume_estimate > min_volume_threshold:
            liquid_pairs.append(result['symbol'])

    logger.warning(f"Фильтрация ликвидности: {len(liquid_pairs)}/{len(symbols)} пар")
    return liquid_pairs


# Функции-обертки для обратной совместимости
async def get_klines_async(symbol: str, interval: str = "15", limit: int = 200) -> List[List[str]]:
    """Обратная совместимость"""
    return await get_klines_async_optimized(symbol, interval, limit)


async def get_usdt_trading_pairs() -> List[str]:
    """Обратная совместимость"""
    return await get_usdt_trading_pairs_optimized()


# Функции для мониторинга производительности
def get_api_stats() -> Dict[str, Any]:
    """Статистика использования API"""
    cache_stats = {
        'klines_cache_size': len(_klines_cache),
        'klines_cache_hit_ratio': 0,  # Можно добавить счетчики
        'instruments_cached': _instruments_cache is not None,
        'total_requests': _api.request_count
    }

    # Статистика кэша
    valid_entries = sum(1 for entry in _klines_cache.values() if entry.is_valid())
    cache_stats['valid_cache_entries'] = valid_entries

    return cache_stats


async def cleanup_api_resources():
    """Очистка всех API ресурсов"""
    global _klines_cache, _instruments_cache, _api

    # Очищаем кэши
    _klines_cache.clear()
    _instruments_cache = None

    # Закрываем сессию
    await _api.cleanup()

    logger.warning("API ресурсы очищены")


# Функция для предварительного прогрева
async def warmup_api():
    """Предварительный прогрев API и кэшей"""
    logger.warning("Прогрев API...")

    try:
        # Получаем список пар (заполняем кэш)
        pairs = await get_usdt_trading_pairs_optimized()
        logger.warning(f"Прогрев: получено {len(pairs)} пар")

        # Прогреваем кэш для топ-10 пар
        top_pairs = pairs[:10]
        warmup_params = [
            {'symbol': pair, 'interval': '5', 'limit': 50}
            for pair in top_pairs
        ]

        results = await batch_get_klines(warmup_params)
        logger.warning(f"Прогрев: кэш заполнен для {len(results)} пар")

        return True

    except Exception as e:
        logger.error(f"Ошибка прогрева API: {e}")
        return False


# Продвинутые функции для оптимизации
class SmartCache:
    """Умный кэш с адаптивным TTL"""

    def __init__(self):
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0
        self.access_patterns = {}

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache and self.cache[key].is_valid():
            self.hit_count += 1
            self.access_patterns[key] = self.access_patterns.get(key, 0) + 1
            return self.cache[key].data

        self.miss_count += 1
        return None

    def set(self, key: str, value: Any, ttl: float = 30.0):
        # Адаптивный TTL на основе частоты доступа
        access_freq = self.access_patterns.get(key, 0)
        if access_freq > 10:  # Часто используемые данные кэшируем дольше
            ttl *= 2

        self.cache[key] = CacheEntry(value, time.time(), ttl)

    def get_hit_ratio(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0

    def cleanup(self):
        current_time = time.time()
        expired_keys = [k for k, v in self.cache.items() if not v.is_valid()]
        for k in expired_keys:
            del self.cache[k]


# Глобальный умный кэш
_smart_cache = SmartCache()


async def get_klines_smart_cached(symbol: str, interval: str = "5", limit: int = 100) -> List[List[str]]:
    """Получение свечей с умным кэшированием"""
    cache_key = f"smart_{symbol}_{interval}_{limit}"

    # Проверяем умный кэш
    cached_data = _smart_cache.get(cache_key)
    if cached_data is not None:
        return cached_data

    # Получаем данные
    try:
        klines = await get_klines_async_optimized(symbol, interval, limit)

        # Сохраняем в умный кэш
        _smart_cache.set(cache_key, klines)

        return klines

    except Exception as e:
        logger.error(f"Ошибка smart cache для {symbol}: {e}")
        raise


# Функция для массового предварительного кэширования
async def preload_market_data(pairs: List[str], intervals: List[str] = ['5', '15']) -> Dict[str, int]:
    """
    Массовая предварительная загрузка данных в кэш

    Returns:
        Статистика загрузки
    """
    start_time = time.time()

    # Создаем все комбинации пар и интервалов
    load_params = []
    for pair in pairs:
        for interval in intervals:
            load_params.append({
                'symbol': pair,
                'interval': interval,
                'limit': 100
            })

    # Массовая загрузка
    results = await batch_get_klines(load_params)

    stats = {
        'total_requested': len(load_params),
        'successful_loads': len(results),
        'load_time': time.time() - start_time,
        'cache_entries': len(_klines_cache)
    }

    logger.warning(
        f"Предварительная загрузка: {stats['successful_loads']}/{stats['total_requested']} "
        f"за {stats['load_time']:.1f}сек"
    )

    return stats


# Экспорт для использования
__all__ = [
    'get_klines_async',
    'get_usdt_trading_pairs',
    'get_klines_async_optimized',
    'get_usdt_trading_pairs_optimized',
    'batch_get_klines',
    'quick_liquidity_scan',
    'get_klines_smart_cached',
    'preload_market_data',
    'warmup_api',
    'cleanup_api_resources',
    'get_api_stats'
]