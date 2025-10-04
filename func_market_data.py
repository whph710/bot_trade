"""
Модуль для получения расширенных рыночных данных
ОБНОВЛЕНО: Адаптивная глубина стакана в зависимости от цены актива
"""

import aiohttp
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def calculate_adaptive_orderbook_step(current_price: float) -> float:
    """
    Рассчитать адекватный шаг для группировки стакана

    Args:
        current_price: текущая цена актива

    Returns:
        Оптимальный шаг для группировки уровней

    Примеры:
        BTC $120,000 -> шаг $10-50 (не $0.01!)
        SHIB $0.00003 -> шаг $0.0000001 (не $0.1!)
    """
    if current_price == 0:
        return 0.01

    # Определяем порядок величины
    import math
    magnitude = math.floor(math.log10(abs(current_price)))

    # Шаг = ~0.01-0.05% от цены
    if current_price >= 10000:  # BTC, ETH дорогие
        step = 10 ** (magnitude - 2)  # $100,000 -> $1000 шаг
    elif current_price >= 100:  # Средние
        step = 10 ** (magnitude - 2)  # $1000 -> $10 шаг
    elif current_price >= 1:  # Дешевые
        step = 10 ** (magnitude - 3)  # $10 -> $0.01 шаг
    elif current_price >= 0.01:  # Очень дешевые
        step = 10 ** (magnitude - 3)  # $0.1 -> $0.001 шаг
    else:  # Мем-коины
        step = 10 ** (magnitude - 2)  # $0.00001 -> $0.0000001 шаг

    return max(step, 10 ** magnitude * 0.0001)  # Минимум 0.01% от цены


def group_orderbook_levels(
    orders: List[List[float]],
    step: float,
    max_levels: int = 10
) -> List[List[float]]:
    """
    Сгруппировать уровни стакана по адаптивному шагу

    Args:
        orders: [[price, size], ...] исходные заявки
        step: шаг группировки
        max_levels: максимум уровней на выходе

    Returns:
        Сгруппированные уровни [[rounded_price, total_size], ...]
    """
    if not orders or step == 0:
        return orders[:max_levels]

    from collections import defaultdict

    grouped = defaultdict(float)

    for price, size in orders:
        # Округляем до ближайшего шага
        rounded_price = round(price / step) * step
        grouped[rounded_price] += size

    # Сортируем и возвращаем топ
    sorted_levels = sorted(grouped.items(), key=lambda x: x[0])
    return [[price, size] for price, size in sorted_levels[:max_levels]]


class MarketDataCollector:
    """Сборщик расширенных рыночных данных с Bybit"""

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.base_url = "https://api.bybit.com"

    async def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        """Получить текущий funding rate"""
        try:
            params = {
                "category": "linear",
                "symbol": symbol
            }

            async with self.session.get(
                    f"{self.base_url}/v5/market/funding/history",
                    params=params
            ) as response:
                if response.status != 200:
                    logger.warning(f"Funding rate HTTP {response.status} для {symbol}")
                    return None

                data = await response.json()

                if data.get("retCode") != 0:
                    logger.warning(f"Funding rate API ошибка для {symbol}")
                    return None

                result_list = data.get("result", {}).get("list", [])
                if not result_list:
                    return None

                latest = result_list[0]

                return {
                    'funding_rate': float(latest.get('fundingRate', 0)),
                    'funding_rate_timestamp': latest.get('fundingRateTimestamp'),
                    'symbol': symbol
                }

        except Exception as e:
            logger.debug(f"Ошибка получения funding rate {symbol}: {e}")
            return None

    async def get_open_interest(self, symbol: str, interval: str = "1h") -> Optional[Dict]:
        """Получить Open Interest"""
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "intervalTime": interval,
                "limit": 24
            }

            async with self.session.get(
                    f"{self.base_url}/v5/market/open-interest",
                    params=params
            ) as response:
                if response.status != 200:
                    logger.warning(f"Open Interest HTTP {response.status} для {symbol}")
                    return None

                data = await response.json()

                if data.get("retCode") != 0:
                    logger.warning(f"Open Interest API ошибка для {symbol}")
                    return None

                result_list = data.get("result", {}).get("list", [])
                if len(result_list) < 2:
                    return None

                current_oi = float(result_list[0].get('openInterest', 0))
                old_oi = float(result_list[-1].get('openInterest', 0))

                if old_oi > 0:
                    oi_change = ((current_oi - old_oi) / old_oi) * 100
                else:
                    oi_change = 0

                if oi_change > 2:
                    trend = 'GROWING'
                elif oi_change < -2:
                    trend = 'DECLINING'
                else:
                    trend = 'STABLE'

                return {
                    'open_interest': current_oi,
                    'oi_change_24h': round(oi_change, 2),
                    'oi_trend': trend,
                    'symbol': symbol
                }

        except Exception as e:
            logger.debug(f"Ошибка получения Open Interest {symbol}: {e}")
            return None

    async def get_orderbook(self, symbol: str, depth: int = 50, current_price: float = None) -> Optional[Dict]:
        """
        Получить Order Book с АДАПТИВНОЙ группировкой

        Args:
            symbol: торговая пара
            depth: сколько уровней запросить от биржи
            current_price: текущая цена для адаптивного шага (опционально)
        """
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "limit": depth
            }

            async with self.session.get(
                    f"{self.base_url}/v5/market/orderbook",
                    params=params
            ) as response:
                if response.status != 200:
                    logger.warning(f"OrderBook HTTP {response.status} для {symbol}")
                    return None

                data = await response.json()

                if data.get("retCode") != 0:
                    logger.warning(f"OrderBook API ошибка для {symbol}")
                    return None

                result = data.get("result", {})

                raw_bids = [[float(p), float(s)] for p, s in result.get('b', [])]
                raw_asks = [[float(p), float(s)] for p, s in result.get('a', [])]

                if not raw_bids or not raw_asks:
                    return None

                best_bid = raw_bids[0][0]
                best_ask = raw_asks[0][0]
                mid_price = (best_bid + best_ask) / 2

                # Используем mid_price если current_price не передан
                # Используем mid_price если current_price не передан
                price_for_calc = current_price if current_price else mid_price

                # Рассчитываем адаптивный шаг
                adaptive_step = calculate_adaptive_orderbook_step(price_for_calc)

                # Группируем уровни
                grouped_bids = group_orderbook_levels(raw_bids, adaptive_step, max_levels=20)
                grouped_asks = group_orderbook_levels(raw_asks, adaptive_step, max_levels=20)

                spread = best_ask - best_bid
                spread_pct = (spread / mid_price) * 100

                return {
                    'bids': grouped_bids,
                    'asks': grouped_asks,
                    'best_bid': best_bid,
                    'best_ask': best_ask,
                    'mid_price': mid_price,
                    'spread': spread,
                    'spread_pct': round(spread_pct, 4),
                    'adaptive_step': adaptive_step,
                    'symbol': symbol
                }

        except Exception as e:
            logger.debug(f"Ошибка получения OrderBook {symbol}: {e}")
            return None

    async def get_taker_buysell_volume(self, symbol: str, interval: str = "5", limit: int = 20) -> Optional[Dict]:
        """Получить Taker Buy/Sell Volume"""
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }

            async with self.session.get(
                    f"{self.base_url}/v5/market/kline",
                    params=params
            ) as response:
                if response.status != 200:
                    return None

                data = await response.json()

                if data.get("retCode") != 0:
                    return None

                klines = data.get("result", {}).get("list", [])

                total_volume = 0
                bullish_volume = 0

                for candle in klines:
                    volume = float(candle[5])
                    open_price = float(candle[1])
                    close_price = float(candle[4])

                    total_volume += volume

                    if close_price > open_price:
                        bullish_volume += volume

                if total_volume == 0:
                    return None

                buy_pressure = bullish_volume / total_volume
                sell_pressure = 1 - buy_pressure

                return {
                    'total_volume': total_volume,
                    'bullish_volume': bullish_volume,
                    'bearish_volume': total_volume - bullish_volume,
                    'buy_pressure': round(buy_pressure, 3),
                    'sell_pressure': round(sell_pressure, 3),
                    'symbol': symbol
                }

        except Exception as e:
            logger.debug(f"Ошибка получения Taker Volume {symbol}: {e}")
            return None

    async def get_market_snapshot(self, symbol: str, current_price: float = None) -> Dict:
        """
        Получить полный snapshot рыночных данных

        Args:
            symbol: торговая пара
            current_price: текущая цена для адаптивного стакана (опционально)
        """
        tasks = [
            self.get_funding_rate(symbol),
            self.get_open_interest(symbol),
            self.get_orderbook(symbol, depth=50, current_price=current_price),
            self.get_taker_buysell_volume(symbol)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        funding, oi, orderbook, taker_volume = results

        return {
            'symbol': symbol,
            'funding_rate': funding if isinstance(funding, dict) else None,
            'open_interest': oi if isinstance(oi, dict) else None,
            'orderbook': orderbook if isinstance(orderbook, dict) else None,
            'taker_volume': taker_volume if isinstance(taker_volume, dict) else None,
            'timestamp': datetime.now().isoformat()
        }

    async def batch_get_funding_rates(self, symbols: List[str]) -> Dict[str, Optional[Dict]]:
        """Массовое получение funding rates"""
        tasks = [self.get_funding_rate(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            symbol: result if isinstance(result, dict) else None
            for symbol, result in zip(symbols, results)
        }

    async def batch_get_open_interest(self, symbols: List[str]) -> Dict[str, Optional[Dict]]:
        """Массовое получение Open Interest"""
        tasks = [self.get_open_interest(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            symbol: result if isinstance(result, dict) else None
            for symbol, result in zip(symbols, results)
        }


# ==================== АНАЛИЗАТОРЫ ДАННЫХ ====================

class MarketDataAnalyzer:
    """Анализ полученных рыночных данных"""

    @staticmethod
    def analyze_funding_rate(funding_data: Optional[Dict]) -> Dict:
        """Анализ funding rate"""
        if not funding_data or 'funding_rate' not in funding_data:
            return {
                'status': 'UNKNOWN',
                'confidence_adjustment': 0,
                'risk_level': 'UNKNOWN',
                'reasoning': 'No funding rate data available'
            }

        rate = funding_data['funding_rate']

        if rate > 0.001:  # >0.10%
            return {
                'status': 'OVERLEVERAGED_LONG',
                'confidence_adjustment': -15,
                'risk_level': 'HIGH',
                'reasoning': f'Funding rate {rate * 100:.3f}% - extreme long leverage, dump risk high'
            }
        elif rate > 0.0005:  # 0.05-0.10%
            return {
                'status': 'ELEVATED_LONG',
                'confidence_adjustment': -8,
                'risk_level': 'MEDIUM',
                'reasoning': f'Funding rate {rate * 100:.3f}% - elevated long positions, caution'
            }
        elif rate < -0.0005:  # < -0.05%
            return {
                'status': 'OVERLEVERAGED_SHORT',
                'confidence_adjustment': +10,
                'risk_level': 'MEDIUM',
                'reasoning': f'Funding rate {rate * 100:.3f}% - shorts squeezed, pump potential'
            }
        else:
            return {
                'status': 'NEUTRAL',
                'confidence_adjustment': 0,
                'risk_level': 'LOW',
                'reasoning': f'Funding rate {rate * 100:.3f}% - neutral leverage'
            }

    @staticmethod
    def analyze_open_interest(oi_data: Optional[Dict], price_direction: str) -> Dict:
        """Анализ Open Interest в контексте движения цены"""
        if not oi_data or 'oi_trend' not in oi_data:
            return {
                'pattern': 'UNKNOWN',
                'confidence_adjustment': 0,
                'reasoning': 'No Open Interest data'
            }

        oi_trend = oi_data['oi_trend']
        oi_change = oi_data['oi_change_24h']

        if price_direction == 'UP':
            if oi_trend == 'GROWING':
                return {
                    'pattern': 'STRONG_UPTREND',
                    'confidence_adjustment': +12,
                    'reasoning': f'Price rising + OI growing ({oi_change:+.1f}%) = strong bullish trend, new buyers'
                }
            elif oi_trend == 'DECLINING':
                return {
                    'pattern': 'WEAK_RALLY',
                    'confidence_adjustment': -8,
                    'reasoning': f'Price rising + OI declining ({oi_change:+.1f}%) = short covering, not sustainable'
                }
            else:
                return {
                    'pattern': 'NEUTRAL_UPTREND',
                    'confidence_adjustment': +3,
                    'reasoning': f'Price rising + OI stable = moderate bullish'
                }

        elif price_direction == 'DOWN':
            if oi_trend == 'GROWING':
                return {
                    'pattern': 'STRONG_DOWNTREND',
                    'confidence_adjustment': -12,
                    'reasoning': f'Price falling + OI growing ({oi_change:+.1f}%) = strong bearish, new shorts'
                }
            elif oi_trend == 'DECLINING':
                return {
                    'pattern': 'WEAK_DECLINE',
                    'confidence_adjustment': +8,
                    'reasoning': f'Price falling + OI declining ({oi_change:+.1f}%) = long unwinding, reversal possible'
                }
            else:
                return {
                    'pattern': 'NEUTRAL_DOWNTREND',
                    'confidence_adjustment': -3,
                    'reasoning': f'Price falling + OI stable = moderate bearish'
                }

        else:  # FLAT
            if oi_trend == 'GROWING':
                return {
                    'pattern': 'ACCUMULATION',
                    'confidence_adjustment': +5,
                    'reasoning': f'Price flat + OI growing ({oi_change:+.1f}%) = position building, breakout likely'
                }
            else:
                return {
                    'pattern': 'CONSOLIDATION',
                    'confidence_adjustment': 0,
                    'reasoning': 'Price and OI both flat = waiting mode'
                }

    @staticmethod
    def analyze_spread(orderbook_data: Optional[Dict]) -> Dict:
        """Анализ спреда для определения ликвидности"""
        if not orderbook_data or 'spread_pct' not in orderbook_data:
            return {
                'liquidity': 'UNKNOWN',
                'tradeable': False,
                'confidence_adjustment': -10,
                'reasoning': 'No orderbook data, skip trading'
            }

        spread_pct = orderbook_data['spread_pct']

        if spread_pct > 0.15:  # >0.15%
            return {
                'liquidity': 'LOW',
                'tradeable': False,
                'confidence_adjustment': -20,
                'reasoning': f'Spread {spread_pct:.3f}% too wide, illiquid market, SKIP'
            }
        elif spread_pct > 0.08:  # 0.08-0.15%
            return {
                'liquidity': 'MEDIUM',
                'tradeable': True,
                'confidence_adjustment': -8,
                'reasoning': f'Spread {spread_pct:.3f}% acceptable but not ideal'
            }
        else:  # <0.08%
            return {
                'liquidity': 'HIGH',
                'tradeable': True,
                'confidence_adjustment': 0,
                'reasoning': f'Spread {spread_pct:.3f}% tight, good liquidity'
            }

    @staticmethod
    def analyze_orderbook_imbalance(orderbook_data: Optional[Dict]) -> Dict:
        """Анализ дисбаланса bid/ask в стакане"""
        if not orderbook_data or not orderbook_data.get('bids') or not orderbook_data.get('asks'):
            return {
                'imbalance': 'UNKNOWN',
                'bid_ask_ratio': 1.0,
                'confidence_adjustment': 0,
                'reasoning': 'No orderbook data'
            }

        bids = orderbook_data['bids'][:10]
        asks = orderbook_data['asks'][:10]

        total_bid_volume = sum(size for _, size in bids)
        total_ask_volume = sum(size for _, size in asks)

        if total_ask_volume == 0:
            return {
                'imbalance': 'UNKNOWN',
                'bid_ask_ratio': 1.0,
                'confidence_adjustment': 0,
                'reasoning': 'Invalid orderbook data'
            }

        ratio = total_bid_volume / total_ask_volume

        if ratio > 1.5:
            return {
                'imbalance': 'STRONG_BID',
                'bid_ask_ratio': round(ratio, 2),
                'confidence_adjustment': +10,
                'reasoning': f'Bid/Ask ratio {ratio:.2f} - strong buy pressure in orderbook'
            }
        elif ratio < 0.67:
            return {
                'imbalance': 'STRONG_ASK',
                'bid_ask_ratio': round(ratio, 2),
                'confidence_adjustment': -10,
                'reasoning': f'Bid/Ask ratio {ratio:.2f} - strong sell pressure in orderbook'
            }
        else:
            return {
                'imbalance': 'BALANCED',
                'bid_ask_ratio': round(ratio, 2),
                'confidence_adjustment': 0,
                'reasoning': f'Bid/Ask ratio {ratio:.2f} - balanced orderbook'
            }

    @staticmethod
    def analyze_taker_volume(taker_data: Optional[Dict]) -> Dict:
        """Анализ агрессивных покупок vs продаж"""
        if not taker_data or 'buy_pressure' not in taker_data:
            return {
                'pressure': 'UNKNOWN',
                'buy_pressure': 0.5,
                'confidence_adjustment': 0,
                'reasoning': 'No taker volume data'
            }

        buy_pressure = taker_data['buy_pressure']

        if buy_pressure > 0.60:
            return {
                'pressure': 'BUY',
                'buy_pressure': buy_pressure,
                'confidence_adjustment': +8,
                'reasoning': f'Buy pressure {buy_pressure * 100:.1f}% - aggressive buyers dominating'
            }
        elif buy_pressure < 0.40:
            return {
                'pressure': 'SELL',
                'buy_pressure': buy_pressure,
                'confidence_adjustment': -8,
                'reasoning': f'Buy pressure {buy_pressure * 100:.1f}% - aggressive sellers dominating'
            }
        else:
            return {
                'pressure': 'NEUTRAL',
                'buy_pressure': buy_pressure,
                'confidence_adjustment': 0,
                'reasoning': f'Buy pressure {buy_pressure * 100:.1f}% - balanced'
            }


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

async def get_comprehensive_market_data(
        session: aiohttp.ClientSession,
        symbol: str,
        price_direction: str = 'FLAT',
        current_price: float = None
) -> Dict:
    """
    Получить и проанализировать все рыночные данные

    Args:
        session: aiohttp сессия
        symbol: торговая пара
        price_direction: направление цены
        current_price: текущая цена для адаптивного стакана
    """
    collector = MarketDataCollector(session)
    analyzer = MarketDataAnalyzer()

    snapshot = await collector.get_market_snapshot(symbol, current_price)

    funding_analysis = analyzer.analyze_funding_rate(snapshot['funding_rate'])
    oi_analysis = analyzer.analyze_open_interest(snapshot['open_interest'], price_direction)
    spread_analysis = analyzer.analyze_spread(snapshot['orderbook'])
    imbalance_analysis = analyzer.analyze_orderbook_imbalance(snapshot['orderbook'])
    taker_analysis = analyzer.analyze_taker_volume(snapshot['taker_volume'])

    total_adjustment = (
            funding_analysis['confidence_adjustment'] +
            oi_analysis['confidence_adjustment'] +
            spread_analysis['confidence_adjustment'] +
            imbalance_analysis['confidence_adjustment'] +
            taker_analysis['confidence_adjustment']
    )

    is_tradeable = spread_analysis['tradeable']

    return {
        'symbol': symbol,
        'timestamp': snapshot['timestamp'],
        'tradeable': is_tradeable,
        'total_confidence_adjustment': total_adjustment,
        'analyses': {
            'funding_rate': funding_analysis,
            'open_interest': oi_analysis,
            'spread': spread_analysis,
            'orderbook_imbalance': imbalance_analysis,
            'taker_volume': taker_analysis
        },
        'raw_data': snapshot
    }