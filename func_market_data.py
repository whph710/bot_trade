"""
Модуль для получения расширенных рыночных данных
Funding rates, Open Interest, Spread, Order Book, etc.
"""

import aiohttp
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MarketDataCollector:
    """Сборщик расширенных рыночных данных с Bybit"""

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.base_url = "https://api.bybit.com"

    async def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        """
        Получить текущий funding rate

        Returns:
            {
                'funding_rate': 0.0001,  # Текущий funding rate
                'next_funding_time': '2025-10-01T16:00:00Z',
                'predicted_rate': 0.00015  # Прогноз следующего
            }
        """
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
        """
        Получить Open Interest (открытые позиции)

        Returns:
            {
                'open_interest': 150000000,  # В USD
                'oi_change_24h': 5.2,  # Изменение за 24ч в %
                'oi_trend': 'GROWING'  # GROWING/DECLINING/STABLE
            }
        """
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "intervalTime": interval,
                "limit": 24  # 24 часа данных
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

                # Определяем тренд OI
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

    async def get_orderbook(self, symbol: str, depth: int = 20) -> Optional[Dict]:
        """
        Получить Order Book (стакан заявок)

        Returns:
            {
                'bids': [[price, size], ...],  # Топ N bid заявок
                'asks': [[price, size], ...],  # Топ N ask заявок
                'mid_price': 50000.5,
                'spread': 0.5,
                'spread_pct': 0.001
            }
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

                bids = [[float(p), float(s)] for p, s in result.get('b', [])]
                asks = [[float(p), float(s)] for p, s in result.get('a', [])]

                if not bids or not asks:
                    return None

                best_bid = bids[0][0]
                best_ask = asks[0][0]
                mid_price = (best_bid + best_ask) / 2
                spread = best_ask - best_bid
                spread_pct = (spread / mid_price) * 100

                return {
                    'bids': bids,
                    'asks': asks,
                    'best_bid': best_bid,
                    'best_ask': best_ask,
                    'mid_price': mid_price,
                    'spread': spread,
                    'spread_pct': round(spread_pct, 4),
                    'symbol': symbol
                }

        except Exception as e:
            logger.debug(f"Ошибка получения OrderBook {symbol}: {e}")
            return None

    async def get_taker_buysell_volume(self, symbol: str, interval: str = "5", limit: int = 20) -> Optional[Dict]:
        """
        Получить Taker Buy/Sell Volume (агрессивные покупки vs продажи)

        Returns:
            {
                'buy_volume': 1500000,
                'sell_volume': 1200000,
                'delta': 300000,
                'buy_pressure': 0.55  # 55% покупок
            }
        """
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

                # Bybit не предоставляет прямо taker buy/sell
                # Но мы можем аппроксимировать через volume и movement
                total_volume = 0
                bullish_volume = 0

                for candle in klines:
                    volume = float(candle[5])
                    open_price = float(candle[1])
                    close_price = float(candle[4])

                    total_volume += volume

                    # Если свеча зеленая - считаем как покупки
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

    async def get_market_snapshot(self, symbol: str) -> Dict:
        """
        Получить полный snapshot рыночных данных для одной пары

        Returns полный набор данных или частичный (что удалось получить)
        """
        tasks = [
            self.get_funding_rate(symbol),
            self.get_open_interest(symbol),
            self.get_orderbook(symbol, depth=20),
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
        """Массовое получение funding rates для множества пар"""
        tasks = [self.get_funding_rate(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            symbol: result if isinstance(result, dict) else None
            for symbol, result in zip(symbols, results)
        }

    async def batch_get_open_interest(self, symbols: List[str]) -> Dict[str, Optional[Dict]]:
        """Массовое получение Open Interest для множества пар"""
        tasks = [self.get_open_interest(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            symbol: result if isinstance(result, dict) else None
            for symbol, result in zip(symbols, results)
        }


# ==================== АНАЛИЗАТОРЫ ДАННЫХ ====================

class MarketDataAnalyzer:
    """Анализ полученных рыночных данных с выдачей рекомендаций"""

    @staticmethod
    def analyze_funding_rate(funding_data: Optional[Dict]) -> Dict:
        """
        Анализ funding rate с выдачей рекомендаций

        Returns:
            {
                'status': 'OVERLEVERAGED_LONG' / 'OVERLEVERAGED_SHORT' / 'NEUTRAL',
                'confidence_adjustment': -15 / 0 / +10,
                'risk_level': 'HIGH' / 'MEDIUM' / 'LOW',
                'reasoning': 'Funding rate 0.15% - extreme long leverage, dump risk'
            }
        """
        if not funding_data or 'funding_rate' not in funding_data:
            return {
                'status': 'UNKNOWN',
                'confidence_adjustment': 0,
                'risk_level': 'UNKNOWN',
                'reasoning': 'No funding rate data available'
            }

        rate = funding_data['funding_rate']

        # Thresholds (funding rate обычно в диапазоне -0.3% до +0.3%)
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
        """
        Анализ Open Interest в контексте движения цены

        Args:
            oi_data: данные OI
            price_direction: 'UP' / 'DOWN' / 'FLAT'

        Returns:
            {
                'pattern': 'STRONG_TREND' / 'WEAK_RALLY' / 'DISTRIBUTION',
                'confidence_adjustment': -10 / 0 / +10,
                'reasoning': '...'
            }
        """
        if not oi_data or 'oi_trend' not in oi_data:
            return {
                'pattern': 'UNKNOWN',
                'confidence_adjustment': 0,
                'reasoning': 'No Open Interest data'
            }

        oi_trend = oi_data['oi_trend']
        oi_change = oi_data['oi_change_24h']

        # Анализ паттернов:
        # 1. Price UP + OI UP = Strong uptrend (new longs entering)
        # 2. Price UP + OI DOWN = Weak rally (shorts covering)
        # 3. Price DOWN + OI UP = Strong downtrend (new shorts)
        # 4. Price DOWN + OI DOWN = Weak decline (longs exiting)

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
        """
        Анализ спреда для определения ликвидности

        Returns:
            {
                'liquidity': 'HIGH' / 'MEDIUM' / 'LOW',
                'tradeable': True / False,
                'confidence_adjustment': -15 / 0,
                'reasoning': '...'
            }
        """
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
        """
        Анализ дисбаланса bid/ask в стакане

        Returns:
            {
                'imbalance': 'STRONG_BID' / 'STRONG_ASK' / 'BALANCED',
                'bid_ask_ratio': 1.5,
                'confidence_adjustment': +8 / -8 / 0,
                'reasoning': '...'
            }
        """
        if not orderbook_data or not orderbook_data.get('bids') or not orderbook_data.get('asks'):
            return {
                'imbalance': 'UNKNOWN',
                'bid_ask_ratio': 1.0,
                'confidence_adjustment': 0,
                'reasoning': 'No orderbook data'
            }

        # Считаем суммарный объем в топ-10 уровнях
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
        elif ratio < 0.67:  # 1/1.5
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
        """
        Анализ агрессивных покупок vs продаж

        Returns:
            {
                'pressure': 'BUY' / 'SELL' / 'NEUTRAL',
                'buy_pressure': 0.65,
                'confidence_adjustment': +10 / -10 / 0,
                'reasoning': '...'
            }
        """
        if not taker_data or 'buy_pressure' not in taker_data:
            return {
                'pressure': 'UNKNOWN',
                'buy_pressure': 0.5,
                'confidence_adjustment': 0,
                'reasoning': 'No taker volume data'
            }

        buy_pressure = taker_data['buy_pressure']

        if buy_pressure > 0.60:  # >60% покупок
            return {
                'pressure': 'BUY',
                'buy_pressure': buy_pressure,
                'confidence_adjustment': +8,
                'reasoning': f'Buy pressure {buy_pressure * 100:.1f}% - aggressive buyers dominating'
            }
        elif buy_pressure < 0.40:  # <40% покупок
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
        price_direction: str = 'FLAT'
) -> Dict:
    """
    Получить и проанализировать все рыночные данные для одной пары

    Args:
        session: aiohttp сессия
        symbol: торговая пара
        price_direction: направление цены ('UP'/'DOWN'/'FLAT')

    Returns:
        Полный анализ с рекомендациями по confidence adjustment
    """
    collector = MarketDataCollector(session)
    analyzer = MarketDataAnalyzer()

    # Собираем данные
    snapshot = await collector.get_market_snapshot(symbol)

    # Анализируем каждый компонент
    funding_analysis = analyzer.analyze_funding_rate(snapshot['funding_rate'])
    oi_analysis = analyzer.analyze_open_interest(snapshot['open_interest'], price_direction)
    spread_analysis = analyzer.analyze_spread(snapshot['orderbook'])
    imbalance_analysis = analyzer.analyze_orderbook_imbalance(snapshot['orderbook'])
    taker_analysis = analyzer.analyze_taker_volume(snapshot['taker_volume'])

    # Считаем общий confidence adjustment
    total_adjustment = (
            funding_analysis['confidence_adjustment'] +
            oi_analysis['confidence_adjustment'] +
            spread_analysis['confidence_adjustment'] +
            imbalance_analysis['confidence_adjustment'] +
            taker_analysis['confidence_adjustment']
    )

    # Проверка на блокирующие факторы
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