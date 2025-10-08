"""
Market data collection module
"""

import aiohttp
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
import math

logger = logging.getLogger(__name__)


def calculate_adaptive_orderbook_step(current_price: float) -> float:
    """Calculate adaptive orderbook grouping step"""
    if current_price == 0:
        return 0.01

    magnitude = math.floor(math.log10(abs(current_price)))

    if current_price >= 10000:
        step = 10 ** (magnitude - 2)
    elif current_price >= 100:
        step = 10 ** (magnitude - 2)
    elif current_price >= 1:
        step = 10 ** (magnitude - 3)
    elif current_price >= 0.01:
        step = 10 ** (magnitude - 3)
    else:
        step = 10 ** (magnitude - 2)

    return max(step, 10 ** magnitude * 0.0001)


def group_orderbook_levels(
    orders: List[List[float]],
    step: float,
    max_levels: int = 10
) -> List[List[float]]:
    """Group orderbook levels by adaptive step"""
    if not orders or step == 0:
        return orders[:max_levels]

    from collections import defaultdict

    grouped = defaultdict(float)

    for price, size in orders:
        rounded_price = round(price / step) * step
        grouped[rounded_price] += size

    sorted_levels = sorted(grouped.items(), key=lambda x: x[0])
    return [[price, size] for price, size in sorted_levels[:max_levels]]


class MarketDataCollector:
    """Market data collector from Bybit"""

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.base_url = "https://api.bybit.com"

    async def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        """Get current funding rate"""
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
                    return None

                data = await response.json()

                if data.get("retCode") != 0:
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
            logger.debug(f"Error getting funding rate {symbol}: {e}")
            return None

    async def get_open_interest(self, symbol: str, interval: str = "1h") -> Optional[Dict]:
        """Get Open Interest"""
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
                    return None

                data = await response.json()

                if data.get("retCode") != 0:
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
            logger.debug(f"Error getting Open Interest {symbol}: {e}")
            return None

    async def get_orderbook(self, symbol: str, depth: int = 50, current_price: float = None) -> Optional[Dict]:
        """Get Order Book with adaptive grouping"""
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
                    return None

                data = await response.json()

                if data.get("retCode") != 0:
                    return None

                result = data.get("result", {})

                raw_bids = [[float(p), float(s)] for p, s in result.get('b', [])]
                raw_asks = [[float(p), float(s)] for p, s in result.get('a', [])]

                if not raw_bids or not raw_asks:
                    return None

                best_bid = raw_bids[0][0]
                best_ask = raw_asks[0][0]
                mid_price = (best_bid + best_ask) / 2

                price_for_calc = current_price if current_price else mid_price

                adaptive_step = calculate_adaptive_orderbook_step(price_for_calc)

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
            logger.debug(f"Error getting OrderBook {symbol}: {e}")
            return None

    async def get_taker_buysell_volume(self, symbol: str, interval: str = "5", limit: int = 20) -> Optional[Dict]:
        """Get Taker Buy/Sell Volume"""
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
            logger.debug(f"Error getting Taker Volume {symbol}: {e}")
            return None

    async def get_market_snapshot(self, symbol: str, current_price: float = None) -> Dict:
        """Get full market data snapshot"""
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