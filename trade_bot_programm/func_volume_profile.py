"""
Volume Profile calculation module
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class VolumeProfileCalculator:
    """Volume Profile calculator"""

    def __init__(self, price_granularity: float = None):
        self.price_granularity = price_granularity

    def calculate_volume_profile(
        self,
        candles: List[List],
        num_bins: int = 50
    ) -> Dict:
        """Calculate Volume Profile from candles"""
        if not candles or len(candles) < 10:
            return self._empty_profile()

        try:
            highs = [float(c[2]) for c in candles]
            lows = [float(c[3]) for c in candles]
            closes = [float(c[4]) for c in candles]
            volumes = [float(c[5]) for c in candles]

            min_price = min(lows)
            max_price = max(highs)

            if min_price == max_price or max_price == 0:
                return self._empty_profile()

            if self.price_granularity:
                bin_size = self.price_granularity
            else:
                price_range = max_price - min_price
                bin_size = price_range / num_bins

            price_levels = np.arange(min_price, max_price + bin_size, bin_size)
            volume_at_price = defaultdict(float)

            for i, candle in enumerate(candles):
                high = float(candle[2])
                low = float(candle[3])
                volume = float(candle[5])

                candle_levels = [p for p in price_levels if low <= p <= high]

                if candle_levels:
                    volume_per_level = volume / len(candle_levels)
                    for level in candle_levels:
                        rounded_level = round(level / bin_size) * bin_size
                        volume_at_price[rounded_level] += volume_per_level

            if not volume_at_price:
                return self._empty_profile()

            sorted_levels = sorted(volume_at_price.items(), key=lambda x: x[1], reverse=True)

            poc_price, poc_volume = sorted_levels[0]

            total_volume = sum(volume_at_price.values())
            value_area_volume_target = total_volume * 0.70

            value_area_prices = []
            accumulated_volume = 0

            for price, volume in sorted_levels:
                value_area_prices.append(price)
                accumulated_volume += volume
                if accumulated_volume >= value_area_volume_target:
                    break

            value_area_high = max(value_area_prices)
            value_area_low = min(value_area_prices)
            value_area_pct = (accumulated_volume / total_volume) * 100

            hvn_zones, lvn_zones = self._identify_volume_nodes(
                volume_at_price,
                poc_volume,
                bin_size
            )

            return {
                'poc': float(poc_price),
                'poc_volume': float(poc_volume),
                'value_area_high': float(value_area_high),
                'value_area_low': float(value_area_low),
                'value_area_volume_pct': round(value_area_pct, 2),
                'total_volume': float(total_volume),
                'profile': {float(k): float(v) for k, v in volume_at_price.items()},
                'hvn_zones': hvn_zones,
                'lvn_zones': lvn_zones,
                'price_range': (float(min_price), float(max_price)),
                'bin_size': float(bin_size)
            }

        except Exception as e:
            logger.error(f"Volume Profile calculation error: {e}")
            return self._empty_profile()

    def _identify_volume_nodes(
        self,
        volume_at_price: Dict[float, float],
        poc_volume: float,
        bin_size: float
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """Identify High and Low Volume Nodes"""
        if not volume_at_price:
            return [], []

        try:
            avg_volume = np.mean(list(volume_at_price.values()))
            hvn_threshold = avg_volume * 1.5
            lvn_threshold = avg_volume * 0.5

            sorted_by_price = sorted(volume_at_price.items())

            hvn_zones = []
            lvn_zones = []

            current_hvn = None
            current_lvn = None

            for price, volume in sorted_by_price:
                if volume >= hvn_threshold:
                    if current_hvn is None:
                        current_hvn = [price, price]
                    else:
                        current_hvn[1] = price
                else:
                    if current_hvn is not None:
                        hvn_zones.append(tuple(current_hvn))
                        current_hvn = None

                if volume <= lvn_threshold:
                    if current_lvn is None:
                        current_lvn = [price, price]
                    else:
                        current_lvn[1] = price
                else:
                    if current_lvn is not None:
                        lvn_zones.append(tuple(current_lvn))
                        current_lvn = None

            if current_hvn:
                hvn_zones.append(tuple(current_hvn))
            if current_lvn:
                lvn_zones.append(tuple(current_lvn))

            return hvn_zones, lvn_zones

        except Exception as e:
            logger.error(f"Volume nodes identification error: {e}")
            return [], []

    def _empty_profile(self) -> Dict:
        """Empty profile on errors"""
        return {
            'poc': 0.0,
            'poc_volume': 0.0,
            'value_area_high': 0.0,
            'value_area_low': 0.0,
            'value_area_volume_pct': 0.0,
            'total_volume': 0.0,
            'profile': {},
            'hvn_zones': [],
            'lvn_zones': [],
            'price_range': (0.0, 0.0),
            'bin_size': 0.0
        }

    def find_nearest_hvn(
        self,
        current_price: float,
        hvn_zones: List[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """Find nearest HVN zone"""
        if not hvn_zones:
            return None

        nearest = None
        min_distance = float('inf')

        for low, high in hvn_zones:
            zone_center = (low + high) / 2
            distance = abs(current_price - zone_center)

            if distance < min_distance:
                min_distance = distance
                nearest = (low, high)

        return nearest

    def is_price_in_hvn(
        self,
        price: float,
        hvn_zones: List[Tuple[float, float]]
    ) -> bool:
        """Check if price is in HVN zone"""
        for low, high in hvn_zones:
            if low <= price <= high:
                return True
        return False

    def is_price_in_lvn(
        self,
        price: float,
        lvn_zones: List[Tuple[float, float]]
    ) -> bool:
        """Check if price is in LVN zone"""
        for low, high in lvn_zones:
            if low <= price <= high:
                return True
        return False


class VolumeProfileAnalyzer:
    """Volume Profile analyzer"""

    @staticmethod
    def analyze_poc_proximity(
        current_price: float,
        poc: float,
        value_area: Tuple[float, float]
    ) -> Dict:
        """Analyze POC proximity"""
        if poc == 0 or current_price == 0:
            return {
                'distance_to_poc_pct': 0,
                'poc_relevance': 'UNKNOWN',
                'expected_behavior': 'NEUTRAL',
                'confidence_adjustment': 0
            }

        distance_pct = abs((current_price - poc) / current_price * 100)

        if distance_pct < 1.0:
            relevance = 'STRONG'
            behavior = 'ATTRACTION'
            adjustment = +8
        elif distance_pct < 2.5:
            relevance = 'MODERATE'
            behavior = 'ATTRACTION'
            adjustment = +5
        elif distance_pct < 5.0:
            relevance = 'WEAK'
            behavior = 'NEUTRAL'
            adjustment = 0
        else:
            relevance = 'EXPIRED'
            behavior = 'NEUTRAL'
            adjustment = 0

        return {
            'distance_to_poc_pct': round(distance_pct, 2),
            'poc_relevance': relevance,
            'expected_behavior': behavior,
            'confidence_adjustment': adjustment
        }

    @staticmethod
    def analyze_value_area_position(
        current_price: float,
        value_area_low: float,
        value_area_high: float
    ) -> Dict:
        """Analyze Value Area position"""
        if value_area_low == 0 or value_area_high == 0:
            return {
                'position': 'UNKNOWN',
                'market_condition': 'UNKNOWN',
                'expected_move': 'RANGE',
                'confidence_adjustment': 0
            }

        if current_price > value_area_high:
            position = 'ABOVE'
            distance_pct = ((current_price - value_area_high) / value_area_high) * 100

            if distance_pct > 3.0:
                condition = 'OVEREXTENDED'
                move = 'REVERT_TO_VA'
                adjustment = -5
            else:
                condition = 'STRONG'
                move = 'CONTINUE'
                adjustment = +5

        elif current_price < value_area_low:
            position = 'BELOW'
            distance_pct = ((value_area_low - current_price) / value_area_low) * 100

            if distance_pct > 3.0:
                condition = 'UNDEREXTENDED'
                move = 'REVERT_TO_VA'
                adjustment = -5
            else:
                condition = 'WEAK'
                move = 'CONTINUE'
                adjustment = +5

        else:
            position = 'INSIDE'
            condition = 'NORMAL'
            move = 'RANGE'
            adjustment = 0

        return {
            'position': position,
            'market_condition': condition,
            'expected_move': move,
            'confidence_adjustment': adjustment
        }

    @staticmethod
    def analyze_volume_nodes(
        current_price: float,
        hvn_zones: List[Tuple[float, float]],
        lvn_zones: List[Tuple[float, float]]
    ) -> Dict:
        """Analyze volume nodes"""
        calc = VolumeProfileCalculator()

        in_hvn = calc.is_price_in_hvn(current_price, hvn_zones)
        in_lvn = calc.is_price_in_lvn(current_price, lvn_zones)
        nearest_hvn = calc.find_nearest_hvn(current_price, hvn_zones)

        if in_hvn:
            behavior = 'SUPPORT'
            adjustment = +5
        elif in_lvn:
            behavior = 'FAST_MOVE'
            adjustment = -3
        else:
            behavior = 'NORMAL'
            adjustment = 0

        return {
            'in_hvn': in_hvn,
            'in_lvn': in_lvn,
            'nearest_hvn': nearest_hvn,
            'price_behavior': behavior,
            'confidence_adjustment': adjustment
        }


def calculate_volume_profile_for_candles(
    candles: List[List],
    num_bins: int = 50
) -> Dict:
    """Calculate Volume Profile"""
    calculator = VolumeProfileCalculator()
    return calculator.calculate_volume_profile(candles, num_bins)


def analyze_volume_profile(
    vp_data: Dict,
    current_price: float
) -> Dict:
    """Full Volume Profile analysis"""
    analyzer = VolumeProfileAnalyzer()

    if not vp_data or vp_data.get('poc', 0) == 0:
        return {
            'total_confidence_adjustment': 0,
            'poc_analysis': {'confidence_adjustment': 0},
            'value_area_analysis': {'confidence_adjustment': 0},
            'volume_nodes_analysis': {'confidence_adjustment': 0}
        }

    poc_analysis = analyzer.analyze_poc_proximity(
        current_price,
        vp_data['poc'],
        (vp_data['value_area_low'], vp_data['value_area_high'])
    )

    va_analysis = analyzer.analyze_value_area_position(
        current_price,
        vp_data['value_area_low'],
        vp_data['value_area_high']
    )

    vn_analysis = analyzer.analyze_volume_nodes(
        current_price,
        vp_data['hvn_zones'],
        vp_data['lvn_zones']
    )

    total_adjustment = (
        poc_analysis['confidence_adjustment'] +
        va_analysis['confidence_adjustment'] +
        vn_analysis['confidence_adjustment']
    )

    return {
        'total_confidence_adjustment': total_adjustment,
        'poc_analysis': poc_analysis,
        'value_area_analysis': va_analysis,
        'volume_nodes_analysis': vn_analysis,
        'raw_vp_data': vp_data
    }


def calculate_round_numbers_proximity(current_price: float) -> Dict:
    """Calculate proximity to round numbers"""
    if current_price == 0:
        return {
            'nearest_round_number': 0,
            'distance_pct': 0,
            'is_close': False,
            'confidence_adjustment': 0,
            'reasoning': 'Invalid price'
        }

    if current_price >= 10000:
        step = 1000
    elif current_price >= 1000:
        step = 100
    elif current_price >= 100:
        step = 10
    elif current_price >= 10:
        step = 1
    else:
        step = 0.1

    lower_round = (current_price // step) * step
    upper_round = lower_round + step

    if abs(current_price - lower_round) < abs(current_price - upper_round):
        nearest_round = lower_round
    else:
        nearest_round = upper_round

    distance_pct = abs((current_price - nearest_round) / current_price * 100)

    if distance_pct < 0.5:
        is_close = True
        adjustment = +8
    elif distance_pct < 1.0:
        is_close = True
        adjustment = +5
    else:
        is_close = False
        adjustment = 0

    return {
        'nearest_round_number': nearest_round,
        'distance_pct': round(distance_pct, 3),
        'is_close': is_close,
        'confidence_adjustment': adjustment,
        'reasoning': f'Price {current_price:.2f} is {distance_pct:.2f}% from round {nearest_round:.2f}'
    }