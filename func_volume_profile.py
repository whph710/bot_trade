"""
Модуль расчета Volume Profile
POC, Value Area, HVN/LVN zones
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class VolumeProfileCalculator:
    """Расчет Volume Profile для свечных данных"""

    def __init__(self, price_granularity: float = None):
        """
        Args:
            price_granularity: шаг цены для группировки (None = auto)
        """
        self.price_granularity = price_granularity

    def calculate_volume_profile(
        self,
        candles: List[List],
        num_bins: int = 50
    ) -> Dict:
        """
        Рассчитать Volume Profile из свечей

        Args:
            candles: список свечей [[timestamp, o, h, l, c, v], ...]
            num_bins: количество ценовых уровней для группировки

        Returns:
            {
                'poc': float,  # Point of Control (цена с максимальным объемом)
                'poc_volume': float,
                'value_area_high': float,
                'value_area_low': float,
                'value_area_volume_pct': float,
                'profile': {price: volume},  # Полный профиль
                'hvn_zones': [(low, high), ...],  # High Volume Nodes
                'lvn_zones': [(low, high), ...],  # Low Volume Nodes
            }
        """
        if not candles or len(candles) < 10:
            return self._empty_profile()

        try:
            # Извлекаем данные
            highs = [float(c[2]) for c in candles]
            lows = [float(c[3]) for c in candles]
            closes = [float(c[4]) for c in candles]
            volumes = [float(c[5]) for c in candles]

            # Определяем ценовой диапазон
            min_price = min(lows)
            max_price = max(highs)

            if min_price == max_price or max_price == 0:
                return self._empty_profile()

            # Определяем размер бина (ценовой уровень)
            if self.price_granularity:
                bin_size = self.price_granularity
            else:
                # Автоматический расчет
                price_range = max_price - min_price
                bin_size = price_range / num_bins

            # Создаем ценовые уровни
            price_levels = np.arange(min_price, max_price + bin_size, bin_size)
            volume_at_price = defaultdict(float)

            # Распределяем объем по ценовым уровням
            for i, candle in enumerate(candles):
                high = float(candle[2])
                low = float(candle[3])
                volume = float(candle[5])

                # Находим уровни которые попадают в диапазон свечи
                candle_levels = [p for p in price_levels if low <= p <= high]

                if candle_levels:
                    # Распределяем объем равномерно по уровням внутри свечи
                    volume_per_level = volume / len(candle_levels)
                    for level in candle_levels:
                        # Округляем до bin_size
                        rounded_level = round(level / bin_size) * bin_size
                        volume_at_price[rounded_level] += volume_per_level

            if not volume_at_price:
                return self._empty_profile()

            # Сортируем по объему
            sorted_levels = sorted(volume_at_price.items(), key=lambda x: x[1], reverse=True)

            # POC - уровень с максимальным объемом
            poc_price, poc_volume = sorted_levels[0]

            # Value Area - 70% объема
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

            # Определяем HVN и LVN зоны
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
            logger.error(f"Ошибка расчета Volume Profile: {e}")
            return self._empty_profile()

    def _identify_volume_nodes(
        self,
        volume_at_price: Dict[float, float],
        poc_volume: float,
        bin_size: float
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Определить High Volume Nodes и Low Volume Nodes

        Args:
            volume_at_price: {price: volume}
            poc_volume: объем на POC
            bin_size: размер ценового бина

        Returns:
            (hvn_zones, lvn_zones)
        """
        if not volume_at_price:
            return [], []

        try:
            # Пороги
            avg_volume = np.mean(list(volume_at_price.values()))
            hvn_threshold = avg_volume * 1.5  # HVN - выше среднего в 1.5 раза
            lvn_threshold = avg_volume * 0.5  # LVN - ниже среднего в 2 раза

            # Сортируем по цене
            sorted_by_price = sorted(volume_at_price.items())

            hvn_zones = []
            lvn_zones = []

            current_hvn = None
            current_lvn = None

            for price, volume in sorted_by_price:
                # HVN detection
                if volume >= hvn_threshold:
                    if current_hvn is None:
                        current_hvn = [price, price]
                    else:
                        current_hvn[1] = price
                else:
                    if current_hvn is not None:
                        hvn_zones.append(tuple(current_hvn))
                        current_hvn = None

                # LVN detection
                if volume <= lvn_threshold:
                    if current_lvn is None:
                        current_lvn = [price, price]
                    else:
                        current_lvn[1] = price
                else:
                    if current_lvn is not None:
                        lvn_zones.append(tuple(current_lvn))
                        current_lvn = None

            # Закрываем последние зоны если есть
            if current_hvn:
                hvn_zones.append(tuple(current_hvn))
            if current_lvn:
                lvn_zones.append(tuple(current_lvn))

            return hvn_zones, lvn_zones

        except Exception as e:
            logger.error(f"Ошибка определения volume nodes: {e}")
            return [], []

    def _empty_profile(self) -> Dict:
        """Пустой профиль при ошибках"""
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
        """
        Найти ближайшую HVN зону к текущей цене

        Returns:
            (low, high) ближайшей HVN зоны или None
        """
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
        """Проверить находится ли цена в HVN зоне"""
        for low, high in hvn_zones:
            if low <= price <= high:
                return True
        return False

    def is_price_in_lvn(
        self,
        price: float,
        lvn_zones: List[Tuple[float, float]]
    ) -> bool:
        """Проверить находится ли цена в LVN зоне"""
        for low, high in lvn_zones:
            if low <= price <= high:
                return True
        return False


# ==================== АНАЛИЗАТОР VOLUME PROFILE ====================

class VolumeProfileAnalyzer:
    """Анализ Volume Profile с торговыми рекомендациями"""

    @staticmethod
    def analyze_poc_proximity(
        current_price: float,
        poc: float,
        value_area: Tuple[float, float]
    ) -> Dict:
        """
        Анализ близости к POC

        Returns:
            {
                'distance_to_poc_pct': 2.5,
                'poc_relevance': 'STRONG' / 'MODERATE' / 'WEAK',
                'expected_behavior': 'ATTRACTION' / 'NEUTRAL',
                'confidence_adjustment': +8 / 0
            }
        """
        if poc == 0 or current_price == 0:
            return {
                'distance_to_poc_pct': 0,
                'poc_relevance': 'UNKNOWN',
                'expected_behavior': 'NEUTRAL',
                'confidence_adjustment': 0
            }

        distance_pct = abs((current_price - poc) / current_price * 100)

        # Определяем релевантность POC
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
        """
        Анализ позиции относительно Value Area

        Returns:
            {
                'position': 'ABOVE' / 'INSIDE' / 'BELOW',
                'market_condition': 'OVEREXTENDED' / 'NORMAL' / 'UNDEREXTENDED',
                'expected_move': 'REVERT_TO_VA' / 'CONTINUE' / 'RANGE',
                'confidence_adjustment': +5 / 0 / -5
            }
        """
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
        """
        Анализ текущей позиции относительно volume nodes

        Returns:
            {
                'in_hvn': True / False,
                'in_lvn': True / False,
                'nearest_hvn': (low, high) or None,
                'price_behavior': 'SUPPORT' / 'RESISTANCE' / 'FAST_MOVE' / 'NORMAL',
                'confidence_adjustment': +5 / 0 / -3
            }
        """
        calc = VolumeProfileCalculator()

        in_hvn = calc.is_price_in_hvn(current_price, hvn_zones)
        in_lvn = calc.is_price_in_lvn(current_price, lvn_zones)
        nearest_hvn = calc.find_nearest_hvn(current_price, hvn_zones)

        if in_hvn:
            behavior = 'SUPPORT'  # HVN действует как поддержка/сопротивление
            adjustment = +5
        elif in_lvn:
            behavior = 'FAST_MOVE'  # LVN зоны проходятся быстро
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


# ==================== ИНТЕГРАЦИОННЫЕ ФУНКЦИИ ====================

def calculate_volume_profile_for_candles(
    candles: List[List],
    num_bins: int = 50
) -> Dict:
    """
    Удобная функция для расчета Volume Profile

    Args:
        candles: свечные данные
        num_bins: количество ценовых уровней

    Returns:
        Полный Volume Profile
    """
    calculator = VolumeProfileCalculator()
    return calculator.calculate_volume_profile(candles, num_bins)


def analyze_volume_profile(
    vp_data: Dict,
    current_price: float
) -> Dict:
    """
    Полный анализ Volume Profile с рекомендациями

    Args:
        vp_data: результат calculate_volume_profile_for_candles()
        current_price: текущая цена

    Returns:
        {
            'total_confidence_adjustment': int,
            'poc_analysis': {...},
            'value_area_analysis': {...},
            'volume_nodes_analysis': {...}
        }
    """
    analyzer = VolumeProfileAnalyzer()

    if not vp_data or vp_data.get('poc', 0) == 0:
        return {
            'total_confidence_adjustment': 0,
            'poc_analysis': {'confidence_adjustment': 0},
            'value_area_analysis': {'confidence_adjustment': 0},
            'volume_nodes_analysis': {'confidence_adjustment': 0}
        }

    # POC анализ
    poc_analysis = analyzer.analyze_poc_proximity(
        current_price,
        vp_data['poc'],
        (vp_data['value_area_low'], vp_data['value_area_high'])
    )

    # Value Area анализ
    va_analysis = analyzer.analyze_value_area_position(
        current_price,
        vp_data['value_area_low'],
        vp_data['value_area_high']
    )

    # Volume Nodes анализ
    vn_analysis = analyzer.analyze_volume_nodes(
        current_price,
        vp_data['hvn_zones'],
        vp_data['lvn_zones']
    )

    # Суммируем adjustments
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


def find_optimal_entry_zones(
    vp_data: Dict,
    current_price: float,
    signal_direction: str
) -> Dict:
    """
    Найти оптимальные зоны входа на основе Volume Profile

    Args:
        vp_data: Volume Profile данные
        current_price: текущая цена
        signal_direction: 'LONG' / 'SHORT'

    Returns:
        {
            'optimal_entry': float,
            'entry_reasoning': str,
            'nearby_support_levels': [prices],
            'nearby_resistance_levels': [prices]
        }
    """
    if not vp_data or vp_data.get('poc', 0) == 0:
        return {
            'optimal_entry': current_price,
            'entry_reasoning': 'No VP data, using current price',
            'nearby_support_levels': [],
            'nearby_resistance_levels': []
        }

    poc = vp_data['poc']
    va_low = vp_data['value_area_low']
    va_high = vp_data['value_area_high']
    hvn_zones = vp_data['hvn_zones']

    # Находим ближайшие HVN уровни
    support_levels = []
    resistance_levels = []

    for low, high in hvn_zones:
        zone_center = (low + high) / 2
        if zone_center < current_price:
            support_levels.append(zone_center)
        else:
            resistance_levels.append(zone_center)

    # Сортируем по близости
    support_levels.sort(reverse=True)  # Ближайшие сверху
    resistance_levels.sort()  # Ближайшие снизу

    # Определяем оптимальный вход
    if signal_direction == 'LONG':
        # Для LONG ищем pullback к поддержке
        if support_levels:
            optimal_entry = support_levels[0]
            reasoning = f'Pullback to nearest HVN support at {optimal_entry:.2f}'
        elif current_price > poc:
            optimal_entry = poc
            reasoning = f'Pullback to POC at {poc:.2f}'
        else:
            optimal_entry = current_price
            reasoning = 'Current price is optimal'

    else:  # SHORT
        # Для SHORT ищем rally к сопротивлению
        if resistance_levels:
            optimal_entry = resistance_levels[0]
            reasoning = f'Rally to nearest HVN resistance at {optimal_entry:.2f}'
        elif current_price < poc:
            optimal_entry = poc
            reasoning = f'Rally to POC at {poc:.2f}'
        else:
            optimal_entry = current_price
            reasoning = 'Current price is optimal'

    return {
        'optimal_entry': optimal_entry,
        'entry_reasoning': reasoning,
        'nearby_support_levels': support_levels[:3],
        'nearby_resistance_levels': resistance_levels[:3],
        'poc_level': poc,
        'value_area': (va_low, va_high)
    }


def calculate_round_numbers_proximity(current_price: float) -> Dict:
    """
    Рассчитать близость к круглым числам (психологические уровни)

    Args:
        current_price: текущая цена

    Returns:
        {
            'nearest_round_number': float,
            'distance_pct': float,
            'is_close': bool,
            'confidence_adjustment': int
        }
    """
    if current_price == 0:
        return {
            'nearest_round_number': 0,
            'distance_pct': 0,
            'is_close': False,
            'confidence_adjustment': 0
        }

    # Определяем масштаб круглых чисел
    if current_price >= 10000:
        step = 1000  # $50000, $51000, etc.
    elif current_price >= 1000:
        step = 100   # $5000, $5100, etc.
    elif current_price >= 100:
        step = 10    # $500, $510, etc.
    elif current_price >= 10:
        step = 1     # $50, $51, etc.
    else:
        step = 0.1   # $5.0, $5.1, etc.

    # Находим ближайшее круглое число
    lower_round = (current_price // step) * step
    upper_round = lower_round + step

    if abs(current_price - lower_round) < abs(current_price - upper_round):
        nearest_round = lower_round
    else:
        nearest_round = upper_round

    # Рассчитываем дистанцию
    distance_pct = abs((current_price - nearest_round) / current_price * 100)

    # Определяем близость
    if distance_pct < 0.5:
        is_close = True
        adjustment = +8  # Близко к магниту
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
        'reasoning': f'Price {current_price:.2f} is {distance_pct:.2f}% from round number {nearest_round:.2f}'
    }


def get_previous_day_levels(candles: List[List]) -> Dict:
    """
    Получить уровни предыдущего дня (PDH, PDL)

    Args:
        candles: свечи (должны покрывать хотя бы 24 часа)

    Returns:
        {
            'previous_day_high': float,
            'previous_day_low': float,
            'previous_day_close': float,
            'current_vs_pdh_pct': float,
            'current_vs_pdl_pct': float
        }
    """
    if not candles or len(candles) < 24:
        return {
            'previous_day_high': 0,
            'previous_day_low': 0,
            'previous_day_close': 0,
            'current_vs_pdh_pct': 0,
            'current_vs_pdl_pct': 0
        }

    try:
        # Берем данные предыдущего дня (свечи от -48 до -24 если есть)
        if len(candles) >= 48:
            prev_day_candles = candles[-48:-24]
        else:
            # Если меньше данных, берем первую половину
            split_point = len(candles) // 2
            prev_day_candles = candles[:split_point]

        # Находим PDH, PDL
        pdh = max(float(c[2]) for c in prev_day_candles)  # High
        pdl = min(float(c[3]) for c in prev_day_candles)  # Low
        pdc = float(prev_day_candles[-1][4])  # Close последней свечи

        # Текущая цена
        current_price = float(candles[-1][4])

        # Рассчитываем дистанции
        if pdh > 0:
            vs_pdh = ((current_price - pdh) / pdh) * 100
        else:
            vs_pdh = 0

        if pdl > 0:
            vs_pdl = ((current_price - pdl) / pdl) * 100
        else:
            vs_pdl = 0

        return {
            'previous_day_high': pdh,
            'previous_day_low': pdl,
            'previous_day_close': pdc,
            'current_vs_pdh_pct': round(vs_pdh, 2),
            'current_vs_pdl_pct': round(vs_pdl, 2),
            'current_price': current_price
        }

    except Exception as e:
        logger.error(f"Ошибка расчета PDH/PDL: {e}")
        return {
            'previous_day_high': 0,
            'previous_day_low': 0,
            'previous_day_close': 0,
            'current_vs_pdh_pct': 0,
            'current_vs_pdl_pct': 0
        }


def analyze_key_levels(
    current_price: float,
    vp_data: Dict,
    candles: List[List]
) -> Dict:
    """
    Комплексный анализ ключевых уровней

    Args:
        current_price: текущая цена
        vp_data: Volume Profile данные
        candles: свечи для расчета PDH/PDL

    Returns:
        {
            'total_confidence_adjustment': int,
            'round_numbers': {...},
            'previous_day_levels': {...},
            'volume_profile_levels': {...},
            'all_key_levels': {
                'support': [levels],
                'resistance': [levels]
            }
        }
    """
    # Круглые числа
    round_analysis = calculate_round_numbers_proximity(current_price)

    # PDH/PDL
    pd_levels = get_previous_day_levels(candles)

    # Анализ близости к PDH/PDL
    pd_adjustment = 0
    if pd_levels['previous_day_high'] > 0:
        pdh_distance = abs(pd_levels['current_vs_pdh_pct'])
        pdl_distance = abs(pd_levels['current_vs_pdl_pct'])

        if pdh_distance < 1.0 or pdl_distance < 1.0:
            pd_adjustment = +8  # Близко к важному уровню

    # Volume Profile уровни
    vp_adjustment = 0
    support_levels = []
    resistance_levels = []

    if vp_data and vp_data.get('poc', 0) > 0:
        poc = vp_data['poc']

        # POC как ключевой уровень
        if poc < current_price:
            support_levels.append(poc)
        else:
            resistance_levels.append(poc)

        # HVN зоны как ключевые уровни
        for low, high in vp_data.get('hvn_zones', []):
            center = (low + high) / 2
            if center < current_price:
                support_levels.append(center)
            else:
                resistance_levels.append(center)

    # PDH/PDL как уровни
    if pd_levels['previous_day_high'] > 0:
        if pd_levels['previous_day_high'] > current_price:
            resistance_levels.append(pd_levels['previous_day_high'])
        if pd_levels['previous_day_low'] < current_price:
            support_levels.append(pd_levels['previous_day_low'])

    # Круглые числа как уровни
    round_num = round_analysis['nearest_round_number']
    if round_num > 0:
        if round_num > current_price:
            resistance_levels.append(round_num)
        elif round_num < current_price:
            support_levels.append(round_num)

    # Сортируем уровни
    support_levels.sort(reverse=True)
    resistance_levels.sort()

    # Суммарный adjustment
    total_adjustment = (
        round_analysis['confidence_adjustment'] +
        pd_adjustment +
        vp_adjustment
    )

    return {
        'total_confidence_adjustment': total_adjustment,
        'round_numbers': round_analysis,
        'previous_day_levels': pd_levels,
        'pd_proximity_adjustment': pd_adjustment,
        'all_key_levels': {
            'support': support_levels[:5],  # Топ 5
            'resistance': resistance_levels[:5]
        }
    }