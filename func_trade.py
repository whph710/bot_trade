import numpy as np
import pandas as pd
import talib
import math
from typing import List, Dict, Any, Optional, Tuple
import logging
from scipy import stats
from scipy.signal import find_peaks
import asyncio

logger = logging.getLogger(__name__)


def calculate_advanced_scalping_indicators(candles: List) -> Dict[str, Any]:
    """
    Расчет продвинутых индикаторов для скальпинга
    Оптимизировано для быстрого выполнения и точности сигналов
    """
    if not candles or len(candles) < 50:
        return {}

    try:
        # Конвертируем в numpy массивы для talib
        timestamps = np.array([int(c[0]) for c in candles])
        opens = np.array([float(c[1]) for c in candles])
        highs = np.array([float(c[2]) for c in candles])
        lows = np.array([float(c[3]) for c in candles])
        closes = np.array([float(c[4]) for c in candles])
        volumes = np.array([float(c[5]) for c in candles])

        current_price = closes[-1]

        # === TEMA ТРЕНД СИСТЕМА (3, 5, 8) ===
        tema3 = talib.TEMA(closes, timeperiod=3)
        tema5 = talib.TEMA(closes, timeperiod=5)
        tema8 = talib.TEMA(closes, timeperiod=8)

        # Выравнивание TEMA (восходящий тренд)
        tema_alignment = (tema3[-1] > tema5[-1] > tema8[-1]) if not np.isnan(
            [tema3[-1], tema5[-1], tema8[-1]]).any() else False

        # Наклон тренда
        tema_slope = 0
        if len(tema3) >= 5 and not np.isnan(tema3[-5:]).any():
            tema_slope = (tema3[-1] - tema3[-5]) / tema3[-5] * 100

        # Направление тренда
        if tema_slope > 0.1:
            trend_direction = 'BULLISH'
        elif tema_slope < -0.1:
            trend_direction = 'BEARISH'
        else:
            trend_direction = 'SIDEWAYS'

        # === MOMENTUM ИНДИКАТОРЫ ===
        rsi = talib.RSI(closes, timeperiod=14)
        rsi_current = rsi[-1] if not np.isnan(rsi[-1]) else 50

        # RSI тренд (последние 3 значения)
        if len(rsi) >= 3 and not np.isnan(rsi[-3:]).any():
            if rsi[-1] > rsi[-2] > rsi[-3]:
                rsi_trend = 'RISING'
            elif rsi[-1] < rsi[-2] < rsi[-3]:
                rsi_trend = 'FALLING'
            else:
                rsi_trend = 'NEUTRAL'
        else:
            rsi_trend = 'NEUTRAL'

        # Stochastic
        stoch_k, stoch_d = talib.STOCH(highs, lows, closes)
        stoch_k_current = stoch_k[-1] if not np.isnan(stoch_k[-1]) else 50
        stoch_d_current = stoch_d[-1] if not np.isnan(stoch_d[-1]) else 50

        # Сигнал Stochastic
        if stoch_k_current > stoch_d_current and stoch_k_current < 20:
            stoch_signal = 'OVERSOLD_CROSS'
        elif stoch_k_current < stoch_d_current and stoch_k_current > 80:
            stoch_signal = 'OVERBOUGHT_CROSS'
        elif stoch_k_current < 20:
            stoch_signal = 'OVERSOLD'
        elif stoch_k_current > 80:
            stoch_signal = 'OVERBOUGHT'
        else:
            stoch_signal = 'NEUTRAL'

        # MACD
        macd_line, macd_signal_line, macd_hist = talib.MACD(closes)

        # MACD сигналы
        macd_signal = 'NEUTRAL'
        if len(macd_line) >= 2 and not np.isnan(
                [macd_line[-2], macd_line[-1], macd_signal_line[-2], macd_signal_line[-1]]).any():
            # Пересечение линий
            if macd_line[-2] <= macd_signal_line[-2] and macd_line[-1] > macd_signal_line[-1]:
                macd_signal = 'CROSS_UP'
            elif macd_line[-2] >= macd_signal_line[-2] and macd_line[-1] < macd_signal_line[-1]:
                macd_signal = 'CROSS_DOWN'
            # Общее направление
            elif macd_line[-1] > macd_signal_line[-1] and macd_line[-1] > 0:
                macd_signal = 'BULLISH'
            elif macd_line[-1] < macd_signal_line[-1] and macd_line[-1] < 0:
                macd_signal = 'BEARISH'

        # === ВОЛАТИЛЬНОСТЬ ===
        atr = talib.ATR(highs, lows, closes, timeperiod=14)
        atr_current = atr[-1] if not np.isnan(atr[-1]) else current_price * 0.01
        atr_percent = (atr_current / current_price) * 100

        # Режим волатильности
        if len(atr) >= 20:
            atr_avg = np.mean(atr[-20:])
            if atr_current > atr_avg * 1.5:
                volatility_regime = 'HIGH'
            elif atr_current < atr_avg * 0.7:
                volatility_regime = 'LOW'
            else:
                volatility_regime = 'MEDIUM'
        else:
            volatility_regime = 'MEDIUM'

        # Скорость изменения цены
        if len(closes) >= 5:
            price_velocity = (closes[-1] - closes[-5]) / closes[-5] * 100
        else:
            price_velocity = 0

        # Ускорение моментума
        if len(closes) >= 10:
            recent_momentum = (closes[-1] - closes[-5]) / closes[-5]
            previous_momentum = (closes[-5] - closes[-10]) / closes[-10]
            momentum_acceleration = recent_momentum - previous_momentum
        else:
            momentum_acceleration = 0

        # === ОБЪЕМНЫЙ АНАЛИЗ ===
        volume_sma = talib.SMA(volumes, timeperiod=20)
        current_volume = volumes[-1]
        avg_volume = volume_sma[-1] if not np.isnan(volume_sma[-1]) else current_volume

        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        volume_spike = volume_ratio >= 2.0

        # Сила объема
        if volume_ratio >= 3.0:
            volume_strength = 3  # Очень сильный
        elif volume_ratio >= 2.0:
            volume_strength = 2  # Сильный
        elif volume_ratio >= 1.5:
            volume_strength = 1  # Умеренный
        else:
            volume_strength = 0  # Слабый

        # Тренд объема
        if len(volume_sma) >= 5:
            volume_trend_slope = (volume_sma[-1] - volume_sma[-5]) / volume_sma[-5]
            if volume_trend_slope > 0.1:
                volume_trend = 'INCREASING'
            elif volume_trend_slope < -0.1:
                volume_trend = 'DECREASING'
            else:
                volume_trend = 'STABLE'
        else:
            volume_trend = 'STABLE'

        # === ДИВЕРГЕНЦИИ ===
        momentum_divergence = False
        if len(closes) >= 20 and len(rsi) >= 20:
            # Ищем дивергенцию между ценой и RSI
            price_highs, _ = find_peaks(closes[-20:], distance=3)
            rsi_highs, _ = find_peaks(rsi[-20:], distance=3)

            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                # Берем два последних пика
                last_price_high = closes[-20:][price_highs[-1]]
                prev_price_high = closes[-20:][price_highs[-2]]
                last_rsi_high = rsi[-20:][rsi_highs[-1]]
                prev_rsi_high = rsi[-20:][rsi_highs[-2]]

                # Медвежья дивергенция: цена растет, RSI падает
                if last_price_high > prev_price_high and last_rsi_high < prev_rsi_high:
                    momentum_divergence = True

        # === УРОВНИ ПОДДЕРЖКИ/СОПРОТИВЛЕНИЯ ===
        support_levels, resistance_levels = _calculate_sr_levels(highs, lows, closes)

        # Проверка близости к уровням
        near_support = any(abs(current_price - level) / current_price < 0.002 for level in support_levels[-3:])
        near_resistance = any(abs(current_price - level) / current_price < 0.002 for level in resistance_levels[-3:])

        # === СИЛЫ ТРЕНДА ===
        # ADX для силы тренда
        adx = talib.ADX(highs, lows, closes, timeperiod=14)
        adx_current = adx[-1] if not np.isnan(adx[-1]) else 20

        if adx_current > 40:
            trend_strength = 3  # Очень сильный
        elif adx_current > 25:
            trend_strength = 2  # Сильный
        elif adx_current > 15:
            trend_strength = 1  # Слабый
        else:
            trend_strength = 0  # Нет тренда

        # === ПАТТЕРНЫ РАЗВОРОТА ===
        reversal_patterns = _detect_reversal_patterns(opens, highs, lows, closes)

        # === ПАТТЕРНЫ ПРОДОЛЖЕНИЯ ===
        continuation_patterns = _detect_continuation_patterns(opens, highs, lows, closes)

        # === ПОТЕНЦИАЛ ПРОБОЯ ===
        breakout_potential = _calculate_breakout_potential(closes, volumes, atr)

        # === ОПРЕДЕЛЕНИЕ ИНСТИТУЦИОНАЛЬНОГО ПОТОКА ===
        institutional_flow = _analyze_institutional_flow(volumes, closes, highs, lows)

        # === ДЕТЕКЦИЯ ПРОБОЯ ВОЛАТИЛЬНОСТИ ===
        volatility_breakout = False
        if len(atr) >= 10:
            recent_atr = np.mean(atr[-3:])
            historical_atr = np.mean(atr[-10:-3])
            if recent_atr > historical_atr * 1.5:
                volatility_breakout = True

        # === ИТОГОВАЯ СТРУКТУРА ===
        return {
            # Тренд
            'tema3_values': _clean_array(tema3),
            'tema5_values': _clean_array(tema5),
            'tema8_values': _clean_array(tema8),
            'tema_alignment': tema_alignment,
            'tema_slope': tema_slope,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,

            # Моментум
            'rsi_values': _clean_array(rsi),
            'rsi_current': rsi_current,
            'rsi_trend': rsi_trend,
            'stoch_k': _clean_array(stoch_k),
            'stoch_d': _clean_array(stoch_d),
            'stoch_signal': stoch_signal,
            'macd_line': _clean_array(macd_line),
            'macd_signal': _clean_array(macd_signal_line),
            'macd_signal': macd_signal,
            'momentum_divergence': momentum_divergence,

            # Волатильность
            'atr_values': _clean_array(atr),
            'atr_current': atr_current,
            'atr_percent': atr_percent,
            'volatility_regime': volatility_regime,
            'price_velocity': price_velocity,
            'momentum_acceleration': momentum_acceleration,
            'volatility_breakout': volatility_breakout,

            # Объемы
            'volume_spike': volume_spike,
            'volume_ratio': volume_ratio,
            'volume_strength': volume_strength,
            'volume_trend': volume_trend,
            'institutional_flow': institutional_flow,

            # Уровни
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'near_support': near_support,
            'near_resistance': near_resistance,

            # Паттерны
            'reversal_patterns': reversal_patterns,
            'continuation_patterns': continuation_patterns,
            'breakout_potential': breakout_potential,

            # ADX
            'adx_current': adx_current,
        }

    except Exception as e:
        logger.error(f"❌ Ошибка расчета индикаторов: {e}")
        return {}


def _clean_array(arr: np.ndarray) -> List[float]:
    """Очистка массива от NaN значений для JSON"""
    if arr is None or len(arr) == 0:
        return []

    cleaned = []
    for val in arr:
        if isinstance(val, (np.integer, np.floating)):
            val = float(val)
        if not (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            cleaned.append(val)
        else:
            cleaned.append(0.0)

    return cleaned


def _calculate_sr_levels(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Tuple[List[float], List[float]]:
    """Расчет уровней поддержки и сопротивления"""
    try:
        # Находим локальные максимумы и минимумы
        resistance_indices, _ = find_peaks(highs, distance=5, prominence=np.std(highs) * 0.5)
        support_indices, _ = find_peaks(-lows, distance=5, prominence=np.std(lows) * 0.5)

        # Получаем уровни
        resistance_levels = [float(highs[i]) for i in resistance_indices[-10:]]  # Последние 10
        support_levels = [float(lows[i]) for i in support_indices[-10:]]

        # Добавляем психологические уровни
        current_price = float(closes[-1])

        # Круглые числа
        if current_price > 1000:
            step = 100
        elif current_price > 100:
            step = 10
        elif current_price > 10:
            step = 1
        else:
            step = 0.1

        # Ближайшие круглые числа
        lower_round = math.floor(current_price / step) * step
        upper_round = math.ceil(current_price / step) * step

        if abs(current_price - lower_round) / current_price < 0.05:
            support_levels.append(lower_round)
        if abs(current_price - upper_round) / current_price < 0.05:
            resistance_levels.append(upper_round)

        # Сортируем и убираем дубликаты
        support_levels = sorted(list(set(support_levels)))
        resistance_levels = sorted(list(set(resistance_levels)))

        return support_levels, resistance_levels

    except Exception as e:
        logger.error(f"❌ Ошибка расчета уровней S/R: {e}")
        return [], []


def _detect_reversal_patterns(opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> List[str]:
    """Детекция паттернов разворота"""
    patterns = []

    if len(closes) < 3:
        return patterns

    try:
        # Последние 3 свечи
        o1, h1, l1, c1 = opens[-3], highs[-3], lows[-3], closes[-3]
        o2, h2, l2, c2 = opens[-2], highs[-2], lows[-2], closes[-2]
        o3, h3, l3, c3 = opens[-1], highs[-1], lows[-1], closes[-1]

        # Размеры тел свечей
        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        body3 = abs(c3 - o3)

        # Средний размер тела
        avg_body = (body1 + body2 + body3) / 3

        # === МОЛОТ И ПОВЕШЕННЫЙ ===
        # Условия: маленькое тело, длинная нижняя тень, короткая верхняя тень
        lower_shadow = min(o3, c3) - l3
        upper_shadow = h3 - max(o3, c3)

        if (body3 < avg_body * 0.3 and
                lower_shadow > body3 * 2 and
                upper_shadow < body3 * 0.5):

            if c3 > o3:  # Зеленая свеча
                patterns.append("HAMMER_BULLISH")
            else:  # Красная свеча
                patterns.append("HANGING_MAN_BEARISH")

        # === ПАДАЮЩАЯ ЗВЕЗДА ===
        if (body3 < avg_body * 0.3 and
                upper_shadow > body3 * 2 and
                lower_shadow < body3 * 0.5):
            patterns.append("SHOOTING_STAR_BEARISH")

        # === ДОДЖИ ===
        if body3 < avg_body * 0.1:
            patterns.append("DOJI_INDECISION")

        # === ПОГЛОЩЕНИЕ ===
        if len(closes) >= 2:
            # Медвежье поглощение
            if (c2 > o2 and  # Предыдущая свеча бычья
                    c3 < o3 and  # Текущая свеча медвежья
                    o3 > c2 and  # Открытие выше закрытия предыдущей
                    c3 < o2):  # Закрытие ниже открытия предыдущей
                patterns.append("BEARISH_ENGULFING")

            # Бычье поглощение
            if (c2 < o2 and  # Предыдущая свеча медвежья
                    c3 > o3 and  # Текущая свеча бычья
                    o3 < c2 and  # Открытие ниже закрытия предыдущей
                    c3 > o2):  # Закрытие выше открытия предыдущей
                patterns.append("BULLISH_ENGULFING")

        # === ХАРАМИ ===
        if len(closes) >= 2:
            if (body3 < body2 * 0.5 and  # Текущее тело меньше предыдущего
                    max(o3, c3) < max(o2, c2) and  # Максимум текущей < максимума предыдущей
                    min(o3, c3) > min(o2, c2)):  # Минимум текущей > минимума предыдущей

                if c2 < o2 and c3 > o3:  # После медвежьей идет бычья
                    patterns.append("BULLISH_HARAMI")
                elif c2 > o2 and c3 < o3:  # После бычьей идет медвежья
                    patterns.append("BEARISH_HARAMI")

        return patterns

    except Exception as e:
        logger.error(f"❌ Ошибка детекции паттернов разворота: {e}")
        return []


def _detect_continuation_patterns(opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> List[
    str]:
    """Детекция паттернов продолжения"""
    patterns = []

    if len(closes) < 5:
        return patterns

    try:
        # === ФЛАГ (последние 10 свечей) ===
        if len(closes) >= 10:
            # Сильный импульс + коррекция
            impulse = closes[-10:-5]
            correction = closes[-5:]

            impulse_change = (impulse[-1] - impulse[0]) / impulse[0]
            correction_range = (max(correction) - min(correction)) / min(correction)

            # Бычий флаг
            if (impulse_change > 0.02 and  # Сильный рост
                    correction_range < 0.015 and  # Небольшая коррекция
                    correction[-1] < correction[0]):  # Нисходящая коррекция
                patterns.append("BULLISH_FLAG")

            # Медвежий флаг
            elif (impulse_change < -0.02 and  # Сильное падение
                  correction_range < 0.015 and  # Небольшая коррекция
                  correction[-1] > correction[0]):  # Восходящая коррекция
                patterns.append("BEARISH_FLAG")

        # === ТРЕУГОЛЬНИКИ ===
        if len(highs) >= 15:
            recent_highs = highs[-15:]
            recent_lows = lows[-15:]

            # Находим максимумы и минимумы
            high_peaks, _ = find_peaks(recent_highs, distance=3)
            low_peaks, _ = find_peaks(-recent_lows, distance=3)

            if len(high_peaks) >= 2 and len(low_peaks) >= 2:
                # Анализируем наклоны линий
                high_slope = (recent_highs[high_peaks[-1]] - recent_highs[high_peaks[0]]) / len(high_peaks)
                low_slope = (recent_lows[low_peaks[-1]] - recent_lows[low_peaks[0]]) / len(low_peaks)

                # Восходящий треугольник
                if abs(high_slope) < recent_highs[-1] * 0.001 and low_slope > 0:
                    patterns.append("ASCENDING_TRIANGLE")

                # Нисходящий треугольник
                elif abs(low_slope) < recent_lows[-1] * 0.001 and high_slope < 0:
                    patterns.append("DESCENDING_TRIANGLE")

                # Симметричный треугольник
                elif high_slope < 0 and low_slope > 0:
                    patterns.append("SYMMETRICAL_TRIANGLE")

        return patterns

    except Exception as e:
        logger.error(f"❌ Ошибка детекции паттернов продолжения: {e}")
        return []


def _calculate_breakout_potential(closes: np.ndarray, volumes: np.ndarray, atr: np.ndarray) -> int:
    """Расчет потенциала пробоя (0-100)"""
    try:
        if len(closes) < 20:
            return 0

        potential = 0

        # 1. Сжатие волатильности (30 баллов)
        if len(atr) >= 10:
            recent_atr = np.mean(atr[-5:])
            historical_atr = np.mean(atr[-20:-5])

            if recent_atr < historical_atr * 0.7:  # Сжатие
                potential += 30
            elif recent_atr < historical_atr * 0.85:
                potential += 15

        # 2. Накопление объема (25 баллов)
        if len(volumes) >= 10:
            recent_volume = np.mean(volumes[-5:])
            historical_volume = np.mean(volumes[-20:-5])

            if recent_volume > historical_volume * 1.2:  # Рост объема
                potential += 25
            elif recent_volume > historical_volume * 1.1:
                potential += 12

        # 3. Консолидация цены (25 баллов)
        recent_range = max(closes[-10:]) - min(closes[-10:])
        price_consolidation = recent_range / closes[-1]

        if price_consolidation < 0.02:  # Узкий диапазон
            potential += 25
        elif price_consolidation < 0.03:
            potential += 15

        # 4. Позиция в диапазоне (20 баллов)
        range_high = max(closes[-20:])
        range_low = min(closes[-20:])
        current_position = (closes[-1] - range_low) / (range_high - range_low)

        if 0.4 <= current_position <= 0.6:  # В середине диапазона
            potential += 20
        elif 0.3 <= current_position <= 0.7:
            potential += 10

        return min(potential, 100)

    except Exception as e:
        logger.error(f"❌ Ошибка расчета потенциала пробоя: {e}")
        return 0


def _analyze_institutional_flow(volumes: np.ndarray, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> str:
    """Анализ институционального потока"""
    try:
        if len(volumes) < 10:
            return 'NEUTRAL'

        # Анализ объема vs движения цены
        recent_volumes = volumes[-5:]
        recent_closes = closes[-5:]

        avg_volume = np.mean(volumes[-20:-5])

        # Определяем направление движения
        price_change = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
        volume_increase = np.mean(recent_volumes) / avg_volume

        # Институциональная покупка
        if price_change > 0.01 and volume_increase > 1.5:
            return 'BUYING'

        # Институциональная продажа
        elif price_change < -0.01 and volume_increase > 1.5:
            return 'SELLING'

        # Накопление (рост объема при боковом движении)
        elif abs(price_change) < 0.005 and volume_increase > 1.3:
            return 'ACCUMULATION'

        # Распределение (спад объема при росте)
        elif price_change > 0.01 and volume_increase < 0.8:
            return 'DISTRIBUTION'

        return 'NEUTRAL'

    except Exception as e:
        logger.error(f"❌ Ошибка анализа институционального потока: {e}")
        return 'NEUTRAL'


async def analyze_market_microstructure(symbol: str) -> Optional[Dict]:
    """
    Анализ микроструктуры рынка для скальпинга
    Требует функций получения стакана заявок (заглушка)
    """
    try:
        # Здесь должен быть вызов к API для получения стакана
        # orderbook = await get_orderbook_async(symbol, limit=20)

        # Заглушка для демонстрации структуры
        microstructure_data = {
            'book_imbalance': 0.0,  # Дисбаланс стакана (-1 до 1)
            'spread_stability': 0.8,  # Стабильность спреда (0-1)
            'depth_quality': 0.7,  # Качество глубины (0-1)
            'large_orders': False,  # Наличие крупных заявок
            'wall_detection': 'NONE',  # Обнаружение стен (BID_WALL/ASK_WALL/NONE)
            'flow_direction': 'NEUTRAL',  # Направление потока (BUY/SELL/NEUTRAL)
            'overall_score': 65  # Общий балл микроструктуры (0-100)
        }

        return microstructure_data

    except Exception as e:
        logger.error(f"❌ Ошибка анализа микроструктуры {symbol}: {e}")
        return None


def detect_liquidity_grab_pattern(candles: List) -> Dict[str, Any]:
    """
    Детекция паттерна стоп-охоты (Liquidity Grab)
    """
    try:
        if len(candles) < 20:
            return {'detected': False, 'pattern': 'NONE'}

        # Конвертируем данные
        highs = np.array([float(c[2]) for c in candles])
        lows = np.array([float(c[3]) for c in candles])
        closes = np.array([float(c[4]) for c in candles])
        volumes = np.array([float(c[5]) for c in candles])

        # Ищем локальные экстремумы за последние 15 свечей
        recent_highs = highs[-15:]
        recent_lows = lows[-15:]
        recent_volumes = volumes[-15:]

        max_high_idx = np.argmax(recent_highs)
        min_low_idx = np.argmin(recent_lows)

        # Текущие данные
        current_high = highs[-1]
        current_low = lows[-1]
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-20:-1])

        # === БЫЧЬЯ СТОП-ОХОТА ===
        # Пробой вниз с возвратом вверх
        if (min_low_idx >= len(recent_lows) - 3 and  # Минимум в последних 3 свечах
                current_low < np.min(lows[-10:-3]) and  # Пробой предыдущих минимумов
                closes[-1] > closes[-3] and  # Возврат выше
                current_volume > avg_volume * 1.5):  # Высокий объем

            return {
                'detected': True,
                'pattern': 'BULLISH_LIQUIDITY_GRAB',
                'entry_level': float(closes[-1]),
                'invalidation_level': float(current_low),
                'target_level': float(np.max(highs[-10:]))
            }

        # === МЕДВЕЖЬЯ СТОП-ОХОТА ===
        # Пробой вверх с возвратом вниз
        elif (max_high_idx >= len(recent_highs) - 3 and  # Максимум в последних 3 свечах
              current_high > np.max(highs[-10:-3]) and  # Пробой предыдущих максимумов
              closes[-1] < closes[-3] and  # Возврат ниже
              current_volume > avg_volume * 1.5):  # Высокий объем

            return {
                'detected': True,
                'pattern': 'BEARISH_LIQUIDITY_GRAB',
                'entry_level': float(closes[-1]),
                'invalidation_level': float(current_high),
                'target_level': float(np.min(lows[-10:]))
            }

        return {'detected': False, 'pattern': 'NONE'}

    except Exception as e:
        logger.error(f"❌ Ошибка детекции стоп-охоты: {e}")
        return {'detected': False, 'pattern': 'NONE'}


def calculate_dynamic_levels(candles: List) -> Dict[str, Any]:
    """
    Расчет динамических уровней для скальпинга
    """
    try:
        if len(candles) < 50:
            return {}

        # Конвертируем данные
        opens = np.array([float(c[1]) for c in candles])
        highs = np.array([float(c[2]) for c in candles])
        lows = np.array([float(c[3]) for c in candles])
        closes = np.array([float(c[4]) for c in candles])
        volumes = np.array([float(c[5]) for c in candles])

        current_price = closes[-1]

        # === PIVOT POINTS (стандартные) ===
        # Используем данные предыдущего дня (последние 24 свечи для 1H или пропорционально)
        day_high = np.max(highs[-24:])
        day_low = np.min(lows[-24:])
        day_close = closes[-24] if len(closes) > 24 else closes[0]

        pivot = (day_high + day_low + day_close) / 3

        r1 = 2 * pivot - day_low
        r2 = pivot + (day_high - day_low)
        r3 = day_high + 2 * (pivot - day_low)

        s1 = 2 * pivot - day_high
        s2 = pivot - (day_high - day_low)
        s3 = day_low - 2 * (day_high - pivot)

        pivot_points = {
            'pivot': float(pivot),
            'r1': float(r1), 'r2': float(r2), 'r3': float(r3),
            's1': float(s1), 's2': float(s2), 's3': float(s3)
        }

        # === VOLUME PROFILE (упрощенный) ===
        # Разбиваем диапазон цен на уровни и считаем объем на каждом
        price_range = np.max(highs) - np.min(lows)
        num_levels = 20
        level_size = price_range / num_levels

        volume_profile = {}
        for i in range(num_levels):
            level_low = np.min(lows) + i * level_size
            level_high = level_low + level_size

            # Считаем объем в этом диапазоне
            level_volume = 0
            for j, (high, low, volume) in enumerate(zip(highs, lows, volumes)):
                if level_low <= high and level_high >= low:
                    # Пропорциональное распределение объема
                    overlap = min(level_high, high) - max(level_low, low)
                    candle_range = high - low
                    if candle_range > 0:
                        level_volume += volume * (overlap / candle_range)

            volume_profile[f"{level_low:.2f}-{level_high:.2f}"] = level_volume

        # Находим Point of Control (максимальный объем)
        poc_level = max(volume_profile.keys(), key=lambda k: volume_profile[k])
        poc_price = float(poc_level.split('-')[0]) + level_size / 2

        # === ЗОНЫ ЛИКВИДНОСТИ ===
        liquidity_zones = []

        # Зоны вокруг крупных объемов
        sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        for level_range, volume in sorted_levels[:5]:  # Топ-5 уровней
            price = float(level_range.split('-')[0]) + level_size / 2

            # Проверяем близость к текущей цене
            distance_percent = abs(price - current_price) / current_price * 100
            if distance_percent < 5:  # В пределах 5%
                liquidity_zones.append({
                    'price': price,
                    'volume': volume,
                    'type': 'HIGH_VOLUME',
                    'distance_percent': distance_percent
                })

        # Зоны вокруг психологических уровней
        psychological_levels = _get_psychological_levels(current_price)
        for level in psychological_levels:
            distance_percent = abs(level - current_price) / current_price * 100
            if distance_percent < 3:  # В пределах 3%
                liquidity_zones.append({
                    'price': level,
                    'volume': 0,
                    'type': 'PSYCHOLOGICAL',
                    'distance_percent': distance_percent
                })

        # === ФРАКТАЛЬНЫЕ УРОВНИ ===
        support_levels, resistance_levels = _calculate_sr_levels(highs, lows, closes)

        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'pivot_points': pivot_points,
            'volume_profile': {
                'poc_price': poc_price,
                'levels': volume_profile
            },
            'liquidity_zones': liquidity_zones
        }

    except Exception as e:
        logger.error(f"❌ Ошибка расчета динамических уровней: {e}")
        return {}


def _get_psychological_levels(price: float) -> List[float]:
    """Получение психологических уровней цены"""
    levels = []

    if price > 1000:
        # Для больших цен - сотни
        base = int(price / 100) * 100
        levels.extend([base - 100, base, base + 100, base + 200])
    elif price > 100:
        # Для средних цен - десятки
        base = int(price / 10) * 10
        levels.extend([base - 20, base - 10, base, base + 10, base + 20])
    elif price > 10:
        # Для малых цен - единицы
        base = int(price)
        levels.extend([base - 2, base - 1, base, base + 1, base + 2])
    else:
        # Для очень малых цен - десятые
        base = round(price, 1)
        levels.extend([base - 0.2, base - 0.1, base, base + 0.1, base + 0.2])

    return [l for l in levels if l > 0]


def detect_scalping_signal(candles: List) -> Dict[str, Any]:
    """
    Быстрая детекция скальпингового сигнала (упрощенная версия)
    Используется для первичного скрининга
    """
    try:
        if not candles or len(candles) < 30:
            return {'signal': 'NO_SIGNAL', 'confidence': 0}

        # Быстрые расчеты
        closes = np.array([float(c[4]) for c in candles])
        volumes = np.array([float(c[5]) for c in candles])
        highs = np.array([float(c[2]) for c in candles])
        lows = np.array([float(c[3]) for c in candles])

        current_price = closes[-1]
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-20:])

        # Быстрый RSI
        rsi = talib.RSI(closes, timeperiod=14)
        rsi_current = rsi[-1] if not np.isnan(rsi[-1]) else 50

        # Быстрые EMA
        ema_fast = talib.EMA(closes, timeperiod=5)
        ema_slow = talib.EMA(closes, timeperiod=13)

        signal = 'NO_SIGNAL'
        confidence = 0
        entry_reasons = []

        # Условия для LONG
        if (rsi_current < 40 and  # Oversold
                ema_fast[-1] > ema_slow[-1] and  # Fast EMA выше медленной
                current_volume > avg_volume * 1.3):  # Объем выше среднего

            signal = 'LONG'
            confidence = 60
            entry_reasons.append("RSI oversold + EMA cross + volume")

            # Дополнительное подтверждение
            if rsi_current < 30:
                confidence += 15
            if current_volume > avg_volume * 2:
                confidence += 10

        # Условия для SHORT
        elif (rsi_current > 60 and  # Overbought
              ema_fast[-1] < ema_slow[-1] and  # Fast EMA ниже медленной
              current_volume > avg_volume * 1.3):  # Объем выше среднего

            signal = 'SHORT'
            confidence = 60
            entry_reasons.append("RSI overbought + EMA cross + volume")

            # Дополнительное подтверждение
            if rsi_current > 70:
                confidence += 15
            if current_volume > avg_volume * 2:
                confidence += 10

        # Волатильность
        atr = talib.ATR(highs, lows, closes, timeperiod=14)
        atr_current = atr[-1] if not np.isnan(atr[-1]) else current_price * 0.01
        atr_percent = (atr_current / current_price) * 100

        volatility_regime = 'MEDIUM'
        if atr_percent > 1.0:
            volatility_regime = 'HIGH'
        elif atr_percent < 0.3:
            volatility_regime = 'LOW'
            confidence = max(0, confidence - 20)  # Снижаем уверенность при низкой волатильности

        return {
            'signal': signal,
            'confidence': min(confidence, 100),
            'entry_reasons': entry_reasons,
            'volatility_regime': volatility_regime,
            'quality_score': confidence,
            'indicators': {
                'rsi_current': rsi_current,
                'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
                'atr_percent': atr_percent,
                'volume_spike': current_volume > avg_volume * 1.5
            }
        }

    except Exception as e:
        logger.error(f"❌ Ошибка детекции быстрого сигнала: {e}")
        return {'signal': 'NO_SIGNAL', 'confidence': 0}