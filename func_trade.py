import numpy as np
from typing import List, Dict, Any, Tuple
import time
import math

# ПАРАМЕТРЫ СОГЛАСНО ИНСТРУКЦИИ
SCALPING_PARAMS = {
    # EMA параметры (5/8/20)
    'ema_fast': 5,
    'ema_medium': 8,
    'ema_slow': 20,

    # RSI(9) для фильтра импульса
    'rsi_period': 9,

    # MACD стандарт (12,26,9)
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,

    # ATR(14) для волатильности
    'atr_period': 14,

    # Volume анализ
    'volume_sma': 20,

    # Bollinger Bands (20,2)
    'bb_period': 20,
    'bb_std': 2,

    # Минимальная уверенность
    'min_confidence': 70
}


def safe_float(value):
    """Безопасное преобразование в float"""
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return 0.0
        return result
    except (ValueError, TypeError):
        return 0.0


def safe_int(value):
    """Безопасное преобразование в int"""
    try:
        result = int(value)
        if math.isnan(result) or math.isinf(result):
            return 0
        return result
    except (ValueError, TypeError):
        return 0


def safe_list(arr):
    """Безопасное преобразование массива в список"""
    try:
        if isinstance(arr, np.ndarray):
            return [safe_float(x) for x in arr.tolist()]
        elif isinstance(arr, list):
            return [safe_float(x) for x in arr]
        else:
            return []
    except:
        return []


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average"""
    ema = np.zeros_like(prices)
    alpha = 2.0 / (period + 1)
    ema[0] = prices[0]

    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema


def calculate_rsi(prices: np.ndarray, period: int = 9) -> np.ndarray:
    """RSI(9) для фильтра импульса согласно инструкции"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gains = np.zeros_like(prices)
    avg_losses = np.zeros_like(prices)

    # Первое значение через SMA
    if len(gains) >= period:
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])

    # EMA для последующих значений
    alpha = 1.0 / period
    for i in range(period + 1, len(prices)):
        avg_gains[i] = alpha * gains[i - 1] + (1 - alpha) * avg_gains[i - 1]
        avg_losses[i] = alpha * losses[i - 1] + (1 - alpha) * avg_losses[i - 1]

    rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses != 0)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, np.ndarray]:
    """MACD стандарт (12,26,9) согласно инструкции"""
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def calculate_atr(candles: List[List[str]], period: int = 14) -> Dict[str, Any]:
    """ATR(14) для волатильности согласно инструкции"""
    if len(candles) < period + 1:
        return {'atr': [], 'current': 0.0}

    highs = np.array([float(c[2]) for c in candles])
    lows = np.array([float(c[3]) for c in candles])
    closes = np.array([float(c[4]) for c in candles])

    tr = np.zeros(len(candles))
    for i in range(1, len(candles)):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )

    atr = np.zeros(len(candles))
    if len(tr) > period:
        atr[period] = np.mean(tr[1:period + 1])

        for i in range(period + 1, len(candles)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return {
        'atr': safe_list(atr),
        'current': safe_float(atr[-1] if len(atr) > 0 else 0.0)
    }


def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, std: float = 2.0) -> Dict[str, np.ndarray]:
    """Bollinger Bands (20,2) согласно инструкции"""
    sma = np.zeros_like(prices)
    upper = np.zeros_like(prices)
    lower = np.zeros_like(prices)

    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1:i + 1]
        sma[i] = np.mean(window)
        std_dev = np.std(window)
        upper[i] = sma[i] + (std_dev * std)
        lower[i] = sma[i] - (std_dev * std)

    return {
        'upper': upper,
        'middle': sma,
        'lower': lower
    }


def calculate_volume_sma(candles: List[List[str]], period: int = 20) -> np.ndarray:
    """Volume SMA(20) для подтверждения пробоев"""
    volumes = np.array([float(c[5]) for c in candles])
    volume_sma = np.zeros_like(volumes)

    for i in range(period - 1, len(volumes)):
        volume_sma[i] = np.mean(volumes[i - period + 1:i + 1])

    return volume_sma


def analyze_higher_timeframe_trend(candles_15m: List[List[str]]) -> Dict[str, Any]:
    """Анализ старшего таймфрейма (15m) для контекста"""
    if len(candles_15m) < 30:
        return {'trend': 'UNKNOWN', 'strength': 0}

    closes = np.array([float(c[4]) for c in candles_15m])
    ema20 = calculate_ema(closes, 20)
    ema50 = calculate_ema(closes, 50)

    current_price = closes[-1]
    ema20_current = ema20[-1]
    ema50_current = ema50[-1]

    # Определение тренда
    if current_price > ema20_current > ema50_current:
        trend = 'UPTREND'
        strength = safe_int(((current_price - ema50_current) / ema50_current) * 1000)
    elif current_price < ema20_current < ema50_current:
        trend = 'DOWNTREND'
        strength = safe_int(((ema50_current - current_price) / ema50_current) * 1000)
    else:
        trend = 'SIDEWAYS'
        strength = 0

    return {
        'trend': trend,
        'strength': min(100, abs(strength)),
        'ema20': safe_float(ema20_current),
        'ema50': safe_float(ema50_current)
    }


def detect_momentum_breakout(candles: List[List[str]], indicators: Dict) -> Dict[str, Any]:
    """Шаблон A: Momentum breakout (импульсный вход)"""
    if len(candles) < 5:
        return {'signal': False, 'confidence': 0}

    closes = np.array([float(c[4]) for c in candles])
    volumes = np.array([float(c[5]) for c in candles])

    current_close = closes[-1]
    ema20 = indicators.get('ema20', [])
    macd_hist = indicators.get('macd_histogram', [])
    volume_sma = indicators.get('volume_sma', [])

    if not ema20 or not macd_hist or not volume_sma:
        return {'signal': False, 'confidence': 0}

    # Условия согласно инструкции
    price_above_ema20 = current_close > ema20[-1]
    macd_positive = macd_hist[-1] > 0
    volume_confirmed = len(volume_sma) > 0 and volumes[-1] > volume_sma[-1]
    atr_valid = indicators.get('atr_current', 0) >= indicators.get('atr_mean', 0) * 0.9

    if price_above_ema20 and macd_positive and volume_confirmed and atr_valid:
        confidence = 85
        return {
            'signal': True,
            'direction': 'LONG',
            'confidence': confidence,
            'type': 'MOMENTUM_BREAKOUT',
            'reasons': ['price_above_ema20', 'macd_positive', 'volume_confirmed']
        }

    return {'signal': False, 'confidence': 0}


def detect_pullback_entry(candles: List[List[str]], indicators: Dict, higher_tf_trend: str) -> Dict[str, Any]:
    """Шаблон B: Pullback to EMA (вход по откату в тренде)"""
    if len(candles) < 10:
        return {'signal': False, 'confidence': 0}

    closes = np.array([float(c[4]) for c in candles])
    current_close = closes[-1]

    ema8 = indicators.get('ema8', [])
    ema20 = indicators.get('ema20', [])
    rsi = indicators.get('rsi', [])
    macd = indicators.get('macd_line', [])

    if not ema8 or not ema20 or not rsi:
        return {'signal': False, 'confidence': 0}

    # Только в направлении старшего тренда
    if higher_tf_trend == 'UPTREND':
        # Откат к EMA в восходящем тренде
        near_ema8 = abs(current_close - ema8[-1]) / current_close < 0.005  # в пределах 0.5%
        rsi_recovered = len(rsi) > 0 and rsi[-1] > 45
        macd_not_against = not macd or macd[-1] > macd[-2]

        if near_ema8 and rsi_recovered:
            return {
                'signal': True,
                'direction': 'LONG',
                'confidence': 75,
                'type': 'PULLBACK_ENTRY',
                'reasons': ['uptrend_pullback', 'near_ema', 'rsi_recovered']
            }

    elif higher_tf_trend == 'DOWNTREND':
        # Откат к EMA в нисходящем тренде
        near_ema8 = abs(current_close - ema8[-1]) / current_close < 0.005
        rsi_weak = len(rsi) > 0 and rsi[-1] < 55

        if near_ema8 and rsi_weak:
            return {
                'signal': True,
                'direction': 'SHORT',
                'confidence': 75,
                'type': 'PULLBACK_ENTRY',
                'reasons': ['downtrend_pullback', 'near_ema', 'rsi_weak']
            }

    return {'signal': False, 'confidence': 0}


def detect_bollinger_squeeze(candles: List[List[str]], indicators: Dict) -> Dict[str, Any]:
    """Шаблон C: Bollinger Squeeze breakout"""
    if len(candles) < 25:
        return {'signal': False, 'confidence': 0}

    closes = np.array([float(c[4]) for c in candles])
    volumes = np.array([float(c[5]) for c in candles])

    bb_upper = indicators.get('bb_upper', [])
    bb_lower = indicators.get('bb_lower', [])
    volume_sma = indicators.get('volume_sma', [])
    atr_values = indicators.get('atr', [])

    if not bb_upper or not bb_lower or not volume_sma:
        return {'signal': False, 'confidence': 0}

    current_close = closes[-1]

    # Проверяем сжатие полос (последние 5 свечей)
    if len(bb_upper) >= 5:
        recent_width = np.mean([(bb_upper[i] - bb_lower[i]) for i in range(-5, 0)])
        historical_width = np.mean([(bb_upper[i] - bb_lower[i]) for i in range(-20, -5)])

        squeeze_detected = recent_width < historical_width * 0.8

        # Пробой верхней/нижней полосы с объемом
        breakout_up = current_close > bb_upper[-1]
        breakout_down = current_close < bb_lower[-1]
        volume_confirmed = volumes[-1] > volume_sma[-1]

        # ATR начинает расти
        atr_rising = len(atr_values) >= 3 and atr_values[-1] > atr_values[-3]

        if squeeze_detected and volume_confirmed and atr_rising:
            if breakout_up:
                return {
                    'signal': True,
                    'direction': 'LONG',
                    'confidence': 80,
                    'type': 'SQUEEZE_BREAKOUT',
                    'reasons': ['bollinger_squeeze', 'upside_breakout', 'volume_spike']
                }
            elif breakout_down:
                return {
                    'signal': True,
                    'direction': 'SHORT',
                    'confidence': 80,
                    'type': 'SQUEEZE_BREAKOUT',
                    'reasons': ['bollinger_squeeze', 'downside_breakout', 'volume_spike']
                }

    return {'signal': False, 'confidence': 0}


def detect_range_scalp(candles: List[List[str]], indicators: Dict) -> Dict[str, Any]:
    """Шаблон D: Range scalp (внутри диапазона)"""
    if len(candles) < 30:
        return {'signal': False, 'confidence': 0}

    closes = np.array([float(c[4]) for c in candles])
    highs = np.array([float(c[2]) for c in candles])
    lows = np.array([float(c[3]) for c in candles])

    # Определяем диапазон (последние 20 свечей)
    recent_high = np.max(highs[-20:])
    recent_low = np.min(lows[-20:])
    range_size = recent_high - recent_low
    current_close = closes[-1]

    # Проверяем, что мы в боковом движении
    range_ratio = range_size / current_close
    if range_ratio < 0.02:  # менее 2% диапазон
        return {'signal': False, 'confidence': 0}

    rsi = indicators.get('rsi', [])
    if not rsi:
        return {'signal': False, 'confidence': 0}

    # Близость к границам диапазона
    near_support = abs(current_close - recent_low) / range_size < 0.1
    near_resistance = abs(current_close - recent_high) / range_size < 0.1

    current_rsi = rsi[-1]

    # Отскок от поддержки
    if near_support and current_rsi < 30:
        return {
            'signal': True,
            'direction': 'LONG',
            'confidence': 70,
            'type': 'RANGE_SCALP',
            'reasons': ['support_bounce', 'rsi_oversold']
        }

    # Отскок от сопротивления
    elif near_resistance and current_rsi > 70:
        return {
            'signal': True,
            'direction': 'SHORT',
            'confidence': 70,
            'type': 'RANGE_SCALP',
            'reasons': ['resistance_rejection', 'rsi_overbought']
        }

    return {'signal': False, 'confidence': 0}


def calculate_indicators_by_instruction(candles: List[List[str]]) -> Dict[str, Any]:
    """
    Расчет индикаторов согласно инструкции:
    EMA(5/8/20), RSI(9), MACD(12,26,9), ATR(14), Bollinger(20,2), Volume SMA(20)
    """
    if len(candles) < 30:
        return {}

    closes = np.array([float(c[4]) for c in candles])

    # EMA согласно инструкции (5/8/20)
    ema5 = calculate_ema(closes, SCALPING_PARAMS['ema_fast'])
    ema8 = calculate_ema(closes, SCALPING_PARAMS['ema_medium'])
    ema20 = calculate_ema(closes, SCALPING_PARAMS['ema_slow'])

    # RSI(9) для фильтра импульса
    rsi = calculate_rsi(closes, SCALPING_PARAMS['rsi_period'])

    # MACD стандарт (12,26,9)
    macd = calculate_macd(closes,
                          SCALPING_PARAMS['macd_fast'],
                          SCALPING_PARAMS['macd_slow'],
                          SCALPING_PARAMS['macd_signal'])

    # ATR(14) для волатильности
    atr_data = calculate_atr(candles, SCALPING_PARAMS['atr_period'])

    # Bollinger Bands (20,2)
    bb = calculate_bollinger_bands(closes,
                                   SCALPING_PARAMS['bb_period'],
                                   SCALPING_PARAMS['bb_std'])

    # Volume SMA(20)
    volume_sma = calculate_volume_sma(candles, SCALPING_PARAMS['volume_sma'])

    return {
        # EMA тренд/сжатие
        'ema5': safe_list(ema5),
        'ema8': safe_list(ema8),
        'ema20': safe_list(ema20),

        # RSI фильтр импульса
        'rsi': safe_list(rsi),
        'rsi_current': safe_float(rsi[-1] if len(rsi) > 0 else 50.0),

        # MACD подтверждение направления
        'macd_line': safe_list(macd['macd']),
        'macd_signal': safe_list(macd['signal']),
        'macd_histogram': safe_list(macd['histogram']),

        # ATR волатильность
        'atr': safe_list(atr_data['atr']),
        'atr_current': atr_data['current'],
        'atr_mean': safe_float(np.mean(atr_data['atr'][-10:]) if len(atr_data['atr']) >= 10 else 0),

        # Bollinger Bands squeeze/breakout
        'bb_upper': safe_list(bb['upper']),
        'bb_middle': safe_list(bb['middle']),
        'bb_lower': safe_list(bb['lower']),

        # Volume подтверждение пробоев
        'volume_sma': safe_list(volume_sma),
        'volume_current': safe_float(float(candles[-1][5]) if candles else 0),
        'volume_ratio': safe_float(
            float(candles[-1][5]) / volume_sma[-1] if len(volume_sma) > 0 and volume_sma[-1] > 0 else 1.0)
    }


def detect_instruction_based_signals(candles_5m: List[List[str]], candles_15m: List[List[str]] = None) -> Dict[
    str, Any]:
    """
    ГЛАВНАЯ ФУНКЦИЯ: Определение сигналов согласно инструкции
    Использует мультитаймфреймный анализ: 15m для контекста, 5m для входа
    """
    if len(candles_5m) < 50:
        return {'signal': 'NO_SIGNAL', 'confidence': 0, 'reason': 'INSUFFICIENT_DATA'}

    # Рассчитываем индикаторы для 5m
    indicators = calculate_indicators_by_instruction(candles_5m)

    if not indicators:
        return {'signal': 'NO_SIGNAL', 'confidence': 0, 'reason': 'CALCULATION_ERROR'}

    # Анализ старшего таймфрейма (15m) для контекста
    higher_tf_trend = 'UNKNOWN'
    if candles_15m and len(candles_15m) >= 30:
        trend_analysis = analyze_higher_timeframe_trend(candles_15m)
        higher_tf_trend = trend_analysis['trend']

    # Применяем шаблоны входа согласно инструкции
    signal_results = []

    # A) Momentum breakout
    momentum_signal = detect_momentum_breakout(candles_5m, indicators)
    if momentum_signal['signal']:
        signal_results.append(momentum_signal)

    # B) Pullback to EMA (только в направлении старшего тренда)
    pullback_signal = detect_pullback_entry(candles_5m, indicators, higher_tf_trend)
    if pullback_signal['signal']:
        signal_results.append(pullback_signal)

    # C) Bollinger Squeeze breakout
    squeeze_signal = detect_bollinger_squeeze(candles_5m, indicators)
    if squeeze_signal['signal']:
        signal_results.append(squeeze_signal)

    # D) Range scalp
    range_signal = detect_range_scalp(candles_5m, indicators)
    if range_signal['signal']:
        signal_results.append(range_signal)

    # Выбираем лучший сигнал
    if not signal_results:
        return {'signal': 'NO_SIGNAL', 'confidence': 0, 'reason': 'NO_PATTERN_FOUND'}

    # Сортируем по уверенности
    signal_results.sort(key=lambda x: x['confidence'], reverse=True)
    best_signal = signal_results[0]

    # Финальная валидация согласно инструкции (3 из 4 чек-пунктов)
    validation_score = 0
    validation_reasons = []

    # 1. Тренд старшего TF совпадает
    if higher_tf_trend in ['UPTREND', 'DOWNTREND']:
        if (higher_tf_trend == 'UPTREND' and best_signal['direction'] == 'LONG') or \
                (higher_tf_trend == 'DOWNTREND' and best_signal['direction'] == 'SHORT'):
            validation_score += 1
            validation_reasons.append('higher_tf_aligned')
    else:
        validation_score += 0.5  # нейтральный тренд

    # 2. Цена + свечная формация (закрытие свечи в нужном направлении)
    closes = np.array([float(c[4]) for c in candles_5m])
    if len(closes) >= 2:
        if best_signal['direction'] == 'LONG' and closes[-1] > closes[-2]:
            validation_score += 1
            validation_reasons.append('price_direction_confirmed')
        elif best_signal['direction'] == 'SHORT' and closes[-1] < closes[-2]:
            validation_score += 1
            validation_reasons.append('price_direction_confirmed')

    # 3. Объём >= средний
    if indicators.get('volume_ratio', 1.0) >= 1.0:
        validation_score += 1
        validation_reasons.append('volume_confirmed')

    # 4. ATR достаточна
    if indicators.get('atr_current', 0) >= indicators.get('atr_mean', 0) * 0.7:
        validation_score += 1
        validation_reasons.append('atr_sufficient')

    # Требуется минимум 3 из 4
    if validation_score < 3:
        return {'signal': 'NO_SIGNAL', 'confidence': 0, 'reason': 'VALIDATION_FAILED'}

    # Финальная уверенность с учетом валидации
    final_confidence = min(100, int(best_signal['confidence'] * (validation_score / 4)))

    if final_confidence < SCALPING_PARAMS['min_confidence']:
        return {'signal': 'NO_SIGNAL', 'confidence': final_confidence, 'reason': 'LOW_CONFIDENCE'}

    return {
        'signal': best_signal['direction'],
        'confidence': final_confidence,
        'pattern_type': best_signal['type'],
        'entry_reasons': best_signal['reasons'],
        'validation_reasons': validation_reasons,
        'validation_score': f"{validation_score}/4",
        'higher_tf_trend': higher_tf_trend,
        'indicators': indicators,
        'atr_current': indicators.get('atr_current', 0),
        'volume_ratio': indicators.get('volume_ratio', 1.0)
    }