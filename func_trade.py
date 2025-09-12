"""
Модуль технического анализа и определения торговых сигналов
Реализует мультитаймфреймный анализ согласно инструкции
ОБНОВЛЕН: поддержка больших объемов данных для финального анализа
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import time
import math
from config import config

# Параметры согласно инструкции из конфигурации
SCALPING_PARAMS = {
    'ema_fast': config.indicators.EMA_FAST,
    'ema_medium': config.indicators.EMA_MEDIUM,
    'ema_slow': config.indicators.EMA_SLOW,
    'rsi_period': config.indicators.RSI_PERIOD,
    'macd_fast': config.indicators.MACD_FAST,
    'macd_slow': config.indicators.MACD_SLOW,
    'macd_signal': config.indicators.MACD_SIGNAL,
    'atr_period': config.indicators.ATR_PERIOD,
    'volume_sma': config.indicators.VOLUME_SMA,
    'bb_period': config.indicators.BB_PERIOD,
    'bb_std': config.indicators.BB_STD,
    'min_confidence': config.trading.MIN_CONFIDENCE
}


def safe_float(value):
    """
    Безопасное преобразование в float

    Args:
        value: Значение для преобразования

    Returns:
        float: Преобразованное значение или 0.0
    """
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return 0.0
        return result
    except (ValueError, TypeError):
        return 0.0


def safe_int(value):
    """
    Безопасное преобразование в int

    Args:
        value: Значение для преобразования

    Returns:
        int: Преобразованное значение или 0
    """
    try:
        result = int(value)
        if math.isnan(result) or math.isinf(result):
            return 0
        return result
    except (ValueError, TypeError):
        return 0


def safe_list(arr):
    """
    Безопасное преобразование массива в список

    Args:
        arr: Массив для преобразования

    Returns:
        list: Преобразованный список
    """
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
    """
    Расчет экспоненциальной скользящей средней (EMA)
    ОБНОВЛЕНО: оптимизировано для больших объемов данных

    Args:
        prices: Массив цен
        period: Период EMA

    Returns:
        np.ndarray: Массив значений EMA
    """
    if len(prices) < period:
        return np.full_like(prices, prices[0] if len(prices) > 0 else 0)

    ema = np.zeros_like(prices)
    alpha = 2.0 / (period + 1)

    # Первое значение = первая цена
    ema[0] = prices[0]

    # Векторизованный расчет для скорости
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema


def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Расчет простой скользящей средней (SMA)
    НОВОЕ: для больших объемов данных

    Args:
        prices: Массив цен
        period: Период SMA

    Returns:
        np.ndarray: Массив значений SMA
    """
    if len(prices) < period:
        return np.full_like(prices, np.mean(prices))

    sma = np.zeros_like(prices)

    # Первый период - обычное среднее
    sma[:period-1] = prices[0]

    # Скользящее среднее
    for i in range(period - 1, len(prices)):
        sma[i] = np.mean(prices[i - period + 1:i + 1])

    return sma


def calculate_rsi(prices: np.ndarray, period: int = 9) -> np.ndarray:
    """
    Расчет RSI(9) для фильтра импульса согласно инструкции
    ОБНОВЛЕНО: оптимизировано для больших массивов данных

    Args:
        prices: Массив цен
        period: Период RSI (по умолчанию 9)

    Returns:
        np.ndarray: Массив значений RSI
    """
    if len(prices) < period + 1:
        return np.full_like(prices, 50.0)

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gains = np.zeros_like(prices)
    avg_losses = np.zeros_like(prices)

    # Первое значение через SMA
    if len(gains) >= period:
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])

    # EMA для последующих значений (векторизованно)
    alpha = 1.0 / period
    for i in range(period + 1, len(prices)):
        avg_gains[i] = alpha * gains[i - 1] + (1 - alpha) * avg_gains[i - 1]
        avg_losses[i] = alpha * losses[i - 1] + (1 - alpha) * avg_losses[i - 1]

    # Избегаем деление на ноль
    rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses != 0)
    rsi = 100 - (100 / (1 + rs))

    # Заполняем начальные значения
    rsi[:period] = 50.0

    return rsi


def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, np.ndarray]:
    """
    Расчет MACD стандарт (12,26,9) согласно инструкции
    ОБНОВЛЕНО: поддержка больших массивов данных

    Args:
        prices: Массив цен
        fast: Быстрый период
        slow: Медленный период
        signal: Период сигнальной линии

    Returns:
        dict: Словарь с MACD линией, сигнальной линией и гистограммой
    """
    if len(prices) < max(fast, slow):
        zero_array = np.zeros_like(prices)
        return {
            'macd': zero_array,
            'signal': zero_array,
            'histogram': zero_array
        }

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
    """
    Расчет ATR(14) для контроля волатильности согласно инструкции
    ОБНОВЛЕНО: оптимизировано для больших объемов данных

    Args:
        candles: Список свечей
        period: Период ATR (по умолчанию 14)

    Returns:
        dict: Словарь с массивом ATR и текущим значением
    """
    if len(candles) < period + 1:
        return {'atr': [0.0] * len(candles), 'current': 0.0}

    highs = np.array([safe_float(c[2]) for c in candles])
    lows = np.array([safe_float(c[3]) for c in candles])
    closes = np.array([safe_float(c[4]) for c in candles])

    # True Range расчет
    tr = np.zeros(len(candles))
    for i in range(1, len(candles)):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )

    # ATR расчет
    atr = np.zeros(len(candles))
    if len(tr) > period:
        # Первое значение ATR = SMA от TR
        atr[period] = np.mean(tr[1:period + 1])

        # Последующие значения = сглаженное среднее
        for i in range(period + 1, len(candles)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return {
        'atr': safe_list(atr),
        'current': safe_float(atr[-1] if len(atr) > 0 else 0.0)
    }


def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, std: float = 2.0) -> Dict[str, np.ndarray]:
    """
    Расчет Bollinger Bands (20,2) согласно инструкции
    ОБНОВЛЕНО: оптимизировано для больших объемов данных

    Args:
        prices: Массив цен
        period: Период для расчета (по умолчанию 20)
        std: Количество стандартных отклонений (по умолчанию 2)

    Returns:
        dict: Словарь с верхней, средней и нижней полосами
    """
    if len(prices) < period:
        middle = np.full_like(prices, np.mean(prices))
        return {
            'upper': middle,
            'middle': middle,
            'lower': middle
        }

    sma = np.zeros_like(prices)
    upper = np.zeros_like(prices)
    lower = np.zeros_like(prices)

    # Заполняем начальные значения
    sma[:period-1] = prices[0]
    upper[:period-1] = prices[0]
    lower[:period-1] = prices[0]

    # Рассчитываем Bollinger Bands
    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1:i + 1]
        sma[i] = np.mean(window)
        std_dev = np.std(window, ddof=0)
        upper[i] = sma[i] + (std_dev * std)
        lower[i] = sma[i] - (std_dev * std)

    return {
        'upper': upper,
        'middle': sma,
        'lower': lower
    }


def calculate_volume_sma(candles: List[List[str]], period: int = 20) -> np.ndarray:
    """
    Расчет Volume SMA(20) для подтверждения пробоев
    ОБНОВЛЕНО: оптимизировано для больших объемов данных

    Args:
        candles: Список свечей
        period: Период SMA (по умолчанию 20)

    Returns:
        np.ndarray: Массив значений Volume SMA
    """
    if len(candles) < period:
        volumes = np.array([safe_float(c[5]) for c in candles])
        return np.full_like(volumes, np.mean(volumes))

    volumes = np.array([safe_float(c[5]) for c in candles])
    volume_sma = np.zeros_like(volumes)

    # Заполняем начальные значения
    volume_sma[:period-1] = volumes[0] if len(volumes) > 0 else 0

    # Рассчитываем скользящее среднее
    for i in range(period - 1, len(volumes)):
        volume_sma[i] = np.mean(volumes[i - period + 1:i + 1])

    return volume_sma


def calculate_indicators_by_instruction(candles: List[List[str]],
                                       full_history: bool = False) -> Dict[str, Any]:
    """
    Расчет всех индикаторов согласно инструкции
    ОБНОВЛЕНО: поддержка полной истории для финального анализа

    Args:
        candles: Список свечей
        full_history: Если True - возвращает полную историю, иначе - краткую

    Returns:
        dict: Словарь со всеми рассчитанными индикаторами
    """
    if len(candles) < 50:
        return {}

    closes = np.array([safe_float(c[4]) for c in candles])
    highs = np.array([safe_float(c[2]) for c in candles])
    lows = np.array([safe_float(c[3]) for c in candles])
    volumes = np.array([safe_float(c[5]) for c in candles])

    # EMA система (5, 8, 20)
    ema5 = calculate_ema(closes, SCALPING_PARAMS['ema_fast'])
    ema8 = calculate_ema(closes, SCALPING_PARAMS['ema_medium'])
    ema20 = calculate_ema(closes, SCALPING_PARAMS['ema_slow'])

    # RSI(9) фильтр импульса
    rsi = calculate_rsi(closes, SCALPING_PARAMS['rsi_period'])

    # MACD(12,26,9) подтверждение направления
    macd_data = calculate_macd(closes, SCALPING_PARAMS['macd_fast'],
                               SCALPING_PARAMS['macd_slow'], SCALPING_PARAMS['macd_signal'])

    # ATR(14) контроль волатильности
    atr_data = calculate_atr(candles, SCALPING_PARAMS['atr_period'])

    # Bollinger Bands (20,2)
    bb_data = calculate_bollinger_bands(closes, SCALPING_PARAMS['bb_period'],
                                       SCALPING_PARAMS['bb_std'])

    # Volume SMA(20)
    volume_sma = calculate_volume_sma(candles, SCALPING_PARAMS['volume_sma'])

    # Дополнительные индикаторы для полного анализа
    if full_history:
        # EMA для старшего таймфрейма
        ema50 = calculate_ema(closes, 50)
        ema100 = calculate_ema(closes, 100)

        # RSI дивергенции
        rsi_sma = calculate_sma(rsi, 5)

        # MACD дополнительные сигналы
        macd_sma = calculate_sma(macd_data['macd'], 5)

        # Дополнительные BB параметры
        bb_width = bb_data['upper'] - bb_data['lower']
        bb_position = (closes - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])

        additional_indicators = {
            'ema50': safe_list(ema50),
            'ema100': safe_list(ema100),
            'rsi_sma': safe_list(rsi_sma),
            'macd_sma': safe_list(macd_sma),
            'bb_width': safe_list(bb_width),
            'bb_position': safe_list(bb_position),
            'price_changes': safe_list(np.diff(closes)),
            'volume_changes': safe_list(np.diff(volumes)),
        }
    else:
        additional_indicators = {}

    base_indicators = {
        'ema5': safe_list(ema5),
        'ema8': safe_list(ema8),
        'ema20': safe_list(ema20),
        'rsi': safe_list(rsi),
        'rsi_current': safe_float(rsi[-1] if len(rsi) > 0 else 50),
        'macd_line': safe_list(macd_data['macd']),
        'macd_signal': safe_list(macd_data['signal']),
        'macd_histogram': safe_list(macd_data['histogram']),
        'atr': atr_data['atr'],
        'atr_current': atr_data['current'],
        'atr_mean': safe_float(np.mean(atr_data['atr'][-20:]) if len(atr_data['atr']) >= 20 else atr_data['current']),
        'bb_upper': safe_list(bb_data['upper']),
        'bb_middle': safe_list(bb_data['middle']),
        'bb_lower': safe_list(bb_data['lower']),
        'volume_sma': safe_list(volume_sma),
        'volume_current': safe_float(volumes[-1]),
        'volume_ratio': safe_float(volumes[-1] / volume_sma[-1] if volume_sma[-1] > 0 else 1.0)
    }

    # Объединяем базовые и дополнительные индикаторы
    base_indicators.update(additional_indicators)
    return base_indicators


def analyze_higher_timeframe_trend(candles_15m: List[List[str]],
                                 detailed: bool = False) -> Dict[str, Any]:
    """
    Анализ старшего таймфрейма (15m) для определения контекста
    ОБНОВЛЕНО: поддержка детального анализа с большими данными

    Args:
        candles_15m: Список 15-минутных свечей
        detailed: Если True - детальный анализ с большими данными

    Returns:
        dict: Информация о тренде старшего таймфрейма
    """
    if len(candles_15m) < 30:
        return {'trend': 'UNKNOWN', 'strength': 0}

    closes = np.array([safe_float(c[4]) for c in candles_15m])
    ema20 = calculate_ema(closes, 20)
    ema50 = calculate_ema(closes, 50)

    current_price = closes[-1]
    ema20_current = ema20[-1]
    ema50_current = ema50[-1]

    # Определение тренда на основе положения цены относительно EMA
    if current_price > ema20_current > ema50_current:
        trend = 'UPTREND'
        strength = safe_int(((current_price - ema50_current) / ema50_current) * 1000)
    elif current_price < ema20_current < ema50_current:
        trend = 'DOWNTREND'
        strength = safe_int(((ema50_current - current_price) / ema50_current) * 1000)
    else:
        trend = 'SIDEWAYS'
        strength = 0

    base_analysis = {
        'trend': trend,
        'strength': min(100, abs(strength)),
        'ema20': safe_float(ema20_current),
        'ema50': safe_float(ema50_current)
    }

    # Детальный анализ для финального анализа
    if detailed and len(candles_15m) >= 100:
        ema100 = calculate_ema(closes, 100)
        ema200 = calculate_ema(closes, 200) if len(closes) >= 200 else ema100

        # Анализ структуры тренда
        recent_highs = np.array([safe_float(c[2]) for c in candles_15m[-20:]])
        recent_lows = np.array([safe_float(c[3]) for c in candles_15m[-20:]])

        # Поддержки и сопротивления
        resistance_levels = []
        support_levels = []

        # Ищем локальные экстремумы
        for i in range(5, len(recent_highs) - 5):
            if all(recent_highs[i] >= recent_highs[i-j] for j in range(1, 6)) and \
               all(recent_highs[i] >= recent_highs[i+j] for j in range(1, 6)):
                resistance_levels.append(safe_float(recent_highs[i]))

            if all(recent_lows[i] <= recent_lows[i-j] for j in range(1, 6)) and \
               all(recent_lows[i] <= recent_lows[i+j] for j in range(1, 6)):
                support_levels.append(safe_float(recent_lows[i]))

        detailed_analysis = {
            'ema100': safe_float(ema100[-1]),
            'ema200': safe_float(ema200[-1]),
            'resistance_levels': resistance_levels[-3:] if resistance_levels else [],
            'support_levels': support_levels[-3:] if support_levels else [],
            'trend_consistency': calculate_trend_consistency(closes, ema20, ema50),
            'volatility_analysis': analyze_volatility_pattern(candles_15m[-50:]),
            'volume_profile': analyze_volume_profile(candles_15m[-50:])
        }

        base_analysis.update(detailed_analysis)

    return base_analysis


def calculate_trend_consistency(closes: np.ndarray, ema20: np.ndarray, ema50: np.ndarray) -> Dict[str, float]:
    """
    НОВОЕ: Расчет консистентности тренда для детального анализа
    """
    if len(closes) < 20:
        return {'score': 0.0, 'direction_changes': 0}

    # Считаем изменения направления EMA20 относительно EMA50
    ema_diff = ema20[-20:] - ema50[-20:]
    direction_changes = 0

    for i in range(1, len(ema_diff)):
        if (ema_diff[i] > 0) != (ema_diff[i-1] > 0):
            direction_changes += 1

    # Чем меньше изменений направления, тем выше консистентность
    consistency_score = max(0.0, 1.0 - (direction_changes / 10.0))

    return {
        'score': safe_float(consistency_score),
        'direction_changes': direction_changes
    }


def analyze_volatility_pattern(candles: List[List[str]]) -> Dict[str, float]:
    """
    НОВОЕ: Анализ паттерна волатильности для детального анализа
    """
    if len(candles) < 20:
        return {'average_range': 0.0, 'volatility_trend': 'STABLE'}

    ranges = []
    for candle in candles:
        high = safe_float(candle[2])
        low = safe_float(candle[3])
        ranges.append((high - low) / safe_float(candle[4]) * 100 if safe_float(candle[4]) > 0 else 0)

    avg_range = np.mean(ranges)
    recent_avg = np.mean(ranges[-10:])
    early_avg = np.mean(ranges[:10])

    if recent_avg > early_avg * 1.2:
        volatility_trend = 'INCREASING'
    elif recent_avg < early_avg * 0.8:
        volatility_trend = 'DECREASING'
    else:
        volatility_trend = 'STABLE'

    return {
        'average_range': safe_float(avg_range),
        'volatility_trend': volatility_trend,
        'recent_vs_early_ratio': safe_float(recent_avg / early_avg if early_avg > 0 else 1.0)
    }


def analyze_volume_profile(candles: List[List[str]]) -> Dict[str, Any]:
    """
    НОВОЕ: Анализ профиля объемов для детального анализа
    """
    if len(candles) < 20:
        return {'average_volume': 0.0, 'volume_trend': 'STABLE'}

    volumes = [safe_float(c[5]) for c in candles]
    avg_volume = np.mean(volumes)
    recent_avg = np.mean(volumes[-10:])
    early_avg = np.mean(volumes[:10])

    if recent_avg > early_avg * 1.3:
        volume_trend = 'INCREASING'
    elif recent_avg < early_avg * 0.7:
        volume_trend = 'DECREASING'
    else:
        volume_trend = 'STABLE'

    # Ищем объемные всплески
    volume_spikes = []
    volume_threshold = avg_volume * 2.0

    for i, vol in enumerate(volumes):
        if vol > volume_threshold:
            volume_spikes.append({
                'index': i,
                'volume': vol,
                'ratio': vol / avg_volume
            })

    return {
        'average_volume': safe_float(avg_volume),
        'volume_trend': volume_trend,
        'recent_vs_early_ratio': safe_float(recent_avg / early_avg if early_avg > 0 else 1.0),
        'volume_spikes': volume_spikes[-5:],  # Последние 5 всплесков
        'spike_frequency': len(volume_spikes) / len(volumes)
    }


# Остальные функции остаются без изменений, но добавляем поддержку детального анализа

def detect_momentum_breakout(candles: List[List[str]], indicators: Dict,
                           detailed: bool = False) -> Dict[str, Any]:
    """
    Шаблон A: Momentum breakout (импульсный вход)
    ОБНОВЛЕНО: поддержка детального анализа
    """
    if len(candles) < 5:
        return {'signal': False, 'confidence': 0}

    closes = np.array([safe_float(c[4]) for c in candles])
    current_close = closes[-1]

    ema20 = indicators.get('ema20', [])
    macd_histogram = indicators.get('macd_histogram', [])
    volume_ratio = indicators.get('volume_ratio', 1.0)

    if not ema20 or not macd_histogram:
        return {'signal': False, 'confidence': 0}

    # Базовые условия momentum breakout
    conditions = {
        'price_above_ema20': current_close > ema20[-1],
        'macd_positive': macd_histogram[-1] > 0,
        'volume_spike': volume_ratio > 1.5,
        'ema_alignment': len(indicators.get('ema5', [])) > 0 and
                         len(indicators.get('ema8', [])) > 0 and
                         indicators['ema5'][-1] > indicators['ema8'][-1] > ema20[-1]
    }

    # Дополнительные условия для детального анализа
    if detailed:
        # Проверяем динамику MACD
        if len(macd_histogram) >= 3:
            conditions['macd_accelerating'] = (macd_histogram[-1] > macd_histogram[-2] > macd_histogram[-3])

        # Проверяем пробой с силой
        if len(ema20) >= 10:
            ema20_slope = (ema20[-1] - ema20[-10]) / ema20[-10] * 100
            conditions['ema20_rising'] = ema20_slope > 0.1

        # Проверяем объемное подтверждение
        volume_sma = indicators.get('volume_sma', [])
        if volume_sma:
            conditions['volume_sustained'] = volume_ratio > 1.2

    signal = all(conditions.values())
    confidence = sum(conditions.values()) * (100 // len(conditions))

    result = {
        'signal': signal,
        'confidence': confidence,
        'pattern': 'MOMENTUM_BREAKOUT',
        'conditions': conditions
    }

    if detailed:
        result['breakout_strength'] = analyze_breakout_strength(candles, indicators)
        result['momentum_quality'] = analyze_momentum_quality(indicators)

    return result


def analyze_breakout_strength(candles: List[List[str]], indicators: Dict) -> Dict[str, float]:
    """
    НОВОЕ: Анализ силы пробоя для детального анализа
    """
    if len(candles) < 10:
        return {'strength': 0.0, 'sustainability': 0.0}

    closes = np.array([safe_float(c[4]) for c in candles])
    volumes = np.array([safe_float(c[5]) for c in candles])

    # Сила пробоя основана на изменении цены и объеме
    price_change = (closes[-1] - closes[-5]) / closes[-5] * 100
    volume_increase = np.mean(volumes[-3:]) / np.mean(volumes[-10:-3])

    strength = min(100.0, abs(price_change) * volume_increase * 10)

    # Устойчивость основана на консистентности движения
    recent_closes = closes[-5:]
    if len(recent_closes) >= 2:
        sustainability = len([i for i in range(1, len(recent_closes))
                            if recent_closes[i] > recent_closes[i-1]]) / (len(recent_closes) - 1) * 100
    else:
        sustainability = 0.0

    return {
        'strength': safe_float(strength),
        'sustainability': safe_float(sustainability),
        'price_change_percent': safe_float(price_change),
        'volume_increase_ratio': safe_float(volume_increase)
    }


def analyze_momentum_quality(indicators: Dict) -> Dict[str, float]:
    """
    НОВОЕ: Анализ качества моментума для детального анализа
    """
    quality_score = 0.0
    components = {}

    # EMA выравнивание
    ema5 = indicators.get('ema5', [])
    ema8 = indicators.get('ema8', [])
    ema20 = indicators.get('ema20', [])

    if ema5 and ema8 and ema20:
        if ema5[-1] > ema8[-1] > ema20[-1]:
            components['ema_alignment'] = 25.0
            quality_score += 25.0

    # MACD сигнал
    macd_line = indicators.get('macd_line', [])
    macd_signal = indicators.get('macd_signal', [])

    if macd_line and macd_signal and len(macd_line) >= 2:
        if macd_line[-1] > macd_signal[-1] and macd_line[-1] > macd_line[-2]:
            components['macd_strength'] = 25.0
            quality_score += 25.0

    # RSI уровень
    rsi_current = indicators.get('rsi_current', 50)
    if 40 <= rsi_current <= 70:  # Здоровый диапазон для лонга
        components['rsi_healthy'] = 25.0
        quality_score += 25.0

    # Объем
    volume_ratio = indicators.get('volume_ratio', 1.0)
    if volume_ratio > 1.2:
        components['volume_confirmation'] = 25.0
        quality_score += 25.0

    return {
        'overall_quality': safe_float(quality_score),
        'components': components
    }


# Аналогично обновляем остальные функции паттернов...
def detect_pullback_entry(candles: List[List[str]], indicators: Dict,
                         detailed: bool = False) -> Dict[str, Any]:
    """
    Шаблон B: Pullback to EMA (откат к EMA)
    ОБНОВЛЕНО: поддержка детального анализа
    """
    if len(candles) < 10:
        return {'signal': False, 'confidence': 0}

    closes = np.array([safe_float(c[4]) for c in candles])
    current_close = closes[-1]

    ema8 = indicators.get('ema8', [])
    ema20 = indicators.get('ema20', [])
    rsi_current = indicators.get('rsi_current', 50)

    if not ema8 or not ema20:
        return {'signal': False, 'confidence': 0}

    # Проверяем близость к EMA
    ema8_proximity = abs(current_close - ema8[-1]) / current_close < config.patterns.PULLBACK_EMA_PROXIMITY
    ema20_proximity = abs(current_close - ema20[-1]) / current_close < config.patterns.PULLBACK_EMA_PROXIMITY

    # Базовые условия pullback entry
    conditions = {
        'near_ema': ema8_proximity or ema20_proximity,
        'rsi_recovery_long': 30 < rsi_current < config.patterns.PULLBACK_RSI_RECOVERY,
        'rsi_recovery_short': config.patterns.PULLBACK_RSI_WEAK < rsi_current < 70,
        'trend_alignment': len(indicators.get('ema5', [])) > 0 and
                           indicators['ema5'][-1] > ema8[-1]
    }

    # Дополнительные условия для детального анализа
    if detailed:
        # Анализ качества пулбека
        if len(closes) >= 20:
            conditions['pullback_depth'] = analyze_pullback_depth(closes, ema8, ema20)
            conditions['bounce_confirmation'] = check_bounce_signals(candles[-5:], indicators)

    # Определяем направление
    if conditions['near_ema'] and conditions['rsi_recovery_long'] and conditions['trend_alignment']:
        signal_type = 'LONG'
        confidence = 75
        signal = True
    elif conditions['near_ema'] and conditions['rsi_recovery_short']:
        signal_type = 'SHORT'
        confidence = 70
        signal = True
    else:
        signal_type = False
        confidence = 0
        signal = False

    result = {
        'signal': signal,
        'signal_type': signal_type,
        'confidence': confidence,
        'pattern': 'PULLBACK_ENTRY',
        'conditions': conditions
    }

    if detailed and signal:
        result['pullback_analysis'] = analyze_pullback_quality(candles, indicators)

    return result


def analyze_pullback_depth(closes: np.ndarray, ema8: List, ema20: List) -> bool:
    """
    НОВОЕ: Анализ глубины пулбека
    """
    if len(closes) < 20 or not ema8 or not ema20:
        return False

    recent_high = np.max(closes[-20:])
    current_price = closes[-1]
    pullback_depth = (recent_high - current_price) / recent_high * 100

    # Здоровый пулбек: 2-8% от недавнего хая
    return 2.0 <= pullback_depth <= 8.0


def check_bounce_signals(recent_candles: List[List[str]], indicators: Dict) -> bool:
    """
    НОВОЕ: Проверка сигналов отскока
    """
    if len(recent_candles) < 3:
        return False

    # Проверяем последние 3 свечи на формирование дна
    lows = [safe_float(c[3]) for c in recent_candles]
    closes = [safe_float(c[4]) for c in recent_candles]

    # Ищем признаки отскока: растущие минимумы или бычьи свечи
    rising_lows = lows[-1] > lows[-2] > lows[-3] if len(lows) >= 3 else False
    bullish_candle = closes[-1] > safe_float(recent_candles[-1][1])  # close > open

    return rising_lows or bullish_candle


def analyze_pullback_quality(candles: List[List[str]], indicators: Dict) -> Dict[str, Any]:
    """
    НОВОЕ: Анализ качества пулбека для детального анализа
    """
    if len(candles) < 20:
        return {'quality_score': 0.0}

    closes = np.array([safe_float(c[4]) for c in candles])
    volumes = np.array([safe_float(c[5]) for c in candles])

    # Анализируем структуру пулбека
    recent_high = np.max(closes[-20:])
    recent_low = np.min(closes[-10:])
    current_price = closes[-1]

    # Глубина отката
    pullback_depth = (recent_high - current_price) / recent_high * 100

    # Объем во время отката (должен быть меньше)
    pullback_volume = np.mean(volumes[-5:])
    trend_volume = np.mean(volumes[-20:-5])
    volume_ratio = pullback_volume / trend_volume if trend_volume > 0 else 1.0

    # Скорость восстановления
    recovery_speed = (current_price - recent_low) / (recent_high - recent_low) * 100 if recent_high != recent_low else 0

    # Качество пулбека (чем больше, тем лучше)
    quality_components = {
        'depth_score': max(0, 100 - abs(pullback_depth - 5) * 10),  # Оптимум ~5%
        'volume_score': max(0, 100 - abs(volume_ratio - 0.7) * 100),  # Меньше объема = лучше
        'recovery_score': min(100, recovery_speed * 2),  # Быстрое восстановление = хорошо
    }

    overall_quality = np.mean(list(quality_components.values()))

    return {
        'quality_score': safe_float(overall_quality),
        'pullback_depth_percent': safe_float(pullback_depth),
        'volume_during_pullback_ratio': safe_float(volume_ratio),
        'recovery_speed_percent': safe_float(recovery_speed),
        'components': quality_components
    }


def detect_squeeze_breakout(candles: List[List[str]], indicators: Dict,
                          detailed: bool = False) -> Dict[str, Any]:
    """
    Шаблон C: Squeeze breakout (сжатие Bollinger)
    ОБНОВЛЕНО: поддержка детального анализа
    """
    if len(candles) < 20:
        return {'signal': False, 'confidence': 0}

    closes = np.array([safe_float(c[4]) for c in candles])
    current_close = closes[-1]

    bb_upper = indicators.get('bb_upper', [])
    bb_lower = indicators.get('bb_lower', [])
    volume_ratio = indicators.get('volume_ratio', 1.0)
    atr = indicators.get('atr', [])

    if not bb_upper or not bb_lower or not atr:
        return {'signal': False, 'confidence': 0}

    # Проверяем сжатие (последние 10 периодов)
    if len(bb_upper) >= 10 and len(bb_lower) >= 10:
        current_width = bb_upper[-1] - bb_lower[-1]
        avg_width = np.mean([bb_upper[i] - bb_lower[i] for i in range(-10, 0)])
        squeeze = current_width < avg_width * config.indicators.BB_SQUEEZE_RATIO
    else:
        squeeze = False

    # Проверяем пробой
    breakout_up = current_close > bb_upper[-1]
    breakout_down = current_close < bb_lower[-1]

    # Базовые условия squeeze breakout
    conditions = {
        'squeeze_detected': squeeze,
        'breakout_occurred': breakout_up or breakout_down,
        'volume_confirmation': volume_ratio > config.indicators.VOLUME_SPIKE_RATIO,
        'atr_rising': len(atr) >= 3 and atr[-1] > atr[-3]
    }

    # Дополнительные условия для детального анализа
    if detailed:
        conditions['squeeze_duration'] = analyze_squeeze_duration(bb_upper, bb_lower)
        conditions['breakout_conviction'] = analyze_breakout_conviction(candles[-5:], bb_upper[-1], bb_lower[-1])

    signal = all(conditions.values())
    confidence = sum(conditions.values()) * 20

    result = {
        'signal': signal,
        'confidence': confidence,
        'pattern': 'SQUEEZE_BREAKOUT',
        'conditions': conditions,
        'breakout_direction': 'LONG' if breakout_up else 'SHORT' if breakout_down else 'NONE'
    }

    if detailed and signal:
        result['squeeze_analysis'] = analyze_squeeze_quality(bb_upper, bb_lower, atr, volume_ratio)

    return result


def analyze_squeeze_duration(bb_upper: List, bb_lower: List) -> bool:
    """
    НОВОЕ: Анализ продолжительности сжатия
    """
    if len(bb_upper) < 20 or len(bb_lower) < 20:
        return False

    # Считаем количество периодов сжатия
    squeeze_periods = 0
    avg_width = np.mean([bb_upper[i] - bb_lower[i] for i in range(-20, -10)])

    for i in range(-10, 0):
        current_width = bb_upper[i] - bb_lower[i]
        if current_width < avg_width * config.indicators.BB_SQUEEZE_RATIO:
            squeeze_periods += 1

    # Оптимальная продолжительность: 5-8 периодов
    return 5 <= squeeze_periods <= 8


def analyze_breakout_conviction(recent_candles: List[List[str]], upper_band: float, lower_band: float) -> bool:
    """
    НОВОЕ: Анализ убежденности пробоя
    """
    if len(recent_candles) < 3:
        return False

    # Проверяем, что пробой произошел с силой
    closes = [safe_float(c[4]) for c in recent_candles]
    volumes = [safe_float(c[5]) for c in recent_candles]

    # Пробой считается убежденным, если:
    # 1. Цена закрылась за пределами полос
    # 2. Объем увеличился
    # 3. Движение продолжается

    last_close = closes[-1]
    breakout_occurred = last_close > upper_band or last_close < lower_band

    volume_increased = len(volumes) >= 2 and volumes[-1] > volumes[-2]

    # Проверяем продолжение движения
    if len(closes) >= 2:
        if last_close > upper_band:
            movement_continues = closes[-1] > closes[-2]
        elif last_close < lower_band:
            movement_continues = closes[-1] < closes[-2]
        else:
            movement_continues = False
    else:
        movement_continues = False

    return breakout_occurred and volume_increased and movement_continues


def analyze_squeeze_quality(bb_upper: List, bb_lower: List, atr: List, volume_ratio: float) -> Dict[str, Any]:
    """
    НОВОЕ: Анализ качества сжатия для детального анализа
    """
    if len(bb_upper) < 20 or len(bb_lower) < 20:
        return {'quality_score': 0.0}

    # Анализ сжатия полос
    recent_widths = [bb_upper[i] - bb_lower[i] for i in range(-10, 0)]
    historical_widths = [bb_upper[i] - bb_lower[i] for i in range(-20, -10)]

    avg_recent_width = np.mean(recent_widths)
    avg_historical_width = np.mean(historical_widths)

    compression_ratio = avg_recent_width / avg_historical_width if avg_historical_width > 0 else 1.0

    # Анализ волатильности
    recent_atr = np.mean(atr[-5:]) if len(atr) >= 5 else 0
    historical_atr = np.mean(atr[-15:-5]) if len(atr) >= 15 else recent_atr

    atr_compression = recent_atr / historical_atr if historical_atr > 0 else 1.0

    # Оценка качества
    quality_components = {
        'bb_compression': max(0, (1 - compression_ratio) * 100),
        'atr_compression': max(0, (1 - atr_compression) * 100),
        'volume_buildup': min(100, volume_ratio * 50),
    }

    overall_quality = np.mean(list(quality_components.values()))

    return {
        'quality_score': safe_float(overall_quality),
        'compression_ratio': safe_float(compression_ratio),
        'atr_compression_ratio': safe_float(atr_compression),
        'volume_ratio': safe_float(volume_ratio),
        'components': quality_components
    }


def detect_range_scalp(candles: List[List[str]], indicators: Dict,
                      detailed: bool = False) -> Dict[str, Any]:
    """
    Шаблон D: Range scalp (скальпинг в боковике)
    ОБНОВЛЕНО: поддержка детального анализа
    """
    if len(candles) < 20:
        return {'signal': False, 'confidence': 0}

    closes = np.array([safe_float(c[4]) for c in candles])
    highs = np.array([safe_float(c[2]) for c in candles])
    lows = np.array([safe_float(c[3]) for c in candles])

    current_close = closes[-1]
    rsi_current = indicators.get('rsi_current', 50)

    # Определяем диапазон (последние 20 свечей)
    range_high = np.max(highs[-20:])
    range_low = np.min(lows[-20:])
    range_size = (range_high - range_low) / current_close * 100

    # Проверяем условия диапазона
    near_resistance = abs(current_close - range_high) / current_close < config.patterns.RANGE_BOUNDARY_PROXIMITY
    near_support = abs(current_close - range_low) / current_close < config.patterns.RANGE_BOUNDARY_PROXIMITY

    # Базовые условия range scalp
    conditions = {
        'range_size_adequate': range_size > config.patterns.RANGE_MIN_SIZE_PERCENT,
        'near_boundary': near_resistance or near_support,
        'rsi_oversold': rsi_current < config.indicators.RSI_OVERSOLD,
        'rsi_overbought': rsi_current > config.indicators.RSI_OVERBOUGHT
    }

    # Дополнительные условия для детального анализа
    if detailed:
        conditions['range_established'] = analyze_range_establishment(candles[-50:])
        conditions['boundary_strength'] = analyze_boundary_strength(candles[-30:], range_high, range_low)

    # Определяем направление
    if conditions['range_size_adequate'] and near_support and conditions['rsi_oversold']:
        signal_type = 'LONG'
        confidence = 70
        signal = True
    elif conditions['range_size_adequate'] and near_resistance and conditions['rsi_overbought']:
        signal_type = 'SHORT'
        confidence = 70
        signal = True
    else:
        signal_type = 'NONE'
        confidence = 0
        signal = False

    result = {
        'signal': signal,
        'signal_type': signal_type,
        'confidence': confidence,
        'pattern': 'RANGE_SCALP',
        'conditions': conditions,
        'range_levels': {'high': range_high, 'low': range_low}
    }

    if detailed and signal:
        result['range_analysis'] = analyze_range_quality(candles, range_high, range_low, indicators)

    return result


def analyze_range_establishment(candles: List[List[str]]) -> bool:
    """
    НОВОЕ: Проверка установившегося диапазона
    """
    if len(candles) < 30:
        return False

    closes = np.array([safe_float(c[4]) for c in candles])
    highs = np.array([safe_float(c[2]) for c in candles])
    lows = np.array([safe_float(c[3]) for c in candles])

    # Проверяем стабильность границ диапазона
    recent_high = np.max(highs[-15:])
    recent_low = np.min(lows[-15:])

    historical_high = np.max(highs[-30:-15])
    historical_low = np.min(lows[-30:-15])

    # Диапазон считается установившимся, если границы стабильны
    high_stability = abs(recent_high - historical_high) / historical_high < 0.02  # < 2%
    low_stability = abs(recent_low - historical_low) / historical_low < 0.02  # < 2%

    return high_stability and low_stability


def analyze_boundary_strength(candles: List[List[str]], range_high: float, range_low: float) -> bool:
    """
    НОВОЕ: Анализ силы границ диапазона
    """
    if len(candles) < 20:
        return False

    highs = [safe_float(c[2]) for c in candles]
    lows = [safe_float(c[3]) for c in candles]

    # Считаем количество касаний границ
    upper_touches = sum(1 for high in highs if abs(high - range_high) / range_high < 0.005)  # < 0.5%
    lower_touches = sum(1 for low in lows if abs(low - range_low) / range_low < 0.005)  # < 0.5%

    # Границы считаются сильными при 2+ касаниях каждой
    return upper_touches >= 2 and lower_touches >= 2


def analyze_range_quality(candles: List[List[str]], range_high: float, range_low: float,
                         indicators: Dict) -> Dict[str, Any]:
    """
    НОВОЕ: Анализ качества диапазона для детального анализа
    """
    if len(candles) < 30:
        return {'quality_score': 0.0}

    closes = np.array([safe_float(c[4]) for c in candles])
    volumes = np.array([safe_float(c[5]) for c in candles])

    # Анализ характеристик диапазона
    range_size_percent = (range_high - range_low) / np.mean(closes) * 100

    # Распределение цен в диапазоне
    price_distribution = []
    for close in closes[-20:]:
        position = (close - range_low) / (range_high - range_low)
        price_distribution.append(position)

    distribution_balance = 1 - abs(np.mean(price_distribution) - 0.5)  # Близость к центру

    # Объемы на границах vs в центре
    boundary_volumes = []
    center_volumes = []

    for i, candle in enumerate(candles[-20:]):
        close = safe_float(candle[4])
        volume = safe_float(candle[5])

        if abs(close - range_high) / range_high < 0.01 or abs(close - range_low) / range_low < 0.01:
            boundary_volumes.append(volume)
        elif 0.3 <= (close - range_low) / (range_high - range_low) <= 0.7:
            center_volumes.append(volume)

    if boundary_volumes and center_volumes:
        volume_profile_ratio = np.mean(boundary_volumes) / np.mean(center_volumes)
    else:
        volume_profile_ratio = 1.0

    # Оценка качества диапазона
    quality_components = {
        'size_adequacy': min(100, range_size_percent * 10),  # Больше размер = лучше
        'distribution_balance': distribution_balance * 100,
        'volume_profile': min(100, volume_profile_ratio * 50),  # Больше объема на границах = лучше
    }

    overall_quality = np.mean(list(quality_components.values()))

    return {
        'quality_score': safe_float(overall_quality),
        'range_size_percent': safe_float(range_size_percent),
        'price_distribution_balance': safe_float(distribution_balance),
        'volume_profile_ratio': safe_float(volume_profile_ratio),
        'components': quality_components
    }


def validate_signal(candles_5m: List[List[str]], candles_15m: List[List[str]],
                    signal_data: Dict, indicators: Dict) -> Dict[str, Any]:
    """
    Валидация сигнала согласно инструкции (3 из 5 проверок)
    БЕЗ ИЗМЕНЕНИЙ - функция уже оптимальна
    """
    if not candles_5m or not candles_15m:
        return {'score': '0/5', 'valid': False, 'checks': {}}

    # Анализ старшего таймфрейма
    htf_analysis = analyze_higher_timeframe_trend(candles_15m)

    # Получаем направление сигнала
    signal_direction = signal_data.get('signal_type', 'NONE')

    # 5 проверок валидации
    checks = {}

    # 1. Тренд 15m совпадает с направлением входа
    if signal_direction == 'LONG':
        checks['higher_tf_trend_aligned'] = htf_analysis['trend'] in ['UPTREND', 'SIDEWAYS']
    elif signal_direction == 'SHORT':
        checks['higher_tf_trend_aligned'] = htf_analysis['trend'] in ['DOWNTREND', 'SIDEWAYS']
    else:
        checks['higher_tf_trend_aligned'] = False

    # 2. Свеча закрылась в нужном направлении
    if len(candles_5m) >= 2:
        prev_close = float(candles_5m[-2][4])
        curr_close = float(candles_5m[-1][4])

        if signal_direction == 'LONG':
            checks['candle_closed_correctly'] = curr_close > prev_close
        elif signal_direction == 'SHORT':
            checks['candle_closed_correctly'] = curr_close < prev_close
        else:
            checks['candle_closed_correctly'] = False
    else:
        checks['candle_closed_correctly'] = False

    # 3. Объем >= Volume SMA(20)
    volume_ratio = indicators.get('volume_ratio', 0)
    checks['volume_confirmed'] = volume_ratio >= config.indicators.VOLUME_MIN_RATIO

    # 4. ATR достаточный для размера позиции
    atr_current = indicators.get('atr_current', 0)
    atr_mean = indicators.get('atr_mean', 0)
    checks['atr_sufficient'] = atr_current >= atr_mean * config.indicators.ATR_OPTIMAL_RATIO

    # 5. Свечной паттерн на последней свече совпадает с направлением
    if len(candles_5m) >= 1:
        last_candle = candles_5m[-1]
        candle_open = float(last_candle[1])
        candle_close = float(last_candle[4])

        if signal_direction == 'LONG':
            checks['candle_pattern_aligned'] = candle_close > candle_open
        elif signal_direction == 'SHORT':
            checks['candle_pattern_aligned'] = candle_close < candle_open
        else:
            checks['candle_pattern_aligned'] = False
    else:
        checks['candle_pattern_aligned'] = False

    # Подсчет результата
    passed_checks = sum(checks.values())
    total_checks = len(checks)

    return {
        'score': f'{passed_checks}/{total_checks}',
        'valid': passed_checks >= config.trading.VALIDATION_CHECKS_REQUIRED,
        'checks': checks,
        'passed': passed_checks,
        'total': total_checks
    }


def detect_instruction_based_signals(candles_5m: List[List[str]],
                                     candles_15m: List[List[str]],
                                     detailed_analysis: bool = False) -> Dict[str, Any]:
    """
    Основная функция определения сигналов согласно инструкции
    Мультитаймфреймный анализ: 15m контекст + 5m вход
    ОБНОВЛЕНО: добавлен параметр detailed_analysis для финального анализа

    Args:
        candles_5m: Свечи 5m таймфрейма для точного входа
        candles_15m: Свечи 15m таймфрейма для контекста
        detailed_analysis: Если True - используется для финального анализа с большими данными

    Returns:
        dict: Полная информация о сигнале
    """
    if not candles_5m or not candles_15m:
        return {
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'pattern_type': 'NONE',
            'validation_score': '0/5'
        }

    # Рассчитываем индикаторы для 5m (с полной историей для детального анализа)
    indicators = calculate_indicators_by_instruction(candles_5m, full_history=detailed_analysis)
    if not indicators:
        return {
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'pattern_type': 'NONE',
            'validation_score': '0/5'
        }

    # Анализируем старший таймфрейм (с детализацией для финального анализа)
    htf_analysis = analyze_higher_timeframe_trend(candles_15m, detailed=detailed_analysis)

    # Проверяем все шаблоны входа согласно приоритету
    patterns = [
        ('MOMENTUM_BREAKOUT', detect_momentum_breakout(candles_5m, indicators, detailed=detailed_analysis)),
        ('SQUEEZE_BREAKOUT', detect_squeeze_breakout(candles_5m, indicators, detailed=detailed_analysis)),
        ('PULLBACK_ENTRY', detect_pullback_entry(candles_5m, indicators, detailed=detailed_analysis)),
        ('RANGE_SCALP', detect_range_scalp(candles_5m, indicators, detailed=detailed_analysis))
    ]

    # Ищем лучший сигнал
    best_signal = None
    best_confidence = 0
    best_pattern = 'NONE'

    for pattern_name, pattern_result in patterns:
        if pattern_result['signal'] and pattern_result['confidence'] > best_confidence:
            best_signal = pattern_result
            best_confidence = pattern_result['confidence']
            best_pattern = pattern_name

    # Если сигнал не найден
    if not best_signal:
        return {
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'pattern_type': 'NONE',
            'higher_tf_trend': htf_analysis['trend'],
            'validation_score': '0/5',
            'indicators': indicators,
            'higher_tf_analysis': htf_analysis if detailed_analysis else {}
        }

    # Определяем направление сигнала
    signal_direction = best_signal.get('signal_type',
                                       best_signal.get('breakout_direction', 'LONG'))

    # Валидируем сигнал (3 из 5 проверок)
    validation = validate_signal(candles_5m, candles_15m,
                                 {'signal_type': signal_direction}, indicators)

    # Применяем модификаторы уверенности из конфига
    final_confidence = best_confidence

    # Модификаторы согласно конфигурации
    if validation['checks'].get('higher_tf_trend_aligned', False):
        final_confidence *= config.scoring.CONFIDENCE_MODIFIERS['higher_tf_aligned']

    if indicators.get('volume_ratio', 1.0) > config.indicators.VOLUME_SPIKE_RATIO:
        final_confidence *= config.scoring.CONFIDENCE_MODIFIERS['volume_spike']

    if (len(indicators.get('ema5', [])) > 0 and len(indicators.get('ema8', [])) > 0 and
            len(indicators.get('ema20', [])) > 0):
        ema5_curr = indicators['ema5'][-1]
        ema8_curr = indicators['ema8'][-1]
        ema20_curr = indicators['ema20'][-1]

        if signal_direction == 'LONG' and ema5_curr > ema8_curr > ema20_curr:
            final_confidence *= config.scoring.CONFIDENCE_MODIFIERS['perfect_ema_alignment']
        elif signal_direction == 'SHORT' and ema5_curr < ema8_curr < ema20_curr:
            final_confidence *= config.scoring.CONFIDENCE_MODIFIERS['perfect_ema_alignment']

    if validation['passed'] == validation['total']:
        final_confidence *= config.scoring.CONFIDENCE_MODIFIERS['validation_perfect']

    # Ограничиваем уверенность
    final_confidence = min(100, int(final_confidence))

    # Проверяем минимальную валидацию
    if not validation['valid']:
        return {
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'pattern_type': best_pattern,
            'higher_tf_trend': htf_analysis['trend'],
            'validation_score': validation['score'],
            'validation_reasons': [k for k, v in validation['checks'].items() if not v],
            'indicators': indicators,
            'higher_tf_analysis': htf_analysis if detailed_analysis else {}
        }

    # Формируем полный результат
    result = {
        'signal': signal_direction,
        'confidence': final_confidence,
        'pattern_type': best_pattern,
        'higher_tf_trend': htf_analysis['trend'],
        'validation_score': validation['score'],
        'atr_current': indicators.get('atr_current', 0),
        'volume_ratio': indicators.get('volume_ratio', 1.0),
        'entry_reasons': [
            f"{best_pattern} pattern detected",
            f"Higher TF trend: {htf_analysis['trend']}",
            f"Volume ratio: {indicators.get('volume_ratio', 1.0):.2f}",
            f"ATR current: {indicators.get('atr_current', 0):.6f}"
        ],
        'validation_reasons': [k for k, v in validation['checks'].items() if v],
        'indicators': indicators
    }

    # Добавляем детальную информацию для финального анализа
    if detailed_analysis:
        result.update({
            'higher_tf_analysis': htf_analysis,
            'pattern_details': best_signal,
            'full_validation': validation,
            'market_context': {
                'timeframe_sync': analyze_timeframe_synchronization(candles_5m, candles_15m),
                'volatility_context': analyze_market_volatility_context(candles_5m, candles_15m),
                'volume_context': analyze_volume_context(candles_5m, candles_15m)
            }
        })

    return result


def analyze_timeframe_synchronization(candles_5m: List[List[str]], candles_15m: List[List[str]]) -> Dict[str, Any]:
    """
    НОВОЕ: Анализ синхронизации таймфреймов для детального анализа
    """
    if len(candles_5m) < 50 or len(candles_15m) < 20:
        return {'sync_score': 0.0, 'sync_quality': 'POOR'}

    # Сравниваем тренды на разных таймфреймах
    closes_5m = np.array([safe_float(c[4]) for c in candles_5m])
    closes_15m = np.array([safe_float(c[4]) for c in candles_15m])

    # EMA для определения направления тренда
    ema20_5m = calculate_ema(closes_5m, 20)
    ema20_15m = calculate_ema(closes_15m, 20)

    # Направление тренда на каждом таймфрейме
    trend_5m = 'UP' if closes_5m[-1] > ema20_5m[-1] else 'DOWN'
    trend_15m = 'UP' if closes_15m[-1] > ema20_15m[-1] else 'DOWN'

    # Синхронизация трендов
    trends_aligned = trend_5m == trend_15m

    # Анализ силы трендов
    strength_5m = abs(closes_5m[-1] - ema20_5m[-1]) / ema20_5m[-1] * 100
    strength_15m = abs(closes_15m[-1] - ema20_15m[-1]) / ema20_15m[-1] * 100

    # Общий скор синхронизации
    sync_score = 0
    if trends_aligned:
        sync_score += 50

    # Бонус за силу трендов
    if strength_5m > 0.5 and strength_15m > 0.5:
        sync_score += 30

    # Бонус за стабильность
    if len(ema20_5m) >= 10 and len(ema20_15m) >= 5:
        stable_5m = all(ema20_5m[i] > ema20_5m[i-1] for i in range(-5, 0)) or \
                   all(ema20_5m[i] < ema20_5m[i-1] for i in range(-5, 0))
        stable_15m = all(ema20_15m[i] > ema20_15m[i-1] for i in range(-3, 0)) or \
                    all(ema20_15m[i] < ema20_15m[i-1] for i in range(-3, 0))

        if stable_5m and stable_15m:
            sync_score += 20

    # Определение качества синхронизации
    if sync_score >= 80:
        sync_quality = 'EXCELLENT'
    elif sync_score >= 60:
        sync_quality = 'GOOD'
    elif sync_score >= 40:
        sync_quality = 'FAIR'
    else:
        sync_quality = 'POOR'

    return {
        'sync_score': safe_float(sync_score),
        'sync_quality': sync_quality,
        'trend_5m': trend_5m,
        'trend_15m': trend_15m,
        'trends_aligned': trends_aligned,
        'strength_5m': safe_float(strength_5m),
        'strength_15m': safe_float(strength_15m)
    }


def analyze_market_volatility_context(candles_5m: List[List[str]], candles_15m: List[List[str]]) -> Dict[str, Any]:
    """
    НОВОЕ: Анализ контекста волатильности рынка
    """
    if len(candles_5m) < 50 or len(candles_15m) < 20:
        return {'volatility_context': 'UNKNOWN'}

    # ATR для обоих таймфреймов
    atr_5m = calculate_atr(candles_5m[-50:], 14)
    atr_15m = calculate_atr(candles_15m[-20:], 14)

    current_atr_5m = atr_5m['current']
    current_atr_15m = atr_15m['current']

    avg_atr_5m = np.mean(atr_5m['atr'][-20:]) if len(atr_5m['atr']) >= 20 else current_atr_5m
    avg_atr_15m = np.mean(atr_15m['atr'][-10:]) if len(atr_15m['atr']) >= 10 else current_atr_15m

    # Сравнение текущей и средней волатильности
    volatility_ratio_5m = current_atr_5m / avg_atr_5m if avg_atr_5m > 0 else 1.0
    volatility_ratio_15m = current_atr_15m / avg_atr_15m if avg_atr_15m > 0 else 1.0

    # Определение контекста волатильности
    if volatility_ratio_5m > 1.5 or volatility_ratio_15m > 1.5:
        context = 'HIGH_VOLATILITY'
    elif volatility_ratio_5m < 0.7 and volatility_ratio_15m < 0.7:
        context = 'LOW_VOLATILITY'
    else:
        context = 'NORMAL_VOLATILITY'

    # Тренд волатильности
    if len(atr_5m['atr']) >= 10:
        recent_atr = np.mean(atr_5m['atr'][-5:])
        historical_atr = np.mean(atr_5m['atr'][-10:-5])

        if recent_atr > historical_atr * 1.2:
            volatility_trend = 'INCREASING'
        elif recent_atr < historical_atr * 0.8:
            volatility_trend = 'DECREASING'
        else:
            volatility_trend = 'STABLE'
    else:
        volatility_trend = 'STABLE'

    return {
        'volatility_context': context,
        'volatility_trend': volatility_trend,
        'current_atr_5m': safe_float(current_atr_5m),
        'current_atr_15m': safe_float(current_atr_15m),
        'volatility_ratio_5m': safe_float(volatility_ratio_5m),
        'volatility_ratio_15m': safe_float(volatility_ratio_15m),
        'suitable_for_scalping': context in ['NORMAL_VOLATILITY', 'HIGH_VOLATILITY'] and volatility_trend != 'DECREASING'
    }


def analyze_volume_context(candles_5m: List[List[str]], candles_15m: List[List[str]]) -> Dict[str, Any]:
    """
    НОВОЕ: Анализ контекста объемов
    """
    if len(candles_5m) < 50 or len(candles_15m) < 20:
        return {'volume_context': 'UNKNOWN'}

    # Объемы для анализа
    volumes_5m = np.array([safe_float(c[5]) for c in candles_5m])
    volumes_15m = np.array([safe_float(c[5]) for c in candles_15m])

    # Средние объемы
    avg_volume_5m = np.mean(volumes_5m[-20:])
    avg_volume_15m = np.mean(volumes_15m[-10:])

    current_volume_5m = volumes_5m[-1]
    current_volume_15m = volumes_15m[-1]

    # Соотношения объемов
    volume_ratio_5m = current_volume_5m / avg_volume_5m if avg_volume_5m > 0 else 1.0
    volume_ratio_15m = current_volume_15m / avg_volume_15m if avg_volume_15m > 0 else 1.0

    # Определение контекста объемов
    if volume_ratio_5m > 2.0 or volume_ratio_15m > 2.0:
        context = 'HIGH_VOLUME'
    elif volume_ratio_5m < 0.5 and volume_ratio_15m < 0.5:
        context = 'LOW_VOLUME'
    else:
        context = 'NORMAL_VOLUME'

    # Тренд объемов
    if len(volumes_5m) >= 10:
        recent_avg = np.mean(volumes_5m[-5:])
        historical_avg = np.mean(volumes_5m[-10:-5])

        if recent_avg > historical_avg * 1.3:
            volume_trend = 'INCREASING'
        elif recent_avg < historical_avg * 0.7:
            volume_trend = 'DECREASING'
        else:
            volume_trend = 'STABLE'
    else:
        volume_trend = 'STABLE'

    # Поиск объемных аномалий
    volume_spikes = []
    spike_threshold = avg_volume_5m * 3.0

    for i, vol in enumerate(volumes_5m[-20:]):
        if vol > spike_threshold:
            volume_spikes.append({
                'position': i - 20,  # Позиция относительно текущей свечи
                'volume': safe_float(vol),
                'ratio': safe_float(vol / avg_volume_5m)
            })

    return {
        'volume_context': context,
        'volume_trend': volume_trend,
        'current_ratio_5m': safe_float(volume_ratio_5m),
        'current_ratio_15m': safe_float(volume_ratio_15m),
        'average_volume_5m': safe_float(avg_volume_5m),
        'average_volume_15m': safe_float(avg_volume_15m),
        'recent_spikes': volume_spikes,
        'spike_count': len(volume_spikes),
        'volume_confirmation': volume_ratio_5m > 1.2 or volume_ratio_15m > 1.2
    }


# НОВЫЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ РАСШИРЕННОГО АНАЛИЗА

def calculate_multi_timeframe_ema_alignment(candles_5m: List[List[str]],
                                          candles_15m: List[List[str]]) -> Dict[str, Any]:
    """
    НОВОЕ: Расчет выравнивания EMA на мультитаймфреймах
    """
    if len(candles_5m) < 50 or len(candles_15m) < 50:
        return {'alignment_score': 0.0, 'alignment_quality': 'POOR'}

    closes_5m = np.array([safe_float(c[4]) for c in candles_5m])
    closes_15m = np.array([safe_float(c[4]) for c in candles_15m])

    # Рассчитываем EMA для обоих таймфреймов
    ema_5m = {
        'ema8': calculate_ema(closes_5m, 8),
        'ema20': calculate_ema(closes_5m, 20),
        'ema50': calculate_ema(closes_5m, 50)
    }

    ema_15m = {
        'ema8': calculate_ema(closes_15m, 8),
        'ema20': calculate_ema(closes_15m, 20),
        'ema50': calculate_ema(closes_15m, 50)
    }

    # Проверяем выравнивание на каждом таймфрейме
    alignment_5m = (ema_5m['ema8'][-1] > ema_5m['ema20'][-1] > ema_5m['ema50'][-1]) or \
                   (ema_5m['ema8'][-1] < ema_5m['ema20'][-1] < ema_5m['ema50'][-1])

    alignment_15m = (ema_15m['ema8'][-1] > ema_15m['ema20'][-1] > ema_15m['ema50'][-1]) or \
                    (ema_15m['ema8'][-1] < ema_15m['ema20'][-1] < ema_15m['ema50'][-1])

    # Проверяем согласованность направлений
    direction_5m = 'UP' if ema_5m['ema8'][-1] > ema_5m['ema50'][-1] else 'DOWN'
    direction_15m = 'UP' if ema_15m['ema8'][-1] > ema_15m['ema50'][-1] else 'DOWN'

    directions_aligned = direction_5m == direction_15m

    # Расчет общего скора
    score = 0
    if alignment_5m:
        score += 30
    if alignment_15m:
        score += 40
    if directions_aligned:
        score += 30

    # Качество выравнивания
    if score >= 80:
        quality = 'EXCELLENT'
    elif score >= 60:
        quality = 'GOOD'
    elif score >= 40:
        quality = 'FAIR'
    else:
        quality = 'POOR'

    return {
        'alignment_score': safe_float(score),
        'alignment_quality': quality,
        'alignment_5m': alignment_5m,
        'alignment_15m': alignment_15m,
        'directions_aligned': directions_aligned,
        'direction_5m': direction_5m,
        'direction_15m': direction_15m
    }


def detect_divergences(candles: List[List[str]], indicators: Dict) -> Dict[str, Any]:
    """
    НОВОЕ: Детекция дивергенций для расширенного анализа
    """
    if len(candles) < 50:
        return {'divergences_found': False, 'divergence_types': []}

    closes = np.array([safe_float(c[4]) for c in candles])
    highs = np.array([safe_float(c[2]) for c in candles])
    lows = np.array([safe_float(c[3]) for c in candles])

    rsi = indicators.get('rsi', [])
    macd_line = indicators.get('macd_line', [])

    divergences = []

    # Поиск дивергенций RSI
    if len(rsi) >= 50:
        rsi_divergence = find_rsi_divergence(closes[-50:], highs[-50:], lows[-50:], rsi[-50:])
        if rsi_divergence:
            divergences.append(rsi_divergence)

    # Поиск дивергенций MACD
    if len(macd_line) >= 50:
        macd_divergence = find_macd_divergence(closes[-50:], highs[-50:], lows[-50:], macd_line[-50:])
        if macd_divergence:
            divergences.append(macd_divergence)

    return {
        'divergences_found': len(divergences) > 0,
        'divergence_types': divergences,
        'divergence_count': len(divergences)
    }


def find_rsi_divergence(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, rsi: List) -> Dict[str, Any]:
    """
    НОВОЕ: Поиск дивергенций RSI
    """
    if len(closes) < 20 or len(rsi) < 20:
        return None

    # Ищем локальные экстремумы цены и RSI
    price_peaks = []
    rsi_peaks = []

    # Поиск пиков (упрощенный алгоритм)
    for i in range(5, len(closes) - 5):
        # Пик цены
        if all(highs[i] >= highs[i-j] for j in range(1, 6)) and \
           all(highs[i] >= highs[i+j] for j in range(1, 6)):
            price_peaks.append({'index': i, 'value': highs[i]})

        # Пик RSI
        if all(rsi[i] >= rsi[i-j] for j in range(1, 6)) and \
           all(rsi[i] >= rsi[i+j] for j in range(1, 6)):
            rsi_peaks.append({'index': i, 'value': rsi[i]})

    # Поиск медвежьей дивергенции (цена растет, RSI падает)
    if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
        last_price_peak = price_peaks[-1]
        prev_price_peak = price_peaks[-2]

        # Найдем соответствующие пики RSI
        last_rsi_peak = None
        prev_rsi_peak = None

        for peak in rsi_peaks:
            if abs(peak['index'] - last_price_peak['index']) <= 3:
                last_rsi_peak = peak
            if abs(peak['index'] - prev_price_peak['index']) <= 3:
                prev_rsi_peak = peak

        if last_rsi_peak and prev_rsi_peak:
            if (last_price_peak['value'] > prev_price_peak['value'] and
                last_rsi_peak['value'] < prev_rsi_peak['value']):

                return {
                    'type': 'RSI_BEARISH_DIVERGENCE',
                    'strength': abs(last_rsi_peak['value'] - prev_rsi_peak['value']),
                    'reliability': 'MODERATE'
                }

    return None


def find_macd_divergence(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, macd: List) -> Dict[str, Any]:
    """
    НОВОЕ: Поиск дивергенций MACD
    """
    if len(closes) < 20 or len(macd) < 20:
        return None

    # Аналогичный поиск для MACD
    # Упрощенная версия - проверяем последние экстремумы
    recent_price_trend = closes[-1] - closes[-10]
    recent_macd_trend = macd[-1] - macd[-10]

    # Дивергенция если направления противоположны
    if (recent_price_trend > 0 and recent_macd_trend < 0) or \
       (recent_price_trend < 0 and recent_macd_trend > 0):

        return {
            'type': 'MACD_DIVERGENCE',
            'price_trend': 'UP' if recent_price_trend > 0 else 'DOWN',
            'macd_trend': 'UP' if recent_macd_trend > 0 else 'DOWN',
            'strength': abs(recent_macd_trend),
            'reliability': 'LOW'  # Упрощенный алгоритм
        }

    return None


def calculate_support_resistance_levels(candles: List[List[str]], lookback: int = 100) -> Dict[str, List[float]]:
    """
    НОВОЕ: Расчет уровней поддержки и сопротивления
    """
    if len(candles) < lookback:
        return {'support_levels': [], 'resistance_levels': []}

    highs = np.array([safe_float(c[2]) for c in candles[-lookback:]])
    lows = np.array([safe_float(c[3]) for c in candles[-lookback:]])
    closes = np.array([safe_float(c[4]) for c in candles[-lookback:]])

    resistance_levels = []
    support_levels = []

    # Поиск локальных экстремумов
    for i in range(5, len(highs) - 5):
        # Локальный максимум
        if all(highs[i] >= highs[i-j] for j in range(1, 6)) and \
           all(highs[i] >= highs[i+j] for j in range(1, 6)):
            resistance_levels.append(highs[i])

        # Локальный минимум
        if all(lows[i] <= lows[i-j] for j in range(1, 6)) and \
           all(lows[i] <= lows[i+j] for j in range(1, 6)):
            support_levels.append(lows[i])

    # Кластеризация уровней (объединяем близкие уровни)
    current_price = closes[-1]
    tolerance = current_price * 0.005  # 0.5% толерантность

    def cluster_levels(levels):
        if not levels:
            return []

        clustered = []
        levels.sort()

        current_cluster = [levels[0]]

        for level in levels[1:]:
            if level - current_cluster[-1] <= tolerance:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]

        clustered.append(np.mean(current_cluster))
        return clustered

    # Фильтруем и кластеризуем
    resistance_levels = [r for r in resistance_levels if r > current_price]
    support_levels = [s for s in support_levels if s < current_price]

    clustered_resistance = cluster_levels(resistance_levels)
    clustered_support = cluster_levels(support_levels)

    # Берем только ближайшие уровни
    nearest_resistance = sorted(clustered_resistance)[:5]
    nearest_support = sorted(clustered_support, reverse=True)[:5]

    return {
        'support_levels': [safe_float(s) for s in nearest_support],
        'resistance_levels': [safe_float(r) for r in nearest_resistance],
        'current_price': safe_float(current_price)
    }