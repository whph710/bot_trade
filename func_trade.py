"""
Упрощенный модуль индикаторов - только необходимые функции
Убраны дублирования и избыточные проверки
"""

import numpy as np
from typing import List, Dict, Any
from config import config


def safe_float(value) -> float:
    """Безопасное преобразование в float"""
    try:
        result = float(value)
        return 0.0 if (np.isnan(result) or np.isinf(result)) else result
    except:
        return 0.0


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Быстрый расчет EMA"""
    if len(prices) < period:
        return np.full_like(prices, prices[0] if len(prices) > 0 else 0)

    ema = np.zeros_like(prices)
    alpha = 2.0 / (period + 1)
    ema[0] = prices[0]

    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema


def calculate_rsi(prices: np.ndarray, period: int = 9) -> np.ndarray:
    """Быстрый расчет RSI"""
    if len(prices) < period + 1:
        return np.full_like(prices, 50.0)

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gains = np.zeros_like(prices)
    avg_losses = np.zeros_like(prices)

    if len(gains) >= period:
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])

    alpha = 1.0 / period
    for i in range(period + 1, len(prices)):
        avg_gains[i] = alpha * gains[i - 1] + (1 - alpha) * avg_gains[i - 1]
        avg_losses[i] = alpha * losses[i - 1] + (1 - alpha) * avg_losses[i - 1]

    rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses != 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 50.0

    return rsi


def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, np.ndarray]:
    """Быстрый расчет MACD"""
    if len(prices) < max(fast, slow):
        zero_array = np.zeros_like(prices)
        return {'line': zero_array, 'signal': zero_array, 'histogram': zero_array}

    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return {'line': macd_line, 'signal': signal_line, 'histogram': histogram}


def calculate_atr(candles: List[List[str]], period: int = 14) -> float:
    """Быстрый расчет текущего ATR"""
    if len(candles) < period + 1:
        return 0.0

    highs = np.array([safe_float(c[2]) for c in candles])
    lows = np.array([safe_float(c[3]) for c in candles])
    closes = np.array([safe_float(c[4]) for c in candles])

    tr = np.zeros(len(candles))
    for i in range(1, len(candles)):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )

    if len(tr) <= period:
        return safe_float(np.mean(tr[1:]))

    atr = np.mean(tr[1:period + 1])
    for i in range(period + 1, len(candles)):
        atr = (atr * (period - 1) + tr[i]) / period

    return safe_float(atr)


def calculate_basic_indicators(candles: List[List[str]]) -> Dict[str, Any]:
    """
    Базовые индикаторы для быстрого сканирования
    Возвращает только текущие значения
    """
    if len(candles) < 20:
        return {}

    closes = np.array([safe_float(c[4]) for c in candles])
    volumes = np.array([safe_float(c[5]) for c in candles])

    # EMA система
    ema5 = calculate_ema(closes, config.EMA_FAST)
    ema8 = calculate_ema(closes, config.EMA_MEDIUM)
    ema20 = calculate_ema(closes, config.EMA_SLOW)

    # RSI
    rsi = calculate_rsi(closes, config.RSI_PERIOD)

    # MACD
    macd = calculate_macd(closes, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL)

    # ATR
    atr = calculate_atr(candles, config.ATR_PERIOD)

    # Объем
    avg_volume = np.mean(volumes[-20:])
    volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0

    return {
        'price': safe_float(closes[-1]),
        'ema5': safe_float(ema5[-1]),
        'ema8': safe_float(ema8[-1]),
        'ema20': safe_float(ema20[-1]),
        'rsi': safe_float(rsi[-1]),
        'macd_line': safe_float(macd['line'][-1]),
        'macd_signal': safe_float(macd['signal'][-1]),
        'macd_histogram': safe_float(macd['histogram'][-1]),
        'atr': atr,
        'volume_ratio': safe_float(volume_ratio)
    }


def calculate_ai_indicators(candles: List[List[str]], history_length: int) -> Dict[str, Any]:
    """
    Индикаторы для ИИ отбора с историей
    """
    if len(candles) < history_length:
        return {}

    closes = np.array([safe_float(c[4]) for c in candles])
    volumes = np.array([safe_float(c[5]) for c in candles])

    # EMA с историей
    ema5 = calculate_ema(closes, config.EMA_FAST)
    ema8 = calculate_ema(closes, config.EMA_MEDIUM)
    ema20 = calculate_ema(closes, config.EMA_SLOW)

    # RSI с историей
    rsi = calculate_rsi(closes, config.RSI_PERIOD)

    # MACD с историей
    macd = calculate_macd(closes, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL)

    # Объемы с историей
    volume_sma = np.convolve(volumes, np.ones(20)/20, mode='valid')
    volume_ratios = volumes[19:] / volume_sma if len(volume_sma) > 0 else [1.0]

    return {
        # Последние значения индикаторов
        'ema5_history': [safe_float(x) for x in ema5[-history_length:]],
        'ema8_history': [safe_float(x) for x in ema8[-history_length:]],
        'ema20_history': [safe_float(x) for x in ema20[-history_length:]],
        'rsi_history': [safe_float(x) for x in rsi[-history_length:]],
        'macd_line_history': [safe_float(x) for x in macd['line'][-history_length:]],
        'macd_signal_history': [safe_float(x) for x in macd['signal'][-history_length:]],
        'macd_histogram_history': [safe_float(x) for x in macd['histogram'][-history_length:]],
        'volume_ratio_history': [safe_float(x) for x in volume_ratios[-history_length:]] if len(volume_ratios) >= history_length else [],

        # Текущие значения
        'current': {
            'price': safe_float(closes[-1]),
            'ema5': safe_float(ema5[-1]),
            'ema8': safe_float(ema8[-1]),
            'ema20': safe_float(ema20[-1]),
            'rsi': safe_float(rsi[-1]),
            'macd_line': safe_float(macd['line'][-1]),
            'macd_histogram': safe_float(macd['histogram'][-1]),
            'volume_ratio': safe_float(volume_ratios[-1]) if volume_ratios else 1.0,
            'atr': calculate_atr(candles, config.ATR_PERIOD)
        }
    }


def check_basic_signal(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Быстрая проверка базового сигнала для первичного отбора
    """
    if not indicators:
        return {'signal': False, 'confidence': 0, 'direction': 'NONE'}

    price = indicators['price']
    ema5 = indicators['ema5']
    ema8 = indicators['ema8']
    ema20 = indicators['ema20']
    rsi = indicators['rsi']
    macd_hist = indicators['macd_histogram']
    volume_ratio = indicators['volume_ratio']
    atr = indicators['atr']

    # Проверяем базовые условия
    conditions = []

    # EMA выравнивание для лонга
    if price > ema5 > ema8 > ema20:
        conditions.append(('LONG', 25))

    # EMA выравнивание для шорта
    if price < ema5 < ema8 < ema20:
        conditions.append(('SHORT', 25))

    # RSI в рабочем диапазоне
    if 30 < rsi < 70:
        conditions.append(('ANY', 15))

    # MACD поддержка
    if abs(macd_hist) > 0.001:  # Активный MACD
        conditions.append(('ANY', 15))

    # Объем подтверждение
    if volume_ratio >= config.MIN_VOLUME_RATIO:
        conditions.append(('ANY', 20))

    # ATR достаточный
    if atr > 0.001:  # Минимальная волатильность
        conditions.append(('ANY', 10))

    if not conditions:
        return {'signal': False, 'confidence': 0, 'direction': 'NONE'}

    # Определяем направление
    long_score = sum(score for direction, score in conditions if direction in ['LONG', 'ANY'])
    short_score = sum(score for direction, score in conditions if direction in ['SHORT', 'ANY'])

    if long_score > short_score and long_score >= config.MIN_CONFIDENCE:
        return {'signal': True, 'confidence': long_score, 'direction': 'LONG'}
    elif short_score > long_score and short_score >= config.MIN_CONFIDENCE:
        return {'signal': True, 'confidence': short_score, 'direction': 'SHORT'}
    else:
        return {'signal': False, 'confidence': max(long_score, short_score), 'direction': 'NONE'}