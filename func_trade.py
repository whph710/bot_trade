"""
Модуль технического анализа с МАКСИМАЛЬНЫМ набором индикаторов
Реализует мультитаймфреймный анализ с полным техническим арсеналом
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import time
import math
from config import config

# Параметры согласно обновленной консервативной конфигурации
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
    """Расчет экспоненциальной скользящей средней (EMA)"""
    if len(prices) < period:
        return np.zeros_like(prices)

    ema = np.zeros_like(prices)
    alpha = 2.0 / (period + 1)

    # Первое значение - SMA
    ema[period-1] = np.mean(prices[:period])

    # EMA для остальных значений
    for i in range(period, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema


def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Расчет простой скользящей средней (SMA)"""
    sma = np.zeros_like(prices)
    for i in range(period - 1, len(prices)):
        sma[i] = np.mean(prices[i - period + 1:i + 1])
    return sma


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Расчет RSI с улучшенной точностью"""
    if len(prices) <= period:
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

    # EMA для последующих значений
    alpha = 1.0 / period
    for i in range(period + 1, len(prices)):
        avg_gains[i] = alpha * gains[i - 1] + (1 - alpha) * avg_gains[i - 1]
        avg_losses[i] = alpha * losses[i - 1] + (1 - alpha) * avg_losses[i - 1]

    # Избежание деления на ноль
    rs = np.divide(avg_gains, avg_losses, out=np.ones_like(avg_gains), where=avg_losses != 0)
    rsi = 100 - (100 / (1 + rs))

    # Заполняем начальные значения
    rsi[:period] = 50.0

    return rsi


def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, np.ndarray]:
    """Расчет MACD с полными данными"""
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


def calculate_stochastic(candles: List[List[str]], k_period: int = 14, d_period: int = 3) -> Dict[str, np.ndarray]:
    """Расчет Stochastic Oscillator"""
    if len(candles) < k_period:
        return {'k': np.array([]), 'd': np.array([])}

    highs = np.array([float(c[2]) for c in candles])
    lows = np.array([float(c[3]) for c in candles])
    closes = np.array([float(c[4]) for c in candles])

    k_values = np.zeros_like(closes)

    for i in range(k_period - 1, len(closes)):
        high_max = np.max(highs[i - k_period + 1:i + 1])
        low_min = np.min(lows[i - k_period + 1:i + 1])

        if high_max != low_min:
            k_values[i] = ((closes[i] - low_min) / (high_max - low_min)) * 100
        else:
            k_values[i] = 50

    # %D - SMA от %K
    d_values = calculate_sma(k_values, d_period)

    return {
        'k': k_values,
        'd': d_values
    }


def calculate_williams_r(candles: List[List[str]], period: int = 14) -> np.ndarray:
    """Расчет Williams %R"""
    if len(candles) < period:
        return np.array([])

    highs = np.array([float(c[2]) for c in candles])
    lows = np.array([float(c[3]) for c in candles])
    closes = np.array([float(c[4]) for c in candles])

    wr_values = np.zeros_like(closes)

    for i in range(period - 1, len(closes)):
        high_max = np.max(highs[i - period + 1:i + 1])
        low_min = np.min(lows[i - period + 1:i + 1])

        if high_max != low_min:
            wr_values[i] = ((high_max - closes[i]) / (high_max - low_min)) * -100
        else:
            wr_values[i] = -50

    return wr_values


def calculate_cci(candles: List[List[str]], period: int = 20) -> np.ndarray:
    """Расчет Commodity Channel Index"""
    if len(candles) < period:
        return np.array([])

    highs = np.array([float(c[2]) for c in candles])
    lows = np.array([float(c[3]) for c in candles])
    closes = np.array([float(c[4]) for c in candles])

    # Typical Price
    tp = (highs + lows + closes) / 3

    cci_values = np.zeros_like(tp)

    for i in range(period - 1, len(tp)):
        tp_sma = np.mean(tp[i - period + 1:i + 1])
        mean_deviation = np.mean(np.abs(tp[i - period + 1:i + 1] - tp_sma))

        if mean_deviation != 0:
            cci_values[i] = (tp[i] - tp_sma) / (0.015 * mean_deviation)
        else:
            cci_values[i] = 0

    return cci_values


def calculate_atr(candles: List[List[str]], period: int = 14) -> Dict[str, Any]:
    """Расчет ATR с дополнительной статистикой"""
    if len(candles) < period + 1:
        return {'atr': [], 'current': 0.0, 'mean': 0.0, 'std': 0.0, 'percentile_75': 0.0}

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

    # Дополнительная статистика
    valid_atr = atr[atr > 0]

    return {
        'atr': safe_list(atr),
        'current': safe_float(atr[-1] if len(atr) > 0 else 0.0),
        'mean': safe_float(np.mean(valid_atr) if len(valid_atr) > 0 else 0.0),
        'std': safe_float(np.std(valid_atr) if len(valid_atr) > 0 else 0.0),
        'percentile_75': safe_float(np.percentile(valid_atr, 75) if len(valid_atr) > 0 else 0.0),
        'percentile_25': safe_float(np.percentile(valid_atr, 25) if len(valid_atr) > 0 else 0.0)
    }


def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, std: float = 2.0) -> Dict[str, np.ndarray]:
    """Расчет Bollinger Bands с дополнительными метриками"""
    sma = np.zeros_like(prices)
    upper = np.zeros_like(prices)
    lower = np.zeros_like(prices)
    width = np.zeros_like(prices)
    position = np.zeros_like(prices)

    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1:i + 1]
        sma[i] = np.mean(window)
        std_dev = np.std(window)
        upper[i] = sma[i] + (std_dev * std)
        lower[i] = sma[i] - (std_dev * std)

        # Ширина полос (нормализованная)
        if sma[i] != 0:
            width[i] = (upper[i] - lower[i]) / sma[i] * 100

        # Позиция цены в полосах (0-100)
        if upper[i] != lower[i]:
            position[i] = ((prices[i] - lower[i]) / (upper[i] - lower[i])) * 100

    return {
        'upper': upper,
        'middle': sma,
        'lower': lower,
        'width': width,
        'position': position
    }


def calculate_volume_indicators(candles: List[List[str]], period: int = 20) -> Dict[str, Any]:
    """Расчет объемных индикаторов"""
    volumes = np.array([float(c[5]) for c in candles])
    closes = np.array([float(c[4]) for c in candles])

    # Volume SMA
    volume_sma = calculate_sma(volumes, period)

    # Volume EMA
    volume_ema = calculate_ema(volumes, period)

    # OBV (On-Balance Volume)
    obv = np.zeros_like(closes)
    if len(closes) > 1:
        obv[0] = volumes[0]
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif closes[i] < closes[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]

    # Volume Rate of Change
    volume_roc = np.zeros_like(volumes)
    for i in range(period, len(volumes)):
        if volumes[i-period] != 0:
            volume_roc[i] = ((volumes[i] - volumes[i-period]) / volumes[i-period]) * 100

    return {
        'volume_sma': safe_list(volume_sma),
        'volume_ema': safe_list(volume_ema),
        'obv': safe_list(obv),
        'volume_roc': safe_list(volume_roc),
        'current_volume': safe_float(volumes[-1]),
        'avg_volume': safe_float(np.mean(volumes[-period:])),
        'volume_ratio': safe_float(volumes[-1] / volume_sma[-1] if volume_sma[-1] > 0 else 1.0)
    }


def calculate_momentum_indicators(prices: np.ndarray, period: int = 10) -> Dict[str, Any]:
    """Расчет индикаторов моментума"""
    # Rate of Change
    roc = np.zeros_like(prices)
    for i in range(period, len(prices)):
        if prices[i-period] != 0:
            roc[i] = ((prices[i] - prices[i-period]) / prices[i-period]) * 100

    # Momentum
    momentum = np.zeros_like(prices)
    for i in range(period, len(prices)):
        momentum[i] = prices[i] - prices[i-period]

    return {
        'roc': safe_list(roc),
        'momentum': safe_list(momentum)
    }


def calculate_trend_strength(prices: np.ndarray, period: int = 14) -> Dict[str, float]:
    """Расчет силы тренда"""
    if len(prices) < period * 2:
        return {'adx': 0.0, 'trend_strength': 'WEAK'}

    # Упрощенный расчет силы тренда
    recent_prices = prices[-period:]
    slope = (recent_prices[-1] - recent_prices[0]) / period

    # Линейная регрессия для определения тренда
    x = np.arange(len(recent_prices))
    z = np.polyfit(x, recent_prices, 1)
    slope_normalized = abs(z[0]) / np.mean(recent_prices) * 100

    # R-squared для определения силы тренда
    p = np.poly1d(z)
    yhat = p(x)
    ybar = np.sum(recent_prices) / len(recent_prices)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((recent_prices - ybar) ** 2)
    r_squared = ssreg / sstot if sstot != 0 else 0

    # Определение силы тренда
    if r_squared > 0.7 and slope_normalized > 1.0:
        trend_strength = 'VERY_STRONG'
    elif r_squared > 0.5 and slope_normalized > 0.5:
        trend_strength = 'STRONG'
    elif r_squared > 0.3:
        trend_strength = 'MODERATE'
    else:
        trend_strength = 'WEAK'

    return {
        'slope': safe_float(slope),
        'slope_normalized': safe_float(slope_normalized),
        'r_squared': safe_float(r_squared),
        'trend_strength': trend_strength
    }


def calculate_support_resistance(candles: List[List[str]], lookback: int = 50) -> Dict[str, Any]:
    """Расчет уровней поддержки и сопротивления"""
    if len(candles) < lookback:
        return {'support_levels': [], 'resistance_levels': []}

    highs = np.array([float(c[2]) for c in candles[-lookback:]])
    lows = np.array([float(c[3]) for c in candles[-lookback:]])
    closes = np.array([float(c[4]) for c in candles[-lookback:]])

    # Простое определение уровней (локальные экстремумы)
    resistance_levels = []
    support_levels = []

    # Поиск локальных максимумов и минимумов
    window = 5
    for i in range(window, len(highs) - window):
        # Сопротивление
        if all(highs[i] >= highs[j] for j in range(i-window, i+window+1) if j != i):
            resistance_levels.append(safe_float(highs[i]))

        # Поддержка
        if all(lows[i] <= lows[j] for j in range(i-window, i+window+1) if j != i):
            support_levels.append(safe_float(lows[i]))

    # Сортируем и берем ключевые уровни
    resistance_levels = sorted(list(set(resistance_levels)))[-5:]  # Топ 5
    support_levels = sorted(list(set(support_levels)), reverse=True)[-5:]  # Топ 5

    current_price = safe_float(closes[-1])

    # Ближайшие уровни
    nearest_resistance = min([r for r in resistance_levels if r > current_price],
                           default=current_price * 1.02)
    nearest_support = max([s for s in support_levels if s < current_price],
                         default=current_price * 0.98)

    return {
        'resistance_levels': resistance_levels,
        'support_levels': support_levels,
        'nearest_resistance': safe_float(nearest_resistance),
        'nearest_support': safe_float(nearest_support),
        'distance_to_resistance': safe_float((nearest_resistance - current_price) / current_price * 100),
        'distance_to_support': safe_float((current_price - nearest_support) / current_price * 100)
    }


def calculate_all_indicators_comprehensive(candles: List[List[str]]) -> Dict[str, Any]:
    """
    МАКСИМАЛЬНО ПОЛНЫЙ расчет всех индикаторов для передачи в ИИ

    Args:
        candles: Список свечей 5m таймфрейма (максимальное количество)

    Returns:
        dict: Словарь со ВСЕМИ рассчитанными индикаторами и максимальной историей
    """
    if len(candles) < 100:  # Минимум для качественного анализа
        return {}

    closes = np.array([float(c[4]) for c in candles])
    opens = np.array([float(c[1]) for c in candles])
    highs = np.array([float(c[2]) for c in candles])
    lows = np.array([float(c[3]) for c in candles])
    volumes = np.array([float(c[5]) for c in candles])
    timestamps = np.array([int(c[0]) for c in candles])

    # 1. ТРЕНДОВЫЕ ИНДИКАТОРЫ (максимальная история)
    ema5 = calculate_ema(closes, config.indicators.EMA_FAST)
    ema8 = calculate_ema(closes, config.indicators.EMA_MEDIUM)
    ema20 = calculate_ema(closes, config.indicators.EMA_SLOW)
    ema50 = calculate_ema(closes, 50)
    ema100 = calculate_ema(closes, 100)
    ema200 = calculate_ema(closes, 200)

    sma20 = calculate_sma(closes, 20)
    sma50 = calculate_sma(closes, 50)
    sma100 = calculate_sma(closes, 100)

    # 2. ОСЦИЛЛЯТОРЫ (полная история)
    rsi = calculate_rsi(closes, config.indicators.RSI_PERIOD)
    rsi_9 = calculate_rsi(closes, 9)  # Быстрый RSI

    stoch_data = calculate_stochastic(candles, config.indicators.STOCH_K_PERIOD,
                                     config.indicators.STOCH_D_PERIOD)
    williams_r = calculate_williams_r(candles, config.indicators.WILLIAMS_PERIOD)
    cci = calculate_cci(candles, config.indicators.CCI_PERIOD)

    # 3. MACD (максимальная конфигурация)
    macd_data = calculate_macd(closes, config.indicators.MACD_FAST,
                               config.indicators.MACD_SLOW, config.indicators.MACD_SIGNAL)

    # 4. ВОЛАТИЛЬНОСТЬ И ATR
    atr_data = calculate_atr(candles, config.indicators.ATR_PERIOD)
    bb_data = calculate_bollinger_bands(closes, config.indicators.BB_PERIOD,
                                       config.indicators.BB_STD)

    # 5. ОБЪЕМНЫЕ ИНДИКАТОРЫ
    volume_data = calculate_volume_indicators(candles, config.indicators.VOLUME_SMA)

    # 6. МОМЕНТУМ
    momentum_data = calculate_momentum_indicators(closes, 14)

    # 7. АНАЛИЗ ТРЕНДА
    trend_data = calculate_trend_strength(closes, 20)

    # 8. ПОДДЕРЖКА И СОПРОТИВЛЕНИЕ
    sr_data = calculate_support_resistance(candles, 100)

    # 9. СВЕЧНОЙ АНАЛИЗ (последние 50 свечей)
    candle_analysis = analyze_candlestick_patterns(candles[-50:])

    # 10. ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ
    price_action = {
        'current_price': safe_float(closes[-1]),
        'open_price': safe_float(opens[-1]),
        'high_price': safe_float(highs[-1]),
        'low_price': safe_float(lows[-1]),
        'price_change': safe_float(closes[-1] - closes[-2] if len(closes) > 1 else 0),
        'price_change_percent': safe_float(((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) > 1 and closes[-2] != 0 else 0),
        'daily_high': safe_float(np.max(highs[-288:])) if len(highs) >= 288 else safe_float(np.max(highs)),  # 288 = 24h на 5m
        'daily_low': safe_float(np.min(lows[-288:])) if len(lows) >= 288 else safe_float(np.min(lows)),
        'daily_range': safe_float((np.max(highs[-288:]) - np.min(lows[-288:])) / closes[-1] * 100) if len(highs) >= 288 else 0
    }

    # КОМПИЛИРУЕМ ВСЕ ДАННЫЕ ДЛЯ ИИ
    comprehensive_indicators = {
        # ОСНОВНЫЕ ТРЕНДОВЫЕ ИНДИКАТОРЫ (последние 200 значений)
        'trend_following': {
            'ema5': safe_list(ema5[-200:]),
            'ema8': safe_list(ema8[-200:]),
            'ema20': safe_list(ema20[-200:]),
            'ema50': safe_list(ema50[-200:]),
            'ema100': safe_list(ema100[-200:]),
            'ema200': safe_list(ema200[-200:]),
            'sma20': safe_list(sma20[-200:]),
            'sma50': safe_list(sma50[-200:]),
            'sma100': safe_list(sma100[-200:]),

            # ТЕКУЩИЕ ЗНАЧЕНИЯ
            'ema5_current': safe_float(ema5[-1]),
            'ema8_current': safe_float(ema8[-1]),
            'ema20_current': safe_float(ema20[-1]),
            'ema50_current': safe_float(ema50[-1]),

            # ВЫРАВНИВАНИЕ EMA
            'ema_alignment_bullish': len(ema5) > 0 and len(ema8) > 0 and len(ema20) > 0 and ema5[-1] > ema8[-1] > ema20[-1],
            'ema_alignment_bearish': len(ema5) > 0 and len(ema8) > 0 and len(ema20) > 0 and ema5[-1] < ema8[-1] < ema20[-1],
            'price_above_emas': closes[-1] > ema5[-1] and closes[-1] > ema8[-1] and closes[-1] > ema20[-1],
            'price_below_emas': closes[-1] < ema5[-1] and closes[-1] < ema8[-1] and closes[-1] < ema20[-1]
        },

        # ОСЦИЛЛЯТОРЫ (последние 100 значений)
        'oscillators': {
            'rsi14': safe_list(rsi[-100:]),
            'rsi9': safe_list(rsi_9[-100:]),
            'rsi14_current': safe_float(rsi[-1]),
            'rsi9_current': safe_float(rsi_9[-1]),
            'rsi_oversold': rsi[-1] < config.indicators.RSI_OVERSOLD,
            'rsi_overbought': rsi[-1] > config.indicators.RSI_OVERBOUGHT,
            'rsi_divergence': detect_rsi_divergence(closes[-20:], rsi[-20:]),

            'stoch_k': safe_list(stoch_data['k'][-100:]),
            'stoch_d': safe_list(stoch_data['d'][-100:]),
            'stoch_k_current': safe_float(stoch_data['k'][-1]) if len(stoch_data['k']) > 0 else 50,
            'stoch_d_current': safe_float(stoch_data['d'][-1]) if len(stoch_data['d']) > 0 else 50,

            'williams_r': safe_list(williams_r[-100:]),
            'williams_r_current': safe_float(williams_r[-1]) if len(williams_r) > 0 else -50,

            'cci': safe_list(cci[-100:]),
            'cci_current': safe_float(cci[-1]) if len(cci) > 0 else 0,
            'cci_extreme': abs(cci[-1]) > 100 if len(cci) > 0 else False
        },

        # MACD (последние 100 значений)
        'macd_analysis': {
            'macd_line': safe_list(macd_data['macd'][-100:]),
            'macd_signal': safe_list(macd_data['signal'][-100:]),
            'macd_histogram': safe_list(macd_data['histogram'][-100:]),
            'macd_current': safe_float(macd_data['macd'][-1]),
            'signal_current': safe_float(macd_data['signal'][-1]),
            'histogram_current': safe_float(macd_data['histogram'][-1]),
            'macd_bullish_crossover': detect_macd_crossover(macd_data, 'bullish'),
            'macd_bearish_crossover': detect_macd_crossover(macd_data, 'bearish'),
            'histogram_increasing': detect_histogram_change(macd_data['histogram']),
            'macd_divergence': detect_macd_divergence(closes[-20:], macd_data['macd'][-20:])
        },

        # ВОЛАТИЛЬНОСТЬ (максимальная история)
        'volatility': {
            'atr': safe_list(atr_data['atr'][-100:]),
            'atr_current': atr_data['current'],
            'atr_mean': atr_data['mean'],
            'atr_std': atr_data['std'],
            'atr_percentile_75': atr_data['percentile_75'],
            'atr_percentile_25': atr_data['percentile_25'],
            'atr_normalized': safe_float((atr_data['current'] / closes[-1]) * 100),
            'volatility_regime': classify_volatility_regime(atr_data),

            # Bollinger Bands (последние 100 значений)
            'bb_upper': safe_list(bb_data['upper'][-100:]),
            'bb_middle': safe_list(bb_data['middle'][-100:]),
            'bb_lower': safe_list(bb_data['lower'][-100:]),
            'bb_width': safe_list(bb_data['width'][-100:]),
            'bb_position': safe_list(bb_data['position'][-100:]),
            'bb_squeeze': detect_bb_squeeze(bb_data),
            'price_bb_position': safe_float(bb_data['position'][-1]),
            'bb_breakout': detect_bb_breakout(closes, bb_data)
        },

        # ОБЪЕМНЫЙ АНАЛИЗ (максимальные данные)
        'volume_analysis': {
            'volume': safe_list(volumes[-200:]),
            'volume_sma': volume_data['volume_sma'][-100:],
            'volume_ema': volume_data['volume_ema'][-100:],
            'obv': volume_data['obv'][-100:],
            'volume_roc': volume_data['volume_roc'][-100:],
            'current_volume': volume_data['current_volume'],
            'avg_volume': volume_data['avg_volume'],
            'volume_ratio': volume_data['volume_ratio'],
            'volume_spike': volume_data['volume_ratio'] > config.indicators.VOLUME_SPIKE_RATIO,
            'volume_trend': detect_volume_trend(volumes[-20:]),
            'volume_price_divergence': detect_volume_price_divergence(closes[-20:], volumes[-20:])
        },

        # МОМЕНТУМ И ТРЕНД
        'momentum_trend': {
            'roc': momentum_data['roc'][-50:],
            'momentum': momentum_data['momentum'][-50:],
            'trend_strength': trend_data['trend_strength'],
            'trend_slope': trend_data['slope'],
            'trend_r_squared': trend_data['r_squared'],
            'price_momentum_5': safe_float(closes[-1] - closes[-6]) if len(closes) >= 6 else 0,
            'price_momentum_10': safe_float(closes[-1] - closes[-11]) if len(closes) >= 11 else 0,
            'price_momentum_20': safe_float(closes[-1] - closes[-21]) if len(closes) >= 21 else 0
        },

        # УРОВНИ ПОДДЕРЖКИ И СОПРОТИВЛЕНИЯ
        'support_resistance': sr_data,

        # СВЕЧНОЙ АНАЛИЗ
        'candlestick_analysis': candle_analysis,

        # ЦЕНОВОЕ ДЕЙСТВИЕ
        'price_action': price_action,

        # ВРЕМЕННЫЕ МЕТКИ (последние 100)
        'timestamps': safe_list(timestamps[-100:]),

        # ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА
        'statistics': {
            'data_points': len(candles),
            'volatility_20_periods': safe_float(np.std(closes[-20:]) / np.mean(closes[-20:]) * 100) if len(closes) >= 20 else 0,
            'max_drawdown_20': calculate_max_drawdown(closes[-20:]),
            'win_rate_signals': calculate_recent_win_rate(closes[-50:]),
            'correlation_volume_price': calculate_price_volume_correlation(closes[-30:], volumes[-30:])
        }
    }

    return comprehensive_indicators


def analyze_candlestick_patterns(candles: List[List[str]]) -> Dict[str, Any]:
    """Анализ свечных паттернов"""
    if len(candles) < 3:
        return {}

    patterns = {
        'doji': [],
        'hammer': [],
        'hanging_man': [],
        'engulfing_bullish': [],
        'engulfing_bearish': [],
        'spinning_top': []
    }

    for i in range(len(candles)):
        candle = candles[i]
        open_price = float(candle[1])
        high_price = float(candle[2])
        low_price = float(candle[3])
        close_price = float(candle[4])

        body = abs(close_price - open_price)
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        total_range = high_price - low_price

        if total_range > 0:
            # Дожи
            if body / total_range < 0.1:
                patterns['doji'].append(i)

            # Молот
            if (lower_shadow > body * 2 and upper_shadow < body * 0.5 and
                close_price > open_price):
                patterns['hammer'].append(i)

            # Повешенный
            if (lower_shadow > body * 2 and upper_shadow < body * 0.5 and
                close_price < open_price):
                patterns['hanging_man'].append(i)

    # Поглощение (нужно минимум 2 свечи)
    for i in range(1, len(candles)):
        prev_candle = candles[i-1]
        curr_candle = candles[i]

        prev_open = float(prev_candle[1])
        prev_close = float(prev_candle[4])
        curr_open = float(curr_candle[1])
        curr_close = float(curr_candle[4])

        # Бычье поглощение
        if (prev_close < prev_open and  # Предыдущая свеча медвежья
            curr_close > curr_open and  # Текущая свеча бычья
            curr_open < prev_close and  # Открытие ниже предыдущего закрытия
            curr_close > prev_open):    # Закрытие выше предыдущего открытия
            patterns['engulfing_bullish'].append(i)

        # Медвежье поглощение
        if (prev_close > prev_open and  # Предыдущая свеча бычья
            curr_close < curr_open and  # Текущая свеча медвежья
            curr_open > prev_close and  # Открытие выше предыдущего закрытия
            curr_close < prev_open):    # Закрытие ниже предыдущего открытия
            patterns['engulfing_bearish'].append(i)

    return patterns


def detect_rsi_divergence(prices: np.ndarray, rsi: np.ndarray) -> bool:
    """Определение дивергенции RSI"""
    if len(prices) < 10 or len(rsi) < 10:
        return False

    # Упрощенное определение дивергенции
    price_trend = prices[-1] - prices[-10]
    rsi_trend = rsi[-1] - rsi[-10]

    # Дивергенция: цена растет, RSI падает (или наоборот)
    return (price_trend > 0 and rsi_trend < 0) or (price_trend < 0 and rsi_trend > 0)


def detect_macd_crossover(macd_data: Dict, direction: str) -> bool:
    """Определение пересечения MACD"""
    macd_line = macd_data['macd']
    signal_line = macd_data['signal']

    if len(macd_line) < 2 or len(signal_line) < 2:
        return False

    if direction == 'bullish':
        return (macd_line[-2] <= signal_line[-2] and macd_line[-1] > signal_line[-1])
    else:  # bearish
        return (macd_line[-2] >= signal_line[-2] and macd_line[-1] < signal_line[-1])


def detect_histogram_change(histogram: np.ndarray) -> bool:
    """Определение роста гистограммы MACD"""
    if len(histogram) < 3:
        return False
    return histogram[-1] > histogram[-2] > histogram[-3]


def detect_macd_divergence(prices: np.ndarray, macd: np.ndarray) -> bool:
    """Определение дивергенции MACD"""
    if len(prices) < 10 or len(macd) < 10:
        return False

    price_trend = prices[-1] - prices[-10]
    macd_trend = macd[-1] - macd[-10]

    return (price_trend > 0 and macd_trend < 0) or (price_trend < 0 and macd_trend > 0)


def classify_volatility_regime(atr_data: Dict) -> str:
    """Классификация режима волатильности"""
    current_atr = atr_data['current']
    mean_atr = atr_data['mean']
    std_atr = atr_data['std']

    if current_atr > mean_atr + std_atr:
        return 'HIGH_VOLATILITY'
    elif current_atr < mean_atr - std_atr:
        return 'LOW_VOLATILITY'
    else:
        return 'NORMAL_VOLATILITY'


def detect_bb_squeeze(bb_data: Dict) -> bool:
    """Определение сжатия Bollinger Bands"""
    width = bb_data['width']
    if len(width) < 20:
        return False

    current_width = width[-1]
    avg_width = np.mean(width[-20:])

    return current_width < avg_width * 0.8


def detect_bb_breakout(prices: np.ndarray, bb_data: Dict) -> Dict[str, bool]:
    """Определение пробоя Bollinger Bands"""
    if len(prices) < 2:
        return {'upper_breakout': False, 'lower_breakout': False}

    upper = bb_data['upper']
    lower = bb_data['lower']

    if len(upper) < 2 or len(lower) < 2:
        return {'upper_breakout': False, 'lower_breakout': False}

    return {
        'upper_breakout': prices[-1] > upper[-1] and prices[-2] <= upper[-2],
        'lower_breakout': prices[-1] < lower[-1] and prices[-2] >= lower[-2]
    }


def detect_volume_trend(volumes: np.ndarray) -> str:
    """Определение тренда объема"""
    if len(volumes) < 10:
        return 'UNKNOWN'

    recent_avg = np.mean(volumes[-5:])
    older_avg = np.mean(volumes[-10:-5])

    if recent_avg > older_avg * 1.1:
        return 'INCREASING'
    elif recent_avg < older_avg * 0.9:
        return 'DECREASING'
    else:
        return 'STABLE'


def detect_volume_price_divergence(prices: np.ndarray, volumes: np.ndarray) -> bool:
    """Определение дивергенции объема и цены"""
    if len(prices) < 10 or len(volumes) < 10:
        return False

    price_trend = (prices[-1] - prices[-10]) / prices[-10]
    volume_trend = (volumes[-1] - volumes[-10]) / volumes[-10]

    # Дивергенция: цена растет, объем падает значительно
    return abs(price_trend) > 0.02 and price_trend * volume_trend < -0.01


def calculate_max_drawdown(prices: np.ndarray) -> float:
    """Расчет максимальной просадки"""
    if len(prices) < 2:
        return 0.0

    peak = prices[0]
    max_dd = 0.0

    for price in prices:
        if price > peak:
            peak = price

        drawdown = (peak - price) / peak
        if drawdown > max_dd:
            max_dd = drawdown

    return safe_float(max_dd * 100)


def calculate_recent_win_rate(prices: np.ndarray) -> float:
    """Расчет процента выигрышных движений"""
    if len(prices) < 10:
        return 50.0

    changes = np.diff(prices)
    positive_changes = np.sum(changes > 0)
    total_changes = len(changes)

    return safe_float((positive_changes / total_changes) * 100)


def calculate_price_volume_correlation(prices: np.ndarray, volumes: np.ndarray) -> float:
    """Корреляция цены и объема"""
    if len(prices) < 10 or len(volumes) < 10:
        return 0.0

    try:
        correlation = np.corrcoef(prices, volumes)[0, 1]
        return safe_float(correlation) if not np.isnan(correlation) else 0.0
    except:
        return 0.0


def analyze_higher_timeframe_trend_comprehensive(candles_15m: List[List[str]]) -> Dict[str, Any]:
    """
    РАСШИРЕННЫЙ анализ старшего таймфрейма с максимальными данными
    """
    if len(candles_15m) < 50:
        return {'trend': 'UNKNOWN', 'strength': 0, 'confidence': 0}

    closes = np.array([float(c[4]) for c in candles_15m])
    highs = np.array([float(c[2]) for c in candles_15m])
    lows = np.array([float(c[3]) for c in candles_15m])
    volumes = np.array([float(c[5]) for c in candles_15m])

    # Множественные EMA для определения тренда
    ema20 = calculate_ema(closes, 20)
    ema50 = calculate_ema(closes, 50)
    ema100 = calculate_ema(closes, 100)

    # RSI на старшем таймфрейме
    rsi_15m = calculate_rsi(closes, 14)

    # MACD на старшем таймфрейме
    macd_15m = calculate_macd(closes, 12, 26, 9)

    # ATR на старшем таймфрейме
    atr_15m = calculate_atr(candles_15m, 14)

    # Анализ тренда
    current_price = closes[-1]
    ema20_current = ema20[-1]
    ema50_current = ema50[-1]
    ema100_current = ema100[-1]

    # Определение направления тренда (множественные критерии)
    trend_signals = {
        'price_above_ema20': current_price > ema20_current,
        'price_above_ema50': current_price > ema50_current,
        'ema20_above_ema50': ema20_current > ema50_current,
        'ema50_above_ema100': ema50_current > ema100_current,
        'rsi_bullish': rsi_15m[-1] > 50,
        'macd_positive': macd_15m['macd'][-1] > 0,
        'macd_above_signal': macd_15m['macd'][-1] > macd_15m['signal'][-1]
    }

    bullish_signals = sum(trend_signals.values())
    total_signals = len(trend_signals)

    # Определение тренда и силы
    if bullish_signals >= 6:  # 6 из 7
        trend = 'STRONG_UPTREND'
        strength = min(100, int((bullish_signals / total_signals) * 100))
    elif bullish_signals >= 4:  # 4-5 из 7
        trend = 'UPTREND'
        strength = min(85, int((bullish_signals / total_signals) * 100))
    elif bullish_signals <= 1:  # 0-1 из 7
        trend = 'STRONG_DOWNTREND'
        strength = min(100, int(((total_signals - bullish_signals) / total_signals) * 100))
    elif bullish_signals <= 3:  # 2-3 из 7
        trend = 'DOWNTREND'
        strength = min(85, int(((total_signals - bullish_signals) / total_signals) * 100))
    else:
        trend = 'SIDEWAYS'
        strength = 50

    # Дополнительный анализ силы тренда
    trend_quality = calculate_trend_strength(closes, 20)

    return {
        'trend': trend,
        'strength': strength,
        'confidence': min(95, strength + int(trend_quality['r_squared'] * 20)),
        'trend_signals': trend_signals,
        'bullish_signals_count': bullish_signals,
        'total_signals_count': total_signals,

        # Подробные данные для ИИ
        'ema_values': {
            'ema20': safe_float(ema20_current),
            'ema50': safe_float(ema50_current),
            'ema100': safe_float(ema100_current)
        },
        'rsi_15m': safe_float(rsi_15m[-1]),
        'macd_15m': {
            'macd': safe_float(macd_15m['macd'][-1]),
            'signal': safe_float(macd_15m['signal'][-1]),
            'histogram': safe_float(macd_15m['histogram'][-1])
        },
        'atr_15m': safe_float(atr_15m['current']),
        'volatility_15m': safe_float((atr_15m['current'] / current_price) * 100),

        # Качество тренда
        'trend_quality': trend_quality,
        'price_momentum': safe_float(((current_price - closes[-10]) / closes[-10]) * 100) if len(closes) >= 10 else 0
    }


def detect_instruction_based_signals_enhanced(candles_5m: List[List[str]],
                                            candles_15m: List[List[str]]) -> Dict[str, Any]:
    """
    УЛУЧШЕННАЯ функция определения сигналов с максимальными данными для ИИ
    Мультитаймфреймный анализ: 15m контекст + 5m вход + ВСЕ индикаторы
    """
    if not candles_5m or not candles_15m:
        return {
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'pattern_type': 'NONE',
            'validation_score': '0/5',
            'comprehensive_data': {}
        }

    # Рассчитываем ВСЕ индикаторы для 5m (максимальные данные)
    comprehensive_indicators = calculate_all_indicators_comprehensive(candles_5m)
    if not comprehensive_indicators:
        return {
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'pattern_type': 'NONE',
            'validation_score': '0/5',
            'comprehensive_data': {}
        }

    # Расширенный анализ старшего таймфрейма
    htf_analysis = analyze_higher_timeframe_trend_comprehensive(candles_15m)

    # Основные шаблоны входа (используем консервативные настройки)
    patterns = [
        ('PULLBACK_ENTRY', detect_pullback_entry_enhanced(candles_5m, comprehensive_indicators)),
        ('MOMENTUM_BREAKOUT', detect_momentum_breakout_enhanced(candles_5m, comprehensive_indicators)),
        ('SQUEEZE_BREAKOUT', detect_squeeze_breakout_enhanced(candles_5m, comprehensive_indicators)),
        ('RANGE_SCALP', detect_range_scalp_enhanced(candles_5m, comprehensive_indicators))
    ]

    # Ищем лучший сигнал с учетом новых приоритетов
    best_signal = None
    best_confidence = 0
    best_pattern = 'NONE'

    for pattern_name, pattern_result in patterns:
        if pattern_result['signal'] and pattern_result['confidence'] >= config.trading.MIN_CONFIDENCE:
            # Учитываем приоритет паттернов из конфига
            priority_bonus = {
                'PULLBACK_ENTRY': 10,
                'MOMENTUM_BREAKOUT': 5,
                'SQUEEZE_BREAKOUT': 3,
                'RANGE_SCALP': 0
            }.get(pattern_name, 0)

            adjusted_confidence = pattern_result['confidence'] + priority_bonus

            if adjusted_confidence > best_confidence:
                best_signal = pattern_result
                best_confidence = adjusted_confidence
                best_pattern = pattern_name

    # Если сигнал не найден
    if not best_signal or best_confidence < config.trading.MIN_CONFIDENCE:
        return {
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'pattern_type': 'NONE',
            'higher_tf_analysis': htf_analysis,
            'validation_score': '0/5',
            'comprehensive_data': comprehensive_indicators
        }

    # Определяем направление сигнала
    signal_direction = best_signal.get('signal_type',
                                     best_signal.get('breakout_direction', 'LONG'))

    # РАСШИРЕННАЯ валидация сигнала (4 из 5 проверок)
    validation = validate_signal_enhanced(candles_5m, candles_15m,
                                        {'signal_type': signal_direction},
                                        comprehensive_indicators, htf_analysis)

    # Применяем все модификаторы уверенности
    final_confidence = apply_confidence_modifiers(best_confidence, validation,
                                                comprehensive_indicators, htf_analysis)

    # Проверяем строгую валидацию (4 из 5)
    if not validation['valid']:
        return {
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'pattern_type': best_pattern,
            'higher_tf_analysis': htf_analysis,
            'validation_score': validation['score'],
            'validation_reasons': [k for k, v in validation['checks'].items() if not v],
            'comprehensive_data': comprehensive_indicators
        }

    # Финальная проверка консервативных критериев
    if not meets_conservative_criteria(comprehensive_indicators, htf_analysis, signal_direction):
        return {
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'pattern_type': best_pattern,
            'higher_tf_analysis': htf_analysis,
            'validation_score': validation['score'],
            'rejection_reason': 'Conservative criteria not met',
            'comprehensive_data': comprehensive_indicators
        }

    return {
        'signal': signal_direction,
        'confidence': min(100, final_confidence),
        'pattern_type': best_pattern,
        'higher_tf_analysis': htf_analysis,
        'validation_score': validation['score'],
        'entry_reasons': generate_entry_reasons(best_pattern, htf_analysis, comprehensive_indicators),
        'validation_reasons': [k for k, v in validation['checks'].items() if v],
        'risk_metrics': calculate_risk_metrics(comprehensive_indicators, candles_5m),
        'comprehensive_data': comprehensive_indicators  # ВСЕ данные для ИИ
    }


def detect_pullback_entry_enhanced(candles: List[List[str]], indicators: Dict) -> Dict[str, Any]:
    """Улучшенный детектор Pullback Entry с консервативными настройками"""
    if len(candles) < 20:
        return {'signal': False, 'confidence': 0}

    closes = np.array([float(c[4]) for c in candles])
    current_close = closes[-1]

    trend_data = indicators['trend_following']
    oscillators = indicators['oscillators']

    # Более строгие условия для pullback
    conditions = {
        'strong_trend_context': trend_data['ema_alignment_bullish'] or trend_data['ema_alignment_bearish'],
        'near_key_ema': abs(current_close - trend_data['ema20_current']) / current_close < config.patterns.PULLBACK_EMA_PROXIMITY,
        'rsi_in_range': config.indicators.RSI_OVERSOLD < oscillators['rsi14_current'] < config.indicators.RSI_OVERBOUGHT,
        'volume_support': indicators['volume_analysis']['volume_ratio'] >= config.indicators.VOLUME_MIN_RATIO,
        'trend_strength_good': indicators.get('momentum_trend', {}).get('trend_strength') in ['STRONG', 'VERY_STRONG']
    }

    if sum(conditions.values()) >= 4:  # 4 из 5 условий
        signal_type = 'LONG' if trend_data['ema_alignment_bullish'] else 'SHORT'
        base_confidence = config.patterns.PATTERN_BASE_CONFIDENCE['PULLBACK_ENTRY']

        # Бонусы за качество сигнала
        quality_bonus = sum([
            5 if conditions['strong_trend_context'] else 0,
            5 if conditions['trend_strength_good'] else 0,
            3 if indicators['volume_analysis']['volume_spike'] else 0
        ])

        return {
            'signal': True,
            'signal_type': signal_type,
            'confidence': min(95, base_confidence + quality_bonus),
            'pattern': 'PULLBACK_ENTRY',
            'conditions': conditions
        }

    return {'signal': False, 'confidence': 0}


def detect_momentum_breakout_enhanced(candles: List[List[str]], indicators: Dict) -> Dict[str, Any]:
    """Улучшенный детектор Momentum Breakout"""
    if len(candles) < 10:
        return {'signal': False, 'confidence': 0}

    closes = np.array([float(c[4]) for c in candles])
    current_close = closes[-1]

    trend_data = indicators['trend_following']
    macd_data = indicators['macd_analysis']
    volume_data = indicators['volume_analysis']

    conditions = {
        'price_above_emas': trend_data['price_above_emas'],
        'macd_bullish': macd_data['histogram_current'] > 0 and macd_data['histogram_increasing'],
        'volume_confirmation': volume_data['volume_ratio'] > config.indicators.VOLUME_SPIKE_RATIO,
        'momentum_positive': indicators['momentum_trend']['price_momentum_5'] > 0,
        'no_overbought': indicators['oscillators']['rsi14_current'] < config.indicators.RSI_OVERBOUGHT
    }

    if sum(conditions.values()) >= 4:  # Строже: 4 из 5
        base_confidence = config.patterns.PATTERN_BASE_CONFIDENCE['MOMENTUM_BREAKOUT']

        return {
            'signal': True,
            'signal_type': 'LONG',
            'confidence': min(90, base_confidence + (sum(conditions.values()) * 2)),
            'pattern': 'MOMENTUM_BREAKOUT',
            'conditions': conditions
        }

    return {'signal': False, 'confidence': 0}


def detect_squeeze_breakout_enhanced(candles: List[List[str]], indicators: Dict) -> Dict[str, Any]:
    """Улучшенный детектор Squeeze Breakout"""
    if len(candles) < 30:
        return {'signal': False, 'confidence': 0}

    volatility_data = indicators['volatility']
    volume_data = indicators['volume_analysis']

    conditions = {
        'bb_squeeze_detected': volatility_data['bb_squeeze'],
        'breakout_occurred': volatility_data['bb_breakout']['upper_breakout'] or volatility_data['bb_breakout']['lower_breakout'],
        'volume_spike': volume_data['volume_spike'],
        'atr_rising': volatility_data['volatility_regime'] != 'LOW_VOLATILITY'
    }

    if sum(conditions.values()) >= 3:  # 3 из 4 (более реалистично)
        signal_type = 'LONG' if volatility_data['bb_breakout']['upper_breakout'] else 'SHORT'
        base_confidence = config.patterns.PATTERN_BASE_CONFIDENCE['SQUEEZE_BREAKOUT']

        return {
            'signal': True,
            'signal_type': signal_type,
            'confidence': min(85, base_confidence + (sum(conditions.values()) * 3)),
            'pattern': 'SQUEEZE_BREAKOUT',
            'conditions': conditions
        }

    return {'signal': False, 'confidence': 0}


def detect_range_scalp_enhanced(candles: List[List[str]], indicators: Dict) -> Dict[str, Any]:
    """Улучшенный детектор Range Scalp с консервативными настройками"""
    if len(candles) < 50:
        return {'signal': False, 'confidence': 0}

    sr_data = indicators['support_resistance']
    oscillators = indicators['oscillators']
    trend_data = indicators.get('momentum_trend', {})

    # Только в слабом тренде или боковике
    if trend_data.get('trend_strength') in ['VERY_STRONG', 'STRONG']:
        return {'signal': False, 'confidence': 0}

    conditions = {
        'near_support': sr_data['distance_to_support'] < 1.0,  # Ближе 1%
        'near_resistance': sr_data['distance_to_resistance'] < 1.0,
        'rsi_extreme': (oscillators['rsi14_current'] < config.indicators.RSI_OVERSOLD or
                       oscillators['rsi14_current'] > config.indicators.RSI_OVERBOUGHT),
        'stoch_extreme': (oscillators['stoch_k_current'] < 25 or oscillators['stoch_k_current'] > 75)
    }

    if (conditions['near_support'] and oscillators['rsi14_current'] < config.indicators.RSI_OVERSOLD):
        signal_type = 'LONG'
        confidence = config.patterns.PATTERN_BASE_CONFIDENCE['RANGE_SCALP']
    elif (conditions['near_resistance'] and oscillators['rsi14_current'] > config.indicators.RSI_OVERBOUGHT):
        signal_type = 'SHORT'
        confidence = config.patterns.PATTERN_BASE_CONFIDENCE['RANGE_SCALP']
    else:
        return {'signal': False, 'confidence': 0}

    return {
        'signal': True,
        'signal_type': signal_type,
        'confidence': confidence,
        'pattern': 'RANGE_SCALP',
        'conditions': conditions
    }


def validate_signal_enhanced(candles_5m: List[List[str]], candles_15m: List[List[str]],
                           signal_data: Dict, indicators: Dict, htf_analysis: Dict) -> Dict[str, Any]:
    """РАСШИРЕННАЯ валидация сигнала (4 из 5 проверок)"""
    if not candles_5m or not candles_15m:
        return {'score': '0/5', 'valid': False, 'checks': {}}

    signal_direction = signal_data.get('signal_type', 'NONE')

    # 5 проверок валидации (усиленные)
    checks = {}

    # 1. Тренд 15m СТРОГО совпадает с направлением входа
    htf_trend = htf_analysis['trend']
    if signal_direction == 'LONG':
        checks['higher_tf_trend_aligned'] = htf_trend in ['UPTREND', 'STRONG_UPTREND']
    elif signal_direction == 'SHORT':
        checks['higher_tf_trend_aligned'] = htf_trend in ['DOWNTREND', 'STRONG_DOWNTREND']
    else:
        checks['higher_tf_trend_aligned'] = False

    # 2. Последние 2 свечи подтверждают направление
    if len(candles_5m) >= 3:
        last_closes = [float(c[4]) for c in candles_5m[-3:]]
        if signal_direction == 'LONG':
            checks['candle_confirmation'] = last_closes[-1] > last_closes[-2] and last_closes[-2] >= last_closes[-3]
        elif signal_direction == 'SHORT':
            checks['candle_confirmation'] = last_closes[-1] < last_closes[-2] and last_closes[-2] <= last_closes[-3]
        else:
            checks['candle_confirmation'] = False
    else:
        checks['candle_confirmation'] = False

    # 3. Объем ЗНАЧИТЕЛЬНО выше среднего
    volume_ratio = indicators['volume_analysis']['volume_ratio']
    checks['volume_strong_confirmation'] = volume_ratio >= config.indicators.VOLUME_MIN_RATIO * 1.2

    # 4. ATR в оптимальном диапазоне (не слишком высокий, не слишком низкий)
    atr_data = indicators['volatility']
    checks['atr_optimal'] = (atr_data['atr_current'] >= atr_data['atr_mean'] * config.indicators.ATR_OPTIMAL_RATIO and
                            atr_data['volatility_regime'] != 'HIGH_VOLATILITY')

    # 5. Множественные осцилляторы подтверждают направление
    oscillators = indicators['oscillators']
    if signal_direction == 'LONG':
        osc_confirmations = sum([
            oscillators['rsi14_current'] > 45,
            oscillators['stoch_k_current'] > 25,
            not oscillators['rsi_overbought'],
            indicators['macd_analysis']['histogram_current'] > 0
        ])
    elif signal_direction == 'SHORT':
        osc_confirmations = sum([
            oscillators['rsi14_current'] < 55,
            oscillators['stoch_k_current'] < 75,
            not oscillators['rsi_oversold'],
            indicators['macd_analysis']['histogram_current'] < 0
        ])
    else:
        osc_confirmations = 0

    checks['oscillators_aligned'] = osc_confirmations >= 3

    # Подсчет результата
    passed_checks = sum(checks.values())
    total_checks = len(checks)

    return {
        'score': f'{passed_checks}/{total_checks}',
        'valid': passed_checks >= config.trading.VALIDATION_CHECKS_REQUIRED,  # 4 из 5
        'checks': checks,
        'passed': passed_checks,
        'total': total_checks
    }


def apply_confidence_modifiers(base_confidence: int, validation: Dict,
                             indicators: Dict, htf_analysis: Dict) -> int:
    """Применение всех модификаторов уверенности"""
    final_confidence = base_confidence

    # Модификаторы из конфига
    if validation['checks'].get('higher_tf_trend_aligned', False):
        final_confidence *= config.scoring.CONFIDENCE_MODIFIERS['higher_tf_aligned']

    if indicators['volume_analysis']['volume_spike']:
        final_confidence *= config.scoring.CONFIDENCE_MODIFIERS['volume_spike']

    if validation['passed'] == validation['total']:
        final_confidence *= config.scoring.CONFIDENCE_MODIFIERS['validation_perfect']

    # Новые модификаторы
    if htf_analysis['trend'] in ['STRONG_UPTREND', 'STRONG_DOWNTREND']:
        final_confidence *= config.scoring.CONFIDENCE_MODIFIERS['strong_trend_context']

    if indicators['volatility']['volatility_regime'] == 'NORMAL_VOLATILITY':
        final_confidence *= config.scoring.CONFIDENCE_MODIFIERS['low_volatility_risk']

    return min(100, int(final_confidence))


def meets_conservative_criteria(indicators: Dict, htf_analysis: Dict, signal_direction: str) -> bool:
    """Проверка консервативных критериев безопасности"""

    # 1. Минимальная ликвидность
    if indicators['volume_analysis']['volume_ratio'] < config.indicators.VOLUME_MIN_RATIO:
        return False

    # 2. Волатильность не экстремальная
    if indicators['volatility']['volatility_regime'] == 'HIGH_VOLATILITY':
        return False

    # 3. RSI не в экстремальных зонах
    rsi_current = indicators['oscillators']['rsi14_current']
    if rsi_current < 25 or rsi_current > 75:
        return False

    # 4. Тренд старшего ТФ поддерживает сигнал
    htf_confidence = htf_analysis.get('confidence', 0)
    if htf_confidence < 60:
        return False

    # 5. Нет сильной дивергенции
    if (indicators['oscillators'].get('rsi_divergence', False) and
        indicators['macd_analysis'].get('macd_divergence', False)):
        return False

    return True


def calculate_risk_metrics(indicators: Dict, candles: List[List[str]]) -> Dict[str, Any]:
    """Расчет метрик риска для сигнала"""
    current_price = float(candles[-1][4])
    atr_current = indicators['volatility']['atr_current']

    # Рекомендуемый стоп-лосс (консервативный)
    stop_loss_atr = atr_current * config.indicators.ATR_MULTIPLIER_STOP
    stop_loss_percent = max(config.trading.MIN_STOP_LOSS_PERCENT,
                           (stop_loss_atr / current_price) * 100)

    # Рекомендуемый тейк-профит
    take_profit_percent = min(config.trading.MAX_TAKE_PROFIT_PERCENT,
                             stop_loss_percent * config.trading.DEFAULT_RISK_REWARD)

    return {
        'stop_loss_percent': safe_float(stop_loss_percent),
        'take_profit_percent': safe_float(take_profit_percent),
        'risk_reward_ratio': safe_float(take_profit_percent / stop_loss_percent),
        'atr_based_stop': safe_float(stop_loss_atr),
        'position_risk_score': calculate_position_risk_score(indicators),
        'recommended_position_size': calculate_recommended_position_size(stop_loss_percent)
    }


def calculate_position_risk_score(indicators: Dict) -> int:
    """Оценка риска позиции от 1 (низкий) до 10 (высокий)"""
    risk_factors = {
        'high_volatility': indicators['volatility']['volatility_regime'] == 'HIGH_VOLATILITY',
        'low_volume': indicators['volume_analysis']['volume_ratio'] < 1.2,
        'extreme_rsi': indicators['oscillators']['rsi14_current'] < 25 or indicators['oscillators']['rsi14_current'] > 75,
        'divergence_present': (indicators['oscillators'].get('rsi_divergence', False) or
                              indicators['macd_analysis'].get('macd_divergence', False)),
        'weak_trend': indicators.get('momentum_trend', {}).get('trend_strength') == 'WEAK'
    }

    risk_score = 1 + sum(risk_factors.values()) * 1.8  # 1-10 scale
    return min(10, int(risk_score))


def calculate_recommended_position_size(stop_loss_percent: float) -> float:
    """Расчет рекомендуемого размера позиции"""
    base_size = config.trading.DEFAULT_POSITION_SIZE_PERCENT

    # Уменьшаем размер при большем стопе
    if stop_loss_percent > 0.8:
        size_multiplier = 0.7
    elif stop_loss_percent > 0.6:
        size_multiplier = 0.85
    else:
        size_multiplier = 1.0

    return min(config.trading.MAX_POSITION_SIZE_PERCENT,
               safe_float(base_size * size_multiplier))


def generate_entry_reasons(pattern: str, htf_analysis: Dict, indicators: Dict) -> List[str]:
    """Генерация детальных причин для входа"""
    reasons = [
        f"{pattern} pattern confirmed with {htf_analysis.get('confidence', 0)}% HTF confidence",
        f"Higher timeframe trend: {htf_analysis.get('trend', 'UNKNOWN')}",
        f"Volume ratio: {indicators['volume_analysis']['volume_ratio']:.2f}x average",
        f"RSI(14): {indicators['oscillators']['rsi14_current']:.1f}",
        f"ATR volatility: {indicators['volatility']['volatility_regime']}",
        f"MACD histogram: {indicators['macd_analysis']['histogram_current']:.6f}",
        f"Trend strength: {indicators.get('momentum_trend', {}).get('trend_strength', 'UNKNOWN')}"
    ]

    # Добавляем специфичные для паттерна причины
    if pattern == 'PULLBACK_ENTRY':
        reasons.append(f"Price near EMA20: {indicators['trend_following']['ema20_current']:.6f}")
        reasons.append("Pullback in established trend")
    elif pattern == 'MOMENTUM_BREAKOUT':
        reasons.append("Strong momentum confirmed by multiple indicators")
        reasons.append(f"Price above all EMAs: {indicators['trend_following']['price_above_emas']}")
    elif pattern == 'SQUEEZE_BREAKOUT':
        reasons.append("Bollinger Bands squeeze detected")
        reasons.append(f"Breakout with volume spike: {indicators['volume_analysis']['volume_spike']}")
    elif pattern == 'RANGE_SCALP':
        reasons.append(f"Near support/resistance levels")
        reasons.append(f"RSI in extreme zone for mean reversion")

    return reasons


# Функции для обратной совместимости со старым кодом
def calculate_indicators_by_instruction(candles: List[List[str]]) -> Dict[str, Any]:
    """
    Обертка для обратной совместимости - использует новую функцию с максимальными данными
    """
    comprehensive_data = calculate_all_indicators_comprehensive(candles)

    if not comprehensive_data:
        return {}

    # Конвертируем в старый формат для совместимости
    return {
        'ema5': comprehensive_data['trend_following']['ema5'],
        'ema8': comprehensive_data['trend_following']['ema8'],
        'ema20': comprehensive_data['trend_following']['ema20'],
        'rsi': comprehensive_data['oscillators']['rsi14'],
        'rsi_current': comprehensive_data['oscillators']['rsi14_current'],
        'macd_line': comprehensive_data['macd_analysis']['macd_line'],
        'macd_signal': comprehensive_data['macd_analysis']['macd_signal'],
        'macd_histogram': comprehensive_data['macd_analysis']['macd_histogram'],
        'atr': comprehensive_data['volatility']['atr'],
        'atr_current': comprehensive_data['volatility']['atr_current'],
        'atr_mean': comprehensive_data['volatility']['atr_mean'],
        'bb_upper': comprehensive_data['volatility']['bb_upper'],
        'bb_middle': comprehensive_data['volatility']['bb_middle'],
        'bb_lower': comprehensive_data['volatility']['bb_lower'],
        'volume_sma': comprehensive_data['volume_analysis']['volume_sma'],
        'volume_current': comprehensive_data['volume_analysis']['current_volume'],
        'volume_ratio': comprehensive_data['volume_analysis']['volume_ratio']
    }


def detect_instruction_based_signals(candles_5m: List[List[str]],
                                   candles_15m: List[List[str]]) -> Dict[str, Any]:
    """
    Обертка для обратной совместимости - использует новую улучшенную функцию
    """
    return detect_instruction_based_signals_enhanced(candles_5m, candles_15m)