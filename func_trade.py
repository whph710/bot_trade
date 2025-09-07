"""
Модуль технического анализа и определения торговых сигналов
Реализует мультитаймфреймный анализ согласно инструкции
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

    Args:
        prices: Массив цен
        period: Период EMA

    Returns:
        np.ndarray: Массив значений EMA
    """
    ema = np.zeros_like(prices)
    alpha = 2.0 / (period + 1)
    ema[0] = prices[0]

    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema


def calculate_rsi(prices: np.ndarray, period: int = 9) -> np.ndarray:
    """
    Расчет RSI(9) для фильтра импульса согласно инструкции

    Args:
        prices: Массив цен
        period: Период RSI (по умолчанию 9)

    Returns:
        np.ndarray: Массив значений RSI
    """
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
    """
    Расчет MACD стандарт (12,26,9) согласно инструкции

    Args:
        prices: Массив цен
        fast: Быстрый период
        slow: Медленный период
        signal: Период сигнальной линии

    Returns:
        dict: Словарь с MACD линией, сигнальной линией и гистограммой
    """
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

    Args:
        candles: Список свечей
        period: Период ATR (по умолчанию 14)

    Returns:
        dict: Словарь с массивом ATR и текущим значением
    """
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
    """
    Расчет Bollinger Bands (20,2) согласно инструкции

    Args:
        prices: Массив цен
        period: Период для расчета (по умолчанию 20)
        std: Количество стандартных отклонений (по умолчанию 2)

    Returns:
        dict: Словарь с верхней, средней и нижней полосами
    """
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
    """
    Расчет Volume SMA(20) для подтверждения пробоев

    Args:
        candles: Список свечей
        period: Период SMA (по умолчанию 20)

    Returns:
        np.ndarray: Массив значений Volume SMA
    """
    volumes = np.array([float(c[5]) for c in candles])
    volume_sma = np.zeros_like(volumes)

    for i in range(period - 1, len(volumes)):
        volume_sma[i] = np.mean(volumes[i - period + 1:i + 1])

    return volume_sma


def calculate_indicators_by_instruction(candles: List[List[str]]) -> Dict[str, Any]:
    """
    Расчет всех индикаторов согласно инструкции

    Args:
        candles: Список свечей 5m таймфрейма

    Returns:
        dict: Словарь со всеми рассчитанными индикаторами
    """
    if len(candles) < 50:
        return {}

    closes = np.array([float(c[4]) for c in candles])
    volumes = np.array([float(c[5]) for c in candles])

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
    bb_data = calculate_bollinger_bands(closes, SCALPING_PARAMS['bb_period'], SCALPING_PARAMS['bb_std'])

    # Volume SMA(20)
    volume_sma = calculate_volume_sma(candles, SCALPING_PARAMS['volume_sma'])

    return {
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
        'atr_mean': safe_float(np.mean(atr_data['atr'][-20:]) if len(atr_data['atr']) >= 20 else 0),
        'bb_upper': safe_list(bb_data['upper']),
        'bb_middle': safe_list(bb_data['middle']),
        'bb_lower': safe_list(bb_data['lower']),
        'volume_sma': safe_list(volume_sma),
        'volume_current': safe_float(volumes[-1]),
        'volume_ratio': safe_float(volumes[-1] / volume_sma[-1] if volume_sma[-1] > 0 else 1.0)
    }


def analyze_higher_timeframe_trend(candles_15m: List[List[str]]) -> Dict[str, Any]:
    """
    Анализ старшего таймфрейма (15m) для определения контекста

    Args:
        candles_15m: Список 15-минутных свечей

    Returns:
        dict: Информация о тренде старшего таймфрейма
    """
    if len(candles_15m) < 30:
        return {'trend': 'UNKNOWN', 'strength': 0}

    closes = np.array([float(c[4]) for c in candles_15m])
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

    return {
        'trend': trend,
        'strength': min(100, abs(strength)),
        'ema20': safe_float(ema20_current),
        'ema50': safe_float(ema50_current)
    }


def detect_momentum_breakout(candles: List[List[str]], indicators: Dict) -> Dict[str, Any]:
    """
    Шаблон A: Momentum breakout (импульсный вход)
    Условия: цена выше EMA20 + MACD гистограмма >0 + объем выше среднего

    Args:
        candles: Список свечей
        indicators: Рассчитанные индикаторы

    Returns:
        dict: Информация о сигнале
    """
    if len(candles) < 5:
        return {'signal': False, 'confidence': 0}

    closes = np.array([float(c[4]) for c in candles])
    current_close = closes[-1]

    ema20 = indicators.get('ema20', [])
    macd_histogram = indicators.get('macd_histogram', [])
    volume_ratio = indicators.get('volume_ratio', 1.0)

    if not ema20 or not macd_histogram:
        return {'signal': False, 'confidence': 0}

    # Проверяем условия momentum breakout
    conditions = {
        'price_above_ema20': current_close > ema20[-1],
        'macd_positive': macd_histogram[-1] > 0,
        'volume_spike': volume_ratio > 1.5,
        'ema_alignment': len(indicators.get('ema5', [])) > 0 and
                         len(indicators.get('ema8', [])) > 0 and
                         indicators['ema5'][-1] > indicators['ema8'][-1] > ema20[-1]
    }

    signal = all(conditions.values())
    confidence = sum(conditions.values()) * 20  # Каждое условие = 20%

    return {
        'signal': signal,
        'confidence': confidence,
        'pattern': 'MOMENTUM_BREAKOUT',
        'conditions': conditions
    }


def detect_pullback_entry(candles: List[List[str]], indicators: Dict) -> Dict[str, Any]:
    """
    Шаблон B: Pullback to EMA (откат к EMA)
    Условия: откат к EMA8-20 в тренде + свечная формация + RSI восстановление

    Args:
        candles: Список свечей
        indicators: Рассчитанные индикаторы

    Returns:
        dict: Информация о сигнале
    """
    if len(candles) < 10:
        return {'signal': False, 'confidence': 0}

    closes = np.array([float(c[4]) for c in candles])
    current_close = closes[-1]

    ema8 = indicators.get('ema8', [])
    ema20 = indicators.get('ema20', [])
    rsi_current = indicators.get('rsi_current', 50)

    if not ema8 or not ema20:
        return {'signal': False, 'confidence': 0}

    # Проверяем близость к EMA
    ema8_proximity = abs(current_close - ema8[-1]) / current_close < config.patterns.PULLBACK_EMA_PROXIMITY
    ema20_proximity = abs(current_close - ema20[-1]) / current_close < config.patterns.PULLBACK_EMA_PROXIMITY

    # Условия pullback entry
    conditions = {
        'near_ema': ema8_proximity or ema20_proximity,
        'rsi_recovery_long': 30 < rsi_current < config.patterns.PULLBACK_RSI_RECOVERY,
        'rsi_recovery_short': config.patterns.PULLBACK_RSI_WEAK < rsi_current < 70,
        'trend_alignment': len(indicators.get('ema5', [])) > 0 and
                           indicators['ema5'][-1] > ema8[-1]
    }

    # Определяем направление
    if conditions['near_ema'] and conditions['rsi_recovery_long'] and conditions['trend_alignment']:
        signal_type = 'LONG'
        confidence = 75
    elif conditions['near_ema'] and conditions['rsi_recovery_short']:
        signal_type = 'SHORT'
        confidence = 70
    else:
        signal_type = False
        confidence = 0

    return {
        'signal': signal_type != False,
        'signal_type': signal_type,
        'confidence': confidence,
        'pattern': 'PULLBACK_ENTRY',
        'conditions': conditions
    }


def detect_squeeze_breakout(candles: List[List[str]], indicators: Dict) -> Dict[str, Any]:
    """
    Шаблон C: Squeeze breakout (сжатие Bollinger)
    Условия: сжатие Bollinger + пробой с объемом + ATR растет

    Args:
        candles: Список свечей
        indicators: Рассчитанные индикаторы

    Returns:
        dict: Информация о сигнале
    """
    if len(candles) < 20:
        return {'signal': False, 'confidence': 0}

    closes = np.array([float(c[4]) for c in candles])
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

    # Условия squeeze breakout
    conditions = {
        'squeeze_detected': squeeze,
        'breakout_occurred': breakout_up or breakout_down,
        'volume_confirmation': volume_ratio > config.indicators.VOLUME_SPIKE_RATIO,
        'atr_rising': len(atr) >= 3 and atr[-1] > atr[-3]
    }

    signal = all(conditions.values())
    confidence = sum(conditions.values()) * 20

    return {
        'signal': signal,
        'confidence': confidence,
        'pattern': 'SQUEEZE_BREAKOUT',
        'conditions': conditions,
        'breakout_direction': 'LONG' if breakout_up else 'SHORT' if breakout_down else 'NONE'
    }


def detect_range_scalp(candles: List[List[str]], indicators: Dict) -> Dict[str, Any]:
    """
    Шаблон D: Range scalp (скальпинг в боковике)
    Условия: боковик + цена у границ + RSI экстремум

    Args:
        candles: Список свечей
        indicators: Рассчитанные индикаторы

    Returns:
        dict: Информация о сигнале
    """
    if len(candles) < 20:
        return {'signal': False, 'confidence': 0}

    closes = np.array([float(c[4]) for c in candles])
    highs = np.array([float(c[2]) for c in candles])
    lows = np.array([float(c[3]) for c in candles])

    current_close = closes[-1]
    rsi_current = indicators.get('rsi_current', 50)

    # Определяем диапазон (последние 20 свечей)
    range_high = np.max(highs[-20:])
    range_low = np.min(lows[-20:])
    range_size = (range_high - range_low) / current_close * 100

    # Проверяем условия диапазона
    near_resistance = abs(current_close - range_high) / current_close < config.patterns.RANGE_BOUNDARY_PROXIMITY
    near_support = abs(current_close - range_low) / current_close < config.patterns.RANGE_BOUNDARY_PROXIMITY

    # Условия range scalp
    conditions = {
        'range_size_adequate': range_size > config.patterns.RANGE_MIN_SIZE_PERCENT,
        'near_boundary': near_resistance or near_support,
        'rsi_oversold': rsi_current < config.indicators.RSI_OVERSOLD,
        'rsi_overbought': rsi_current > config.indicators.RSI_OVERBOUGHT
    }

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

    return {
        'signal': signal,
        'signal_type': signal_type,
        'confidence': confidence,
        'pattern': 'RANGE_SCALP',
        'conditions': conditions,
        'range_levels': {'high': range_high, 'low': range_low}
    }


def validate_signal(candles_5m: List[List[str]], candles_15m: List[List[str]],
                    signal_data: Dict, indicators: Dict) -> Dict[str, Any]:
    """
    Валидация сигнала согласно инструкции (5 из 5 проверок)

    Args:
        candles_5m: Свечи 5m
        candles_15m: Свечи 15m
        signal_data: Данные о сигнале
        indicators: Рассчитанные индикаторы

    Returns:
        dict: Результат валидации
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
                                     candles_15m: List[List[str]]) -> Dict[str, Any]:
    """
    Основная функция определения сигналов согласно инструкции
    Мультитаймфреймный анализ: 15m контекст + 5m вход

    Args:
        candles_5m: Свечи 5m таймфрейма для точного входа
        candles_15m: Свечи 15m таймфрейма для контекста

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

    # Рассчитываем индикаторы для 5m
    indicators = calculate_indicators_by_instruction(candles_5m)
    if not indicators:
        return {
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'pattern_type': 'NONE',
            'validation_score': '0/5'
        }

    # Анализируем старший таймфрейм
    htf_analysis = analyze_higher_timeframe_trend(candles_15m)

    # Проверяем все шаблоны входа согласно приоритету
    patterns = [
        ('MOMENTUM_BREAKOUT', detect_momentum_breakout(candles_5m, indicators)),
        ('SQUEEZE_BREAKOUT', detect_squeeze_breakout(candles_5m, indicators)),
        ('PULLBACK_ENTRY', detect_pullback_entry(candles_5m, indicators)),
        ('RANGE_SCALP', detect_range_scalp(candles_5m, indicators))
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
            'indicators': indicators
        }

    # Определяем направление сигнала
    signal_direction = best_signal.get('signal_type',
                                       best_signal.get('breakout_direction', 'LONG'))

    # Валидируем сигнал (5 из 5 проверок)
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
            'indicators': indicators
        }

    return {
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