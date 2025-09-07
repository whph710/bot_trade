"""
Упрощенный модуль технических индикаторов
Только самые прибыльные и проверенные индикаторы
"""

import numpy as np
from typing import List, Dict, Tuple
from config import config


def calculate_ema(prices: List[float], period: int) -> List[float]:
    """Экспоненциальная скользящая средняя"""
    if len(prices) < period:
        return [0.0] * len(prices)

    prices_array = np.array(prices, dtype=float)
    ema = np.zeros_like(prices_array)

    # Первое значение = простая средняя
    ema[period - 1] = np.mean(prices_array[:period])

    # Коэффициент сглаживания
    alpha = 2.0 / (period + 1)

    # Рассчитываем остальные значения
    for i in range(period, len(prices)):
        ema[i] = alpha * prices_array[i] + (1 - alpha) * ema[i - 1]

    return ema.tolist()


def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """Индекс относительной силы"""
    if len(prices) < period + 1:
        return [50.0] * len(prices)

    prices_array = np.array(prices, dtype=float)
    deltas = np.diff(prices_array)

    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    rsi = np.full(len(prices), 50.0)

    # Первое значение RSI
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    # Остальные значения
    for i in range(period + 1, len(prices)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi.tolist()


def calculate_atr(candles: List[List], period: int = 14) -> List[float]:
    """Average True Range - волатильность"""
    if len(candles) < period + 1:
        return [0.0] * len(candles)

    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    closes = [float(c[4]) for c in candles]

    true_ranges = [0.0]  # Первое значение всегда 0

    for i in range(1, len(candles)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i - 1])
        tr3 = abs(lows[i] - closes[i - 1])
        true_ranges.append(max(tr1, tr2, tr3))

    atr = [0.0] * len(candles)

    # Первое значение ATR - простая средняя
    if len(true_ranges) >= period + 1:
        atr[period] = np.mean(true_ranges[1:period + 1])

        # Остальные значения - экспоненциальная средняя
        for i in range(period + 1, len(candles)):
            atr[i] = (atr[i - 1] * (period - 1) + true_ranges[i]) / period

    return atr


def calculate_volume_ratio(candles: List[List], period: int = 20) -> float:
    """Отношение текущего объема к среднему"""
    if len(candles) < period:
        return 1.0

    volumes = [float(c[5]) for c in candles]
    current_volume = volumes[-1]
    avg_volume = np.mean(volumes[-period:])

    return current_volume / avg_volume if avg_volume > 0 else 1.0


def analyze_trend(candles: List[List]) -> Dict:
    """
    Простой но эффективный анализ тренда
    Основан на EMA и структуре цены
    """
    if len(candles) < config.trading.EMA_SLOW:
        return {
            'direction': 'UNKNOWN',
            'strength': 0,
            'quality': 'POOR'
        }

    closes = [float(c[4]) for c in candles]
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]

    ema_fast = calculate_ema(closes, config.trading.EMA_FAST)
    ema_slow = calculate_ema(closes, config.trading.EMA_SLOW)

    current_price = closes[-1]
    current_fast = ema_fast[-1]
    current_slow = ema_slow[-1]

    # Определяем направление тренда
    if current_price > current_fast > current_slow:
        direction = 'UP'
        strength = ((current_price - current_slow) / current_slow) * 100
    elif current_price < current_fast < current_slow:
        direction = 'DOWN'
        strength = ((current_slow - current_price) / current_slow) * 100
    else:
        direction = 'SIDEWAYS'
        strength = 0

    # Оценка качества тренда
    if strength > 2.0:
        quality = 'STRONG'
    elif strength > 1.0:
        quality = 'MODERATE'
    elif strength > 0.5:
        quality = 'WEAK'
    else:
        quality = 'POOR'

    return {
        'direction': direction,
        'strength': min(100, abs(strength)),
        'quality': quality,
        'ema_fast': current_fast,
        'ema_slow': current_slow
    }


def detect_breakout(candles: List[List]) -> Dict:
    """
    Детекция пробоя - один из самых прибыльных сигналов
    """
    if len(candles) < 20:
        return {'signal': False, 'direction': 'NONE', 'strength': 0}

    closes = [float(c[4]) for c in candles]
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    volumes = [float(c[5]) for c in candles]

    current_price = closes[-1]
    current_volume = volumes[-1]
    avg_volume = np.mean(volumes[-20:])

    # Ищем уровни сопротивления/поддержки за последние 20 свечей
    recent_highs = highs[-20:]
    recent_lows = lows[-20:]

    resistance_level = np.percentile(recent_highs, 90)
    support_level = np.percentile(recent_lows, 10)

    volume_confirmed = current_volume > avg_volume * config.trading.MIN_VOLUME_RATIO

    # Пробой сопротивления (LONG сигнал)
    if current_price > resistance_level and volume_confirmed:
        strength = ((current_price - resistance_level) / resistance_level) * 100
        return {
            'signal': True,
            'direction': 'LONG',
            'strength': min(100, strength * 50),
            'level': resistance_level
        }

    # Пробой поддержки (SHORT сигнал)
    elif current_price < support_level and volume_confirmed:
        strength = ((support_level - current_price) / support_level) * 100
        return {
            'signal': True,
            'direction': 'SHORT',
            'strength': min(100, strength * 50),
            'level': support_level
        }

    return {'signal': False, 'direction': 'NONE', 'strength': 0}


def detect_pullback(candles: List[List]) -> Dict:
    """
    Детекция пуллбэка - высокоточный сигнал входа
    """
    if len(candles) < config.trading.EMA_SLOW + 10:
        return {'signal': False, 'direction': 'NONE', 'confidence': 0}

    closes = [float(c[4]) for c in candles]
    rsi = calculate_rsi(closes)
    ema_fast = calculate_ema(closes, config.trading.EMA_FAST)
    ema_slow = calculate_ema(closes, config.trading.EMA_SLOW)

    current_price = closes[-1]
    current_rsi = rsi[-1]
    current_fast = ema_fast[-1]
    current_slow = ema_slow[-1]

    # Условия для LONG пуллбэка
    long_conditions = [
        current_fast > current_slow,  # Основной тренд вверх
        30 < current_rsi < 50,  # RSI в зоне перепроданности но не критичной
        current_price < current_fast,  # Цена ниже быстрой EMA (пуллбэк)
        abs(current_price - current_fast) / current_price < 0.02  # Неглубокий пуллбэк
    ]

    # Условия для SHORT пуллбэка
    short_conditions = [
        current_fast < current_slow,  # Основной тренд вниз
        50 < current_rsi < 70,  # RSI в зоне перекупленности но не критичной
        current_price > current_fast,  # Цена выше быстрой EMA (пуллбэк)
        abs(current_price - current_fast) / current_price < 0.02  # Неглубокий пуллбэк
    ]

    if sum(long_conditions) >= 3:
        return {
            'signal': True,
            'direction': 'LONG',
            'confidence': sum(long_conditions) * 25,
            'entry_level': current_fast
        }

    if sum(short_conditions) >= 3:
        return {
            'signal': True,
            'direction': 'SHORT',
            'confidence': sum(short_conditions) * 25,
            'entry_level': current_fast
        }

    return {'signal': False, 'direction': 'NONE', 'confidence': 0}


def calculate_all_indicators(candles: List[List]) -> Dict:
    """
    Расчет всех необходимых индикаторов
    Оптимизировано для производительности
    """
    if len(candles) < config.trading.EMA_SLOW:
        return {}

    closes = [float(c[4]) for c in candles]

    # Основные индикаторы
    ema_fast = calculate_ema(closes, config.trading.EMA_FAST)
    ema_slow = calculate_ema(closes, config.trading.EMA_SLOW)
    rsi = calculate_rsi(closes, config.trading.RSI_PERIOD)
    atr = calculate_atr(candles, config.trading.ATR_PERIOD)
    volume_ratio = calculate_volume_ratio(candles, config.trading.VOLUME_PERIOD)

    # Анализ тренда и паттернов
    trend_analysis = analyze_trend(candles)
    breakout_analysis = detect_breakout(candles)
    pullback_analysis = detect_pullback(candles)

    return {
        'ema_fast': ema_fast,
        'ema_slow': ema_slow,
        'rsi': rsi,
        'atr': atr,
        'volume_ratio': volume_ratio,
        'current_price': closes[-1],
        'current_rsi': rsi[-1] if rsi else 50,
        'current_atr': atr[-1] if atr else 0,
        'trend': trend_analysis,
        'breakout': breakout_analysis,
        'pullback': pullback_analysis
    }


def generate_trading_signal(candles_5m: List[List], candles_15m: List[List]) -> Dict:
    """
    Генерация торгового сигнала на основе мультитаймфреймного анализа
    Упрощенная но эффективная логика
    """
    if len(candles_5m) < 50 or len(candles_15m) < 30:
        return {
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'reason': 'Insufficient data'
        }

    # Анализ высшего таймфрейма (15m) - определяем основной тренд
    htf_indicators = calculate_all_indicators(candles_15m)
    htf_trend = htf_indicators.get('trend', {})

    # Анализ младшего таймфрейма (5m) - ищем точку входа
    ltf_indicators = calculate_all_indicators(candles_5m)

    # Проверяем базовые условия
    volume_ok = ltf_indicators.get('volume_ratio', 0) >= config.trading.MIN_VOLUME_RATIO
    atr_ok = ltf_indicators.get('current_atr', 0) > 0

    if not (volume_ok and atr_ok):
        return {
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'reason': 'Volume or volatility too low'
        }

    # Ищем лучшие сигналы
    signals = []

    # 1. Пробой с подтверждением тренда
    breakout = ltf_indicators.get('breakout', {})
    if (breakout.get('signal') and
            breakout.get('direction') == htf_trend.get('direction', 'UNKNOWN').replace('UP', 'LONG').replace('DOWN',
                                                                                                             'SHORT')):
        signals.append({
            'type': 'BREAKOUT',
            'direction': breakout['direction'],
            'confidence': min(95, breakout.get('strength', 0) + htf_trend.get('strength', 0)),
            'reason': 'Breakout confirmed by higher timeframe trend'
        })

    # 2. Пуллбэк в направлении тренда
    pullback = ltf_indicators.get('pullback', {})
    if (pullback.get('signal') and
            pullback.get('direction') == htf_trend.get('direction', 'UNKNOWN').replace('UP', 'LONG').replace('DOWN',
                                                                                                             'SHORT')):
        signals.append({
            'type': 'PULLBACK',
            'direction': pullback['direction'],
            'confidence': min(90, pullback.get('confidence', 0) + (htf_trend.get('strength', 0) // 2)),
            'reason': 'Pullback entry in strong trend'
        })

    # 3. Простой тренд-следующий сигнал
    if htf_trend.get('quality') in ['STRONG', 'MODERATE']:
        ltf_trend = ltf_indicators.get('trend', {})
        if (ltf_trend.get('direction') == htf_trend.get('direction') and
                ltf_indicators.get('current_rsi', 50) not in [range(20), range(80, 101)]):
            direction = 'LONG' if htf_trend['direction'] == 'UP' else 'SHORT'
            signals.append({
                'type': 'TREND_FOLLOW',
                'direction': direction,
                'confidence': min(85, htf_trend.get('strength', 0)),
                'reason': 'Multi-timeframe trend alignment'
            })

    # Выбираем лучший сигнал
    if not signals:
        return {
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'reason': 'No qualifying patterns found'
        }

    best_signal = max(signals, key=lambda x: x['confidence'])

    if best_signal['confidence'] < config.trading.MIN_CONFIDENCE:
        return {
            'signal': 'NO_SIGNAL',
            'confidence': best_signal['confidence'],
            'reason': f"Confidence too low: {best_signal['confidence']}"
        }

    return {
        'signal': best_signal['direction'],
        'confidence': best_signal['confidence'],
        'pattern': best_signal['type'],
        'reason': best_signal['reason'],
        'indicators': {
            'htf_trend': htf_trend,
            'current_price': ltf_indicators.get('current_price'),
            'current_rsi': ltf_indicators.get('current_rsi'),
            'current_atr': ltf_indicators.get('current_atr'),
            'volume_ratio': ltf_indicators.get('volume_ratio')
        }
    }