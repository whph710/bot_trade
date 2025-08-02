import numpy as np
from typing import List, Dict, Any, Tuple
import time

# СКАЛЬПИНГОВЫЕ ПАРАМЕТРЫ (УСКОРЕННЫЕ ДЛЯ 15M)
SCALPING_PARAMS = {
    'tema_fast': 3,  # Очень быстрая TEMA
    'tema_medium': 5,  # Средняя TEMA
    'tema_slow': 8,  # Медленная TEMA
    'rsi_period': 5,  # Быстрый RSI
    'stoch_k': 5,  # Stochastic %K
    'stoch_d': 3,  # Stochastic %D
    'macd_fast': 5,  # Быстрая MACD
    'macd_slow': 13,  # Медленная MACD
    'macd_signal': 5,  # Сигнальная MACD
    'atr_period': 8,  # Период ATR
    'volume_lookback': 8,  # Анализ объемов
    'min_confidence': 70  # Минимальная уверенность
}


def calculate_tema(prices: np.ndarray, period: int) -> np.ndarray:
    """Triple Exponential Moving Average - самый быстрый тренд"""
    ema1 = calculate_ema(prices, period)
    ema2 = calculate_ema(ema1, period)
    ema3 = calculate_ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average"""
    ema = np.zeros_like(prices)
    alpha = 2.0 / (period + 1)
    ema[0] = prices[0]

    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema


def calculate_fast_rsi(prices: np.ndarray, period: int = 5) -> np.ndarray:
    """Быстрый RSI для скальпинга"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gains = np.zeros_like(prices)
    avg_losses = np.zeros_like(prices)

    # Используем SMA для первого значения
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


def calculate_stochastic(candles: List[List[str]], k_period: int = 5, d_period: int = 3) -> Dict[str, np.ndarray]:
    """Stochastic Oscillator для momentum"""
    highs = np.array([float(c[2]) for c in candles])
    lows = np.array([float(c[3]) for c in candles])
    closes = np.array([float(c[4]) for c in candles])

    k_values = np.zeros_like(closes)

    for i in range(k_period - 1, len(closes)):
        highest_high = np.max(highs[i - k_period + 1:i + 1])
        lowest_low = np.min(lows[i - k_period + 1:i + 1])

        if highest_high != lowest_low:
            k_values[i] = (closes[i] - lowest_low) / (highest_high - lowest_low) * 100
        else:
            k_values[i] = 50

    # %D - сглаженная версия %K
    d_values = calculate_ema(k_values, d_period)

    return {'k': k_values, 'd': d_values}


def calculate_macd_fast(prices: np.ndarray, fast: int = 5, slow: int = 13, signal: int = 5) -> Dict[str, np.ndarray]:
    """Быстрая MACD для скальпинга"""
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


def calculate_price_velocity(prices: np.ndarray, period: int = 3) -> float:
    """Скорость изменения цены"""
    if len(prices) < period:
        return 0.0

    recent_change = abs(prices[-1] - prices[-period])
    avg_price = np.mean(prices[-period:])

    return (recent_change / avg_price * 100) if avg_price > 0 else 0.0


def calculate_momentum_acceleration(prices: np.ndarray) -> float:
    """Ускорение моментума"""
    if len(prices) < 6:
        return 0.0

    # Momentum за последние 3 свечи
    recent_momentum = prices[-1] - prices[-4]
    # Momentum за предыдущие 3 свечи
    prev_momentum = prices[-4] - prices[-7] if len(prices) >= 7 else 0

    return abs(recent_momentum - prev_momentum) / prices[-1] * 100 if prices[-1] > 0 else 0.0


def analyze_volume_spike(candles: List[List[str]], lookback: int = 8) -> Dict[str, Any]:
    """Анализ объемных всплесков для скальпинга"""
    if len(candles) < lookback:
        return {'spike': False, 'ratio': 1.0, 'strength': 0}

    volumes = np.array([float(c[5]) for c in candles])
    current_volume = volumes[-1]
    avg_volume = np.mean(volumes[-lookback:-1])  # Исключаем текущую свечу

    if avg_volume == 0:
        return {'spike': False, 'ratio': 1.0, 'strength': 0}

    ratio = current_volume / avg_volume

    return {
        'spike': ratio > 1.5,  # Снижен порог для скальпинга
        'ratio': ratio,
        'strength': min(100, (ratio - 1) * 50)  # 0-100 шкала
    }


def find_micro_support_resistance(candles: List[List[str]], window: int = 8) -> Dict[str, List[float]]:
    """Поиск микро-уровней поддержки/сопротивления"""
    if len(candles) < window:
        return {'support': [], 'resistance': []}

    highs = np.array([float(c[2]) for c in candles])
    lows = np.array([float(c[3]) for c in candles])

    support_levels = []
    resistance_levels = []

    # Ищем локальные экстремумы в окне
    for i in range(2, len(lows) - 2):
        # Локальный минимум
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i - 2] and lows[i] <= lows[i + 1] and lows[i] <= lows[i + 2]:
            support_levels.append(float(lows[i]))

        # Локальный максимум
        if highs[i] >= highs[i - 1] and highs[i] >= highs[i - 2] and highs[i] >= highs[i + 1] and highs[i] >= highs[
            i + 2]:
            resistance_levels.append(float(highs[i]))

    return {
        'support': sorted(set(support_levels))[-3:],  # Последние 3 уровня
        'resistance': sorted(set(resistance_levels))[-3:]
    }


def calculate_scalping_indicators(candles: List[List[str]]) -> Dict[str, Any]:
    """
    ГЛАВНАЯ ФУНКЦИЯ: Расчет всех индикаторов для скальпинга 15M
    Оптимизирована для удержания 3-4 свечи
    """
    if len(candles) < 20:
        return {}

    # Извлекаем цены
    prices = np.array([float(c[4]) for c in candles])

    # Быстрые трендовые индикаторы
    tema3 = calculate_tema(prices, SCALPING_PARAMS['tema_fast'])
    tema5 = calculate_tema(prices, SCALPING_PARAMS['tema_medium'])
    tema8 = calculate_tema(prices, SCALPING_PARAMS['tema_slow'])

    # Momentum индикаторы
    rsi = calculate_fast_rsi(prices, SCALPING_PARAMS['rsi_period'])
    stoch = calculate_stochastic(candles, SCALPING_PARAMS['stoch_k'], SCALPING_PARAMS['stoch_d'])
    macd = calculate_macd_fast(prices, SCALPING_PARAMS['macd_fast'],
                               SCALPING_PARAMS['macd_slow'], SCALPING_PARAMS['macd_signal'])

    # Волатильность
    atr_data = calculate_atr_fast(candles, SCALPING_PARAMS['atr_period'])

    # Объемный анализ
    volume_data = analyze_volume_spike(candles, SCALPING_PARAMS['volume_lookback'])

    # Микроструктура
    levels = find_micro_support_resistance(candles[-16:])  # Только последние 16 свечей

    # Динамические метрики
    price_velocity = calculate_price_velocity(prices[-8:])
    momentum_accel = calculate_momentum_acceleration(prices[-8:])

    return {
        # Тренд
        'tema3_values': tema3.tolist(),
        'tema5_values': tema5.tolist(),
        'tema8_values': tema8.tolist(),
        'tema_alignment': tema3[-1] > tema5[-1] > tema8[-1] if len(tema3) > 0 else False,
        'tema_slope': (tema3[-1] - tema3[-2]) / tema3[-2] * 100 if len(tema3) >= 2 else 0,

        # Momentum
        'rsi_values': rsi.tolist(),
        'rsi_current': rsi[-1] if len(rsi) > 0 else 50,
        'stoch_k': stoch['k'].tolist(),
        'stoch_d': stoch['d'].tolist(),
        'stoch_signal': 'OVERSOLD' if stoch['k'][-1] < 20 else 'OVERBOUGHT' if stoch['k'][-1] > 80 else 'NEUTRAL',

        # MACD
        'macd_line': macd['macd'].tolist(),
        'macd_signal': macd['signal'].tolist(),
        'macd_histogram': macd['histogram'].tolist(),
        'macd_crossover': detect_macd_crossover(macd),

        # Волатильность
        'atr_values': atr_data['atr'],
        'atr_current': atr_data['current'],
        'volatility_regime': classify_volatility(atr_data['current'], prices[-1]),

        # Объемы
        'volume_spike': volume_data['spike'],
        'volume_ratio': volume_data['ratio'],
        'volume_strength': volume_data['strength'],

        # Уровни
        'support_levels': levels['support'],
        'resistance_levels': levels['resistance'],
        'near_support': is_near_level(prices[-1], levels['support'], 0.3),
        'near_resistance': is_near_level(prices[-1], levels['resistance'], 0.3),

        # Микроструктура
        'price_velocity': price_velocity,
        'momentum_acceleration': momentum_accel,
        'trend_strength': calculate_trend_strength(tema3, tema5, tema8),

        # Общие метрики
        'signal_quality': calculate_signal_quality_fast(candles, prices, rsi, stoch, volume_data)
    }


def calculate_atr_fast(candles: List[List[str]], period: int = 8) -> Dict[str, Any]:
    """Быстрый расчет ATR"""
    if len(candles) < period + 1:
        return {'atr': [], 'current': 0}

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
    atr[period] = np.mean(tr[1:period + 1])

    for i in range(period + 1, len(candles)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return {
        'atr': atr.tolist(),
        'current': atr[-1] if len(atr) > 0 else 0
    }


def detect_macd_crossover(macd: Dict[str, np.ndarray]) -> str:
    """Определение пересечения MACD"""
    if len(macd['macd']) < 2 or len(macd['signal']) < 2:
        return 'NONE'

    macd_line = macd['macd']
    signal_line = macd['signal']

    # Бычье пересечение
    if macd_line[-2] <= signal_line[-2] and macd_line[-1] > signal_line[-1]:
        return 'BULLISH'

    # Медвежье пересечение
    if macd_line[-2] >= signal_line[-2] and macd_line[-1] < signal_line[-1]:
        return 'BEARISH'

    return 'NONE'


def classify_volatility(atr: float, price: float) -> str:
    """Классификация режима волатильности"""
    if price == 0:
        return 'UNKNOWN'

    atr_percent = (atr / price) * 100

    if atr_percent > 2.0:
        return 'HIGH'
    elif atr_percent > 1.0:
        return 'MEDIUM'
    else:
        return 'LOW'


def is_near_level(price: float, levels: List[float], threshold: float = 0.3) -> bool:
    """Проверка близости к уровню"""
    if not levels or price == 0:
        return False

    for level in levels:
        distance = abs(price - level) / price * 100
        if distance <= threshold:
            return True

    return False


def calculate_trend_strength(tema3: np.ndarray, tema5: np.ndarray, tema8: np.ndarray) -> int:
    """Расчет силы тренда (0-100)"""
    if len(tema3) < 5:
        return 0

    # Проверяем выравнивание TEMA
    alignment_score = 0
    if tema3[-1] > tema5[-1] > tema8[-1]:  # Восходящий тренд
        alignment_score = 40
    elif tema3[-1] < tema5[-1] < tema8[-1]:  # Нисходящий тренд
        alignment_score = 40

    # Проверяем консистентность направления
    consistency_score = 0
    tema3_rising = all(tema3[i] > tema3[i - 1] for i in range(-3, 0))
    tema3_falling = all(tema3[i] < tema3[i - 1] for i in range(-3, 0))

    if tema3_rising or tema3_falling:
        consistency_score = 30

    # Проверяем угол наклона
    slope_score = 0
    if len(tema3) >= 5:
        slope = abs((tema3[-1] - tema3[-5]) / tema3[-5] * 100)
        slope_score = min(30, slope * 10)

    return int(alignment_score + consistency_score + slope_score)


def calculate_signal_quality_fast(candles: List[List[str]], prices: np.ndarray,
                                  rsi: np.ndarray, stoch: Dict, volume_data: Dict) -> int:
    """Быстрая оценка качества сигнала (0-100)"""
    quality = 0

    # Объемное подтверждение (25 баллов)
    if volume_data['spike']:
        quality += 25
    elif volume_data['ratio'] > 1.2:
        quality += 15

    # RSI в оптимальной зоне (25 баллов)
    current_rsi = rsi[-1] if len(rsi) > 0 else 50
    if 30 < current_rsi < 70:
        quality += 25
    elif current_rsi <= 30 or current_rsi >= 70:
        quality += 15  # Экстремальные значения тоже хороши для скальпинга

    # Stochastic подтверждение (25 баллов)
    if len(stoch['k']) > 0 and len(stoch['d']) > 0:
        k_current = stoch['k'][-1]
        d_current = stoch['d'][-1]

        if k_current < 20 and k_current > d_current:  # Выход из перепроданности
            quality += 25
        elif k_current > 80 and k_current < d_current:  # Выход из перекупленности
            quality += 25
        elif 20 < k_current < 80:  # Нейтральная зона
            quality += 15

    # Ценовая активность (25 баллов)
    if len(prices) >= 5:
        recent_range = max(prices[-5:]) - min(prices[-5:])
        avg_price = np.mean(prices[-5:])
        activity = (recent_range / avg_price * 100) if avg_price > 0 else 0

        if 0.5 < activity < 3.0:  # Оптимальная активность для скальпинга
            quality += 25
        elif activity > 0.2:
            quality += 15

    return min(100, quality)


def detect_scalping_signal(candles: List[List[str]]) -> Dict[str, Any]:
    """
    ОСНОВНАЯ ФУНКЦИЯ ОПРЕДЕЛЕНИЯ СИГНАЛОВ ДЛЯ СКАЛЬПИНГА
    Заменяет enhanced_signal_detection для более быстрой работы
    """
    if len(candles) < 30:
        return {'signal': 'NO_SIGNAL', 'confidence': 0, 'reason': 'INSUFFICIENT_DATA'}

    # Рассчитываем индикаторы
    indicators = calculate_scalping_indicators(candles)

    if not indicators:
        return {'signal': 'NO_SIGNAL', 'confidence': 0, 'reason': 'CALCULATION_ERROR'}

    # Определяем сигнал на основе быстрых индикаторов
    signal_type = 'NO_SIGNAL'
    confidence = 0
    entry_reasons = []

    # BULLISH условия
    bullish_conditions = 0
    if indicators.get('tema_alignment') and indicators.get('tema_slope', 0) > 0.1:
        bullish_conditions += 1
        entry_reasons.append('TEMA_BULLISH')

    if indicators.get('macd_crossover') == 'BULLISH':
        bullish_conditions += 1
        entry_reasons.append('MACD_BULLISH')

    if indicators.get('rsi_current', 50) < 40:  # RSI показывает потенциал роста
        bullish_conditions += 1
        entry_reasons.append('RSI_OVERSOLD')

    if indicators.get('stoch_signal') == 'OVERSOLD' and indicators.get('stoch_k', [])[-1] > \
            indicators.get('stoch_d', [])[-1]:
        bullish_conditions += 1
        entry_reasons.append('STOCH_REVERSAL')

    # BEARISH условия
    bearish_conditions = 0
    if not indicators.get('tema_alignment') and indicators.get('tema_slope', 0) < -0.1:
        bearish_conditions += 1
        entry_reasons.append('TEMA_BEARISH')

    if indicators.get('macd_crossover') == 'BEARISH':
        bearish_conditions += 1
        entry_reasons.append('MACD_BEARISH')

    if indicators.get('rsi_current', 50) > 60:  # RSI показывает потенциал падения
        bearish_conditions += 1
        entry_reasons.append('RSI_OVERBOUGHT')

    if indicators.get('stoch_signal') == 'OVERBOUGHT' and indicators.get('stoch_k', [])[-1] < \
            indicators.get('stoch_d', [])[-1]:
        bearish_conditions += 1
        entry_reasons.append('STOCH_REVERSAL')

    # Определяем сигнал
    if bullish_conditions >= 2:
        signal_type = 'LONG'
        confidence = min(100, bullish_conditions * 25 + indicators.get('signal_quality', 0) * 0.3)
    elif bearish_conditions >= 2:
        signal_type = 'SHORT'
        confidence = min(100, bearish_conditions * 25 + indicators.get('signal_quality', 0) * 0.3)

    # Дополнительные проверки качества
    if signal_type != 'NO_SIGNAL':
        # Объемное подтверждение обязательно
        if not indicators.get('volume_spike', False) and indicators.get('volume_ratio', 1.0) < 1.3:
            confidence *= 0.7

        # Проверка близости к уровням (может быть как плюсом, так и минусом)
        if indicators.get('near_support') or indicators.get('near_resistance'):
            if signal_type == 'LONG' and indicators.get('near_support'):
                confidence *= 1.1  # Отскок от поддержки
            elif signal_type == 'SHORT' and indicators.get('near_resistance'):
                confidence *= 1.1  # Отскок от сопротивления
            else:
                confidence *= 0.9  # Торговля против уровня

        # Режим волатильности
        volatility = indicators.get('volatility_regime', 'MEDIUM')
        if volatility == 'HIGH':
            confidence *= 1.1  # Высокая волатильность хороша для скальпинга
        elif volatility == 'LOW':
            confidence *= 0.8  # Низкая волатильность хуже

    # Финальная проверка минимального качества
    if confidence < SCALPING_PARAMS['min_confidence']:
        signal_type = 'NO_SIGNAL'
        confidence = 0

    return {
        'signal': signal_type,
        'confidence': int(confidence),
        'entry_reasons': entry_reasons,
        'indicators': indicators,
        'quality_score': indicators.get('signal_quality', 0),
        'volatility_regime': indicators.get('volatility_regime', 'MEDIUM')
    }