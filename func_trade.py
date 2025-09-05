import numpy as np
from typing import List, Dict, Any, Tuple
import time
import math

# ЭКСТРЕМАЛЬНЫЕ ПАРАМЕТРЫ ДЛЯ 5M СКАЛЬПИНГА (МАКСИМАЛЬНЫЙ ПРОФИТ)
SCALPING_PARAMS = {
    # TSI (True Strength Index) - лучший momentum для скальпинга
    'tsi_double_smooth': 13,
    'tsi_single_smooth': 25,

    # Быстрый MACD (Linda Raschke настройки для скальпинга)
    'macd_fast': 3,  # Сверхбыстрый
    'macd_slow': 10,  # Быстрый
    'macd_signal': 16,  # Сигнальный

    # VWAP - критичен для определения справедливой цены
    'vwap_period': 20,

    # DeMarker - отличный осциллятор для разворотов
    'demarker_period': 8,

    # Ускоренный RSI
    'rsi_period': 3,  # Сверхбыстрый для 5M

    # EMA для тренда
    'ema_fast': 8,  # Быстрая
    'ema_slow': 21,  # Медленная

    # Супер-быстрый ATR
    'atr_period': 5,

    # Объемы
    'volume_lookback': 5,
    'min_confidence': 75  # Повышен для качества
}


def safe_float(value):
    """Безопасное преобразование в float с обработкой NaN"""
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


def safe_bool(value):
    """Безопасное преобразование в bool"""
    try:
        return bool(value)
    except:
        return False


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
    if len(prices) < period:
        return np.zeros_like(prices)

    ema = np.zeros_like(prices)
    alpha = 2.0 / (period + 1)
    ema[0] = prices[0]

    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema


def calculate_tsi(prices: np.ndarray, double_smooth: int = 13, single_smooth: int = 25) -> np.ndarray:
    """True Strength Index - лучший momentum индикатор для скальпинга"""
    if len(prices) < double_smooth + single_smooth:
        return np.zeros_like(prices)

    # Рассчитываем изменения цены
    price_changes = np.diff(prices, prepend=prices[0])

    # Двойное сглаживание цен
    pc_smooth1 = calculate_ema(price_changes, single_smooth)
    pc_smooth2 = calculate_ema(pc_smooth1, double_smooth)

    # Двойное сглаживание абсолютных изменений
    abs_pc_smooth1 = calculate_ema(np.abs(price_changes), single_smooth)
    abs_pc_smooth2 = calculate_ema(abs_pc_smooth1, double_smooth)

    # TSI = 100 * (smooth2 / abs_smooth2)
    tsi = np.zeros_like(prices)
    for i in range(len(prices)):
        if abs_pc_smooth2[i] != 0:
            tsi[i] = 100 * (pc_smooth2[i] / abs_pc_smooth2[i])

    return tsi


def calculate_linda_macd(prices: np.ndarray, fast: int = 3, slow: int = 10, signal: int = 16) -> Dict[str, np.ndarray]:
    """Linda Raschke MACD настройки - экстремально быстрый для скальпинга"""
    if len(prices) < slow:
        return {'macd': np.zeros_like(prices), 'signal': np.zeros_like(prices), 'histogram': np.zeros_like(prices)}

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


def calculate_vwap(candles: List[List[str]], period: int = 20) -> np.ndarray:
    """Volume Weighted Average Price - критично для справедливой цены"""
    if len(candles) < period:
        return np.zeros(len(candles))

    vwap = np.zeros(len(candles))

    for i in range(period, len(candles)):
        total_volume = 0
        total_pv = 0

        for j in range(i - period + 1, i + 1):
            high = safe_float(candles[j][2])
            low = safe_float(candles[j][3])
            close = safe_float(candles[j][4])
            volume = safe_float(candles[j][5])

            typical_price = (high + low + close) / 3
            total_pv += typical_price * volume
            total_volume += volume

        if total_volume > 0:
            vwap[i] = total_pv / total_volume

    return vwap


def calculate_demarker(candles: List[List[str]], period: int = 8) -> np.ndarray:
    """DeMarker индикатор - отличный для определения разворотов"""
    if len(candles) < period + 1:
        return np.zeros(len(candles))

    de_max = []
    de_min = []

    for i in range(1, len(candles)):
        high = safe_float(candles[i][2])
        low = safe_float(candles[i][3])
        prev_high = safe_float(candles[i - 1][2])
        prev_low = safe_float(candles[i - 1][3])

        de_max.append(max(0, high - prev_high))
        de_min.append(max(0, prev_low - low))

    de_max = np.array(de_max)
    de_min = np.array(de_min)

    demarker = np.zeros(len(candles))

    for i in range(period - 1, len(de_max)):
        sma_max = np.mean(de_max[i - period + 1:i + 1])
        sma_min = np.mean(de_min[i - period + 1:i + 1])

        if sma_max + sma_min > 0:
            demarker[i + 1] = sma_max / (sma_max + sma_min)

    return demarker


def calculate_super_rsi(prices: np.ndarray, period: int = 3) -> np.ndarray:
    """Сверхбыстрый RSI для 5M скальпинга"""
    if len(prices) < period + 1:
        return np.full_like(prices, 50.0)

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    rsi = np.full_like(prices, 50.0)

    # Простое скользящее среднее для первого значения
    if len(gains) >= period:
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[period] = 100 - (100 / (1 + rs))

        # EMA для последующих значений
        alpha = 1.0 / period
        for i in range(period + 1, len(prices)):
            avg_gain = alpha * gains[i - 1] + (1 - alpha) * avg_gain
            avg_loss = alpha * losses[i - 1] + (1 - alpha) * avg_loss

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))

    return rsi


def calculate_price_momentum(prices: np.ndarray, period: int = 3) -> float:
    """Моментум цены за период"""
    if len(prices) < period:
        return 0.0

    return safe_float((prices[-1] - prices[-period]) / prices[-period] * 100 if prices[-period] > 0 else 0.0)


def calculate_volume_acceleration(volumes: List[float], period: int = 3) -> float:
    """Ускорение объема"""
    if len(volumes) < period * 2:
        return 0.0

    recent_avg = np.mean(volumes[-period:])
    prev_avg = np.mean(volumes[-period * 2:-period])

    if prev_avg > 0:
        return safe_float((recent_avg - prev_avg) / prev_avg * 100)
    return 0.0


def analyze_volume_profile(candles: List[List[str]], lookback: int = 5) -> Dict[str, Any]:
    """Усиленный анализ объемов для 5M"""
    if len(candles) < lookback:
        return {'spike': False, 'ratio': 1.0, 'strength': 0, 'acceleration': 0.0}

    volumes = [safe_float(c[5]) for c in candles]
    current_volume = volumes[-1]
    avg_volume = np.mean(volumes[-lookback - 1:-1])  # Исключаем текущую свечу

    if avg_volume == 0:
        return {'spike': False, 'ratio': 1.0, 'strength': 0, 'acceleration': 0.0}

    ratio = current_volume / avg_volume
    acceleration = calculate_volume_acceleration(volumes, 3)

    return {
        'spike': safe_bool(ratio > 1.8),  # Повышен порог для 5M
        'ratio': safe_float(ratio),
        'strength': safe_int(min(100, (ratio - 1) * 40)),
        'acceleration': safe_float(acceleration)
    }


def calculate_scalping_indicators(candles: List[List[str]]) -> Dict[str, Any]:
    """
    ГЛАВНАЯ ФУНКЦИЯ: Расчет ЛУЧШИХ индикаторов для 5M скальпинга
    Заточена под максимальный профит
    """
    if len(candles) < 30:
        return {}

    # Извлекаем цены
    prices = np.array([safe_float(c[4]) for c in candles])
    highs = np.array([safe_float(c[2]) for c in candles])
    lows = np.array([safe_float(c[3]) for c in candles])
    volumes = [safe_float(c[5]) for c in candles]

    # TSI - КОРОЛЬ momentum индикаторов
    tsi = calculate_tsi(prices, SCALPING_PARAMS['tsi_double_smooth'], SCALPING_PARAMS['tsi_single_smooth'])

    # Linda Raschke MACD - сверхбыстрые настройки
    macd = calculate_linda_macd(prices, SCALPING_PARAMS['macd_fast'],
                                SCALPING_PARAMS['macd_slow'], SCALPING_PARAMS['macd_signal'])

    # VWAP - справедливая цена
    vwap = calculate_vwap(candles, SCALPING_PARAMS['vwap_period'])

    # DeMarker - лучший для разворотов
    demarker = calculate_demarker(candles, SCALPING_PARAMS['demarker_period'])

    # Сверхбыстрый RSI
    rsi = calculate_super_rsi(prices, SCALPING_PARAMS['rsi_period'])

    # Быстрые EMA
    ema_fast = calculate_ema(prices, SCALPING_PARAMS['ema_fast'])
    ema_slow = calculate_ema(prices, SCALPING_PARAMS['ema_slow'])

    # ATR для волатильности
    atr_data = calculate_atr_fast(candles, SCALPING_PARAMS['atr_period'])

    # Объемный анализ
    volume_data = analyze_volume_profile(candles, SCALPING_PARAMS['volume_lookback'])

    # Моментум метрики
    price_momentum = calculate_price_momentum(prices, 3)

    # Логические сигналы
    tsi_current = safe_float(tsi[-1] if len(tsi) > 0 else 0.0)
    tsi_bullish = safe_bool(len(tsi) >= 2 and tsi[-1] > tsi[-2] and tsi_current > 0)
    tsi_bearish = safe_bool(len(tsi) >= 2 and tsi[-1] < tsi[-2] and tsi_current < 0)

    ema_trend = 'BULLISH' if len(ema_fast) > 0 and len(ema_slow) > 0 and ema_fast[-1] > ema_slow[-1] else 'BEARISH'

    vwap_position = 'ABOVE' if len(vwap) > 0 and vwap[-1] > 0 and prices[-1] > vwap[-1] else 'BELOW'

    demarker_current = safe_float(demarker[-1] if len(demarker) > 0 else 0.5)
    demarker_signal = 'OVERSOLD' if demarker_current < 0.3 else ('OVERBOUGHT' if demarker_current > 0.7 else 'NEUTRAL')

    macd_crossover = detect_macd_crossover(macd)

    return {
        # TSI - основной momentum
        'tsi_values': safe_list(tsi),
        'tsi_current': tsi_current,
        'tsi_bullish': tsi_bullish,
        'tsi_bearish': tsi_bearish,

        # Linda Raschke MACD
        'macd_line': safe_list(macd['macd']),
        'macd_signal': safe_list(macd['signal']),
        'macd_histogram': safe_list(macd['histogram']),
        'macd_crossover': macd_crossover,

        # VWAP
        'vwap_values': safe_list(vwap),
        'vwap_position': vwap_position,
        'vwap_distance': safe_float((prices[-1] - vwap[-1]) / vwap[-1] * 100 if len(vwap) > 0 and vwap[-1] > 0 else 0),

        # DeMarker
        'demarker_values': safe_list(demarker),
        'demarker_current': demarker_current,
        'demarker_signal': demarker_signal,

        # Быстрый RSI
        'rsi_values': safe_list(rsi),
        'rsi_current': safe_float(rsi[-1] if len(rsi) > 0 else 50.0),

        # EMA тренд
        'ema_fast': safe_list(ema_fast),
        'ema_slow': safe_list(ema_slow),
        'ema_trend': ema_trend,
        'ema_distance': safe_float(
            (ema_fast[-1] - ema_slow[-1]) / ema_slow[-1] * 100 if len(ema_fast) > 0 and len(ema_slow) > 0 and ema_slow[
                -1] > 0 else 0),

        # Волатильность
        'atr_values': safe_list(atr_data['atr']),
        'atr_current': safe_float(atr_data['current']),
        'volatility_regime': classify_volatility(atr_data['current'], prices[-1]),

        # Объемы
        'volume_spike': volume_data['spike'],
        'volume_ratio': volume_data['ratio'],
        'volume_strength': volume_data['strength'],
        'volume_acceleration': volume_data['acceleration'],

        # Моментум
        'price_momentum': price_momentum,
        'momentum_strength': safe_int(abs(price_momentum) * 20),  # 0-100 шкала

        # Общие метрики
        'signal_quality': safe_int(
            calculate_signal_quality_extreme(candles, tsi, macd, demarker, volume_data, vwap, prices))
    }


def calculate_atr_fast(candles: List[List[str]], period: int = 5) -> Dict[str, Any]:
    """Сверхбыстрый ATR для 5M"""
    if len(candles) < period + 1:
        return {'atr': [], 'current': 0.0}

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

    atr = np.zeros(len(candles))
    atr[period] = np.mean(tr[1:period + 1])

    for i in range(period + 1, len(candles)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return {
        'atr': safe_list(atr),
        'current': safe_float(atr[-1] if len(atr) > 0 else 0.0)
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
    """Классификация режима волатильности для 5M"""
    if price == 0:
        return 'UNKNOWN'

    atr_percent = (atr / price) * 100

    if atr_percent > 1.5:  # Пороги снижены для 5M
        return 'HIGH'
    elif atr_percent > 0.7:
        return 'MEDIUM'
    else:
        return 'LOW'


def calculate_signal_quality_extreme(candles: List[List[str]], tsi: np.ndarray, macd: Dict,
                                     demarker: np.ndarray, volume_data: Dict, vwap: np.ndarray,
                                     prices: np.ndarray) -> int:
    """ЭКСТРЕМАЛЬНАЯ оценка качества сигнала для максимального профита"""
    quality = 0

    # TSI подтверждение (30 баллов)
    if len(tsi) >= 2:
        tsi_current = safe_float(tsi[-1])
        if abs(tsi_current) > 5 and abs(tsi_current) < 25:  # Оптимальная зона для входа
            quality += 30
        elif abs(tsi_current) > 25:  # Экстремальные значения
            quality += 20

    # MACD подтверждение (25 баллов)
    macd_cross = detect_macd_crossover(macd)
    if macd_cross in ['BULLISH', 'BEARISH']:
        quality += 25
    elif len(macd['histogram']) > 0 and abs(macd['histogram'][-1]) > 0.001:
        quality += 15

    # DeMarker экстремумы (20 баллов)
    if len(demarker) > 0:
        dm_current = safe_float(demarker[-1])
        if dm_current < 0.3 or dm_current > 0.7:  # Экстремальные зоны
            quality += 20
        elif 0.4 < dm_current < 0.6:  # Нейтральная зона
            quality += 10

    # VWAP позиция (15 баллов)
    if len(vwap) > 0 and vwap[-1] > 0:
        distance = abs((prices[-1] - vwap[-1]) / vwap[-1] * 100)
        if 0.1 < distance < 1.0:  # Оптимальное отклонение
            quality += 15

    # Объемное подтверждение (10 баллов)
    if volume_data.get('spike', False):
        quality += 10
    elif volume_data.get('ratio', 1.0) > 1.5:
        quality += 5

    return min(100, safe_int(quality))


def detect_scalping_signal(candles: List[List[str]]) -> Dict[str, Any]:
    """
    ЭКСТРЕМАЛЬНЫЙ ДЕТЕКТОР СИГНАЛОВ для 5M скальпинга
    Настроен на максимальный профит
    """
    if len(candles) < 30:
        return {'signal': 'NO_SIGNAL', 'confidence': 0, 'reason': 'INSUFFICIENT_DATA'}

    # Рассчитываем индикаторы
    indicators = calculate_scalping_indicators(candles)

    if not indicators:
        return {'signal': 'NO_SIGNAL', 'confidence': 0, 'reason': 'CALCULATION_ERROR'}

    # Определяем сигнал
    signal_type = 'NO_SIGNAL'
    confidence = 0
    entry_reasons = []

    # BULLISH условия (экстремально жёсткие для максимального профита)
    bullish_score = 0

    # TSI бычий
    if indicators.get('tsi_bullish', False) and indicators.get('tsi_current', 0) > 5:
        bullish_score += 2
        entry_reasons.append('TSI_BULLISH_MOMENTUM')

    # MACD бычье пересечение
    if indicators.get('macd_crossover', 'NONE') == 'BULLISH':
        bullish_score += 2
        entry_reasons.append('MACD_BULLISH_CROSS')

    # EMA тренд + дистанция
    if indicators.get('ema_trend', 'BEARISH') == 'BULLISH' and indicators.get('ema_distance', 0) > 0.2:
        bullish_score += 1
        entry_reasons.append('EMA_STRONG_BULLISH')

    # DeMarker перепроданность
    if indicators.get('demarker_signal', 'NEUTRAL') == 'OVERSOLD':
        bullish_score += 2
        entry_reasons.append('DEMARKER_OVERSOLD')

    # VWAP поддержка
    if indicators.get('vwap_position', 'BELOW') == 'ABOVE' and indicators.get('vwap_distance', 0) > 0.1:
        bullish_score += 1
        entry_reasons.append('VWAP_SUPPORT')

    # BEARISH условия
    bearish_score = 0

    # TSI медвежий
    if indicators.get('tsi_bearish', False) and indicators.get('tsi_current', 0) < -5:
        bearish_score += 2
        entry_reasons.append('TSI_BEARISH_MOMENTUM')

    # MACD медвежье пересечение
    if indicators.get('macd_crossover', 'NONE') == 'BEARISH':
        bearish_score += 2
        entry_reasons.append('MACD_BEARISH_CROSS')

    # EMA тренд + дистанция
    if indicators.get('ema_trend', 'BULLISH') == 'BEARISH' and indicators.get('ema_distance', 0) < -0.2:
        bearish_score += 1
        entry_reasons.append('EMA_STRONG_BEARISH')

    # DeMarker перекупленность
    if indicators.get('demarker_signal', 'NEUTRAL') == 'OVERBOUGHT':
        bearish_score += 2
        entry_reasons.append('DEMARKER_OVERBOUGHT')

    # VWAP сопротивление
    if indicators.get('vwap_position', 'ABOVE') == 'BELOW' and indicators.get('vwap_distance', 0) < -0.1:
        bearish_score += 1
        entry_reasons.append('VWAP_RESISTANCE')

    # Определяем финальный сигнал (повышены требования)
    if bullish_score >= 4:  # Минимум 4 подтверждения
        signal_type = 'LONG'
        confidence = min(100, bullish_score * 15 + indicators.get('signal_quality', 0) * 0.4)
    elif bearish_score >= 4:  # Минимум 4 подтверждения
        signal_type = 'SHORT'
        confidence = min(100, bearish_score * 15 + indicators.get('signal_quality', 0) * 0.4)

    # Дополнительные фильтры качества
    if signal_type != 'NO_SIGNAL':
        # Объемное подтверждение ОБЯЗАТЕЛЬНО
        if not indicators.get('volume_spike', False) and indicators.get('volume_ratio', 1.0) < 1.8:
            confidence *= 0.5  # Жёсткое наказание за слабый объем

        # Высокая волатильность = бонус
        volatility = indicators.get('volatility_regime', 'MEDIUM')
        if volatility == 'HIGH':
            confidence *= 1.2
        elif volatility == 'LOW':
            confidence *= 0.7

        # Моментум должен быть сильным
        momentum_strength = indicators.get('momentum_strength', 0)
        if momentum_strength < 30:
            confidence *= 0.8

    # Финальная проверка минимального качества
    if confidence < SCALPING_PARAMS['min_confidence']:
        signal_type = 'NO_SIGNAL'
        confidence = 0

    return {
        'signal': signal_type,
        'confidence': safe_int(confidence),
        'entry_reasons': entry_reasons,
        'indicators': indicators,
        'quality_score': indicators.get('signal_quality', 0),
        'volatility_regime': indicators.get('volatility_regime', 'MEDIUM'),
        'momentum_strength': indicators.get('momentum_strength', 0)
    }