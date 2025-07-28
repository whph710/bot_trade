import numpy as np
from typing import List, Dict, Any, Tuple


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Расчет Exponential Moving Average (точно как в Pine Script)

    Args:
        prices: Массив цен
        period: Период EMA

    Returns:
        Массив значений EMA
    """
    ema = np.zeros_like(prices)
    alpha = 2.0 / (period + 1)

    # Первое значение EMA равно первой цене
    ema[0] = prices[0]

    # Расчет последующих значений EMA
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema


def double_smooth(src: np.ndarray, long_period: int, short_period: int) -> np.ndarray:
    """
    Двойное сглаживание с помощью EMA (как в Pine Script)

    Args:
        src: Исходные данные
        long_period: Длинный период для первого сглаживания
        short_period: Короткий период для второго сглаживания

    Returns:
        Массив двойно сглаженных значений
    """
    # Первое сглаживание
    first_smooth = calculate_ema(src, long_period)

    # Второе сглаживание
    second_smooth = calculate_ema(first_smooth, short_period)

    return second_smooth


def calculate_tsi(candles: List[List[str]],
                  long_length: int = 25,
                  short_length: int = 13,
                  signal_length: int = 13) -> Dict[str, np.ndarray]:
    """
    Расчет True Strength Index (TSI) - индикатор момента тренда

    Args:
        candles: Данные свечей в формате Bybit [timestamp, open, high, low, close, volume, turnover]
        long_length: Длинный период сглаживания (по умолчанию 25)
        short_length: Короткий период сглаживания (по умолчанию 13)
        signal_length: Период сигнальной линии (по умолчанию 13)

    Returns:
        Словарь с массивами:
        - 'tsi': значения TSI
        - 'signal': сигнальная линия (EMA от TSI)
    """
    if len(candles) < max(long_length, short_length, signal_length) + 10:
        return {
            'tsi': np.array([]),
            'signal': np.array([])
        }

    # Извлекаем цены закрытия
    prices = np.array([float(candle[4]) for candle in candles])

    # Вычисляем изменения цены (price change)
    pc = np.diff(prices, prepend=prices[0])  # Первое значение = 0

    # Двойное сглаживание изменений цены
    double_smoothed_pc = double_smooth(pc, long_length, short_length)

    # Двойное сглаживание абсолютных изменений цены
    abs_pc = np.abs(pc)
    double_smoothed_abs_pc = double_smooth(abs_pc, long_length, short_length)

    # Рассчитываем TSI (избегаем деление на ноль)
    tsi_values = np.zeros_like(double_smoothed_pc)
    non_zero_mask = double_smoothed_abs_pc != 0
    tsi_values[non_zero_mask] = 100 * (double_smoothed_pc[non_zero_mask] /
                                       double_smoothed_abs_pc[non_zero_mask])

    # Сигнальная линия - EMA от TSI
    signal_line = calculate_ema(tsi_values, signal_length)

    return {
        'tsi': tsi_values,
        'signal': signal_line
    }


def find_support_resistance_levels(candles: List[List[str]], window: int = 20) -> Dict[str, List[float]]:
    """
    Поиск уровней поддержки/сопротивления через локальные минимумы/максимумы

    Args:
        candles: Данные свечей
        window: Окно для поиска локальных экстремумов

    Returns:
        Словарь с уровнями поддержки и сопротивления
    """
    if len(candles) < window * 2:
        return {'support': [], 'resistance': []}

    highs = np.array([float(candle[2]) for candle in candles])
    lows = np.array([float(candle[3]) for candle in candles])

    support_levels = []
    resistance_levels = []

    # Поиск локальных минимумов (поддержка)
    for i in range(window, len(lows) - window):
        if lows[i] == min(lows[i - window:i + window + 1]):
            support_levels.append(float(lows[i]))

    # Поиск локальных максимумов (сопротивление)
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i - window:i + window + 1]):
            resistance_levels.append(float(highs[i]))

    # Убираем дубликаты и сортируем
    support_levels = sorted(list(set(support_levels)))
    resistance_levels = sorted(list(set(resistance_levels)))

    return {
        'support': support_levels,
        'resistance': resistance_levels
    }


def calculate_rsi_with_divergence(candles: List[List[str]], period: int = 14) -> Dict[str, Any]:
    """
    Расчет RSI с поиском дивергенций

    Args:
        candles: Данные свечей
        period: Период RSI

    Returns:
        Словарь с RSI и найденными дивергенциями
    """
    if len(candles) < period + 10:
        return {'rsi': [], 'divergences': []}

    prices = np.array([float(candle[4]) for candle in candles])

    # Расчет RSI
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gains = np.zeros_like(prices)
    avg_losses = np.zeros_like(prices)

    # Первое значение - простое среднее
    avg_gains[period] = np.mean(gains[:period])
    avg_losses[period] = np.mean(losses[:period])

    # Последующие значения - сглаженное среднее
    for i in range(period + 1, len(prices)):
        avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_losses[i] = (avg_losses[i - 1] * (period - 1) + losses[i - 1]) / period

    rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses != 0)
    rsi = 100 - (100 / (1 + rs))

    # Поиск дивергенций (упрощенный)
    divergences = []
    if len(rsi) > 50:
        # Ищем последние 20 свечей для дивергенций
        recent_prices = prices[-20:]
        recent_rsi = rsi[-20:]

        # Бычья дивергенция: цена падает, RSI растет
        if len(recent_prices) > 10:
            price_trend = recent_prices[-1] - recent_prices[-10]
            rsi_trend = recent_rsi[-1] - recent_rsi[-10]

            if price_trend < 0 and rsi_trend > 0 and recent_rsi[-1] < 30:
                divergences.append({'type': 'bullish', 'strength': abs(rsi_trend)})
            elif price_trend > 0 and rsi_trend < 0 and recent_rsi[-1] > 70:
                divergences.append({'type': 'bearish', 'strength': abs(rsi_trend)})

    return {
        'rsi': rsi.tolist(),
        'divergences': divergences
    }


def get_volume_anomalies(candles: List[List[str]], threshold: float = 1.5) -> Dict[str, Any]:
    """
    Поиск объемных аномалий и построение объемного профиля

    Args:
        candles: Данные свечей
        threshold: Порог для определения объемного всплеска

    Returns:
        Словарь с объемными данными
    """
    if len(candles) < 20:
        return {'spikes': [], 'profile': []}

    volumes = np.array([float(candle[5]) for candle in candles])
    prices = np.array([float(candle[4]) for candle in candles])

    # Поиск объемных всплесков
    if len(volumes) > 20:
        avg_volume = np.mean(volumes[-20:])
        spikes = []

        for i in range(len(volumes)):
            if volumes[i] > avg_volume * threshold:
                spikes.append({
                    'index': i,
                    'volume': float(volumes[i]),
                    'price': float(prices[i]),
                    'ratio': float(volumes[i] / avg_volume)
                })
    else:
        spikes = []

    # Упрощенный объемный профиль (последние 100 свечей)
    profile = []
    if len(candles) > 50:
        recent_candles = candles[-100:] if len(candles) > 100 else candles
        price_min = min(float(c[3]) for c in recent_candles)
        price_max = max(float(c[2]) for c in recent_candles)

        # Разбиваем диапазон на 20 уровней
        price_levels = np.linspace(price_min, price_max, 20)

        for i in range(len(price_levels) - 1):
            level_volume = 0
            for candle in recent_candles:
                candle_low = float(candle[3])
                candle_high = float(candle[2])
                candle_volume = float(candle[5])

                # Если свеча пересекает уровень, добавляем объем
                if candle_low <= price_levels[i + 1] and candle_high >= price_levels[i]:
                    level_volume += candle_volume

            if level_volume > 0:
                profile.append({
                    'price_level': float((price_levels[i] + price_levels[i + 1]) / 2),
                    'volume': float(level_volume)
                })

    return {
        'spikes': spikes[-10:],  # Последние 10 всплесков
        'profile': sorted(profile, key=lambda x: x['volume'], reverse=True)[:10]  # Топ-10 уровней
    }


def simulate_higher_timeframes(candles: List[List[str]]) -> Dict[str, str]:
    """
    Симуляция анализа старших таймфреймов из 15-минутных данных

    Args:
        candles: 15-минутные свечи

    Returns:
        Словарь с направлениями трендов на старших ТФ
    """
    if len(candles) < 100:
        return {'1h_direction': 'UNKNOWN', '4h_direction': 'UNKNOWN'}

    prices = np.array([float(candle[4]) for candle in candles])

    # Симуляция 1H (4 свечи 15M = 1H)
    h1_direction = 'UNKNOWN'
    if len(prices) >= 16:  # Минимум 4 часа данных
        h1_prices = []
        for i in range(0, len(prices), 4):
            if i + 3 < len(prices):
                h1_prices.append(prices[i + 3])  # Цена закрытия часовой свечи

        if len(h1_prices) >= 4:
            recent_h1 = h1_prices[-4:]
            if recent_h1[-1] > recent_h1[0]:
                h1_direction = 'BULLISH'
            elif recent_h1[-1] < recent_h1[0]:
                h1_direction = 'BEARISH'
            else:
                h1_direction = 'SIDEWAYS'

    # Симуляция 4H (16 свечей 15M = 4H)
    h4_direction = 'UNKNOWN'
    if len(prices) >= 64:  # Минимум 16 часов данных
        h4_prices = []
        for i in range(0, len(prices), 16):
            if i + 15 < len(prices):
                h4_prices.append(prices[i + 15])  # Цена закрытия 4-часовой свечи

        if len(h4_prices) >= 4:
            recent_h4 = h4_prices[-4:]
            if recent_h4[-1] > recent_h4[0]:
                h4_direction = 'BULLISH'
            elif recent_h4[-1] < recent_h4[0]:
                h4_direction = 'BEARISH'
            else:
                h4_direction = 'SIDEWAYS'

    return {
        '1h_direction': h1_direction,
        '4h_direction': h4_direction
    }


def calculate_atr_stops(candles: List[List[str]], period: int = 14) -> Dict[str, Any]:
    """
    Расчет ATR и динамических стоп-лоссов

    Args:
        candles: Данные свечей
        period: Период ATR

    Returns:
        Словарь с ATR и уровнями стопов
    """
    if len(candles) < period + 1:
        return {'atr': [], 'stop_levels': []}

    highs = np.array([float(candle[2]) for candle in candles])
    lows = np.array([float(candle[3]) for candle in candles])
    closes = np.array([float(candle[4]) for candle in candles])

    # Расчет True Range
    tr = np.zeros(len(candles))
    for i in range(1, len(candles)):
        tr[i] = max(
            highs[i] - lows[i],  # High - Low
            abs(highs[i] - closes[i - 1]),  # High - Previous Close
            abs(lows[i] - closes[i - 1])  # Low - Previous Close
        )

    # Расчет ATR
    atr = np.zeros(len(candles))
    atr[period] = np.mean(tr[1:period + 1])

    for i in range(period + 1, len(candles)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    # Динамические стоп-лоссы
    stop_levels = []
    if len(atr) > 0:
        current_price = closes[-1]
        current_atr = atr[-1]

        stop_levels = {
            'long_stop': current_price - (current_atr * 2),  # Стоп для лонга
            'short_stop': current_price + (current_atr * 2),  # Стоп для шорта
            'trailing_multiplier': 2.0,
            'current_volatility': current_atr / current_price * 100 if current_price > 0 else 0
        }

    return {
        'atr': atr.tolist(),
        'stop_levels': stop_levels
    }


def calculate_all_indicators_extended(candles: List[List[str]],
                                      ema1: int = 7,
                                      ema2: int = 14,
                                      ema3: int = 28,
                                      tsi_long: int = 25,
                                      tsi_short: int = 13,
                                      tsi_signal: int = 13) -> Dict[str, Any]:
    """
    Главная функция расчета всех индикаторов (базовых + новых)

    Args:
        candles: Данные свечей
        ema1, ema2, ema3: Периоды EMA
        tsi_long, tsi_short, tsi_signal: Параметры TSI

    Returns:
        Словарь со всеми рассчитанными индикаторами
    """
    if not candles:
        return {}

    # Существующие индикаторы
    base_indicators = calculate_indicators_for_candles(
        candles, ema1, ema2, ema3, tsi_long, tsi_short, tsi_signal
    )

    # Новые индикаторы
    levels = find_support_resistance_levels(candles)
    rsi_data = calculate_rsi_with_divergence(candles)
    volume_data = get_volume_anomalies(candles)
    mtf_data = simulate_higher_timeframes(candles)
    atr_data = calculate_atr_stops(candles)

    # Извлекаем цены для дополнительного анализа
    prices = [float(candle[4]) for candle in candles]
    current_price = prices[-1] if prices else 0

    # Анализ близости к уровням
    nearby_support = []
    nearby_resistance = []

    if current_price > 0:
        # Ищем ближайшие уровни поддержки (ниже текущей цены)
        for level in levels['support']:
            if level < current_price:
                distance = abs(current_price - level) / current_price * 100
                if distance <= 2.0:  # В пределах 2%
                    nearby_support.append({'level': level, 'distance_percent': distance})

        # Ищем ближайшие уровни сопротивления (выше текущей цены)
        for level in levels['resistance']:
            if level > current_price:
                distance = abs(level - current_price) / current_price * 100
                if distance <= 2.0:  # В пределах 2%
                    nearby_resistance.append({'level': level, 'distance_percent': distance})

    # Объединяем все данные
    return {
        # Базовые индикаторы (EMA + TSI)
        **base_indicators,

        # Уровневый анализ
        'support_levels': levels['support'][-5:],  # Последние 5 уровней
        'resistance_levels': levels['resistance'][-5:],  # Последние 5 уровней
        'nearby_support': nearby_support,
        'nearby_resistance': nearby_resistance,

        # RSI + дивергенции
        'rsi_values': rsi_data['rsi'],
        'rsi_divergences': rsi_data['divergences'],
        'rsi_current': rsi_data['rsi'][-1] if rsi_data['rsi'] else 50,

        # Объемный анализ
        'volume_spikes': volume_data['spikes'],
        'volume_profile': volume_data['profile'],

        # Мультитаймфрейм
        'h1_trend': mtf_data['1h_direction'],
        'h4_trend': mtf_data['4h_direction'],
        'mtf_confluence': mtf_data['1h_direction'] == mtf_data['4h_direction'],

        # ATR и стопы
        'atr_values': atr_data['atr'],
        'dynamic_stops': atr_data['stop_levels'],
        'current_atr': atr_data['atr'][-1] if atr_data['atr'] else 0
    }


def check_ema_tsi_signal(candles: List[List[str]],
                         ema1_period: int = 7,
                         ema2_period: int = 14,
                         ema3_period: int = 28,
                         tsi_long: int = 25,
                         tsi_short: int = 13,
                         tsi_signal: int = 13) -> str:
    """
    Проверка EMA + TSI сигнала точно по индикатору Pine Script

    Логика:
    - LONG: EMA7 > EMA14 > EMA28 И TSI пересекает сигнальную линию снизу вверх на ПОСЛЕДНЕЙ свече
    - SHORT: EMA7 < EMA14 < EMA28 И TSI пересекает сигнальную линию сверху вниз на ПОСЛЕДНЕЙ свече

    Args:
        candles: Данные свечей от Bybit
        ema1_period: Период первой EMA (по умолчанию 7)
        ema2_period: Период второй EMA (по умолчанию 14)
        ema3_period: Период третьей EMA (по умолчанию 28)
        tsi_long: Длинный период TSI (по умолчанию 25)
        tsi_short: Короткий период TSI (по умолчанию 13)
        tsi_signal: Период сигнальной линии TSI (по умолчанию 13)

    Returns:
        'LONG', 'SHORT', или 'NO_SIGNAL'
    """
    # Проверяем достаточность данных
    required_data = max(ema3_period, tsi_long, 50) + 20
    if len(candles) < required_data:
        return 'NO_SIGNAL'

    # Извлекаем цены закрытия
    prices = np.array([float(candle[4]) for candle in candles])

    # Рассчитываем EMA
    ema1 = calculate_ema(prices, ema1_period)
    ema2 = calculate_ema(prices, ema2_period)
    ema3 = calculate_ema(prices, ema3_period)

    # Рассчитываем TSI
    tsi_data = calculate_tsi(candles, tsi_long, tsi_short, tsi_signal)

    if len(tsi_data['tsi']) < 2:  # Нужно минимум 2 точки для проверки пересечения
        return 'NO_SIGNAL'

    tsi_values = tsi_data['tsi']
    signal_values = tsi_data['signal']

    # Получаем значения на последней свече
    current_ema1 = ema1[-1]
    current_ema2 = ema2[-1]
    current_ema3 = ema3[-1]

    current_tsi = tsi_values[-1]
    current_signal = signal_values[-1]
    prev_tsi = tsi_values[-2]
    prev_signal = signal_values[-2]

    # Проверяем EMA условия
    ema_uptrend = current_ema1 > current_ema2 > current_ema3  # EMA 7 > EMA 14 > EMA 28
    ema_downtrend = current_ema1 < current_ema2 < current_ema3  # EMA 7 < EMA 14 < EMA 28

    # Проверяем TSI пересечения на ТЕКУЩЕЙ свече
    tsi_crossover_up = prev_tsi <= prev_signal and current_tsi > current_signal  # Пересечение снизу вверх
    tsi_crossover_down = prev_tsi >= prev_signal and current_tsi < current_signal  # Пересечение сверху вниз

    # Проверяем финальные сигналы (точно по Pine Script логике)
    if ema_uptrend and tsi_crossover_up:
        return 'LONG'
    elif ema_downtrend and tsi_crossover_down:
        return 'SHORT'

    return 'NO_SIGNAL'


def get_signal_details(candles: List[List[str]],
                       ema1_period: int = 7,
                       ema2_period: int = 14,
                       ema3_period: int = 28,
                       tsi_long: int = 25,
                       tsi_short: int = 13,
                       tsi_signal: int = 13) -> Dict[str, Any]:
    """
    Получение детальной информации о сигнале

    Args:
        candles: Данные свечей от Bybit
        ema1_period: Период первой EMA
        ema2_period: Период второй EMA
        ema3_period: Период третьей EMA
        tsi_long: Длинный период TSI
        tsi_short: Короткий период TSI
        tsi_signal: Период сигнальной линии TSI

    Returns:
        Словарь с детальной информацией
    """
    required_data = max(ema3_period, tsi_long, 50) + 20

    if len(candles) < required_data:
        return {
            'signal': 'NO_SIGNAL',
            'reason': 'INSUFFICIENT_DATA',
            'last_price': float(candles[-1][4]) if candles else 0,
            'ema1': 0,
            'ema2': 0,
            'ema3': 0,
            'tsi_value': 0,
            'tsi_signal_value': 0,
            'ema_alignment': 'UNKNOWN',
            'tsi_crossover': False
        }

    # Извлекаем цены закрытия
    prices = np.array([float(candle[4]) for candle in candles])

    # Рассчитываем EMA
    ema1 = calculate_ema(prices, ema1_period)
    ema2 = calculate_ema(prices, ema2_period)
    ema3 = calculate_ema(prices, ema3_period)

    # Рассчитываем TSI
    tsi_data = calculate_tsi(candles, tsi_long, tsi_short, tsi_signal)

    if len(tsi_data['tsi']) < 2:
        return {
            'signal': 'NO_SIGNAL',
            'reason': 'INSUFFICIENT_TSI_DATA',
            'last_price': prices[-1],
            'ema1': ema1[-1],
            'ema2': ema2[-1],
            'ema3': ema3[-1],
            'tsi_value': 0,
            'tsi_signal_value': 0,
            'ema_alignment': 'UNKNOWN',
            'tsi_crossover': False
        }

    tsi_values = tsi_data['tsi']
    signal_values = tsi_data['signal']

    # Получаем текущие значения
    current_ema1 = ema1[-1]
    current_ema2 = ema2[-1]
    current_ema3 = ema3[-1]
    current_tsi = tsi_values[-1]
    current_signal = signal_values[-1]
    prev_tsi = tsi_values[-2]
    prev_signal = signal_values[-2]

    # Определяем выравнивание EMA
    if current_ema1 > current_ema2 > current_ema3:
        ema_alignment = 'BULLISH'
    elif current_ema1 < current_ema2 < current_ema3:
        ema_alignment = 'BEARISH'
    else:
        ema_alignment = 'NEUTRAL'

    # Проверяем пересечения TSI
    tsi_crossover_up = prev_tsi <= prev_signal and current_tsi > current_signal
    tsi_crossover_down = prev_tsi >= prev_signal and current_tsi < current_signal
    tsi_crossover = tsi_crossover_up or tsi_crossover_down

    # Получаем финальный сигнал
    signal = check_ema_tsi_signal(candles, ema1_period, ema2_period, ema3_period,
                                  tsi_long, tsi_short, tsi_signal)

    # Определяем причину
    if signal == 'LONG':
        reason = 'EMA_BULLISH_TSI_CROSSOVER_UP'
    elif signal == 'SHORT':
        reason = 'EMA_BEARISH_TSI_CROSSOVER_DOWN'
    elif ema_alignment == 'NEUTRAL':
        reason = 'EMA_NOT_ALIGNED'
    elif not tsi_crossover:
        reason = 'NO_TSI_CROSSOVER'
    else:
        reason = 'NO_SIGNAL_CONDITION_MET'

    return {
        'signal': signal,
        'reason': reason,
        'last_price': prices[-1],
        'ema1': current_ema1,
        'ema2': current_ema2,
        'ema3': current_ema3,
        'tsi_value': current_tsi,
        'tsi_signal_value': current_signal,
        'ema_alignment': ema_alignment,
        'tsi_crossover': tsi_crossover,
        'tsi_crossover_direction': 'UP' if tsi_crossover_up else ('DOWN' if tsi_crossover_down else 'NONE')
    }


def calculate_indicators_for_candles(candles: List[List[str]],
                                     ema1_period: int = 7,
                                     ema2_period: int = 14,
                                     ema3_period: int = 28,
                                     tsi_long: int = 25,
                                     tsi_short: int = 13,
                                     tsi_signal: int = 13) -> Dict[str, Any]:
    """
    Рассчитывает все индикаторы для свечей и возвращает их значения

    Args:
        candles: Данные свечей
        ema1_period: Период первой EMA
        ema2_period: Период второй EMA
        ema3_period: Период третьей EMA
        tsi_long: Длинный период TSI
        tsi_short: Короткий период TSI
        tsi_signal: Период сигнальной линии TSI

    Returns:
        Словарь с массивами значений индикаторов
    """
    if not candles:
        return {}

    # Извлекаем цены закрытия
    prices = np.array([float(candle[4]) for candle in candles])

    # Рассчитываем EMA
    ema1_values = calculate_ema(prices, ema1_period)
    ema2_values = calculate_ema(prices, ema2_period)
    ema3_values = calculate_ema(prices, ema3_period)

    # Рассчитываем TSI
    tsi_data = calculate_tsi(candles, tsi_long, tsi_short, tsi_signal)

    return {
        'ema1_values': ema1_values.tolist(),
        'ema2_values': ema2_values.tolist(),
        'ema3_values': ema3_values.tolist(),
        'tsi_values': tsi_data['tsi'].tolist(),
        'tsi_signal_values': tsi_data['signal'].tolist()
    }


# Добавляем новые функции из задания

def find_support_resistance_levels(candles: List[List[str]], window: int = 20) -> Dict[str, List[float]]:
    """
    Поиск уровней поддержки/сопротивления через локальные минимумы/максимумы

    Args:
        candles: Данные свечей
        window: Окно для поиска локальных экстремумов

    Returns:
        Словарь с уровнями поддержки и сопротивления
    """
    if len(candles) < window * 2:
        return {'support': [], 'resistance': []}

    highs = np.array([float(candle[2]) for candle in candles])
    lows = np.array([float(candle[3]) for candle in candles])

    support_levels = []
    resistance_levels = []

    # Поиск локальных минимумов (поддержка)
    for i in range(window, len(lows) - window):
        if lows[i] == min(lows[i - window:i + window + 1]):
            support_levels.append(float(lows[i]))

    # Поиск локальных максимумов (сопротивление)
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i - window:i + window + 1]):
            resistance_levels.append(float(highs[i]))

    # Убираем дубликаты и сортируем
    support_levels = sorted(list(set(support_levels)))
    resistance_levels = sorted(list(set(resistance_levels)))

    return {
        'support': support_levels,
        'resistance': resistance_levels
    }


def calculate_rsi_with_divergence(candles: List[List[str]], period: int = 14) -> Dict[str, Any]:
    """
    Расчет RSI с поиском дивергенций

    Args:
        candles: Данные свечей
        period: Период RSI

    Returns:
        Словарь с RSI и найденными дивергенциями
    """
    if len(candles) < period + 10:
        return {'rsi': [], 'divergences': []}

    prices = np.array([float(candle[4]) for candle in candles])

    # Расчет RSI
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gains = np.zeros_like(prices)
    avg_losses = np.zeros_like(prices)

    # Первое значение - простое среднее
    avg_gains[period] = np.mean(gains[:period])
    avg_losses[period] = np.mean(losses[:period])

    # Последующие значения - сглаженное среднее
    for i in range(period + 1, len(prices)):
        avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_losses[i] = (avg_losses[i - 1] * (period - 1) + losses[i - 1]) / period

    rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses != 0)
    rsi = 100 - (100 / (1 + rs))

    # Поиск дивергенций (упрощенный)
    divergences = []
    if len(rsi) > 50:
        # Ищем последние 20 свечей для дивергенций
        recent_prices = prices[-20:]
        recent_rsi = rsi[-20:]

        # Бычья дивергенция: цена падает, RSI растет
        if len(recent_prices) > 10:
            price_trend = recent_prices[-1] - recent_prices[-10]
            rsi_trend = recent_rsi[-1] - recent_rsi[-10]

            if price_trend < 0 and rsi_trend > 0 and recent_rsi[-1] < 30:
                divergences.append({'type': 'bullish', 'strength': abs(rsi_trend)})
            elif price_trend > 0 and rsi_trend < 0 and recent_rsi[-1] > 70:
                divergences.append({'type': 'bearish', 'strength': abs(rsi_trend)})

    return {
        'rsi': rsi.tolist(),
        'divergences': divergences
    }


def get_volume_anomalies(candles: List[List[str]], threshold: float = 1.5) -> Dict[str, Any]:
    """
    Поиск объемных аномалий и построение объемного профиля

    Args:
        candles: Данные свечей
        threshold: Порог для определения объемного всплеска

    Returns:
        Словарь с объемными данными
    """
    if len(candles) < 20:
        return {'spikes': [], 'profile': []}

    volumes = np.array([float(candle[5]) for candle in candles])
    prices = np.array([float(candle[4]) for candle in candles])

    # Поиск объемных всплесков
    if len(volumes) > 20:
        avg_volume = np.mean(volumes[-20:])
        spikes = []

        for i in range(len(volumes)):
            if volumes[i] > avg_volume * threshold:
                spikes.append({
                    'index': i,
                    'volume': float(volumes[i]),
                    'price': float(prices[i]),
                    'ratio': float(volumes[i] / avg_volume)
                })
    else:
        spikes = []

    # Упрощенный объемный профиль (последние 100 свечей)
    profile = []
    if len(candles) > 50:
        recent_candles = candles[-100:] if len(candles) > 100 else candles
        price_min = min(float(c[3]) for c in recent_candles)
        price_max = max(float(c[2]) for c in recent_candles)

        # Разбиваем диапазон на 20 уровней
        price_levels = np.linspace(price_min, price_max, 20)

        for i in range(len(price_levels) - 1):
            level_volume = 0
            for candle in recent_candles:
                candle_low = float(candle[3])
                candle_high = float(candle[2])
                candle_volume = float(candle[5])

                # Если свеча пересекает уровень, добавляем объем
                if candle_low <= price_levels[i + 1] and candle_high >= price_levels[i]:
                    level_volume += candle_volume

            if level_volume > 0:
                profile.append({
                    'price_level': float((price_levels[i] + price_levels[i + 1]) / 2),
                    'volume': float(level_volume)
                })

    return {
        'spikes': spikes[-10:],  # Последние 10 всплесков
        'profile': sorted(profile, key=lambda x: x['volume'], reverse=True)[:10]  # Топ-10 уровней
    }


def simulate_higher_timeframes(candles: List[List[str]]) -> Dict[str, str]:
    """
    Симуляция анализа старших таймфреймов из 15-минутных данных

    Args:
        candles: 15-минутные свечи

    Returns:
        Словарь с направлениями трендов на старших ТФ
    """
    if len(candles) < 100:
        return {'1h_direction': 'UNKNOWN', '4h_direction': 'UNKNOWN'}

    prices = np.array([float(candle[4]) for candle in candles])

    # Симуляция 1H (4 свечи 15M = 1H)
    h1_direction = 'UNKNOWN'
    if len(prices) >= 16:  # Минимум 4 часа данных
        h1_prices = []
        for i in range(0, len(prices), 4):
            if i + 3 < len(prices):
                h1_prices.append(prices[i + 3])  # Цена закрытия часовой свечи

        if len(h1_prices) >= 4:
            recent_h1 = h1_prices[-4:]
            if recent_h1[-1] > recent_h1[0]:
                h1_direction = 'BULLISH'
            elif recent_h1[-1] < recent_h1[0]:
                h1_direction = 'BEARISH'
            else:
                h1_direction = 'SIDEWAYS'

    # Симуляция 4H (16 свечей 15M = 4H)
    h4_direction = 'UNKNOWN'
    if len(prices) >= 64:  # Минимум 16 часов данных
        h4_prices = []
        for i in range(0, len(prices), 16):
            if i + 15 < len(prices):
                h4_prices.append(prices[i + 15])  # Цена закрытия 4-часовой свечи

        if len(h4_prices) >= 4:
            recent_h4 = h4_prices[-4:]
            if recent_h4[-1] > recent_h4[0]:
                h4_direction = 'BULLISH'
            elif recent_h4[-1] < recent_h4[0]:
                h4_direction = 'BEARISH'
            else:
                h4_direction = 'SIDEWAYS'

    return {
        '1h_direction': h1_direction,
        '4h_direction': h4_direction
    }


def calculate_atr_stops(candles: List[List[str]], period: int = 14) -> Dict[str, Any]:
    """
    Расчет ATR и динамических стоп-лоссов

    Args:
        candles: Данные свечей
        period: Период ATR

    Returns:
        Словарь с ATR и уровнями стопов
    """
    if len(candles) < period + 1:
        return {'atr': [], 'stop_levels': []}

    highs = np.array([float(candle[2]) for candle in candles])
    lows = np.array([float(candle[3]) for candle in candles])
    closes = np.array([float(candle[4]) for candle in candles])

    # Расчет True Range
    tr = np.zeros(len(candles))
    for i in range(1, len(candles)):
        tr[i] = max(
            highs[i] - lows[i],  # High - Low
            abs(highs[i] - closes[i - 1]),  # High - Previous Close
            abs(lows[i] - closes[i - 1])  # Low - Previous Close
        )

    # Расчет ATR
    atr = np.zeros(len(candles))
    atr[period] = np.mean(tr[1:period + 1])

    for i in range(period + 1, len(candles)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    # Динамические стоп-лоссы
    stop_levels = []
    if len(atr) > 0:
        current_price = closes[-1]
        current_atr = atr[-1]

        stop_levels = {
            'long_stop': current_price - (current_atr * 2),  # Стоп для лонга
            'short_stop': current_price + (current_atr * 2),  # Стоп для шорта
            'trailing_multiplier': 2.0,
            'current_volatility': current_atr / current_price * 100 if current_price > 0 else 0
        }

    return {
        'atr': atr.tolist(),
        'stop_levels': stop_levels
    }


def calculate_all_indicators_extended(candles: List[List[str]],
                                      ema1: int = 7,
                                      ema2: int = 14,
                                      ema3: int = 28,
                                      tsi_long: int = 25,
                                      tsi_short: int = 13,
                                      tsi_signal: int = 13) -> Dict[str, Any]:
    """
    Главная функция расчета всех индикаторов (базовых + новых)

    Args:
        candles: Данные свечей
        ema1, ema2, ema3: Периоды EMA
        tsi_long, tsi_short, tsi_signal: Параметры TSI

    Returns:
        Словарь со всеми рассчитанными индикаторами
    """
    if not candles:
        return {}

    # Существующие индикаторы
    base_indicators = calculate_indicators_for_candles(
        candles, ema1, ema2, ema3, tsi_long, tsi_short, tsi_signal
    )

    # Новые индикаторы
    levels = find_support_resistance_levels(candles)
    rsi_data = calculate_rsi_with_divergence(candles)
    volume_data = get_volume_anomalies(candles)
    mtf_data = simulate_higher_timeframes(candles)
    atr_data = calculate_atr_stops(candles)

    # Извлекаем цены для дополнительного анализа
    prices = [float(candle[4]) for candle in candles]
    current_price = prices[-1] if prices else 0

    # Анализ близости к уровням
    nearby_support = []
    nearby_resistance = []

    if current_price > 0:
        # Ищем ближайшие уровни поддержки (ниже текущей цены)
        for level in levels['support']:
            if level < current_price:
                distance = abs(current_price - level) / current_price * 100
                if distance <= 2.0:  # В пределах 2%
                    nearby_support.append({'level': level, 'distance_percent': distance})

        # Ищем ближайшие уровни сопротивления (выше текущей цены)
        for level in levels['resistance']:
            if level > current_price:
                distance = abs(level - current_price) / current_price * 100
                if distance <= 2.0:  # В пределах 2%
                    nearby_resistance.append({'level': level, 'distance_percent': distance})

    # Объединяем все данные
    return {
        # Базовые индикаторы (EMA + TSI)
        **base_indicators,

        # Уровневый анализ
        'support_levels': levels['support'][-5:],  # Последние 5 уровней
        'resistance_levels': levels['resistance'][-5:],  # Последние 5 уровней
        'nearby_support': nearby_support,
        'nearby_resistance': nearby_resistance,

        # RSI + дивергенции
        'rsi_values': rsi_data['rsi'],
        'rsi_divergences': rsi_data['divergences'],
        'rsi_current': rsi_data['rsi'][-1] if rsi_data['rsi'] else 50,

        # Объемный анализ
        'volume_spikes': volume_data['spikes'],
        'volume_profile': volume_data['profile'],

        # Мультитаймфрейм
        'h1_trend': mtf_data['1h_direction'],
        'h4_trend': mtf_data['4h_direction'],
        'mtf_confluence': mtf_data['1h_direction'] == mtf_data['4h_direction'],

        # ATR и стопы
        'atr_values': atr_data['atr'],
        'dynamic_stops': atr_data['stop_levels'],
        'current_atr': atr_data['atr'][-1] if atr_data['atr'] else 0
    }