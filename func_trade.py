import numpy as np
from typing import List, Dict, Any


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