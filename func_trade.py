import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


def calculate_atr(candles: List[List[str]], period: int = 14) -> float:
    """
    Вычисление ATR (Average True Range) для определения волатильности

    Args:
        candles: Данные свечей в формате Bybit [timestamp, open, high, low, close, volume, turnover]
        period: Период для расчета ATR (по умолчанию 14)

    Returns:
        Значение ATR
    """
    if len(candles) < period:
        return 0.0

    true_ranges = []
    for i in range(1, len(candles)):
        high = float(candles[i][2])
        low = float(candles[i][3])
        prev_close = float(candles[i - 1][4])

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)

    return sum(true_ranges[-period:]) / period if true_ranges else 0.0


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

    TSI является двойно сглаженным индикатором RSI, который использует
    экспоненциальные скользящие средние для фильтрации шума цены.

    Args:
        candles: Данные свечей в формате Bybit [timestamp, open, high, low, close, volume, turnover]
        long_length: Длинный период сглаживания (по умолчанию 25)
        short_length: Короткий период сглаживания (по умолчанию 13)
        signal_length: Период сигнальной линии (по умолчанию 13)

    Returns:
        Словарь с массивами:
        - 'tsi': значения TSI
        - 'signal': сигнальная линия (EMA от TSI)
        - 'histogram': разность между TSI и сигнальной линией
    """
    if len(candles) < max(long_length, short_length, signal_length) + 10:
        return {
            'tsi': np.array([]),
            'signal': np.array([]),
            'histogram': np.array([])
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

    # Гистограмма - разность между TSI и сигнальной линией
    histogram = tsi_values - signal_line

    return {
        'tsi': tsi_values,
        'signal': signal_line,
        'histogram': histogram
    }


def check_tsi_confirmation(candles: List[List[str]],
                           signal_type: str,
                           long_length: int = 25,
                           short_length: int = 13,
                           signal_length: int = 13) -> bool:
    """
    Проверка подтверждения сигнала с помощью TSI
    ИЗМЕНЕНО: Проверяет пересечение ТОЛЬКО на последней свече

    Args:
        candles: Данные свечей от Bybit
        signal_type: Тип сигнала ('LONG' или 'SHORT')
        long_length: Длинный период TSI
        short_length: Короткий период TSI
        signal_length: Период сигнальной линии

    Returns:
        True если TSI подтверждает сигнал
    """
    tsi_data = calculate_tsi(candles, long_length, short_length, signal_length)

    if len(tsi_data['tsi']) < 2:  # Нужно минимум 2 точки для проверки пересечения
        return False

    tsi_values = tsi_data['tsi']
    signal_values = tsi_data['signal']

    # Проверяем ТОЛЬКО последнее пересечение (индексы -2, -1)
    current_tsi = tsi_values[-1]
    current_signal = signal_values[-1]
    prev_tsi = tsi_values[-2]
    prev_signal = signal_values[-2]

    if signal_type == 'LONG':
        # Для LONG: TSI должен пересечь сигнальную линию снизу вверх на последней свече
        if prev_tsi <= prev_signal and current_tsi > current_signal:
            return True
    elif signal_type == 'SHORT':
        # Для SHORT: TSI должен пересечь сигнальную линию сверху вниз на последней свече
        if prev_tsi >= prev_signal and current_tsi < current_signal:
            return True

    return False


def calculate_three_ema(prices: np.ndarray,
                        fast_period: int = 7,
                        medium_period: int = 14,
                        slow_period: int = 28) -> Dict[str, np.ndarray]:
    """
    Расчет трех EMA для анализа тренда

    Args:
        prices: Массив цен закрытия
        fast_period: Период быстрой EMA (по умолчанию 7)
        medium_period: Период средней EMA (по умолчанию 14)
        slow_period: Период медленной EMA (по умолчанию 28)

    Returns:
        Словарь с тремя массивами EMA
    """
    return {
        'ema_fast': calculate_ema(prices, fast_period),
        'ema_medium': calculate_ema(prices, medium_period),
        'ema_slow': calculate_ema(prices, slow_period)
    }


def check_ema_alignment(ema_fast: np.ndarray,
                        ema_medium: np.ndarray,
                        ema_slow: np.ndarray,
                        signal_type: str,
                        index: int) -> bool:
    """
    Проверка выравнивания EMA для определения направления тренда

    Args:
        ema_fast: Быстрая EMA
        ema_medium: Средняя EMA
        ema_slow: Медленная EMA
        signal_type: Тип сигнала ('LONG' или 'SHORT')
        index: Индекс для проверки

    Returns:
        True если выравнивание поддерживает сигнал
    """
    if index < 0 or index >= len(ema_fast):
        return False

    fast = ema_fast[index]
    medium = ema_medium[index]
    slow = ema_slow[index]

    if signal_type == 'LONG':
        # Для LONG: Fast > Medium > Slow (бычье выравнивание)
        return fast > medium > slow
    elif signal_type == 'SHORT':
        # Для SHORT: Fast < Medium < Slow (медвежье выравнивание)
        return fast < medium < slow

    return False


def parse_bybit_candles(raw_candles: List[List[str]]) -> List[List[float]]:
    """
    Парсинг данных свечей Bybit в формат OHLCV

    Args:
        raw_candles: Сырые данные свечей от Bybit

    Returns:
        Список свечей в формате OHLCV
    """
    parsed_candles = []

    # Свечи уже в правильном порядке (от старых к новым) после обработки в get_klines_async
    for candle in raw_candles:
        # Извлекаем данные OHLCV
        open_price = float(candle[1])
        high_price = float(candle[2])
        low_price = float(candle[3])
        close_price = float(candle[4])
        volume = float(candle[5])

        parsed_candles.append([open_price, high_price, low_price, close_price, volume])

    return parsed_candles


def analyze_last_candle(bybit_candles: List[List[str]],
                        ema_fast: int = 7,
                        ema_medium: int = 14,
                        ema_slow: int = 28,
                        use_tsi_filter: bool = True,
                        tsi_long: int = 25,
                        tsi_short: int = 13,
                        tsi_signal: int = 13) -> str:
    """
    Анализ последней свечи для получения EMA сигнала с TSI подтверждением

    Args:
        bybit_candles: Данные свечей от Bybit
        ema_fast: Период быстрой EMA
        ema_medium: Период средней EMA
        ema_slow: Период медленной EMA
        use_tsi_filter: Использовать TSI фильтр (по умолчанию True)
        tsi_long: Длинный период TSI
        tsi_short: Короткий период TSI
        tsi_signal: Период сигнальной линии TSI

    Returns:
        Строка с результатом: 'LONG', 'SHORT', или 'NO_SIGNAL'
    """
    required_data = max(ema_slow, tsi_long, 50) + 20

    if len(bybit_candles) < required_data:
        return 'NO_SIGNAL'

    # Парсим свечи
    candles = parse_bybit_candles(bybit_candles)

    # Извлекаем цены закрытия
    df = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume'])
    prices = df['close'].values

    # Рассчитываем EMA
    ema_data = calculate_three_ema(prices, ema_fast, ema_medium, ema_slow)

    # Проверяем EMA сигнал на последней свече
    last_idx = len(prices) - 1
    ema_signal = 'NO_SIGNAL'

    if check_ema_alignment(ema_data['ema_fast'], ema_data['ema_medium'],
                           ema_data['ema_slow'], 'LONG', last_idx):
        ema_signal = 'LONG'
    elif check_ema_alignment(ema_data['ema_fast'], ema_data['ema_medium'],
                             ema_data['ema_slow'], 'SHORT', last_idx):
        ema_signal = 'SHORT'

    # Если нет EMA сигнала, возвращаем NO_SIGNAL
    if ema_signal == 'NO_SIGNAL':
        return 'NO_SIGNAL'

    # Если TSI фильтр отключен, возвращаем EMA сигнал
    if not use_tsi_filter:
        return ema_signal

    # Проверяем подтверждение TSI
    tsi_confirmed = check_tsi_confirmation(
        bybit_candles, ema_signal, tsi_long, tsi_short, tsi_signal
    )

    # Возвращаем сигнал только если он подтвержден TSI
    return ema_signal if tsi_confirmed else 'NO_SIGNAL'


def get_detailed_signal_info(bybit_candles: List[List[str]],
                             ema_fast: int = 7,
                             ema_medium: int = 14,
                             ema_slow: int = 28,
                             use_tsi_filter: bool = True,
                             tsi_long: int = 25,
                             tsi_short: int = 13,
                             tsi_signal: int = 13) -> Dict:
    """
    Получение детальной информации о EMA сигнале с TSI подтверждением

    Args:
        bybit_candles: Данные свечей от Bybit
        ema_fast: Период быстрой EMA
        ema_medium: Период средней EMA
        ema_slow: Период медленной EMA
        use_tsi_filter: Использовать TSI фильтр
        tsi_long: Длинный период TSI
        tsi_short: Короткий период TSI
        tsi_signal: Период сигнальной линии TSI

    Returns:
        Словарь с детальной информацией о сигнале
    """
    required_data = max(ema_slow, tsi_long, 50) + 20

    if len(bybit_candles) < required_data:
        return {
            'signal': 'NO_SIGNAL',
            'reason': 'INSUFFICIENT_DATA',
            'last_price': float(bybit_candles[0][4]) if bybit_candles else 0,
            'ema_alignment': 'UNKNOWN',
            'tsi_confirmed': False,
            'tsi_used': use_tsi_filter
        }

    # Парсим свечи
    candles = parse_bybit_candles(bybit_candles)

    # Извлекаем цены закрытия
    df = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume'])
    prices = df['close'].values

    # Рассчитываем EMA
    ema_data = calculate_three_ema(prices, ema_fast, ema_medium, ema_slow)

    # Проверяем EMA сигнал на последней свече
    last_idx = len(prices) - 1
    ema_alignment = 'NEUTRAL'
    ema_signal_type = 'NO_SIGNAL'

    if last_idx >= 0:
        if check_ema_alignment(ema_data['ema_fast'], ema_data['ema_medium'],
                               ema_data['ema_slow'], 'LONG', last_idx):
            ema_alignment = 'BULLISH'
            ema_signal_type = 'LONG'
        elif check_ema_alignment(ema_data['ema_fast'], ema_data['ema_medium'],
                                 ema_data['ema_slow'], 'SHORT', last_idx):
            ema_alignment = 'BEARISH'
            ema_signal_type = 'SHORT'

    # Проверяем TSI подтверждение
    tsi_confirmed = False
    tsi_data = None
    final_signal = ema_signal_type

    if use_tsi_filter and ema_signal_type != 'NO_SIGNAL':
        tsi_confirmed = check_tsi_confirmation(
            bybit_candles, ema_signal_type, tsi_long, tsi_short, tsi_signal
        )

        # Получаем TSI данные для дополнительной информации
        tsi_result = calculate_tsi(bybit_candles, tsi_long, tsi_short, tsi_signal)
        if len(tsi_result['tsi']) > 0:
            tsi_data = {
                'tsi_value': tsi_result['tsi'][-1],
                'tsi_signal_value': tsi_result['signal'][-1],
                'tsi_histogram': tsi_result['histogram'][-1]
            }

        # Финальный сигнал только если подтвержден TSI
        if not tsi_confirmed:
            final_signal = 'NO_SIGNAL'

    # Определяем причину
    if final_signal == 'NO_SIGNAL':
        if ema_signal_type == 'NO_SIGNAL':
            reason = 'NO_EMA_ALIGNMENT'
        elif use_tsi_filter and not tsi_confirmed:
            reason = 'TSI_NOT_CONFIRMED'
        else:
            reason = 'UNKNOWN'
    else:
        reason = f'EMA_ALIGNMENT_{ema_alignment}'
        if use_tsi_filter and tsi_confirmed:
            reason += '_TSI_CONFIRMED'

    result = {
        'signal': final_signal,
        'reason': reason,
        'last_price': prices[-1],
        'ema_alignment': ema_alignment,
        'ema_fast_value': ema_data['ema_fast'][-1],
        'ema_medium_value': ema_data['ema_medium'][-1],
        'ema_slow_value': ema_data['ema_slow'][-1],
        'tsi_used': use_tsi_filter,
        'tsi_confirmed': tsi_confirmed
    }

    # Добавляем TSI данные если доступны
    if tsi_data:
        result.update(tsi_data)

    return result


# Дополнительная функция для анализа последней свечи с TSI (совместимость с paste.txt)
def analyze_last_candle_with_tsi(bybit_candles: List[List[str]],
                                 long_length: int = 25,
                                 short_length: int = 13,
                                 signal_length: int = 13) -> str:
    """
    Анализ последней свечи с использованием только TSI индикатора
    ИЗМЕНЕНО: Теперь проверяет пересечение ТОЛЬКО на последней свече

    Args:
        bybit_candles: Данные свечей от Bybit
        long_length: Длинный период TSI
        short_length: Короткий период TSI
        signal_length: Период сигнальной линии

    Returns:
        Строка с результатом: 'LONG', 'SHORT', или 'NO_SIGNAL'
    """
    required_data = max(long_length, short_length, signal_length) + 20

    if len(bybit_candles) < required_data:
        return 'NO_SIGNAL'

    tsi_data = calculate_tsi(bybit_candles, long_length, short_length, signal_length)

    if len(tsi_data['tsi']) < 2:  # Нужно минимум 2 точки
        return 'NO_SIGNAL'

    tsi_values = tsi_data['tsi']
    signal_values = tsi_data['signal']

    # Проверяем ТОЛЬКО последнее пересечение
    current_tsi = tsi_values[-1]
    current_signal = signal_values[-1]
    prev_tsi = tsi_values[-2]
    prev_signal = signal_values[-2]

    # Пересечение TSI и сигнальной линии на последней свече
    if prev_tsi <= prev_signal and current_tsi > current_signal:
        return 'LONG'
    elif prev_tsi >= prev_signal and current_tsi < current_signal:
        return 'SHORT'

    return 'NO_SIGNAL'