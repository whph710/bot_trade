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
                        ema_slow: int = 28) -> str:
    """
    Анализ последней свечи для получения EMA сигнала

    Args:
        bybit_candles: Данные свечей от Bybit
        ema_fast: Период быстрой EMA
        ema_medium: Период средней EMA
        ema_slow: Период медленной EMA

    Returns:
        Строка с результатом: 'LONG', 'SHORT', или 'NO_SIGNAL'
    """
    required_data = max(ema_slow, 50) + 10

    if len(bybit_candles) < required_data:
        return 'NO_SIGNAL'

    # Парсим свечи
    candles = parse_bybit_candles(bybit_candles)

    # Извлекаем цены закрытия
    df = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume'])
    prices = df['close'].values

    # Рассчитываем EMA
    ema_data = calculate_three_ema(prices, ema_fast, ema_medium, ema_slow)

    # Проверяем сигнал на последней свече
    last_idx = len(prices) - 1

    if check_ema_alignment(ema_data['ema_fast'], ema_data['ema_medium'],
                           ema_data['ema_slow'], 'LONG', last_idx):
        return 'LONG'
    elif check_ema_alignment(ema_data['ema_fast'], ema_data['ema_medium'],
                             ema_data['ema_slow'], 'SHORT', last_idx):
        return 'SHORT'

    return 'NO_SIGNAL'


def get_detailed_signal_info(bybit_candles: List[List[str]],
                             ema_fast: int = 7,
                             ema_medium: int = 14,
                             ema_slow: int = 28) -> Dict:
    """
    Получение детальной информации о EMA сигнале

    Args:
        bybit_candles: Данные свечей от Bybit
        ema_fast: Период быстрой EMA
        ema_medium: Период средней EMA
        ema_slow: Период медленной EMA

    Returns:
        Словарь с детальной информацией о сигнале
    """
    required_data = max(ema_slow, 50) + 10

    if len(bybit_candles) < required_data:
        return {
            'signal': 'NO_SIGNAL',
            'reason': 'INSUFFICIENT_DATA',
            'last_price': float(bybit_candles[0][4]) if bybit_candles else 0,
            'ema_alignment': 'UNKNOWN'
        }

    # Парсим свечи
    candles = parse_bybit_candles(bybit_candles)

    # Извлекаем цены закрытия
    df = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume'])
    prices = df['close'].values

    # Рассчитываем EMA
    ema_data = calculate_three_ema(prices, ema_fast, ema_medium, ema_slow)

    # Проверяем сигнал на последней свече
    last_idx = len(prices) - 1
    ema_alignment = 'NEUTRAL'
    signal_type = 'NO_SIGNAL'

    if last_idx >= 0:
        if check_ema_alignment(ema_data['ema_fast'], ema_data['ema_medium'],
                               ema_data['ema_slow'], 'LONG', last_idx):
            ema_alignment = 'BULLISH'
            signal_type = 'LONG'
        elif check_ema_alignment(ema_data['ema_fast'], ema_data['ema_medium'],
                                 ema_data['ema_slow'], 'SHORT', last_idx):
            ema_alignment = 'BEARISH'
            signal_type = 'SHORT'

    return {
        'signal': signal_type,
        'reason': f'EMA_ALIGNMENT_{ema_alignment}',
        'last_price': prices[-1],
        'ema_alignment': ema_alignment,
        'ema_fast_value': ema_data['ema_fast'][-1],
        'ema_medium_value': ema_data['ema_medium'][-1],
        'ema_slow_value': ema_data['ema_slow'][-1]
    }