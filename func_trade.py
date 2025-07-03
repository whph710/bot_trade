import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


class EMAIndicator:
    def __init__(self, ema_fast: int = 7, ema_medium: int = 14, ema_slow: int = 28):
        """
        Simple EMA Indicator for filtering pairs

        Args:
            ema_fast: Fast EMA period (default 7)
            ema_medium: Medium EMA period (default 14)
            ema_slow: Slow EMA period (default 28)
        """
        self.ema_fast = ema_fast
        self.ema_medium = ema_medium
        self.ema_slow = ema_slow

    def calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average (точно как в Pine Script)

        Args:
            prices: Array of prices
            period: EMA period

        Returns:
            Array of EMA values
        """
        ema = np.zeros_like(prices)
        alpha = 2.0 / (period + 1)

        # Первое значение EMA равно первой цене
        ema[0] = prices[0]

        # Расчет последующих значений EMA
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

        return ema

    def parse_bybit_candles(self, raw_candles: List[List[str]]) -> List[List[float]]:
        """
        Parse Bybit candle data format to OHLCV floats
        """
        parsed_candles = []

        # Реверсируем порядок, так как Bybit возвращает новейшие первыми
        for candle in reversed(raw_candles):
            # Извлекаем данные OHLCV
            open_price = float(candle[1])
            high_price = float(candle[2])
            low_price = float(candle[3])
            close_price = float(candle[4])
            volume = float(candle[5])

            parsed_candles.append([open_price, high_price, low_price, close_price, volume])

        return parsed_candles

    def check_ema_alignment(self, ema_fast: np.ndarray, ema_medium: np.ndarray, ema_slow: np.ndarray,
                            signal_type: str, index: int) -> bool:
        """
        Check if EMA alignment supports the signal direction

        Args:
            ema_fast: Fast EMA values
            ema_medium: Medium EMA values
            ema_slow: Slow EMA values
            signal_type: 'LONG' or 'SHORT'
            index: Index to check

        Returns:
            True if EMA alignment supports the signal
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

    def get_last_candle_signal(self, raw_candles: List[List[str]]) -> Dict:
        """
        Get EMA alignment signal for the last candle
        """
        required_data = max(self.ema_slow, 50) + 10

        if len(raw_candles) < required_data:
            return {
                'signal': 'NO_SIGNAL',
                'reason': 'INSUFFICIENT_DATA',
                'last_price': float(raw_candles[0][4]) if raw_candles else 0,
                'ema_alignment': 'UNKNOWN'
            }

        # Парсим свечи
        candles = self.parse_bybit_candles(raw_candles)

        # Рассчитываем EMA
        results = self.generate_signals(candles)

        # Проверяем сигнал на последней свече
        last_idx = len(results['prices']) - 1
        ema_alignment = 'NEUTRAL'
        signal_type = 'NO_SIGNAL'

        if last_idx >= 0:
            # Проверяем текущее выравнивание EMA
            if self.check_ema_alignment(results['ema_fast'], results['ema_medium'],
                                        results['ema_slow'], 'LONG', last_idx):
                ema_alignment = 'BULLISH'
                signal_type = 'LONG'
            elif self.check_ema_alignment(results['ema_fast'], results['ema_medium'],
                                          results['ema_slow'], 'SHORT', last_idx):
                ema_alignment = 'BEARISH'
                signal_type = 'SHORT'

        return {
            'signal': signal_type,
            'reason': f'EMA_ALIGNMENT_{ema_alignment}',
            'last_price': results['prices'][-1],
            'ema_alignment': ema_alignment,
            'ema_fast_value': results['ema_fast'][-1],
            'ema_medium_value': results['ema_medium'][-1],
            'ema_slow_value': results['ema_slow'][-1]
        }

    def generate_signals(self, candles: List[List[float]]) -> Dict:
        """
        Generate EMA-based signals
        """
        df = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume'])
        prices = df['close'].values

        # Рассчитываем 3 EMA
        ema_fast = self.calculate_ema(prices, self.ema_fast)
        ema_medium = self.calculate_ema(prices, self.ema_medium)
        ema_slow = self.calculate_ema(prices, self.ema_slow)

        # Определяем текущее выравнивание
        current_alignment = 'NEUTRAL'
        last_idx = len(prices) - 1

        if last_idx >= 0:
            if self.check_ema_alignment(ema_fast, ema_medium, ema_slow, 'LONG', last_idx):
                current_alignment = 'BULLISH'
            elif self.check_ema_alignment(ema_fast, ema_medium, ema_slow, 'SHORT', last_idx):
                current_alignment = 'BEARISH'

        return {
            'ema_fast': ema_fast,
            'ema_medium': ema_medium,
            'ema_slow': ema_slow,
            'current_alignment': current_alignment,
            'prices': prices
        }


def analyze_last_candle(bybit_candles: List[List[str]],
                        ema_fast: int = 7,
                        ema_medium: int = 14,
                        ema_slow: int = 28) -> str:
    """
    Простая функция для получения EMA сигнала на последней свече

    Args:
        bybit_candles: Данные свечей от Bybit
        ema_fast: Период быстрой EMA
        ema_medium: Период средней EMA
        ema_slow: Период медленной EMA

    Returns:
        Строка с результатом: 'LONG', 'SHORT', или 'NO_SIGNAL'
    """
    indicator = EMAIndicator(
        ema_fast=ema_fast,
        ema_medium=ema_medium,
        ema_slow=ema_slow
    )
    result = indicator.get_last_candle_signal(bybit_candles)
    return result['signal']


def get_detailed_signal_info(bybit_candles: List[List[str]],
                             ema_fast: int = 7,
                             ema_medium: int = 14,
                             ema_slow: int = 28) -> Dict:
    """
    Получает детальную информацию о EMA сигнале на последней свече

    Args:
        bybit_candles: Данные свечей от Bybit
        ema_fast: Период быстрой EMA
        ema_medium: Период средней EMA
        ema_slow: Период медленной EMA

    Returns:
        Словарь с детальной информацией о сигнале
    """
    indicator = EMAIndicator(
        ema_fast=ema_fast,
        ema_medium=ema_medium,
        ema_slow=ema_slow
    )
    return indicator.get_last_candle_signal(bybit_candles)