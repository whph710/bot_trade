import pandas as pd
import numpy as np
import math
from typing import List, Dict, Tuple


class CVDNadarayaWatsonEMAIndicator:
    def __init__(self,
                 cvd_ma_length: int = 50,
                 bandwidth: float = 7.0,
                 mult: float = 1.1,
                 repaint: bool = True,
                 max_bars_back: int = 500,
                 ema_fast: int = 7,
                 ema_medium: int = 14,
                 ema_slow: int = 28):
        """
        CVD + Nadaraya-Watson + 3 EMA Combined Indicator

        Args:
            cvd_ma_length: Length for CVD moving average
            bandwidth: Bandwidth parameter for Nadaraya-Watson
            mult: Multiplier for envelope width
            repaint: Whether to use repainting mode
            max_bars_back: Maximum bars to look back
            ema_fast: Fast EMA period (default 7)
            ema_medium: Medium EMA period (default 14)
            ema_slow: Slow EMA period (default 28)
        """
        self.cvd_ma_length = cvd_ma_length
        self.bandwidth = bandwidth
        self.mult = mult
        self.repaint = repaint
        self.max_bars_back = max_bars_back
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

    def calculate_bull_bear_power(self, candles: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate bull and bear power from OHLCV data (точно как в Pine Script)
        """
        df = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume'])

        bull_power = np.zeros(len(df))
        bear_power = np.zeros(len(df))

        for i in range(len(df)):
            close_curr = df.iloc[i]['close']
            open_curr = df.iloc[i]['open']
            high_curr = df.iloc[i]['high']
            low_curr = df.iloc[i]['low']

            # Получаем предыдущий close
            close_prev = df.iloc[i - 1]['close'] if i > 0 else open_curr

            # Bull Power расчет (точно как в Pine Script)
            if close_curr < open_curr:  # Красная свеча
                if close_prev < open_curr:
                    bull_power[i] = max(high_curr - close_prev, close_curr - low_curr)
                else:
                    bull_power[i] = max(high_curr - open_curr, close_curr - low_curr)
            elif close_curr > open_curr:  # Зеленая свеча
                if close_prev > open_curr:
                    bull_power[i] = high_curr - low_curr
                else:
                    bull_power[i] = max(open_curr - close_prev, high_curr - low_curr)
            else:  # Doji
                if high_curr - close_curr > close_curr - low_curr:
                    if close_prev < open_curr:
                        bull_power[i] = max(high_curr - close_prev, close_curr - low_curr)
                    else:
                        bull_power[i] = high_curr - open_curr
                elif high_curr - close_curr < close_curr - low_curr:
                    if close_prev > open_curr:
                        bull_power[i] = high_curr - low_curr
                    else:
                        bull_power[i] = max(open_curr - close_prev, high_curr - low_curr)
                else:
                    if close_prev > open_curr:
                        bull_power[i] = max(high_curr - open_curr, close_curr - low_curr)
                    elif close_prev < open_curr:
                        bull_power[i] = max(open_curr - close_prev, high_curr - low_curr)
                    else:
                        bull_power[i] = high_curr - low_curr

            # Bear Power расчет (точно как в Pine Script)
            if close_curr < open_curr:  # Красная свеча
                if close_prev > open_curr:
                    bear_power[i] = max(close_prev - open_curr, high_curr - low_curr)
                else:
                    bear_power[i] = high_curr - low_curr
            elif close_curr > open_curr:  # Зеленая свеча
                if close_prev > open_curr:
                    bear_power[i] = max(close_prev - low_curr, high_curr - close_curr)
                else:
                    bear_power[i] = max(open_curr - low_curr, high_curr - close_curr)
            else:  # Doji
                if high_curr - close_curr > close_curr - low_curr:
                    if close_prev > open_curr:
                        bear_power[i] = max(close_prev - open_curr, high_curr - low_curr)
                    else:
                        bear_power[i] = high_curr - low_curr
                elif high_curr - close_curr < close_curr - low_curr:
                    if close_prev > open_curr:
                        bear_power[i] = max(close_prev - low_curr, high_curr - close_curr)
                    else:
                        bear_power[i] = open_curr - low_curr
                else:
                    if close_prev > open_curr:
                        bear_power[i] = max(close_prev - open_curr, high_curr - low_curr)
                    elif close_prev < open_curr:
                        bear_power[i] = max(open_curr - low_curr, high_curr - close_curr)
                    else:
                        bear_power[i] = high_curr - low_curr

        return bull_power, bear_power

    def calculate_cvd(self, candles: List[List[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Cumulative Volume Delta (CVD)
        """
        df = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume'])

        bull_power, bear_power = self.calculate_bull_bear_power(candles)

        # Расчет распределения объема
        total_power = bull_power + bear_power
        # Избегаем деления на ноль
        total_power = np.where(total_power == 0, 1, total_power)

        bull_volume = (bull_power / total_power) * df['volume'].values
        bear_volume = (bear_power / total_power) * df['volume'].values

        # Расчет дельты и кумулятивной дельты объема
        delta = bull_volume - bear_volume
        cvd = np.cumsum(delta)

        # Расчет скользящего среднего CVD (SMA как в Pine Script)
        cvd_ma = pd.Series(cvd).rolling(window=self.cvd_ma_length, min_periods=1).mean().values

        # Направление CVD
        cvd_bullish = cvd > cvd_ma
        cvd_bearish = cvd < cvd_ma

        return cvd, cvd_ma, cvd_bullish, cvd_bearish

    def gauss_kernel(self, x: float, h: float) -> float:
        """Gaussian kernel function"""
        return math.exp(-(x * x) / (h * h * 2))

    def calculate_nadaraya_watson(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Nadaraya-Watson estimator and envelope (точно как в Pine Script)
        """
        n = len(prices)
        max_lookback = min(self.max_bars_back, n)

        nw_estimate = np.zeros(n)
        mae_values = np.zeros(n)

        if self.repaint:
            # Режим перерисовки - используем все доступные данные
            for i in range(n):
                sum_weighted = 0.0
                sum_weights = 0.0

                # Используем симметричное окно вокруг текущей точки
                for j in range(max(0, i - max_lookback + 1), min(n, i + max_lookback)):
                    weight = self.gauss_kernel(i - j, self.bandwidth)
                    sum_weighted += prices[j] * weight
                    sum_weights += weight

                if sum_weights > 0:
                    nw_estimate[i] = sum_weighted / sum_weights
                else:
                    nw_estimate[i] = prices[i]

                # Расчет MAE для конверта
                errors = []
                for j in range(max(0, i - max_lookback + 1), min(n, i + 1)):
                    if j < len(nw_estimate):
                        errors.append(abs(prices[j] - nw_estimate[j]))

                if errors:
                    mae_values[i] = np.mean(errors) * self.mult
                else:
                    mae_values[i] = 0
        else:
            # Режим без перерисовки - используем только исторические данные
            coefs = np.array([self.gauss_kernel(i, self.bandwidth) for i in range(max_lookback)])
            coefs_sum = np.sum(coefs)

            for i in range(n):
                sum_weighted = 0.0
                available_bars = min(i + 1, max_lookback)

                for j in range(available_bars):
                    if i - j >= 0:
                        sum_weighted += prices[i - j] * coefs[j]

                nw_estimate[i] = sum_weighted / coefs_sum

                # Расчет MAE
                errors = []
                for j in range(max(0, i - available_bars + 1), i + 1):
                    errors.append(abs(prices[j] - nw_estimate[j]))

                mae_values[i] = np.mean(errors) * self.mult if errors else 0

        upper_envelope = nw_estimate + mae_values
        lower_envelope = nw_estimate - mae_values

        return nw_estimate, upper_envelope, lower_envelope

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
        Get signal for the last candle with EMA confirmation
        """
        required_data = max(self.cvd_ma_length, self.ema_slow, self.max_bars_back) + 50

        if len(raw_candles) < required_data:
            return {
                'signal': 'NO_SIGNAL',
                'reason': 'INSUFFICIENT_DATA',
                'last_price': float(raw_candles[0][4]) if raw_candles else 0,
                'cvd_status': 'UNKNOWN',
                'ema_alignment': 'UNKNOWN'
            }

        # Парсим свечи
        candles = self.parse_bybit_candles(raw_candles)

        # Рассчитываем все индикаторы
        results = self.generate_signals(candles)

        # Проверяем сигнал на последней свече
        last_idx = len(results['prices']) - 1

        # Проверяем наличие сигнала на последней свече
        last_candle_signal = 'NO_SIGNAL'
        signal_reason = 'NO_CROSS'
        ema_alignment = 'NEUTRAL'

        if last_idx > 0:
            last_price = results['prices'][last_idx]
            prev_price = results['prices'][last_idx - 1]

            upper_last = results['upper_envelope'][last_idx]
            lower_last = results['lower_envelope'][last_idx]
            upper_prev = results['upper_envelope'][last_idx - 1]
            lower_prev = results['lower_envelope'][last_idx - 1]

            cvd_bullish_last = results['cvd_bullish'][last_idx]
            cvd_bearish_last = results['cvd_bearish'][last_idx]

            # Проверяем сигналы CVD + NW сначала
            cvd_nw_signal = None

            # Проверяем сигнал на LONG (цена пересекает снизу нижнюю границу конверта + CVD бычий)
            if (prev_price >= lower_prev and last_price < lower_last and cvd_bullish_last):
                cvd_nw_signal = 'LONG'
                signal_reason = 'PRICE_CROSS_UNDER_LOWER_ENVELOPE_CVD_BULLISH'

            # Проверяем сигнал на SHORT (цена пересекает сверху верхнюю границу конверта + CVD медвежий)
            elif (prev_price <= upper_prev and last_price > upper_last and cvd_bearish_last):
                cvd_nw_signal = 'SHORT'
                signal_reason = 'PRICE_CROSS_OVER_UPPER_ENVELOPE_CVD_BEARISH'

            # Если есть сигнал CVD + NW, проверяем подтверждение EMA
            if cvd_nw_signal:
                ema_supports_signal = self.check_ema_alignment(
                    results['ema_fast'],
                    results['ema_medium'],
                    results['ema_slow'],
                    cvd_nw_signal,
                    last_idx
                )

                if ema_supports_signal:
                    last_candle_signal = cvd_nw_signal
                    ema_alignment = 'BULLISH' if cvd_nw_signal == 'LONG' else 'BEARISH'
                    signal_reason += '_EMA_CONFIRMED'
                else:
                    signal_reason += '_EMA_NOT_ALIGNED'
                    ema_alignment = 'CONFLICTING'
            else:
                # Проверяем текущее выравнивание EMA даже без сигнала
                if self.check_ema_alignment(results['ema_fast'], results['ema_medium'],
                                            results['ema_slow'], 'LONG', last_idx):
                    ema_alignment = 'BULLISH'
                elif self.check_ema_alignment(results['ema_fast'], results['ema_medium'],
                                              results['ema_slow'], 'SHORT', last_idx):
                    ema_alignment = 'BEARISH'
                else:
                    ema_alignment = 'NEUTRAL'

        return {
            'signal': last_candle_signal,
            'reason': signal_reason,
            'last_price': results['prices'][-1],
            'cvd_status': results['current_cvd_status'],
            'ema_alignment': ema_alignment,
            'upper_envelope': results['upper_envelope'][-1],
            'lower_envelope': results['lower_envelope'][-1],
            'nadaraya_watson': results['nadaraya_watson'][-1],
            'cvd_value': results['cvd'][-1],
            'cvd_ma_value': results['cvd_ma'][-1],
            'ema_fast_value': results['ema_fast'][-1],
            'ema_medium_value': results['ema_medium'][-1],
            'ema_slow_value': results['ema_slow'][-1]
        }

    def generate_signals(self, candles: List[List[float]]) -> Dict:
        """
        Generate trading signals based on CVD, Nadaraya-Watson envelope and EMA confirmation
        """
        df = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume'])
        prices = df['close'].values

        # Рассчитываем CVD
        cvd, cvd_ma, cvd_bullish, cvd_bearish = self.calculate_cvd(candles)

        # Рассчитываем Nadaraya-Watson envelope
        nw_estimate, upper_envelope, lower_envelope = self.calculate_nadaraya_watson(prices)

        # Рассчитываем 3 EMA
        ema_fast = self.calculate_ema(prices, self.ema_fast)
        ema_medium = self.calculate_ema(prices, self.ema_medium)
        ema_slow = self.calculate_ema(prices, self.ema_slow)

        # Генерируем сигналы с подтверждением EMA
        long_signals = []
        short_signals = []

        for i in range(1, len(prices)):
            # Сигнал на LONG: цена пересекает снизу нижнюю границу конверта И CVD бычий И EMA выравнивание бычье
            if (prices[i - 1] >= lower_envelope[i - 1] and
                    prices[i] < lower_envelope[i] and
                    cvd_bullish[i] and
                    self.check_ema_alignment(ema_fast, ema_medium, ema_slow, 'LONG', i)):
                long_signals.append(i)

            # Сигнал на SHORT: цена пересекает сверху верхнюю границу конверта И CVD медвежий И EMA выравнивание медвежье
            if (prices[i - 1] <= upper_envelope[i - 1] and
                    prices[i] > upper_envelope[i] and
                    cvd_bearish[i] and
                    self.check_ema_alignment(ema_fast, ema_medium, ema_slow, 'SHORT', i)):
                short_signals.append(i)

        return {
            'cvd': cvd,
            'cvd_ma': cvd_ma,
            'cvd_bullish': cvd_bullish,
            'cvd_bearish': cvd_bearish,
            'nadaraya_watson': nw_estimate,
            'upper_envelope': upper_envelope,
            'lower_envelope': lower_envelope,
            'ema_fast': ema_fast,
            'ema_medium': ema_medium,
            'ema_slow': ema_slow,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'current_cvd_status': 'BULLISH' if cvd_bullish[-1] else 'BEARISH',
            'prices': prices
        }


def analyze_last_candle(bybit_candles: List[List[str]],
                        cvd_ma_length: int = 50,
                        bandwidth: float = 7.0,
                        mult: float = 1.1,
                        ema_fast: int = 7,
                        ema_medium: int = 14,
                        ema_slow: int = 28) -> str:
    """
    Простая функция для получения сигнала на последней свече с подтверждением EMA

    Args:
        bybit_candles: Данные свечей от Bybit
        cvd_ma_length: Длина MA для CVD
        bandwidth: Пропускная способность NW
        mult: Множитель для конверта
        ema_fast: Период быстрой EMA
        ema_medium: Период средней EMA
        ema_slow: Период медленной EMA

    Returns:
        Строка с результатом: 'LONG', 'SHORT', или 'NO_SIGNAL'
    """
    indicator = CVDNadarayaWatsonEMAIndicator(
        cvd_ma_length=cvd_ma_length,
        bandwidth=bandwidth,
        mult=mult,
        repaint=True,
        ema_fast=ema_fast,
        ema_medium=ema_medium,
        ema_slow=ema_slow
    )
    result = indicator.get_last_candle_signal(bybit_candles)
    return result['signal']


def get_detailed_signal_info(bybit_candles: List[List[str]],
                             cvd_ma_length: int = 50,
                             bandwidth: float = 7.0,
                             mult: float = 1.1,
                             ema_fast: int = 7,
                             ema_medium: int = 14,
                             ema_slow: int = 28) -> Dict:
    """
    Получает детальную информацию о сигнале на последней свече с данными EMA

    Args:
        bybit_candles: Данные свечей от Bybit
        cvd_ma_length: Длина MA для CVD
        bandwidth: Пропускная способность NW
        mult: Множитель для конверта
        ema_fast: Период быстрой EMA
        ema_medium: Период средней EMA
        ema_slow: Период медленной EMA

    Returns:
        Словарь с детальной информацией о сигнале
    """
    indicator = CVDNadarayaWatsonEMAIndicator(
        cvd_ma_length=cvd_ma_length,
        bandwidth=bandwidth,
        mult=mult,
        repaint=True,
        ema_fast=ema_fast,
        ema_medium=ema_medium,
        ema_slow=ema_slow
    )
    return indicator.get_last_candle_signal(bybit_candles)