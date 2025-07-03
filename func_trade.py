import pandas as pd
import numpy as np
import math
from typing import List, Dict, Tuple, Optional


class CVDNadarayaWatsonIndicator:
    def __init__(self,
                 cvd_ma_length: int = 50,
                 bandwidth: float = 7.0,
                 mult: float = 1.1,
                 repaint: bool = True,
                 max_bars_back: int = 500):
        """
        CVD + Nadaraya-Watson Envelope Combined Indicator

        Args:
            cvd_ma_length: Length for CVD moving average
            bandwidth: Bandwidth parameter for Nadaraya-Watson
            mult: Multiplier for envelope width
            repaint: Whether to use repainting mode
            max_bars_back: Maximum bars to look back
        """
        self.cvd_ma_length = cvd_ma_length
        self.bandwidth = bandwidth
        self.mult = mult
        self.repaint = repaint
        self.max_bars_back = max_bars_back

    def calculate_bull_bear_power(self, candles: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate bull and bear power from OHLCV data

        Args:
            candles: List of candles [open, high, low, close, volume, ...]

        Returns:
            Tuple of (bull_power, bear_power) arrays
        """
        df = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume'])

        bull_power = np.zeros(len(df))
        bear_power = np.zeros(len(df))

        for i in range(len(df)):
            close_curr = df.iloc[i]['close']
            open_curr = df.iloc[i]['open']
            high_curr = df.iloc[i]['high']
            low_curr = df.iloc[i]['low']

            # Get previous close
            close_prev = df.iloc[i - 1]['close'] if i > 0 else open_curr

            # Bull Power calculation
            if close_curr < open_curr:  # Red candle
                if close_prev < open_curr:
                    bull_power[i] = max(high_curr - close_prev, close_curr - low_curr)
                else:
                    bull_power[i] = max(high_curr - open_curr, close_curr - low_curr)
            elif close_curr > open_curr:  # Green candle
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

            # Bear Power calculation
            if close_curr < open_curr:  # Red candle
                if close_prev > open_curr:
                    bear_power[i] = max(close_prev - open_curr, high_curr - low_curr)
                else:
                    bear_power[i] = high_curr - low_curr
            elif close_curr > open_curr:  # Green candle
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

        Returns:
            Tuple of (cvd, cvd_ma, cvd_bullish, cvd_bearish)
        """
        df = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume'])

        bull_power, bear_power = self.calculate_bull_bear_power(candles)

        # Calculate volume distribution
        total_power = bull_power + bear_power
        # Avoid division by zero
        total_power = np.where(total_power == 0, 1, total_power)

        bull_volume = (bull_power / total_power) * df['volume'].values
        bear_volume = (bear_power / total_power) * df['volume'].values

        # Calculate delta and cumulative volume delta
        delta = bull_volume - bear_volume
        cvd = np.cumsum(delta)

        # Calculate CVD moving average
        cvd_ma = pd.Series(cvd).rolling(window=self.cvd_ma_length, min_periods=1).mean().values

        # CVD direction
        cvd_bullish = cvd > cvd_ma
        cvd_bearish = cvd < cvd_ma

        return cvd, cvd_ma, cvd_bullish, cvd_bearish

    def gauss_kernel(self, x: float, h: float) -> float:
        """Gaussian kernel function"""
        return math.exp(-(x * x) / (h * h * 2))

    def calculate_nadaraya_watson(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Nadaraya-Watson estimator and envelope

        Returns:
            Tuple of (nw_estimate, upper_envelope, lower_envelope)
        """
        n = len(prices)
        max_lookback = min(self.max_bars_back, n)

        nw_estimate = np.zeros(n)
        mae_values = np.zeros(n)

        if self.repaint:
            # Repainting mode - use all available data
            for i in range(n):
                sum_weighted = 0.0
                sum_weights = 0.0

                for j in range(max(0, i - max_lookback + 1), min(n, i + max_lookback)):
                    weight = self.gauss_kernel(i - j, self.bandwidth)
                    sum_weighted += prices[j] * weight
                    sum_weights += weight

                if sum_weights > 0:
                    nw_estimate[i] = sum_weighted / sum_weights
                else:
                    nw_estimate[i] = prices[i]

                # Calculate MAE for envelope
                errors = []
                for j in range(max(0, i - max_lookback + 1), min(n, i + 1)):
                    if j < len(nw_estimate):
                        errors.append(abs(prices[j] - nw_estimate[j]))

                if errors:
                    mae_values[i] = np.mean(errors) * self.mult
                else:
                    mae_values[i] = 0
        else:
            # Non-repainting mode - use only historical data
            coefs = np.array([self.gauss_kernel(i, self.bandwidth) for i in range(max_lookback)])
            coefs_sum = np.sum(coefs)

            for i in range(n):
                sum_weighted = 0.0
                available_bars = min(i + 1, max_lookback)

                for j in range(available_bars):
                    if i - j >= 0:
                        sum_weighted += prices[i - j] * coefs[j]

                nw_estimate[i] = sum_weighted / coefs_sum

                # Calculate MAE
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

        Args:
            raw_candles: List of candles in Bybit format
                        [timestamp, open, high, low, close, volume, turnover]

        Returns:
            List of [open, high, low, close, volume] as floats
        """
        parsed_candles = []

        # Reverse the order since Bybit returns newest first, we need oldest first
        for candle in reversed(raw_candles):
            # Extract OHLCV data (skip timestamp and turnover)
            open_price = float(candle[1])
            high_price = float(candle[2])
            low_price = float(candle[3])
            close_price = float(candle[4])
            volume = float(candle[5])

            parsed_candles.append([open_price, high_price, low_price, close_price, volume])

        return parsed_candles

    def get_last_candle_signal(self, raw_candles: List[List[str]]) -> Dict:
        """
        Get signal for the last candle

        Args:
            raw_candles: List of candles in Bybit format

        Returns:
            Dictionary with signal information for the last candle
        """
        if len(raw_candles) < self.cvd_ma_length + 50:  # Need enough data for calculations
            return {
                'signal': 'NO_SIGNAL',
                'reason': 'INSUFFICIENT_DATA',
                'last_price': float(raw_candles[0][4]) if raw_candles else 0,
                'cvd_status': 'UNKNOWN'
            }

        # Parse candles
        candles = self.parse_bybit_candles(raw_candles)

        # Calculate all indicators
        results = self.generate_signals(candles)

        # Check signal on the last candle
        last_idx = len(results['prices']) - 1

        # Check if there's a signal on the last candle
        last_candle_signal = 'NO_SIGNAL'
        signal_reason = 'NO_CROSS'

        if last_idx > 0:
            last_price = results['prices'][last_idx]
            prev_price = results['prices'][last_idx - 1]

            upper_last = results['upper_envelope'][last_idx]
            lower_last = results['lower_envelope'][last_idx]
            upper_prev = results['upper_envelope'][last_idx - 1]
            lower_prev = results['lower_envelope'][last_idx - 1]

            cvd_bullish_last = results['cvd_bullish'][last_idx]
            cvd_bearish_last = results['cvd_bearish'][last_idx]

            # Check for LONG signal (price crosses under lower envelope + CVD bullish)
            if (prev_price >= lower_prev and last_price < lower_last and cvd_bullish_last):
                last_candle_signal = 'LONG'
                signal_reason = 'PRICE_CROSS_UNDER_LOWER_ENVELOPE_CVD_BULLISH'

            # Check for SHORT signal (price crosses over upper envelope + CVD bearish)
            elif (prev_price <= upper_prev and last_price > upper_last and cvd_bearish_last):
                last_candle_signal = 'SHORT'
                signal_reason = 'PRICE_CROSS_OVER_UPPER_ENVELOPE_CVD_BEARISH'

            # Check if price is near envelope but no cross
            elif last_price <= lower_last and cvd_bullish_last:
                signal_reason = 'PRICE_BELOW_LOWER_ENVELOPE_CVD_BULLISH'
            elif last_price >= upper_last and cvd_bearish_last:
                signal_reason = 'PRICE_ABOVE_UPPER_ENVELOPE_CVD_BEARISH'
            elif last_price <= lower_last and cvd_bearish_last:
                signal_reason = 'PRICE_BELOW_LOWER_ENVELOPE_CVD_BEARISH'
            elif last_price >= upper_last and cvd_bullish_last:
                signal_reason = 'PRICE_ABOVE_UPPER_ENVELOPE_CVD_BULLISH'

        return {
            'signal': last_candle_signal,
            'reason': signal_reason,
            'last_price': results['prices'][-1],
            'cvd_status': results['current_cvd_status'],
            'upper_envelope': results['upper_envelope'][-1],
            'lower_envelope': results['lower_envelope'][-1],
            'nadaraya_watson': results['nadaraya_watson'][-1],
            'cvd_value': results['cvd'][-1],
            'cvd_ma_value': results['cvd_ma'][-1]
        }

    def generate_signals(self, candles: List[List[float]]) -> Dict:
        """
        Generate trading signals based on CVD and Nadaraya-Watson envelope

        Args:
            candles: List of OHLCV candles (parsed format)

        Returns:
            Dictionary containing all indicator values and signals
        """
        df = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume'])
        prices = df['close'].values

        # Calculate CVD
        cvd, cvd_ma, cvd_bullish, cvd_bearish = self.calculate_cvd(candles)

        # Calculate Nadaraya-Watson envelope
        nw_estimate, upper_envelope, lower_envelope = self.calculate_nadaraya_watson(prices)

        # Generate signals
        long_signals = []
        short_signals = []

        for i in range(1, len(prices)):
            # Long signal: price crosses under lower envelope AND CVD is bullish
            if (prices[i - 1] >= lower_envelope[i - 1] and
                    prices[i] < lower_envelope[i] and
                    cvd_bullish[i]):
                long_signals.append(i)

            # Short signal: price crosses over upper envelope AND CVD is bearish
            if (prices[i - 1] <= upper_envelope[i - 1] and
                    prices[i] > upper_envelope[i] and
                    cvd_bearish[i]):
                short_signals.append(i)

        return {
            'cvd': cvd,
            'cvd_ma': cvd_ma,
            'cvd_bullish': cvd_bullish,
            'cvd_bearish': cvd_bearish,
            'nadaraya_watson': nw_estimate,
            'upper_envelope': upper_envelope,
            'lower_envelope': lower_envelope,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'current_cvd_status': 'BULLISH' if cvd_bullish[-1] else 'BEARISH',
            'prices': prices
        }


def calculate_atr(candles: list) -> float:
    """
    Вычисляет Average True Range (ATR) на основе свечных данных.

    Args:
        candles: Список данных свечей формата
               [['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'], ...]

    Returns:
        float: Значение ATR или 0, если недостаточно данных
    """
    if len(candles) < 2:
        return 0.0  # Возвращаем 0, если недостаточно данных

    tr_values = []
    for i in range(1, len(candles)):
        try:
            high = float(candles[i][2])
            low = float(candles[i][3])
            prev_close = float(candles[i - 1][4])
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)
        except (IndexError, ValueError):
            # Пропускаем итерацию, если данные некорректные
            continue

    if not tr_values:
        return 0.0  # Возвращаем 0, если не удалось вычислить TR

    atr = sum(tr_values) / len(tr_values)
    return atr


def detect_candlestick_signals(data: dict) -> dict:
    """
    Определяет сигналы на основе паттернов японских свечей

    Args:
        data: Словарь с данными свечей по символам

    Returns:
        dict: Словарь с сигналами 'long' и 'short' для каждого символа
    """
    longs, shorts = set(), set()

    for symbol, content in data.items():
        candles = content['candles']
        if len(candles) != 3:
            continue

        o = []
        h = []
        l = []
        c = []
        for ts, op, hi, lo, cl, *_ in candles:
            op, hi, lo, cl = map(float, (op, hi, lo, cl))
            o.append(op)
            h.append(hi)
            l.append(lo)
            c.append(cl)

        bodies = [abs(c[i] - o[i]) for i in range(3)]
        upper = [h[i] - max(o[i], c[i]) for i in range(3)]
        lower = [min(o[i], c[i]) - l[i] for i in range(3)]
        is_bull = [c[i] > o[i] for i in range(3)]
        is_bear = [not b for b in is_bull]
        mid1 = (o[0] + c[0]) / 2

        # Hammer → long
        if lower[2] >= 2 * bodies[2] and upper[2] <= bodies[2]:
            longs.add(symbol)
        # Shooting Star → short
        if upper[2] >= 2 * bodies[2] and lower[2] <= bodies[2]:
            shorts.add(symbol)
        # Bullish Engulfing → long
        if is_bear[0] and is_bull[1] and o[1] < c[0] and c[1] > o[0]:
            longs.add(symbol)
        # Bearish Engulfing → short
        if is_bull[0] and is_bear[1] and o[1] > c[0] and c[1] < o[0]:
            shorts.add(symbol)
        # Piercing Line → long
        if is_bear[0] and is_bull[1] and o[1] < l[0] and c[1] > mid1:
            longs.add(symbol)
        # Dark Cloud Cover → short
        if is_bull[0] and is_bear[1] and o[1] > h[0] and c[1] < mid1:
            shorts.add(symbol)
        # Morning Star → long
        if is_bear[0] and bodies[1] < (h[1] - l[1]) * 0.3 and is_bull[2] and c[2] > mid1:
            longs.add(symbol)
        # Evening Star → short
        if is_bull[0] and bodies[1] < (h[1] - l[1]) * 0.3 and is_bear[2] and c[2] < mid1:
            shorts.add(symbol)
        # Three White Soldiers → long
        if all(is_bull) and c[0] < c[1] < c[2]:
            longs.add(symbol)
        # Three Black Crows → short
        if all(is_bear) and c[0] > c[1] > c[2]:
            shorts.add(symbol)

    return {
        'long': sorted(longs),
        'short': sorted(shorts)
    }


def compute_cvd_nw_signals(data, cvd_ma_length=100, bandwidth=7.0, mult=1.0, repaint=True):
    """
    Новая функция для вычисления сигналов на основе CVD + Nadaraya-Watson

    Args:
        data: список строк формата [timestamp, open, high, low, close, volume, ...]
        cvd_ma_length: длина периода скользящей средней для CVD
        bandwidth: параметр bandwidth для Nadaraya-Watson
        mult: множитель для конверта
        repaint: использовать ли режим repaint

    Returns:
        список сигналов 'long'/'short'/None для каждой свечи
    """
    # Создаем индикатор
    indicator = CVDNadarayaWatsonIndicator(
        cvd_ma_length=cvd_ma_length,
        bandwidth=bandwidth,
        mult=mult,
        repaint=repaint
    )

    # Парсим данные в нужный формат
    candles = indicator.parse_bybit_candles(data)

    # Если недостаточно данных
    if len(candles) < cvd_ma_length + 50:
        return [None] * len(data)

    # Генерируем сигналы
    results = indicator.generate_signals(candles)

    # Создаем массив сигналов для каждой свечи
    signals = [None] * len(data)

    # Заполняем сигналы на основе индексов long/short сигналов
    for idx in results['long_signals']:
        # Учитываем, что data в обратном порядке (newest first)
        signal_idx = len(data) - 1 - idx
        if 0 <= signal_idx < len(signals):
            signals[signal_idx] = 'long'

    for idx in results['short_signals']:
        # Учитываем, что data в обратном порядке (newest first)
        signal_idx = len(data) - 1 - idx
        if 0 <= signal_idx < len(signals):
            signals[signal_idx] = 'short'

    return signals


def analyze_last_candle(bybit_candles: List[List[str]],
                        cvd_ma_length: int = 100,
                        bandwidth: float = 7.0,
                        mult: float = 1.0) -> str:
    """
    Простая функция для получения сигнала на последней свече

    Args:
        bybit_candles: Данные свечей от Bybit
        cvd_ma_length: Длина MA для CVD
        bandwidth: Пропускная способность NW
        mult: Множитель для конверта

    Returns:
        Строка с результатом: 'LONG', 'SHORT', или 'NO_SIGNAL'
    """
    indicator = CVDNadarayaWatsonIndicator(cvd_ma_length, bandwidth, mult, True)
    result = indicator.get_last_candle_signal(bybit_candles)
    return result['signal']


def get_detailed_signal_info(bybit_candles: List[List[str]],
                             cvd_ma_length: int = 100,
                             bandwidth: float = 7.0,
                             mult: float = 1.0) -> Dict:
    """
    Получает детальную информацию о сигнале на последней свече

    Args:
        bybit_candles: Данные свечей от Bybit
        cvd_ma_length: Длина MA для CVD
        bandwidth: Пропускная способность NW
        mult: Множитель для конверта

    Returns:
        Словарь с детальной информацией о сигнале
    """
    indicator = CVDNadarayaWatsonIndicator(cvd_ma_length, bandwidth, mult, True)
    return indicator.get_last_candle_signal(bybit_candles)


# Для совместимости со старым кодом
def compute_cvd_signals(data, period_ma=100):
    """
    Заменяем старую функцию CVD на новую логику CVD + Nadaraya-Watson

    Args:
        data: список строк формата [timestamp, open, high, low, close, volume, ...]
        period_ma: длина периода скользящей средней (используется как cvd_ma_length)

    Returns:
        список сигналов 'long'/'short'/None для каждой свечи
    """
    return compute_cvd_nw_signals(data, cvd_ma_length=period_ma)


def compute_trend_signals(data, period_ma=100, ema_short=9, ema_medium=21, ema_long=50,
                          rsi_period=14, rsi_overbought=70, rsi_oversold=30,
                          macd_fast=12, macd_slow=26, macd_signal=9):
    """
    Заменяем тренд сигналы на CVD + Nadaraya-Watson

    Args:
        data: список строк формата [timestamp, open, high, low, close, volume, ...]
        period_ma: длина периода скользящей средней (используется как cvd_ma_length)
        остальные параметры игнорируются для совместимости

    Returns:
        список сигналов 'long'/'short'/None для каждой свечи
    """
    return compute_cvd_nw_signals(data, cvd_ma_length=period_ma)


# Пример использования
if __name__ == "__main__":
    # Пример данных от Bybit (как в вашем формате)
    sample_bybit_data = [
        ["1751452440000", "107718.3", "107718.4", "107718.3", "107718.3", "2.073", "223300.2326"],
        ["1751452380000", "107717.4", "107719.6", "107704.9", "107718.3", "21.662", "2333275.5709"],
        ["1751452320000", "107737.7", "107737.7", "107695.2", "107717.4", "43.68", "4704647.5787"],
        # ... больше данных нужно для корректной работы
    ]

    # Простой способ получить сигнал
    signal = analyze_last_candle(sample_bybit_data)
    print(f"Простой сигнал: {signal}")

    # Детальная информация
    detailed_info = get_detailed_signal_info(sample_bybit_data)
    print(f"Детальная информация: {detailed_info}")

    # Сигналы для всех свечей
    all_signals = compute_cvd_nw_signals(sample_bybit_data)
    print(f"Все сигналы: {all_signals}")