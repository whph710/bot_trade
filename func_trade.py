"""
Модуль индикаторов и торговой логики
Упрощённый, без кэширования
"""

import numpy as np
from typing import List, Dict, Any, Optional
from config import config
import logging

logger = logging.getLogger(__name__)


def safe_float(value) -> float:
    """Безопасное преобразование в float"""
    try:
        if isinstance(value, np.ndarray):
            value = value[-1] if len(value) > 0 else 0.0

        if value is None:
            return 0.0

        result = float(value)
        return 0.0 if (np.isnan(result) or np.isinf(result)) else result
    except (ValueError, TypeError):
        return 0.0


def validate_candles(candles: List[List[str]], min_length: int = 10) -> bool:
    """Валидация свечных данных"""
    if not candles or len(candles) < min_length:
        return False

    try:
        for candle in candles[:3]:
            if not isinstance(candle, list) or len(candle) < 6:
                return False

            try:
                open_p = float(candle[1])
                high = float(candle[2])
                low = float(candle[3])
                close = float(candle[4])
                volume = float(candle[5])

                if any(p <= 0 for p in [open_p, high, low, close]):
                    return False

                if high < max(open_p, close) or low > min(open_p, close):
                    return False

                if volume < 0:
                    return False

            except (ValueError, IndexError):
                return False

        return True

    except Exception as e:
        logger.debug(f"Candle validation error: {e}")
        return False


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Расчёт EMA"""
    if len(prices) < period:
        return np.full_like(prices, prices[0] if len(prices) > 0 else 0)

    try:
        prices = np.array([safe_float(p) for p in prices])

        if np.all(prices == 0) or len(prices) == 0:
            return np.zeros_like(prices)

        ema = np.zeros_like(prices, dtype=np.float64)
        alpha = 2.0 / (period + 1)

        ema[0] = next((p for p in prices if p > 0), prices[0])

        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

        return ema
    except Exception as e:
        logger.error(f"EMA calculation error: {e}")
        return np.full_like(prices, prices[0] if len(prices) > 0 else 0)


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Расчёт RSI"""
    if len(prices) < period + 1:
        return np.full_like(prices, 50.0)

    try:
        prices = np.array([safe_float(p) for p in prices])

        if np.all(prices == 0) or len(prices) < 2:
            return np.full_like(prices, 50.0)

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        gains = np.concatenate([[0], gains])
        losses = np.concatenate([[0], losses])

        avg_gains = np.zeros_like(prices)
        avg_losses = np.zeros_like(prices)

        if len(gains) >= period:
            avg_gains[period] = np.mean(gains[1:period+1])
            avg_losses[period] = np.mean(losses[1:period+1])

        alpha = 1.0 / period
        for i in range(period + 1, len(prices)):
            avg_gains[i] = alpha * gains[i] + (1 - alpha) * avg_gains[i - 1]
            avg_losses[i] = alpha * losses[i] + (1 - alpha) * avg_losses[i - 1]

        rsi = np.full_like(prices, 50.0)
        for i in range(period, len(prices)):
            if avg_losses[i] != 0:
                rs = avg_gains[i] / avg_losses[i]
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100 if avg_gains[i] > 0 else 50

        return np.clip(rsi, 0, 100)
    except Exception as e:
        logger.error(f"RSI calculation error: {e}")
        return np.full_like(prices, 50.0)


def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, np.ndarray]:
    """Расчёт MACD"""
    zero_array = np.zeros_like(prices)

    if len(prices) < max(fast, slow):
        return {'line': zero_array, 'signal': zero_array, 'histogram': zero_array}

    try:
        ema_fast = calculate_ema(prices, fast)
        ema_slow = calculate_ema(prices, slow)

        macd_line = ema_fast - ema_slow
        signal_line = calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line

        return {'line': macd_line, 'signal': signal_line, 'histogram': histogram}
    except Exception as e:
        logger.error(f"MACD calculation error: {e}")
        return {'line': zero_array, 'signal': zero_array, 'histogram': zero_array}


def calculate_atr(candles: List[List[str]], period: int = 14) -> float:
    """Расчёт ATR"""
    if not validate_candles(candles, period + 1):
        return 0.0

    try:
        highs = np.array([safe_float(c[2]) for c in candles])
        lows = np.array([safe_float(c[3]) for c in candles])
        closes = np.array([safe_float(c[4]) for c in candles])

        if np.any(highs <= 0) or np.any(lows <= 0) or np.any(closes <= 0):
            return 0.0

        tr = np.zeros(len(candles))
        for i in range(1, len(candles)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )

        if len(tr) <= period:
            return safe_float(np.mean(tr[1:]))

        atr = np.mean(tr[1:period + 1])
        for i in range(period + 1, len(candles)):
            atr = (atr * (period - 1) + tr[i]) / period

        return safe_float(atr)
    except Exception as e:
        logger.error(f"ATR calculation error: {e}")
        return 0.0


def calculate_volume_ratios(volumes: np.ndarray, window: int = 20) -> np.ndarray:
    """Расчёт отношения объёмов"""
    try:
        volumes = np.array([safe_float(v) for v in volumes])

        if len(volumes) < window:
            return np.ones_like(volumes)

        ratios = np.ones_like(volumes)

        for i in range(window, len(volumes)):
            avg_volume = np.mean(volumes[max(0, i-window):i])
            if avg_volume > 0:
                ratios[i] = volumes[i] / avg_volume
            else:
                ratios[i] = 1.0

        return ratios
    except Exception as e:
        logger.error(f"Volume ratios calculation error: {e}")
        return np.ones_like(volumes)


def calculate_basic_indicators(candles: List[List[str]]) -> Dict[str, Any]:
    """Базовые индикаторы (для быстрого сканирования)"""
    if not validate_candles(candles, 20):
        return {}

    try:
        closes = np.array([safe_float(c[4]) for c in candles])
        volumes = np.array([safe_float(c[5]) for c in candles])

        if len(closes) < 20 or np.all(closes == 0) or np.all(volumes == 0):
            return {}

        ema5 = calculate_ema(closes, config.EMA_FAST)
        ema8 = calculate_ema(closes, config.EMA_MEDIUM)
        ema20 = calculate_ema(closes, config.EMA_SLOW)

        rsi = calculate_rsi(closes, config.RSI_PERIOD)
        macd = calculate_macd(closes, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL)
        atr = calculate_atr(candles, config.ATR_PERIOD)
        volume_ratios = calculate_volume_ratios(volumes)

        return {
            'price': safe_float(closes[-1]),
            'ema5': safe_float(ema5[-1]),
            'ema8': safe_float(ema8[-1]),
            'ema20': safe_float(ema20[-1]),
            'rsi': safe_float(rsi[-1]),
            'macd_line': safe_float(macd['line'][-1]),
            'macd_signal': safe_float(macd['signal'][-1]),
            'macd_histogram': safe_float(macd['histogram'][-1]),
            'atr': safe_float(atr),
            'volume_ratio': safe_float(volume_ratios[-1])
        }

    except Exception as e:
        logger.error(f"Basic indicators calculation error: {e}")
        return {}


def calculate_ai_indicators(candles: List[List[str]], history_length: int) -> Dict[str, Any]:
    """Индикаторы с историей для AI анализа"""
    if not validate_candles(candles, max(history_length, 20)):
        return {}

    try:
        closes = np.array([safe_float(c[4]) for c in candles])
        volumes = np.array([safe_float(c[5]) for c in candles])

        if len(closes) < history_length or np.all(closes == 0):
            return {}

        ema5 = calculate_ema(closes, config.EMA_FAST)
        ema8 = calculate_ema(closes, config.EMA_MEDIUM)
        ema20 = calculate_ema(closes, config.EMA_SLOW)
        rsi = calculate_rsi(closes, config.RSI_PERIOD)
        macd = calculate_macd(closes, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL)
        volume_ratios = calculate_volume_ratios(volumes)

        def safe_history(arr, length):
            """Безопасное извлечение истории"""
            try:
                if len(arr) >= length:
                    return [safe_float(x) for x in arr[-length:]]
                else:
                    first_val = safe_float(arr[0]) if len(arr) > 0 else 0.0
                    result = [first_val] * (length - len(arr))
                    result.extend([safe_float(x) for x in arr])
                    return result
            except Exception:
                return [0.0] * length

        return {
            'ema5_history': safe_history(ema5, history_length),
            'ema8_history': safe_history(ema8, history_length),
            'ema20_history': safe_history(ema20, history_length),
            'rsi_history': safe_history(rsi, history_length),
            'macd_line_history': safe_history(macd['line'], history_length),
            'macd_signal_history': safe_history(macd['signal'], history_length),
            'macd_histogram_history': safe_history(macd['histogram'], history_length),
            'volume_ratio_history': safe_history(volume_ratios, history_length),
            'current': {
                'price': safe_float(closes[-1]),
                'ema5': safe_float(ema5[-1]),
                'ema8': safe_float(ema8[-1]),
                'ema20': safe_float(ema20[-1]),
                'rsi': safe_float(rsi[-1]),
                'macd_line': safe_float(macd['line'][-1]),
                'macd_histogram': safe_float(macd['histogram'][-1]),
                'volume_ratio': safe_float(volume_ratios[-1]),
                'atr': safe_float(calculate_atr(candles, config.ATR_PERIOD))
            }
        }

    except Exception as e:
        logger.error(f"AI indicators calculation error: {e}")
        return {}


def check_basic_signal(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """Проверка базового сигнала"""
    if not indicators:
        return {'signal': False, 'confidence': 0, 'direction': 'NONE'}

    try:
        price = safe_float(indicators.get('price', 0))
        ema5 = safe_float(indicators.get('ema5', 0))
        ema8 = safe_float(indicators.get('ema8', 0))
        ema20 = safe_float(indicators.get('ema20', 0))
        rsi = safe_float(indicators.get('rsi', 50))
        macd_hist = safe_float(indicators.get('macd_histogram', 0))
        volume_ratio = safe_float(indicators.get('volume_ratio', 1.0))
        atr = safe_float(indicators.get('atr', 0))

        if price <= 0 or ema5 <= 0 or ema8 <= 0 or ema20 <= 0:
            return {'signal': False, 'confidence': 0, 'direction': 'NONE'}

        if not (0 <= rsi <= 100):
            rsi = 50

        conditions = []

        # EMA alignment
        if price > ema5 and ema5 > ema8 and ema8 > ema20:
            conditions.append(('LONG', 25))

        if price < ema5 and ema5 < ema8 and ema8 < ema20:
            conditions.append(('SHORT', 25))

        # RSI
        if 30.0 < rsi < 70.0:
            conditions.append(('ANY', 15))

        # MACD
        if abs(macd_hist) > 0.001:
            conditions.append(('ANY', 15))

        # Volume
        if volume_ratio >= config.MIN_VOLUME_RATIO:
            conditions.append(('ANY', 20))

        # ATR
        if atr > 0.001:
            conditions.append(('ANY', 10))

        if not conditions:
            return {'signal': False, 'confidence': 0, 'direction': 'NONE'}

        long_score = sum(score for direction, score in conditions if direction in ['LONG', 'ANY'])
        short_score = sum(score for direction, score in conditions if direction in ['SHORT', 'ANY'])

        if long_score > short_score and long_score >= config.MIN_CONFIDENCE:
            return {'signal': True, 'confidence': int(long_score), 'direction': 'LONG'}
        elif short_score > long_score and short_score >= config.MIN_CONFIDENCE:
            return {'signal': True, 'confidence': int(short_score), 'direction': 'SHORT'}
        else:
            return {'signal': False, 'confidence': int(max(long_score, short_score)), 'direction': 'NONE'}

    except Exception as e:
        logger.error(f"Signal check error: {e}")
        return {'signal': False, 'confidence': 0, 'direction': 'NONE'}