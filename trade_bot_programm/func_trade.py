"""
Technical indicators with Zero-Lag MA Trend Levels
Файл: trade_bot_programm/func_trade.py

ИЗМЕНЕНИЯ:
- Интегрирован индикатор Zero-Lag MA (ZLMA)
- Stage 1 сигнал = треугольник пробоя коробки (из Pine Script)
- Логика: EMA → ZLMA → ATR → коробки → треугольники
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from config import config
import logging

logger = logging.getLogger(__name__)


def safe_float(value) -> float:
    """Safe float conversion"""
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
    """Validate candle data"""
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
    """Calculate EMA"""
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


def calculate_zlma(prices: np.ndarray, length: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Zero-Lag Moving Average (ZLMA)

    Returns:
        (ema, zlma) - regular EMA and Zero-Lag MA
    """
    if len(prices) < length:
        return np.zeros_like(prices), np.zeros_like(prices)

    try:
        # Step 1: Regular EMA
        ema = calculate_ema(prices, length)

        # Step 2: Correction = close + (close - EMA)
        correction = prices + (prices - ema)

        # Step 3: ZLMA = EMA of correction
        zlma = calculate_ema(correction, length)

        return ema, zlma
    except Exception as e:
        logger.error(f"ZLMA calculation error: {e}")
        return np.zeros_like(prices), np.zeros_like(prices)


def calculate_atr_wilder(candles: List[List[str]], period: int = 200) -> np.ndarray:
    """
    Calculate ATR using Wilder's smoothing (RMA-like via ewm)
    Matching Pine Script ta.atr(200)
    """
    if not validate_candles(candles, period + 1):
        return np.zeros(len(candles))

    try:
        highs = np.array([safe_float(c[2]) for c in candles])
        lows = np.array([safe_float(c[3]) for c in candles])
        closes = np.array([safe_float(c[4]) for c in candles])

        if np.any(highs <= 0) or np.any(lows <= 0) or np.any(closes <= 0):
            return np.zeros(len(candles))

        # True Range calculation
        tr = np.zeros(len(candles))
        for i in range(1, len(candles)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )

        # Wilder's smoothing using ewm with alpha=1/period
        # This approximates Pine Script's ta.atr()
        atr = np.zeros(len(candles))

        # Initialize with SMA for first period
        if len(tr) > period:
            atr[period] = np.mean(tr[1:period + 1])

            # Apply Wilder's smoothing
            alpha = 1.0 / period
            for i in range(period + 1, len(candles)):
                atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

        return atr
    except Exception as e:
        logger.error(f"ATR calculation error: {e}")
        return np.zeros(len(candles))


def detect_zlma_trend_boxes(
    candles: List[List[str]],
    length: int = 15,
    atr_length: int = 200
) -> Dict[str, Any]:
    """
    Detect Zero-Lag MA trend boxes and triangle breakouts

    Logic from Pine Script:
    1. Calculate EMA and ZLMA
    2. Detect crossovers: ZLMA crosses EMA
    3. Create boxes: top/bottom defined by ZLMA ± ATR
    4. Detect triangles: price breaks box boundaries

    Returns:
        {
            'last_signal': 'UP' | 'DOWN' | 'NONE',
            'last_triangle': 'UP' | 'DOWN' | 'NONE',
            'box_active': bool,
            'box_top': float,
            'box_bottom': float,
            'ema': float,
            'zlma': float,
            'confidence': int
        }
    """
    if not validate_candles(candles, max(length, atr_length) + 10):
        return {
            'last_signal': 'NONE',
            'last_triangle': 'NONE',
            'box_active': False,
            'box_top': 0,
            'box_bottom': 0,
            'ema': 0,
            'zlma': 0,
            'confidence': 0
        }

    try:
        closes = np.array([safe_float(c[4]) for c in candles])
        highs = np.array([safe_float(c[2]) for c in candles])
        lows = np.array([safe_float(c[3]) for c in candles])

        # Calculate ZLMA and ATR
        ema, zlma = calculate_zlma(closes, length)
        atr = calculate_atr_wilder(candles, atr_length)

        # Detect crossovers (signals)
        signal_up = np.zeros(len(candles), dtype=bool)
        signal_dn = np.zeros(len(candles), dtype=bool)

        for i in range(1, len(candles)):
            # signalUp: ZLMA crosses above EMA
            if zlma[i] > ema[i] and zlma[i - 1] <= ema[i - 1]:
                signal_up[i] = True
            # signalDn: ZLMA crosses below EMA
            elif zlma[i] < ema[i] and zlma[i - 1] >= ema[i - 1]:
                signal_dn[i] = True

        check_signals = signal_up | signal_dn

        # Build boxes (track active box)
        active_box = None
        triangles = []

        for i in range(len(candles)):
            # If new signal occurs
            if signal_up[i] or signal_dn[i]:
                # Create new box
                if signal_up[i]:
                    top = zlma[i]
                    bottom = zlma[i] - atr[i]
                    side = 'up'
                else:
                    top = zlma[i] + atr[i]
                    bottom = zlma[i]
                    side = 'dn'

                active_box = {
                    'start_idx': i,
                    'top': top,
                    'bottom': bottom,
                    'side': side
                }
                continue

            # If there's an active box, check for triangle breakouts
            if active_box is not None and i >= 1:
                # Check: no signals on current and previous bar
                if not check_signals[i] and not check_signals[i - 1]:
                    box_bottom = active_box['bottom']
                    box_top = active_box['top']

                    # Down triangle: high crosses under box bottom
                    if (highs[i - 1] >= box_bottom and
                        highs[i] < box_bottom and
                        ema[i] > zlma[i]):
                        triangles.append({
                            'index': i - 1,
                            'type': 'DOWN',
                            'price': highs[i - 1]
                        })

                    # Up triangle: low crosses over box top
                    if (lows[i - 1] <= box_top and
                        lows[i] > box_top and
                        ema[i] < zlma[i]):
                        triangles.append({
                            'index': i - 1,
                            'type': 'UP',
                            'price': lows[i - 1]
                        })

        # Get last signal and triangle
        last_signal = 'NONE'
        last_signal_idx = -1

        for i in range(len(candles) - 1, -1, -1):
            if signal_up[i]:
                last_signal = 'UP'
                last_signal_idx = i
                break
            elif signal_dn[i]:
                last_signal = 'DOWN'
                last_signal_idx = i
                break

        last_triangle = 'NONE'
        last_triangle_idx = -1

        if triangles:
            last_tri = triangles[-1]
            last_triangle = last_tri['type']
            last_triangle_idx = last_tri['index']

        # Calculate confidence based on recency
        confidence = 0
        current_idx = len(candles) - 1

        # Triangle gets priority if recent (last 3 bars)
        if last_triangle_idx >= current_idx - 3:
            bars_ago = current_idx - last_triangle_idx
            # 0 bars ago = 100, 1 bar = 85, 2 bars = 70, 3 bars = 55
            confidence = max(55, 100 - bars_ago * 15)

        # If no recent triangle, check signal
        elif last_signal_idx >= current_idx - 5:
            bars_ago = current_idx - last_signal_idx
            # Signals are weaker than triangles
            confidence = max(50, 80 - bars_ago * 10)

        return {
            'last_signal': last_signal,
            'last_triangle': last_triangle,
            'box_active': active_box is not None,
            'box_top': active_box['top'] if active_box else 0,
            'box_bottom': active_box['bottom'] if active_box else 0,
            'ema': safe_float(ema[-1]),
            'zlma': safe_float(zlma[-1]),
            'confidence': confidence,
            'last_signal_idx': last_signal_idx,
            'last_triangle_idx': last_triangle_idx,
            'triangles_count': len(triangles)
        }

    except Exception as e:
        logger.error(f"ZLMA trend box detection error: {e}")
        return {
            'last_signal': 'NONE',
            'last_triangle': 'NONE',
            'box_active': False,
            'box_top': 0,
            'box_bottom': 0,
            'ema': 0,
            'zlma': 0,
            'confidence': 0
        }


def calculate_basic_indicators(candles: List[List[str]]) -> Dict[str, Any]:
    """
    Calculate indicators for Stage 1 with ZLMA integration

    Returns indicators + ZLMA trend box data
    """
    if not validate_candles(candles, 20):
        return {}

    try:
        closes = np.array([safe_float(c[4]) for c in candles])
        volumes = np.array([safe_float(c[5]) for c in candles])

        if len(closes) < 20 or np.all(closes == 0) or np.all(volumes == 0):
            return {}

        # Calculate ZLMA trend boxes
        zlma_data = detect_zlma_trend_boxes(
            candles,
            length=config.EMA_TREND,  # 15 по умолчанию
            atr_length=200
        )

        # Volume ratio for confirmation
        volume_ratios = calculate_volume_ratios(volumes, config.VOLUME_WINDOW)

        return {
            'price': safe_float(closes[-1]),
            'volume_ratio': safe_float(volume_ratios[-1]),
            'zlma_data': zlma_data,
            # History for Stage 2/3
            'volume_history': [safe_float(x) for x in volume_ratios[-10:]]
        }

    except Exception as e:
        logger.error(f"Basic indicators calculation error: {e}")
        return {}


def check_basic_signal(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check for trading signal based on ZLMA triangle breakouts

    Logic:
    - PRIMARY: Triangle breakout (last 3 bars)
    - SECONDARY: Recent signal (last 5 bars)
    - Volume confirmation required
    """
    if not indicators:
        return {'signal': False, 'confidence': 0, 'direction': 'NONE'}

    try:
        zlma_data = indicators.get('zlma_data', {})
        volume_ratio = safe_float(indicators.get('volume_ratio', 1.0))

        if not zlma_data:
            return {'signal': False, 'confidence': 0, 'direction': 'NONE'}

        last_triangle = zlma_data.get('last_triangle', 'NONE')
        last_signal = zlma_data.get('last_signal', 'NONE')
        base_confidence = zlma_data.get('confidence', 0)

        # Determine direction
        direction = 'NONE'

        # Triangle gets priority
        if last_triangle == 'UP':
            direction = 'LONG'
        elif last_triangle == 'DOWN':
            direction = 'SHORT'
        # Fallback to signal if no triangle
        elif last_signal == 'UP':
            direction = 'LONG'
        elif last_signal == 'DOWN':
            direction = 'SHORT'

        if direction == 'NONE':
            return {'signal': False, 'confidence': 0, 'direction': 'NONE'}

        # Volume confirmation
        if volume_ratio < config.MIN_VOLUME_RATIO:
            return {'signal': False, 'confidence': 0, 'direction': 'NONE'}

        # Volume boost
        volume_score = min((volume_ratio - 1.0) * 25, 20)

        # Final confidence
        confidence = int(base_confidence + volume_score)
        confidence = max(0, min(100, confidence))

        # Check minimum threshold
        if confidence < config.MIN_CONFIDENCE:
            return {'signal': False, 'confidence': confidence, 'direction': 'NONE'}

        return {
            'signal': True,
            'confidence': confidence,
            'direction': direction
        }

    except Exception as e:
        logger.error(f"Signal check error: {e}")
        return {'signal': False, 'confidence': 0, 'direction': 'NONE'}


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (без изменений)
# ═══════════════════════════════════════════════════════════════

def calculate_volume_ratios(volumes: np.ndarray, window: int = 20) -> np.ndarray:
    """Calculate volume ratios"""
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


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI"""
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
    """Calculate MACD"""
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
    """Calculate ATR (for Stage 2/3)"""
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


def calculate_ai_indicators(candles: List[List[str]], history_length: int) -> Dict[str, Any]:
    """Calculate indicators for AI analysis (Stage 2/3) - without EMA5/8"""
    if not validate_candles(candles, max(history_length, 20)):
        return {}

    try:
        closes = np.array([safe_float(c[4]) for c in candles])
        volumes = np.array([safe_float(c[5]) for c in candles])

        if len(closes) < history_length or np.all(closes == 0):
            return {}

        ema20 = calculate_ema(closes, config.EMA_TREND)
        rsi = calculate_rsi(closes, config.RSI_PERIOD)
        macd = calculate_macd(closes, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL)
        volume_ratios = calculate_volume_ratios(volumes, config.VOLUME_WINDOW)

        def safe_history(arr, length):
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
            'ema20_history': safe_history(ema20, history_length),
            'rsi_history': safe_history(rsi, history_length),
            'macd_line_history': safe_history(macd['line'], history_length),
            'macd_signal_history': safe_history(macd['signal'], history_length),
            'macd_histogram_history': safe_history(macd['histogram'], history_length),
            'volume_ratio_history': safe_history(volume_ratios, history_length),
            'current': {
                'price': safe_float(closes[-1]),
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