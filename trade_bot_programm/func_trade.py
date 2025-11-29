"""
Technical indicators with Triple EMA Strategy
Файл: trade_bot_programm/func_trade.py

ИЗМЕНЕНИЯ:
- Удалены функции ZLMA (calculate_zlma, detect_zlma_trend_boxes)
- Добавлены функции Triple EMA (detect_ema_triple_signal)
- Обновлена логика calculate_basic_indicators и check_basic_signal
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


def detect_ema_triple_signal(
    candles: List[List[str]],
    ema_fast: int = 9,
    ema_medium: int = 21,
    ema_slow: int = 50
) -> Dict[str, Any]:
    """
    Detect Triple EMA signals (классический 9/21/50)

    Returns:
        {
            'alignment': 'BULLISH' | 'BEARISH' | 'NEUTRAL',
            'crossover': 'GOLDEN' | 'DEATH' | 'NONE',
            'pullback': 'BULLISH_BOUNCE' | 'BEARISH_BOUNCE' | 'NONE',
            'compression': 'BREAKOUT_UP' | 'BREAKOUT_DOWN' | 'COMPRESSED' | 'NONE',
            'ema9': float,
            'ema21': float,
            'ema50': float,
            'confidence': int,
            'details': str
        }
    """
    if not validate_candles(candles, max(ema_fast, ema_medium, ema_slow) + 10):
        return {
            'alignment': 'NEUTRAL',
            'crossover': 'NONE',
            'pullback': 'NONE',
            'compression': 'NONE',
            'ema9': 0,
            'ema21': 0,
            'ema50': 0,
            'confidence': 0,
            'details': 'Insufficient candle data'
        }

    try:
        closes = np.array([safe_float(c[4]) for c in candles])
        highs = np.array([safe_float(c[2]) for c in candles])
        lows = np.array([safe_float(c[3]) for c in candles])
        volumes = np.array([safe_float(c[5]) for c in candles])

        # Calculate EMAs
        ema9 = calculate_ema(closes, ema_fast)
        ema21 = calculate_ema(closes, ema_medium)
        ema50 = calculate_ema(closes, ema_slow)

        current_price = closes[-1]
        current_ema9 = ema9[-1]
        current_ema21 = ema21[-1]
        current_ema50 = ema50[-1]

        # Volume ratio
        volume_ratios = calculate_volume_ratios(volumes, config.VOLUME_WINDOW)
        current_volume_ratio = safe_float(volume_ratios[-1])

        # 1. CHECK ALIGNMENT
        alignment = 'NEUTRAL'
        alignment_score = 0

        gap_9_21 = abs((current_ema9 - current_ema21) / current_ema21 * 100)
        gap_21_50 = abs((current_ema21 - current_ema50) / current_ema50 * 100)

        if current_ema9 > current_ema21 > current_ema50:
            alignment = 'BULLISH'
            # Perfect alignment если зазоры >0.5%
            if gap_9_21 >= config.EMA_MIN_GAP_PCT and gap_21_50 >= config.EMA_MIN_GAP_PCT:
                alignment_score = 15
            else:
                alignment_score = 10

        elif current_ema9 < current_ema21 < current_ema50:
            alignment = 'BEARISH'
            if gap_9_21 >= config.EMA_MIN_GAP_PCT and gap_21_50 >= config.EMA_MIN_GAP_PCT:
                alignment_score = 15
            else:
                alignment_score = 10

        # 2. CHECK CROSSOVERS (последние 5 свечей)
        crossover = 'NONE'
        crossover_score = 0
        lookback = min(config.EMA_CROSSOVER_LOOKBACK, len(ema9) - 1)

        for i in range(1, lookback + 1):
            idx = -i
            # Golden Cross: EMA9 crosses EMA21 upward
            if ema9[idx] > ema21[idx] and ema9[idx - 1] <= ema21[idx - 1]:
                # Проверяем что EMA21 уже выше EMA50
                if ema21[idx] > ema50[idx]:
                    crossover = 'GOLDEN'
                    crossover_score = 12
                    break

            # Death Cross: EMA9 crosses EMA21 downward
            elif ema9[idx] < ema21[idx] and ema9[idx - 1] >= ema21[idx - 1]:
                if ema21[idx] < ema50[idx]:
                    crossover = 'DEATH'
                    crossover_score = 12
                    break

        # 3. CHECK PULLBACK TO EMA21
        pullback = 'NONE'
        pullback_score = 0

        # Проверяем последние 3 свечи
        for i in range(1, min(4, len(candles))):
            idx = -i
            low_price = lows[idx]
            high_price = highs[idx]
            close_price = closes[idx]
            ema21_value = ema21[idx]

            # Допуск ±1.5% от EMA21
            touch_upper = ema21_value * (1 + config.PULLBACK_TOUCH_PCT / 100)
            touch_lower = ema21_value * (1 - config.PULLBACK_TOUCH_PCT / 100)

            # Bullish bounce: цена коснулась EMA21 снизу и отскочила вверх
            if alignment == 'BULLISH' and touch_lower <= low_price <= touch_upper:
                if current_price > ema21_value and current_volume_ratio >= config.PULLBACK_BOUNCE_VOLUME:
                    pullback = 'BULLISH_BOUNCE'
                    pullback_score = 10
                    break

            # Bearish bounce: цена коснулась EMA21 сверху и отскочила вниз
            elif alignment == 'BEARISH' and touch_lower <= high_price <= touch_upper:
                if current_price < ema21_value and current_volume_ratio >= config.PULLBACK_BOUNCE_VOLUME:
                    pullback = 'BEARISH_BOUNCE'
                    pullback_score = 10
                    break

        # 4. CHECK COMPRESSION
        compression = 'NONE'
        compression_score = 0

        # Расстояние между EMA9 и EMA50
        total_spread = abs((current_ema9 - current_ema50) / current_ema50 * 100)

        if total_spread <= config.COMPRESSION_MAX_SPREAD_PCT:
            compression = 'COMPRESSED'
            # Проверяем breakout с объёмом
            if current_volume_ratio >= config.COMPRESSION_BREAKOUT_VOLUME:
                if current_price > max(current_ema9, current_ema21, current_ema50):
                    compression = 'BREAKOUT_UP'
                    compression_score = 12
                elif current_price < min(current_ema9, current_ema21, current_ema50):
                    compression = 'BREAKOUT_DOWN'
                    compression_score = 12

        # 5. ДОПОЛНИТЕЛЬНЫЕ БОНУСЫ
        bonus_score = 0

        # EMA slope согласован
        if alignment == 'BULLISH':
            if ema9[-1] > ema9[-5] and ema21[-1] > ema21[-5] and ema50[-1] > ema50[-5]:
                bonus_score += 10
        elif alignment == 'BEARISH':
            if ema9[-1] < ema9[-5] and ema21[-1] < ema21[-5] and ema50[-1] < ema50[-5]:
                bonus_score += 10

        # Цена выше всех EMA (для LONG) или ниже всех (для SHORT)
        if alignment == 'BULLISH' and current_price > current_ema9:
            bonus_score += 8
        elif alignment == 'BEARISH' and current_price < current_ema9:
            bonus_score += 8

        # Расстояние от EMA50 <3% (не перерастянуто)
        distance_from_ema50 = abs((current_price - current_ema50) / current_ema50 * 100)
        if distance_from_ema50 < 3.0:
            bonus_score += 8

        # Volume spike
        if current_volume_ratio >= 1.5:
            bonus_score += 8

        # 6. ШТРАФЫ
        penalty = 0

        # Flat EMA (горизонтальная)
        ema21_slope = abs((ema21[-1] - ema21[-10]) / ema21[-10] * 100)
        if ema21_slope < 0.5:
            penalty -= 10

        # Overextension (>5% от EMA50)
        if distance_from_ema50 > 5.0:
            penalty -= 10

        # Volume dead
        recent_volume = [safe_float(v) for v in volume_ratios[-3:]]
        if all(v < 0.8 for v in recent_volume):
            penalty -= 10

        # Whipsaw zone (частые пересечения)
        crosses_count = 0
        for i in range(1, min(11, len(ema9))):
            if (ema9[-i] > ema21[-i] and ema9[-i-1] <= ema21[-i-1]) or \
               (ema9[-i] < ema21[-i] and ema9[-i-1] >= ema21[-i-1]):
                crosses_count += 1
        if crosses_count >= 3:
            penalty -= 12

        # 7. CALCULATE CONFIDENCE
        base_confidence = 50
        total_confidence = base_confidence + alignment_score + crossover_score + pullback_score + compression_score + bonus_score + penalty
        total_confidence = max(0, min(100, total_confidence))

        # 8. BUILD DETAILS
        details_parts = []
        if alignment != 'NEUTRAL':
            details_parts.append(f"Alignment: {alignment} ({alignment_score:+d})")
        if crossover != 'NONE':
            details_parts.append(f"Crossover: {crossover} ({crossover_score:+d})")
        if pullback != 'NONE':
            details_parts.append(f"Pullback: {pullback} ({pullback_score:+d})")
        if compression != 'NONE' and compression != 'COMPRESSED':
            details_parts.append(f"Compression: {compression} ({compression_score:+d})")
        if bonus_score > 0:
            details_parts.append(f"Bonuses: {bonus_score:+d}")
        if penalty < 0:
            details_parts.append(f"Penalties: {penalty:+d}")

        details = '; '.join(details_parts) if details_parts else 'No significant patterns'

        return {
            'alignment': alignment,
            'crossover': crossover,
            'pullback': pullback,
            'compression': compression,
            'ema9': safe_float(current_ema9),
            'ema21': safe_float(current_ema21),
            'ema50': safe_float(current_ema50),
            'confidence': int(total_confidence),
            'details': details,
            'distance_from_ema50_pct': round(distance_from_ema50, 2),
            'volume_ratio': round(current_volume_ratio, 2)
        }

    except Exception as e:
        logger.error(f"Triple EMA detection error: {e}")
        return {
            'alignment': 'NEUTRAL',
            'crossover': 'NONE',
            'pullback': 'NONE',
            'compression': 'NONE',
            'ema9': 0,
            'ema21': 0,
            'ema50': 0,
            'confidence': 0,
            'details': f'Error: {str(e)[:50]}'
        }


def calculate_basic_indicators(candles: List[List[str]]) -> Dict[str, Any]:
    """
    Calculate indicators for Stage 1 with Triple EMA

    Returns indicators + Triple EMA signal data
    """
    if not validate_candles(candles, 60):
        return {}

    try:
        closes = np.array([safe_float(c[4]) for c in candles])
        volumes = np.array([safe_float(c[5]) for c in candles])

        if len(closes) < 60 or np.all(closes == 0) or np.all(volumes == 0):
            return {}

        # Calculate Triple EMA signal
        ema_signal = detect_ema_triple_signal(
            candles,
            ema_fast=config.EMA_FAST,
            ema_medium=config.EMA_MEDIUM,
            ema_slow=config.EMA_SLOW
        )

        # Volume ratio for confirmation
        volume_ratios = calculate_volume_ratios(volumes, config.VOLUME_WINDOW)

        return {
            'price': safe_float(closes[-1]),
            'volume_ratio': safe_float(volume_ratios[-1]),
            'ema_signal': ema_signal,
            # History for Stage 2/3
            'volume_history': [safe_float(x) for x in volume_ratios[-10:]]
        }

    except Exception as e:
        logger.error(f"Basic indicators calculation error: {e}")
        return {}


def check_basic_signal(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check for trading signal based on Triple EMA

    Logic:
    - PRIMARY: Perfect alignment (BULLISH/BEARISH)
    - SECONDARY: Golden/Death Cross
    - TERTIARY: Pullback bounce
    - BONUS: Compression breakout
    - Volume confirmation required
    """
    if not indicators:
        return {'signal': False, 'confidence': 0, 'direction': 'NONE'}

    try:
        ema_signal = indicators.get('ema_signal', {})
        volume_ratio = safe_float(indicators.get('volume_ratio', 1.0))

        if not ema_signal:
            return {'signal': False, 'confidence': 0, 'direction': 'NONE'}

        alignment = ema_signal.get('alignment', 'NEUTRAL')
        crossover = ema_signal.get('crossover', 'NONE')
        pullback = ema_signal.get('pullback', 'NONE')
        compression = ema_signal.get('compression', 'NONE')
        base_confidence = ema_signal.get('confidence', 0)

        # Determine direction
        direction = 'NONE'

        # Priority 1: Alignment
        if alignment == 'BULLISH':
            direction = 'LONG'
        elif alignment == 'BEARISH':
            direction = 'SHORT'

        # Priority 2: Crossovers
        if crossover == 'GOLDEN':
            direction = 'LONG'
        elif crossover == 'DEATH':
            direction = 'SHORT'

        # Priority 3: Pullback
        if pullback == 'BULLISH_BOUNCE':
            direction = 'LONG'
        elif pullback == 'BEARISH_BOUNCE':
            direction = 'SHORT'

        # Priority 4: Compression breakout
        if compression == 'BREAKOUT_UP':
            direction = 'LONG'
        elif compression == 'BREAKOUT_DOWN':
            direction = 'SHORT'

        if direction == 'NONE':
            return {'signal': False, 'confidence': 0, 'direction': 'NONE'}

        # Volume confirmation - ИСПРАВЛЕНО: единый порог
        if volume_ratio < config.MIN_VOLUME_RATIO:
            return {'signal': False, 'confidence': 0, 'direction': 'NONE'}

        # Volume boost
        volume_score = min((volume_ratio - 1.0) * 10, 15)

        # Final confidence
        confidence = int(base_confidence + volume_score)
        confidence = max(0, min(100, confidence))

        # Check minimum threshold
        if confidence < config.MIN_CONFIDENCE:
            return {'signal': False, 'confidence': confidence, 'direction': 'NONE'}

        return {
            'signal': True,
            'confidence': confidence,
            'direction': direction,
            'details': ema_signal.get('details', '')
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
    """Calculate ATR"""
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
    """Calculate indicators for AI analysis (Stage 2/3)"""
    if not validate_candles(candles, max(history_length, 60)):
        return {}

    try:
        closes = np.array([safe_float(c[4]) for c in candles])
        volumes = np.array([safe_float(c[5]) for c in candles])

        if len(closes) < history_length or np.all(closes == 0):
            return {}

        # Calculate Triple EMA
        ema9 = calculate_ema(closes, config.EMA_FAST)
        ema21 = calculate_ema(closes, config.EMA_MEDIUM)
        ema50 = calculate_ema(closes, config.EMA_SLOW)

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
            'ema9_history': safe_history(ema9, history_length),
            'ema21_history': safe_history(ema21, history_length),
            'ema50_history': safe_history(ema50, history_length),
            'rsi_history': safe_history(rsi, history_length),
            'macd_line_history': safe_history(macd['line'], history_length),
            'macd_signal_history': safe_history(macd['signal'], history_length),
            'macd_histogram_history': safe_history(macd['histogram'], history_length),
            'volume_ratio_history': safe_history(volume_ratios, history_length),
            'current': {
                'price': safe_float(closes[-1]),
                'ema9': safe_float(ema9[-1]),
                'ema21': safe_float(ema21[-1]),
                'ema50': safe_float(ema50[-1]),
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