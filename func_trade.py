"""
Refactor: Technical analysis module
Сгруппировано в класс TechnicalAnalyzer — все self._... теперь методы класса,
повторяющиеся утилиты объединены, дубли calculate_indicators_by_instruction удалён.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import math
from config import config


class TechnicalAnalyzer:
    def __init__(self, params: Dict = None):
        self.params = params or {
            'ema_fast': config.indicators.EMA_FAST,
            'ema_medium': config.indicators.EMA_MEDIUM,
            'ema_slow': config.indicators.EMA_SLOW,
            'rsi_period': config.indicators.RSI_PERIOD,
            'macd_fast': config.indicators.MACD_FAST,
            'macd_slow': config.indicators.MACD_SLOW,
            'macd_signal': config.indicators.MACD_SIGNAL,
            'atr_period': config.indicators.ATR_PERIOD,
            'volume_sma': config.indicators.VOLUME_SMA,
            'bb_period': config.indicators.BB_PERIOD,
            'bb_std': config.indicators.BB_STD,
            'min_confidence': config.trading.MIN_CONFIDENCE
        }

    # --------------------------
    # Safe converters / utils
    # --------------------------
    @staticmethod
    def safe_float(value) -> float:
        try:
            r = float(value)
            if math.isnan(r) or math.isinf(r):
                return 0.0
            return r
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def safe_int(value) -> int:
        try:
            r = int(value)
            if math.isnan(r) or math.isinf(r):
                return 0
            return r
        except (ValueError, TypeError):
            return 0

    def safe_list(self, arr) -> list:
        try:
            if isinstance(arr, np.ndarray):
                return [self.safe_float(x) for x in arr.tolist()]
            elif isinstance(arr, list):
                return [self.safe_float(x) for x in arr]
            else:
                return []
        except Exception:
            return []

    # --------------------------
    # Core indicators
    # --------------------------
    @staticmethod
    def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
        prices = np.asarray(prices, dtype=float)
        ema = np.zeros_like(prices)
        if len(prices) == 0:
            return ema
        alpha = 2.0 / (period + 1)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        return ema

    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        prices = np.asarray(prices, dtype=float)
        rsi = np.zeros_like(prices)
        if len(prices) <= period:
            return rsi
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gains = np.zeros_like(prices)
        avg_losses = np.zeros_like(prices)
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        alpha = 1.0 / period
        for i in range(period + 1, len(prices)):
            avg_gains[i] = alpha * gains[i - 1] + (1 - alpha) * avg_gains[i - 1]
            avg_losses[i] = alpha * losses[i - 1] + (1 - alpha) * avg_losses[i - 1]
        rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses != 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, np.ndarray]:
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return {'macd': macd_line, 'signal': signal_line, 'histogram': histogram}

    def calculate_atr(self, candles: List[List[str]], period: int = 14) -> Dict[str, Any]:
        if len(candles) < period + 1:
            return {'atr': [], 'current': 0.0, 'mean': 0.0, 'trend': 'unknown'}
        highs = np.array([float(c[2]) for c in candles], dtype=float)
        lows = np.array([float(c[3]) for c in candles], dtype=float)
        closes = np.array([float(c[4]) for c in candles], dtype=float)
        tr = np.zeros(len(candles))
        for i in range(1, len(candles)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
        atr = np.zeros(len(candles))
        if len(tr) > period:
            atr[period] = np.mean(tr[1:period + 1])
            for i in range(period + 1, len(candles)):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        current_atr = self.safe_float(atr[-1] if len(atr) > 0 else 0.0)
        mean_atr = self.safe_float(np.mean(atr[-20:]) if len(atr) >= 20 else current_atr)
        if len(atr) >= 10:
            recent_atr = atr[-10:]
            old_atr = atr[-20:-10] if len(atr) >= 20 else recent_atr
            if np.mean(recent_atr) > np.mean(old_atr) * 1.05:
                atr_trend = 'increasing'
            elif np.mean(recent_atr) < np.mean(old_atr) * 0.95:
                atr_trend = 'decreasing'
            else:
                atr_trend = 'stable'
        else:
            atr_trend = 'unknown'
        return {
            'atr': self.safe_list(atr),
            'current': current_atr,
            'mean': mean_atr,
            'trend': atr_trend,
            'volatility_percentile': self.safe_float(np.percentile(atr[-50:], 70) if len(atr) >= 50 else current_atr)
        }

    def calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std: float = 2.0) -> Dict[str, Any]:
        prices = np.asarray(prices, dtype=float)
        sma = np.zeros_like(prices)
        upper = np.zeros_like(prices)
        lower = np.zeros_like(prices)
        bandwidth = np.zeros_like(prices)
        for i in range(period - 1, len(prices)):
            window = prices[i - period + 1:i + 1]
            sma[i] = np.mean(window)
            std_dev = np.std(window)
            upper[i] = sma[i] + (std_dev * std)
            lower[i] = sma[i] - (std_dev * std)
            if sma[i] != 0:
                bandwidth[i] = (upper[i] - lower[i]) / sma[i]
        squeeze_detected = len(bandwidth) >= 20 and bandwidth[-1] < np.mean(bandwidth[-20:]) * 0.8
        return {'upper': upper, 'middle': sma, 'lower': lower, 'bandwidth': bandwidth, 'squeeze_detected': squeeze_detected}

    def calculate_volume_sma(self, candles: List[List[str]], period: int = 20) -> Dict[str, Any]:
        volumes = np.array([float(c[5]) for c in candles], dtype=float)
        volume_sma = np.zeros_like(volumes)
        for i in range(period - 1, len(volumes)):
            volume_sma[i] = np.mean(volumes[i - period + 1:i + 1])
        current_volume = self.safe_float(volumes[-1]) if len(volumes) > 0 else 0.0
        current_sma = self.safe_float(volume_sma[-1]) if len(volume_sma) > 0 else 1.0
        volume_ratio = current_volume / current_sma if current_sma > 0 else 1.0
        volume_spikes = []
        if len(volume_sma) >= 10:
            for i in range(len(volume_sma) - 10, len(volume_sma)):
                if volume_sma[i] > 0 and volumes[i] / volume_sma[i] > 1.5:
                    volume_spikes.append(i)
        return {
            'volume_sma': self.safe_list(volume_sma),
            'current_ratio': volume_ratio,
            'volume_trend': 'high' if volume_ratio > 1.3 else 'normal' if volume_ratio > 0.8 else 'low',
            'recent_spikes': len(volume_spikes),
            'volume_momentum': self.safe_float(np.mean(volumes[-5:]) / np.mean(volumes[-15:-5]) if len(volumes) >= 15 else 1.0),
            'volume_current': current_volume
        }

    # --------------------------
    # Helper / analysis methods (previously self._...)
    # --------------------------
    def _determine_market_phase(self, closes: np.ndarray, ema5: np.ndarray, ema20: np.ndarray, bb_data: Dict) -> str:
        if len(closes) < 20:
            return 'unknown'
        recent_closes = closes[-20:]
        recent_ema5 = ema5[-20:]
        recent_ema20 = ema20[-20:]
        if len(recent_ema5) > 0 and len(recent_ema20) > 0:
            if recent_ema5[-1] > recent_ema20[-1] and recent_ema5[-10] < recent_ema20[-10]:
                return 'trending_up'
            if recent_ema5[-1] < recent_ema20[-1] and recent_ema5[-10] > recent_ema20[-10]:
                return 'trending_down'
        if bb_data.get('squeeze_detected', False):
            return 'consolidation'
        price_range = np.max(recent_closes) - np.min(recent_closes)
        avg_price = np.mean(recent_closes)
        if price_range / avg_price < 0.02:
            return 'low_volatility'
        return 'mixed'

    def _calculate_bb_position(self, price: float, bb_data: Dict, index: int) -> str:
        try:
            upper = bb_data['upper'][index]
            middle = bb_data['middle'][index]
            lower = bb_data['lower'][index]
            if price >= upper:
                return 'above_upper'
            if price >= middle:
                return 'upper_half'
            if price >= lower:
                return 'lower_half'
            return 'below_lower'
        except Exception:
            return 'unknown'

    def _analyze_rsi_regime(self, rsi: np.ndarray) -> str:
        if len(rsi) < 10:
            return 'unknown'
        current_rsi = rsi[-1]
        if current_rsi > 80:
            return 'extremely_overbought'
        if current_rsi > 70:
            return 'overbought'
        if current_rsi < 20:
            return 'extremely_oversold'
        if current_rsi < 30:
            return 'oversold'
        if 45 <= current_rsi <= 55:
            return 'neutral'
        return 'normal'

    def _analyze_volume_profile(self, volumes: np.ndarray, volume_sma: List) -> str:
        if len(volumes) < 20 or len(volume_sma) < 20:
            return 'unknown'
        recent_volumes = volumes[-10:]
        recent_sma = np.array(volume_sma[-10:], dtype=float)
        ratios = [recent_volumes[i] / recent_sma[i] for i in range(len(recent_volumes)) if recent_sma[i] > 0]
        if not ratios:
            return 'unknown'
        avg_ratio = np.mean(ratios)
        if avg_ratio > 1.5:
            return 'high_activity'
        if avg_ratio > 1.1:
            return 'above_average'
        if avg_ratio < 0.7:
            return 'low_activity'
        return 'average'

    def _calculate_breakout_strength(self, indicators: Dict) -> float:
        volume_ratio = indicators.get('volume_ratio', 1.0)
        atr_current = indicators.get('atr_current', 0)
        atr_mean = indicators.get('atr_mean', atr_current) or 1.0
        volume_factor = min(2.0, volume_ratio / 1.5)
        volatility_factor = min(1.5, atr_current / atr_mean) if atr_mean > 0 else 1.0
        return self.safe_float((volume_factor + volatility_factor) / 2)

    def _estimate_momentum_target(self, price: float, indicators: Dict) -> float:
        atr_current = indicators.get('atr_current', 0)
        if atr_current == 0:
            return price * 1.005
        multiplier = 2.0
        return self.safe_float(price + (atr_current * multiplier))

    def _analyze_pullback_depth(self, candles: List, ema20: List) -> bool:
        try:
            if len(candles) < 10 or len(ema20) < 10:
                return False
            recent_closes = [float(c[4]) for c in candles[-10:]]
            recent_ema = ema20[-10:]
            distances = [abs(close - ema) / ema for close, ema in zip(recent_closes, recent_ema) if ema > 0]
            if not distances:
                return False
            max_distance = max(distances)
            return 0.005 <= max_distance <= 0.02
        except Exception:
            return False

    def _assess_pullback_quality(self, indicators: Dict) -> str:
        try:
            rsi = indicators.get('rsi', [])
            volume_ratio = indicators.get('volume_ratio', 1.0)
            atr_trend = indicators.get('atr_trend', 'unknown')
            score = 0
            if len(rsi) > 0:
                r = rsi[-1]
                if 35 <= r <= 65:
                    score += 1
            if 0.7 <= volume_ratio <= 1.2:
                score += 1
            if atr_trend in ['stable', 'decreasing']:
                score += 1
            if score >= 2:
                return 'high'
            if score == 1:
                return 'medium'
            return 'low'
        except Exception:
            return 'unknown'

    def _measure_consolidation_duration(self, bb_bandwidth: List) -> bool:
        try:
            if len(bb_bandwidth) < 10:
                return False
            avg_bandwidth = np.mean(bb_bandwidth[-20:]) if len(bb_bandwidth) >= 20 else np.mean(bb_bandwidth)
            narrow_periods = sum(1 for bw in bb_bandwidth[-10:] if bw < avg_bandwidth * 0.8)
            return narrow_periods >= 5
        except Exception:
            return False

    def _analyze_pre_breakout_volume(self, indicators: Dict) -> bool:
        try:
            volume_sma = indicators.get('volume_sma', [])
            if len(volume_sma) < 5:
                return False
            recent = volume_sma[-3:]
            return all(recent[i] >= recent[i - 1] for i in range(1, len(recent)))
        except Exception:
            return False

    def _estimate_squeeze_target(self, price: float, bb_bandwidth: List, indicators: Dict) -> float:
        try:
            if len(bb_bandwidth) < 20:
                return price * 1.008
            avg_bandwidth = np.mean(bb_bandwidth[-20:])
            current_bandwidth = bb_bandwidth[-1]
            expansion_potential = avg_bandwidth - current_bandwidth
            target_move = expansion_potential * price * 0.5
            return self.safe_float(price + target_move)
        except Exception:
            return self.safe_float(price * 1.008)

    def _count_range_tests(self, candles: List, range_high: float, range_low: float) -> Dict:
        try:
            tolerance = (range_high - range_low) * 0.02
            res = support = 0
            for c in candles:
                high = float(c[2]); low = float(c[3])
                if abs(high - range_high) <= tolerance:
                    res += 1
                if abs(low - range_low) <= tolerance:
                    support += 1
            return {'resistance': res, 'support': support, 'total': res + support}
        except Exception:
            return {'resistance': 0, 'support': 0, 'total': 0}

    def _estimate_range_age(self, candles: List, range_high: float, range_low: float) -> int:
        try:
            age = 0
            tolerance = (range_high - range_low) * 0.05
            for i in range(len(candles) - 1, -1, -1):
                high = float(candles[i][2]); low = float(candles[i][3])
                if (low >= range_low - tolerance and high <= range_high + tolerance):
                    age += 1
                else:
                    break
            return age
        except Exception:
            return 0

    def _analyze_candle_pattern_strength(self, candles: List, direction: str) -> float:
        try:
            if len(candles) < 3:
                return 0.0
            strength = 0.0
            for c in candles:
                open_p = float(c[1]); high = float(c[2]); low = float(c[3]); close = float(c[4])
                body = abs(close - open_p) / open_p if open_p > 0 else 0
                if direction == 'LONG':
                    if close > open_p:
                        strength += body * 2
                        lower_shadow = (open_p - low) / open_p if open_p > low else 0
                        strength += lower_shadow * 0.5
                elif direction == 'SHORT':
                    if close < open_p:
                        strength += body * 2
                        upper_shadow = (high - open_p) / open_p if high > open_p else 0
                        strength += upper_shadow * 0.5
            return min(1.0, strength / len(candles))
        except Exception:
            return 0.0

    def _check_trend_consistency(self, indicators: Dict) -> bool:
        try:
            a = indicators.get('ema_alignment_history', [])
            if len(a) < 5:
                return False
            recent = a[-5:]
            consistency = abs(sum(recent)) / len(recent)
            return consistency >= 0.6
        except Exception:
            return False

    def _check_momentum_alignment(self, indicators: Dict, direction: str) -> bool:
        try:
            macd_hist = indicators.get('macd_histogram', [])
            rsi = indicators.get('rsi', [])
            if len(macd_hist) < 3 or len(rsi) < 3:
                return False
            macd_trend = macd_hist[-1] > macd_hist[-3]
            r = rsi[-1]
            if direction == 'LONG':
                return macd_trend and 40 <= r <= 80
            if direction == 'SHORT':
                return (not macd_trend) and 20 <= r <= 60
            return False
        except Exception:
            return False

    def _check_level_respect(self, candles: List, indicators: Dict) -> bool:
        try:
            if len(candles) < 10:
                return False
            swing_highs = indicators.get('swing_highs', [])
            swing_lows = indicators.get('swing_lows', [])
            if not swing_highs and not swing_lows:
                return False
            recent = candles[-5:]
            interactions = 0
            for c in recent:
                high = float(c[2]); low = float(c[3])
                for _, lvl in swing_highs[-3:]:
                    if abs(high - lvl) / lvl < 0.005:
                        interactions += 1
                for _, lvl in swing_lows[-3:]:
                    if abs(low - lvl) / lvl < 0.005:
                        interactions += 1
            return interactions >= 1
        except Exception:
            return False

    def _assess_pattern_quality(self, pattern_name: str, pattern_result: Dict, indicators: Dict) -> float:
        try:
            base = 1.0
            volume_ratio = indicators.get('volume_ratio', 1.0)
            atr_trend = indicators.get('atr_trend', 'unknown')
            if volume_ratio > 1.2:
                base += 0.1
            elif volume_ratio < 0.8:
                base -= 0.2
            if atr_trend in ['increasing', 'stable']:
                base += 0.05
            elif atr_trend == 'decreasing':
                base -= 0.1
            if pattern_name == 'MOMENTUM_BREAKOUT':
                base *= pattern_result.get('breakout_strength', 1.0)
            elif pattern_name == 'PULLBACK_ENTRY':
                pq = pattern_result.get('pullback_quality', 'medium')
                if pq == 'high':
                    base += 0.15
                elif pq == 'low':
                    base -= 0.15
            elif pattern_name == 'SQUEEZE_BREAKOUT':
                base += min(0.2, pattern_result.get('squeeze_intensity', 0))
            elif pattern_name == 'RANGE_SCALP':
                rq = pattern_result.get('range_quality', {}).get('tests', {}).get('total', 0)
                if rq >= 4:
                    base += 0.1
            return max(0.5, min(1.5, base))
        except Exception:
            return 1.0

    def _check_perfect_ema_alignment(self, indicators: Dict, direction: str) -> bool:
        try:
            ema5 = indicators.get('ema5', []); ema8 = indicators.get('ema8', []); ema20 = indicators.get('ema20', [])
            if not (ema5 and ema8 and ema20):
                return False
            c5, c8, c20 = ema5[-1], ema8[-1], ema20[-1]
            if direction == 'LONG':
                return c5 > c8 > c20
            if direction == 'SHORT':
                return c5 < c8 < c20
            return False
        except Exception:
            return False

    # --------------------------
    # High-level aggregators
    # --------------------------
    def calculate_extended_indicators(self, candles: List[List[str]]) -> Dict[str, Any]:
        if len(candles) < 50:
            return {}
        closes = np.array([float(c[4]) for c in candles], dtype=float)
        highs = np.array([float(c[2]) for c in candles], dtype=float)
        lows = np.array([float(c[3]) for c in candles], dtype=float)
        volumes = np.array([float(c[5]) for c in candles], dtype=float)

        ema5 = self.calculate_ema(closes, self.params['ema_fast'])
        ema8 = self.calculate_ema(closes, self.params['ema_medium'])
        ema20 = self.calculate_ema(closes, self.params['ema_slow'])

        ema_alignment = []
        if len(ema5) >= 20:
            for i in range(len(ema5) - 20, len(ema5)):
                if ema5[i] > ema8[i] > ema20[i]:
                    ema_alignment.append(1)
                elif ema5[i] < ema8[i] < ema20[i]:
                    ema_alignment.append(-1)
                else:
                    ema_alignment.append(0)

        rsi = self.calculate_rsi(closes, self.params['rsi_period'])
        macd_data = self.calculate_macd(closes, self.params['macd_fast'], self.params['macd_slow'], self.params['macd_signal'])
        macd_crossovers = []
        md = macd_data['macd']; sig = macd_data['signal']
        for i in range(1, len(md)):
            prev = md[i - 1] - sig[i - 1]; curr = md[i] - sig[i]
            if prev <= 0 < curr:
                macd_crossovers.append(('bullish', i))
            elif prev >= 0 > curr:
                macd_crossovers.append(('bearish', i))

        atr_data = self.calculate_atr(candles, self.params['atr_period'])
        bb_data = self.calculate_bollinger_bands(closes, self.params['bb_period'], self.params['bb_std'])
        volume_data = self.calculate_volume_sma(candles, self.params['volume_sma'])

        # price momentum
        price_momentum = []
        if len(closes) >= 10:
            for i in range(5, len(closes)):
                price_momentum.append(self.safe_float((closes[i] - closes[i - 5]) / closes[i - 5] * 100))

        # swings
        swing_highs = []; swing_lows = []
        if len(candles) >= 10:
            for i in range(5, len(candles) - 5):
                if all(highs[i] >= highs[j] for j in range(i - 2, i + 3) if j != i):
                    swing_highs.append((i, self.safe_float(highs[i])))
                if all(lows[i] <= lows[j] for j in range(i - 2, i + 3) if j != i):
                    swing_lows.append((i, self.safe_float(lows[i])))

        market_phase = self._determine_market_phase(closes, ema5, ema20, bb_data)
        return {
            'ema5': self.safe_list(ema5), 'ema8': self.safe_list(ema8), 'ema20': self.safe_list(ema20),
            'rsi': self.safe_list(rsi), 'rsi_current': self.safe_float(rsi[-1] if len(rsi) else 50),
            'macd_line': self.safe_list(macd_data['macd']), 'macd_signal': self.safe_list(macd_data['signal']),
            'macd_histogram': self.safe_list(macd_data['histogram']),
            'atr': atr_data['atr'], 'atr_current': atr_data['current'], 'atr_mean': atr_data['mean'], 'atr_trend': atr_data['trend'],
            'bb_upper': self.safe_list(bb_data['upper']), 'bb_middle': self.safe_list(bb_data['middle']), 'bb_lower': self.safe_list(bb_data['lower']),
            'volume_sma': volume_data['volume_sma'], 'volume_current': self.safe_float(volumes[-1]), 'volume_ratio': volume_data['current_ratio'],
            'volume_trend': volume_data['volume_trend'],
            'ema_alignment_history': ema_alignment[-20:], 'macd_crossovers': macd_crossovers[-5:],
            'bb_squeeze_detected': bb_data['squeeze_detected'], 'bb_bandwidth': self.safe_list(bb_data['bandwidth']),
            'volume_spikes': volume_data['recent_spikes'], 'volume_momentum': volume_data['volume_momentum'],
            'price_momentum': price_momentum[-20:], 'swing_highs': swing_highs[-10:], 'swing_lows': swing_lows[-10:],
            'trend_strength': self.safe_float(np.mean(ema_alignment[-10:]) if ema_alignment else 0),
            'volatility_regime': 'high' if atr_data['current'] > atr_data['mean'] * 1.2 else 'normal',
            'market_phase': market_phase,
            'price_position_bb': self._calculate_bb_position(closes[-1], bb_data, len(closes) - 1),
            'rsi_regime': self._analyze_rsi_regime(rsi),
            'volume_profile': self._analyze_volume_profile(volumes, volume_data['volume_sma']),
        }

    def analyze_higher_timeframe_trend(self, candles_15m: List[List[str]]) -> Dict[str, Any]:
        if len(candles_15m) < 30:
            return {'trend': 'UNKNOWN', 'strength': 0, 'quality': 'poor'}
        closes = np.array([float(c[4]) for c in candles_15m], dtype=float)
        highs = np.array([float(c[2]) for c in candles_15m], dtype=float)
        lows = np.array([float(c[3]) for c in candles_15m], dtype=float)
        ema20 = self.calculate_ema(closes, 20); ema50 = self.calculate_ema(closes, 50)
        current = closes[-1]; e20 = ema20[-1]; e50 = ema50[-1]
        if current > e20 > e50:
            trend = 'UPTREND'
            strength = self.safe_int(((current - e50) / e50) * 1000)
            if len(ema20) >= 10:
                slope = (ema20[-1] - ema20[-10]) / ema20[-10] * 100
                quality = 'strong' if slope > 0.5 else 'moderate' if slope > 0.2 else 'weak'
            else:
                quality = 'unknown'
        elif current < e20 < e50:
            trend = 'DOWNTREND'
            strength = self.safe_int(((e50 - current) / e50) * 1000)
            if len(ema20) >= 10:
                slope = (ema20[-1] - ema20[-10]) / ema20[-10] * 100
                quality = 'strong' if slope < -0.5 else 'moderate' if slope < -0.2 else 'weak'
            else:
                quality = 'unknown'
        else:
            trend = 'SIDEWAYS'; strength = 0; quality = 'ranging'
        recent_range = np.max(highs[-20:]) - np.min(lows[-20:])
        price_volatility = recent_range / current * 100 if current != 0 else 0
        return {
            'trend': trend, 'strength': min(100, abs(strength)), 'quality': quality,
            'ema20': self.safe_float(e20), 'ema50': self.safe_float(e50),
            'volatility_percent': self.safe_float(price_volatility),
            'trend_duration': self._estimate_trend_duration(ema20, ema50),
            'momentum': self.safe_float((closes[-1] - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 else 0)
        }

    def _estimate_trend_duration(self, ema20: np.ndarray, ema50: np.ndarray) -> int:
        if len(ema20) < 10 or len(ema50) < 10:
            return 0
        duration = 0
        current_trend = 1 if ema20[-1] > ema50[-1] else -1
        for i in range(len(ema20) - 2, -1, -1):
            trend_at_i = 1 if ema20[i] > ema50[i] else -1
            if trend_at_i == current_trend:
                duration += 1
            else:
                break
        return duration

    # --------------------------
    # Pattern detectors (examples refactored)
    # --------------------------
    def detect_momentum_breakout(self, candles: List[List[str]], indicators: Dict) -> Dict[str, Any]:
        if len(candles) < 5:
            return {'signal': False, 'confidence': 0}
        closes = np.array([float(c[4]) for c in candles], dtype=float)
        current_close = closes[-1]
        ema20 = indicators.get('ema20', [])
        macd_hist = indicators.get('macd_histogram', [])
        volume_ratio = indicators.get('volume_ratio', 1.0)
        ema_alignment = indicators.get('ema_alignment_history', [])
        if not ema20 or not macd_hist:
            return {'signal': False, 'confidence': 0}
        conditions = {
            'price_above_ema20': current_close > ema20[-1],
            'macd_positive': macd_hist[-1] > 0,
            'macd_increasing': len(macd_hist) >= 3 and macd_hist[-1] > macd_hist[-3],
            'volume_spike': volume_ratio > 1.5,
            'ema_alignment': len(indicators.get('ema5', [])) > 0 and indicators['ema5'][-1] > indicators['ema8'][-1] > ema20[-1],
            'trend_consistency': len(ema_alignment) > 0 and sum(ema_alignment[-5:]) > 2,
            'momentum_acceleration': indicators.get('trend_strength', 0) > 0.3
        }
        signal = sum(conditions.values()) >= 4
        confidence = sum(conditions.values()) * 15
        if conditions['macd_increasing'] and conditions['volume_spike']:
            confidence += 10
        if indicators.get('market_phase') in ['trending_up', 'breakout']:
            confidence += 5
        return {
            'signal': signal, 'confidence': min(100, int(confidence)), 'pattern': 'MOMENTUM_BREAKOUT',
            'conditions': conditions,
            'breakout_strength': self._calculate_breakout_strength(indicators),
            'expected_target': self._estimate_momentum_target(current_close, indicators)
        }

    def detect_pullback_entry(self, candles: List[List[str]], indicators: Dict) -> Dict[str, Any]:
        if len(candles) < 10:
            return {'signal': False, 'confidence': 0}
        closes = np.array([float(c[4]) for c in candles], dtype=float)
        current_close = closes[-1]
        ema8 = indicators.get('ema8', []); ema20 = indicators.get('ema20', []); rsi = indicators.get('rsi', [])
        rsi_current = indicators.get('rsi_current', 50)
        if not ema8 or not ema20 or not rsi:
            return {'signal': False, 'confidence': 0}
        atr_current = indicators.get('atr_current', 0)
        ema8_proximity = abs(current_close - ema8[-1]) < atr_current * 0.5
        ema20_proximity = abs(current_close - ema20[-1]) < atr_current * 0.8
        conditions = {
            'near_ema8': ema8_proximity,
            'near_ema20': ema20_proximity,
            'rsi_recovery_long': 35 < rsi_current < 55 and len(rsi) >= 5 and rsi[-1] > rsi[-3],
            'rsi_recovery_short': 45 < rsi_current < 65 and len(rsi) >= 5 and rsi[-1] < rsi[-3],
            'trend_alignment': len(indicators.get('ema5', [])) > 0 and indicators['ema5'][-1] > ema8[-1],
            'pullback_depth': self._analyze_pullback_depth(candles, ema20),
            'volume_support': indicators.get('volume_ratio', 1.0) > 0.8,
            'market_structure': indicators.get('market_phase') not in ['consolidation']
        }
        signal_type = False; confidence = 0
        if conditions['near_ema8'] and conditions['rsi_recovery_long'] and conditions['trend_alignment']:
            signal_type = 'LONG'; confidence = 75
        elif conditions['near_ema20'] and conditions['rsi_recovery_short']:
            signal_type = 'SHORT'; confidence = 70
        elif conditions['near_ema8'] and conditions['pullback_depth']:
            signal_type = 'LONG' if conditions['trend_alignment'] else False; confidence = 65
        return {
            'signal': signal_type is not False, 'signal_type': signal_type, 'confidence': confidence,
            'pattern': 'PULLBACK_ENTRY', 'conditions': conditions,
            'pullback_quality': self._assess_pullback_quality(indicators),
            'risk_level': 'low' if conditions['near_ema8'] else 'medium'
        }

    def detect_squeeze_breakout(self, candles: List[List[str]], indicators: Dict) -> Dict[str, Any]:
        if len(candles) < 20:
            return {'signal': False, 'confidence': 0}
        closes = np.array([float(c[4]) for c in candles], dtype=float)
        current_close = closes[-1]
        bb_upper = indicators.get('bb_upper', []); bb_lower = indicators.get('bb_lower', [])
        bb_bandwidth = indicators.get('bb_bandwidth', [])
        volume_ratio = indicators.get('volume_ratio', 1.0)
        atr_trend = indicators.get('atr_trend', 'unknown')
        squeeze_detected = indicators.get('bb_squeeze_detected', False)
        if not bb_upper or not bb_lower:
            return {'signal': False, 'confidence': 0}
        current_bandwidth = bb_bandwidth[-1] if bb_bandwidth else 0
        avg_bandwidth = np.mean(bb_bandwidth[-20:]) if len(bb_bandwidth) >= 20 else current_bandwidth
        squeeze_intensity = (avg_bandwidth - current_bandwidth) / avg_bandwidth if avg_bandwidth > 0 else 0
        breakout_up = current_close > bb_upper[-1]
        breakout_down = current_close < bb_lower[-1]
        approaching_upper = abs(current_close - bb_upper[-1]) / current_close < 0.005
        approaching_lower = abs(current_close - bb_lower[-1]) / current_close < 0.005
        conditions = {
            'squeeze_detected': squeeze_detected or squeeze_intensity > 0.2,
            'squeeze_intensity_high': squeeze_intensity > 0.3,
            'breakout_occurred': breakout_up or breakout_down,
            'approaching_breakout': approaching_upper or approaching_lower,
            'volume_confirmation': volume_ratio > config.indicators.VOLUME_SPIKE_RATIO,
            'atr_supporting': atr_trend in ['increasing', 'stable'],
            'consolidation_duration': self._measure_consolidation_duration(bb_bandwidth),
            'pre_breakout_volume': self._analyze_pre_breakout_volume(indicators)
        }
        signal = (conditions['squeeze_detected'] and (conditions['breakout_occurred'] or conditions['approaching_breakout']) and conditions['volume_confirmation'])
        confidence = sum(conditions.values()) * 12.5
        if breakout_up or (approaching_upper and volume_ratio > 1.3):
            breakout_direction = 'LONG'
        elif breakout_down or (approaching_lower and volume_ratio > 1.3):
            breakout_direction = 'SHORT'
        else:
            breakout_direction = 'NONE'
        return {
            'signal': signal, 'confidence': min(100, int(confidence)), 'pattern': 'SQUEEZE_BREAKOUT',
            'conditions': conditions, 'breakout_direction': breakout_direction,
            'squeeze_intensity': self.safe_float(squeeze_intensity),
            'expected_move': self._estimate_squeeze_target(current_close, bb_bandwidth, indicators)
        }

    def detect_range_scalp(self, candles: List[List[str]], indicators: Dict) -> Dict[str, Any]:
        if len(candles) < 30:
            return {'signal': False, 'confidence': 0}
        closes = np.array([float(c[4]) for c in candles], dtype=float)
        highs = np.array([float(c[2]) for c in candles], dtype=float)
        lows = np.array([float(c[3]) for c in candles], dtype=float)
        current_close = closes[-1]
        rsi_current = indicators.get('rsi_current', 50)
        rsi_regime = indicators.get('rsi_regime', 'unknown')
        market_phase = indicators.get('market_phase', 'unknown')
        lookback = min(40, len(candles))
        range_high = np.max(highs[-lookback:]); range_low = np.min(lows[-lookback:])
        range_size = (range_high - range_low) / current_close * 100 if current_close != 0 else 0
        range_tests = self._count_range_tests(candles[-lookback:], range_high, range_low)
        range_age = self._estimate_range_age(candles, range_high, range_low)
        range_position = (current_close - range_low) / (range_high - range_low) if range_high != range_low else 0.5
        near_resistance = range_position > 0.85
        near_support = range_position < 0.15
        conditions = {
            'range_size_adequate': range_size > config.patterns.RANGE_MIN_SIZE_PERCENT,
            'range_established': range_tests['support'] >= 2 and range_tests['resistance'] >= 2,
            'near_resistance': near_resistance,
            'near_support': near_support,
            'rsi_overbought': rsi_regime in ['overbought', 'extremely_overbought'],
            'rsi_oversold': rsi_regime in ['oversold', 'extremely_oversold'],
            'market_ranging': market_phase in ['consolidation', 'low_volatility', 'ranging'],
            'volume_normal': 0.7 <= indicators.get('volume_ratio', 1.0) <= 1.3,
            'range_mature': range_age > 10
        }
        if (conditions['range_established'] and conditions['near_support'] and conditions['rsi_oversold'] and conditions['range_size_adequate']):
            signal_type = 'LONG'; confidence = 75; signal = True
        elif (conditions['range_established'] and conditions['near_resistance'] and conditions['rsi_overbought'] and conditions['range_size_adequate']):
            signal_type = 'SHORT'; confidence = 75; signal = True
        else:
            signal_type = 'NONE'; confidence = sum([v for k, v in conditions.items() if k not in ['near_resistance', 'near_support']]) * 10; signal = False
        return {
            'signal': signal, 'signal_type': signal_type, 'confidence': min(100, confidence),
            'pattern': 'RANGE_SCALP', 'conditions': conditions,
            'range_levels': {'high': self.safe_float(range_high), 'low': self.safe_float(range_low), 'position': self.safe_float(range_position)},
            'range_quality': {'tests': range_tests, 'age': range_age, 'size_percent': self.safe_float(range_size)}
        }

    # validate_signal and detect_instruction_based_signals kept similar, but calling class methods
    def validate_signal(self, candles_5m: List[List[str]], candles_15m: List[List[str]], signal_data: Dict, indicators: Dict) -> Dict[str, Any]:
        if not candles_5m or not candles_15m:
            return {'score': '0/5', 'valid': False, 'checks': {}}
        htf_analysis = self.analyze_higher_timeframe_trend(candles_15m)
        signal_direction = signal_data.get('signal_type', signal_data.get('breakout_direction', 'NONE'))
        checks = {}
        # 1 higher tf alignment
        if signal_direction == 'LONG':
            checks['higher_tf_trend_aligned'] = (htf_analysis['trend'] in ['UPTREND', 'SIDEWAYS'] and htf_analysis['quality'] != 'weak')
        elif signal_direction == 'SHORT':
            checks['higher_tf_trend_aligned'] = (htf_analysis['trend'] in ['DOWNTREND', 'SIDEWAYS'] and htf_analysis['quality'] != 'weak')
        else:
            checks['higher_tf_trend_aligned'] = False
        # 2 candle closed correctly
        if len(candles_5m) >= 2:
            prev = float(candles_5m[-2][4]); curr = float(candles_5m[-1][4]); open_p = float(candles_5m[-1][1])
            body = abs(curr - open_p) / open_p * 100 if open_p != 0 else 0
            if signal_direction == 'LONG':
                checks['candle_closed_correctly'] = (curr > prev and curr > open_p and body > 0.1)
            elif signal_direction == 'SHORT':
                checks['candle_closed_correctly'] = (curr < prev and curr < open_p and body > 0.1)
            else:
                checks['candle_closed_correctly'] = False
        else:
            checks['candle_closed_correctly'] = False
        # 3 volume
        volume_ratio = indicators.get('volume_ratio', 0)
        volume_trend = indicators.get('volume_trend', 'low')
        checks['volume_confirmed'] = (volume_ratio >= config.indicators.VOLUME_MIN_RATIO and volume_trend != 'low')
        # 4 ATR
        atr_current = indicators.get('atr_current', 0)
        atr_mean = indicators.get('atr_mean', 0)
        volatility_regime = indicators.get('volatility_regime', 'normal')
        checks['atr_sufficient'] = (atr_current >= atr_mean * config.indicators.ATR_OPTIMAL_RATIO and volatility_regime != 'low')
        # 5 candle pattern strength
        if len(candles_5m) >= 3:
            recent = candles_5m[-3:]
            checks['candle_pattern_aligned'] = self._analyze_candle_pattern_strength(recent, signal_direction) > 0.5
        else:
            checks['candle_pattern_aligned'] = False
        quality_checks = {
            'trend_consistency': self._check_trend_consistency(indicators),
            'momentum_aligned': self._check_momentum_alignment(indicators, signal_direction),
            'level_respect': self._check_level_respect(candles_5m, indicators)
        }
        passed = sum(checks.values())
        quality_bonus = sum(quality_checks.values())
        return {
            'score': f'{passed}/{len(checks)}',
            'valid': passed >= config.trading.VALIDATION_CHECKS_REQUIRED,
            'checks': checks,
            'quality_checks': quality_checks,
            'passed': passed,
            'total': len(checks),
            'quality_score': quality_bonus,
            'overall_quality': 'high' if quality_bonus >= 2 else 'medium' if quality_bonus >= 1 else 'low',
            'htf_analysis': htf_analysis
        }

    def detect_instruction_based_signals(self, candles_5m: List[List[str]], candles_15m: List[List[str]]) -> Dict[str, Any]:
        if not candles_5m or not candles_15m:
            return {'signal': 'NO_SIGNAL', 'confidence': 0, 'pattern_type': 'NONE', 'validation_score': '0/5', 'reason': 'insufficient_data'}
        indicators = self.calculate_extended_indicators(candles_5m)
        if not indicators:
            return {'signal': 'NO_SIGNAL', 'confidence': 0, 'pattern_type': 'NONE', 'validation_score': '0/5', 'reason': 'indicator_calculation_failed'}
        htf_analysis = self.analyze_higher_timeframe_trend(candles_15m)
        patterns = [
            ('MOMENTUM_BREAKOUT', self.detect_momentum_breakout(candles_5m, indicators)),
            ('SQUEEZE_BREAKOUT', self.detect_squeeze_breakout(candles_5m, indicators)),
            ('PULLBACK_ENTRY', self.detect_pullback_entry(candles_5m, indicators)),
            ('RANGE_SCALP', self.detect_range_scalp(candles_5m, indicators))
        ]
        best = None; best_conf = 0; best_name = 'NONE'
        for name, res in patterns:
            if res.get('signal') and res.get('confidence', 0) > best_conf:
                quality = self._assess_pattern_quality(name, res, indicators)
                adj = res['confidence'] * quality
                if adj > best_conf:
                    best = res; best_conf = adj; best_name = name
        if not best:
            return {
                'signal': 'NO_SIGNAL', 'confidence': 0, 'pattern_type': 'NONE',
                'higher_tf_trend': htf_analysis['trend'], 'validation_score': '0/5',
                'reason': 'no_qualifying_patterns', 'indicators': indicators, 'htf_analysis': htf_analysis,
                'pattern_scores': {n: r.get('confidence', 0) for n, r in patterns}
            }
        signal_direction = best.get('signal_type', best.get('breakout_direction', 'LONG'))
        validation = self.validate_signal(candles_5m, candles_15m, {'signal_type': signal_direction}, indicators)
        final_conf = best_conf
        modifiers = getattr(config, 'scoring', {}).CONFIDENCE_MODIFIERS if hasattr(config, 'scoring') else {}
        if validation['checks'].get('higher_tf_trend_aligned', False):
            final_conf *= modifiers.get('higher_tf_aligned', 1.0)
        if indicators.get('volume_ratio', 1.0) > config.indicators.VOLUME_SPIKE_RATIO:
            final_conf *= modifiers.get('volume_spike', 1.0)
        if self._check_perfect_ema_alignment(indicators, signal_direction):
            final_conf *= modifiers.get('perfect_ema_alignment', 1.0)
        if validation['passed'] == validation['total']:
            final_conf *= modifiers.get('validation_perfect', 1.0)
        if validation.get('quality_score', 0) >= 2:
            final_conf *= 1.1
        final_conf = min(100, int(final_conf))
        if not validation['valid'] or final_conf < config.trading.MIN_CONFIDENCE:
            return {
                'signal': 'NO_SIGNAL', 'confidence': 0, 'pattern_type': best_name,
                'higher_tf_trend': htf_analysis['trend'], 'validation_score': validation['score'],
                'validation_reasons': [k for k, v in validation['checks'].items() if not v],
                'reason': 'failed_validation' if not validation['valid'] else 'low_confidence',
                'indicators': indicators, 'htf_analysis': htf_analysis
            }
        return {
            'signal': signal_direction, 'confidence': final_conf, 'pattern_type': best_name,
            'higher_tf_trend': htf_analysis['trend'], 'validation_score': validation['score'],
            'atr_current': indicators.get('atr_current', 0), 'volume_ratio': indicators.get('volume_ratio', 1.0),
            'entry_reasons': [
                f"{best_name} pattern detected (quality: {validation.get('overall_quality', 'unknown')})",
                f"Higher TF trend: {htf_analysis['trend']} ({htf_analysis.get('quality', 'unknown')})",
                f"Volume ratio: {indicators.get('volume_ratio', 1.0):.2f}",
                f"ATR trend: {indicators.get('atr_trend', 'unknown')}",
                f"Market phase: {indicators.get('market_phase', 'unknown')}"
            ],
            'validation_reasons': [k for k, v in validation['checks'].items() if v],
            'quality_metrics': {
                'pattern_quality': validation.get('overall_quality', 'unknown'),
                'trend_strength': htf_analysis.get('strength', 0),
                'volatility_regime': indicators.get('volatility_regime', 'unknown'),
                'market_phase': indicators.get('market_phase', 'unknown')
            },
            'indicators': indicators, 'htf_analysis': htf_analysis, 'pattern_details': best
        }


# Convenience alias to match original API
_default_analyzer = TechnicalAnalyzer()

def calculate_indicators_by_instruction(candles: List[List[str]]) -> Dict[str, Any]:
    return _default_analyzer.calculate_extended_indicators(candles)

def detect_instruction_based_signals(candles_5m: List[List[str]], candles_15m: List[List[str]]) -> Dict[str, Any]:
    return _default_analyzer.detect_instruction_based_signals(candles_5m, candles_15m)
