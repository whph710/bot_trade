"""
Correlation analysis module - FIXED: check_btc_alignment теперь статический метод
"""

import numpy as np
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Correlation analysis between cryptocurrencies"""

    SECTORS = {
        'LAYER1': ['ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'ADAUSDT', 'DOTUSDT', 'ATOMUSDT', 'NEARUSDT'],
        'DEFI': ['AAVEUSDT', 'UNIUSDT', 'SUSHIUSDT', 'CRVUSDT', 'COMPUSDT', 'MKRUSDT', 'SNXUSDT'],
        'LAYER2': ['MATICUSDT', 'ARBUSDT', 'OPUSDT'],
        'MEME': ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT'],
        'EXCHANGE': ['BNBUSDT', 'FTMUSDT', 'CAKEUSDT'],
        'AI': ['FETUSDT', 'AGIXUSDT', 'OCEUSDT'],
        'STORAGE': ['FILUSDT', 'ARUSDT', 'STORJUSDT']
    }

    def __init__(self):
        self.price_cache = {}

    def get_sector(self, symbol: str) -> Optional[str]:
        """Get sector for symbol"""
        for sector, symbols in self.SECTORS.items():
            if symbol in symbols:
                return sector
        return None

    def get_sector_peers(self, symbol: str, max_peers: int = 5) -> List[str]:
        """Get peers from same sector"""
        sector = self.get_sector(symbol)
        if not sector:
            return []

        peers = [s for s in self.SECTORS[sector] if s != symbol]
        return peers[:max_peers]

    def cache_prices(self, symbol: str, prices: List[float]):
        """Cache price data"""
        self.price_cache[symbol] = prices

    def calculate_correlation(self, prices1: List[float], prices2: List[float]) -> float:
        """Calculate Pearson correlation"""
        if len(prices1) != len(prices2) or len(prices1) < 10:
            return 0.0

        try:
            arr1 = np.array(prices1, dtype=np.float64)
            arr2 = np.array(prices2, dtype=np.float64)

            mask = np.isfinite(arr1) & np.isfinite(arr2)
            arr1 = arr1[mask]
            arr2 = arr2[mask]

            if len(arr1) < 10:
                return 0.0

            corr_matrix = np.corrcoef(arr1, arr2)
            correlation = corr_matrix[0, 1]

            if np.isnan(correlation) or np.isinf(correlation):
                return 0.0

            return float(correlation)

        except Exception as e:
            logger.debug(f"Correlation calculation error: {e}")
            return 0.0

    def calculate_btc_correlation(
            self,
            symbol: str,
            symbol_prices: List[float],
            btc_prices: List[float],
            window: int = 24
    ) -> Dict:
        """Calculate correlation with BTC"""
        if len(symbol_prices) < window or len(btc_prices) < window:
            return {
                'correlation': 0.0,
                'correlation_strength': 'UNKNOWN',
                'is_correlated': False,
                'reasoning': 'Insufficient data'
            }

        recent_symbol = symbol_prices[-window:]
        recent_btc = btc_prices[-window:]

        corr = self.calculate_correlation(recent_symbol, recent_btc)

        abs_corr = abs(corr)

        if abs_corr > 0.7:
            strength = 'STRONG'
            is_correlated = True
        elif abs_corr > 0.4:
            strength = 'MODERATE'
            is_correlated = True
        else:
            strength = 'WEAK'
            is_correlated = False

        reasoning = f'{symbol} correlation with BTC: {corr:.3f} ({strength})'

        return {
            'correlation': round(corr, 3),
            'correlation_strength': strength,
            'is_correlated': is_correlated,
            'reasoning': reasoning
        }

    def detect_correlation_anomaly(
            self,
            symbol: str,
            symbol_change_pct: float,
            btc_change_pct: float,
            correlation: float
    ) -> Dict:
        """Detect correlation anomaly"""
        if abs(correlation) < 0.5:
            return {
                'anomaly_detected': False,
                'anomaly_type': 'NONE',
                'expected_direction': 'NEUTRAL',
                'confidence_adjustment': 0,
                'reasoning': f'Weak correlation {correlation:.2f}'
            }

        if correlation > 0.5:
            expected_move_sign = np.sign(btc_change_pct)
        elif correlation < -0.5:
            expected_move_sign = -np.sign(btc_change_pct)
        else:
            expected_move_sign = 0

        actual_move_sign = np.sign(symbol_change_pct)

        if expected_move_sign != 0 and actual_move_sign != 0:
            if expected_move_sign == actual_move_sign:
                expected_magnitude = abs(btc_change_pct) * abs(correlation)
                actual_magnitude = abs(symbol_change_pct)

                if actual_magnitude > expected_magnitude * 1.5:
                    return {
                        'anomaly_detected': True,
                        'anomaly_type': 'DECOUPLING_STRENGTH',
                        'expected_direction': 'UP' if actual_move_sign > 0 else 'DOWN',
                        'confidence_adjustment': +10,
                        'reasoning': f'{symbol} moving {actual_magnitude:.1f}% vs expected {expected_magnitude:.1f}%'
                    }
                else:
                    return {
                        'anomaly_detected': False,
                        'anomaly_type': 'NONE',
                        'expected_direction': 'NEUTRAL',
                        'confidence_adjustment': 0,
                        'reasoning': f'{symbol} following BTC normally'
                    }
            else:
                return {
                    'anomaly_detected': True,
                    'anomaly_type': 'DECOUPLING_WEAKNESS',
                    'expected_direction': 'NEUTRAL',
                    'confidence_adjustment': -15,
                    'reasoning': f'{symbol} {symbol_change_pct:+.1f}% vs BTC {btc_change_pct:+.1f}% divergence'
                }

        return {
            'anomaly_detected': False,
            'anomaly_type': 'NONE',
            'expected_direction': 'NEUTRAL',
            'confidence_adjustment': 0,
            'reasoning': 'No significant movement'
        }

    def analyze_sector_performance(
            self,
            symbol: str,
            sector_prices_data: Dict[str, List[float]]
    ) -> Dict:
        """Analyze sector performance"""
        sector = self.get_sector(symbol)

        if not sector or not sector_prices_data:
            return {
                'sector': 'UNKNOWN',
                'sector_trend': 'UNKNOWN',
                'symbol_vs_sector': 'UNKNOWN',
                'confidence_adjustment': 0,
                'reasoning': 'No sector data'
            }

        sector_changes = []

        for peer_symbol, prices in sector_prices_data.items():
            if peer_symbol == symbol or len(prices) < 2:
                continue

            change_pct = ((prices[-1] - prices[0]) / prices[0]) * 100
            sector_changes.append(change_pct)

        if not sector_changes:
            return {
                'sector': sector,
                'sector_trend': 'UNKNOWN',
                'symbol_vs_sector': 'UNKNOWN',
                'confidence_adjustment': 0,
                'reasoning': 'Insufficient peer data'
            }

        avg_sector_change = np.mean(sector_changes)

        if avg_sector_change > 1.0:
            sector_trend = 'UP'
        elif avg_sector_change < -1.0:
            sector_trend = 'DOWN'
        else:
            sector_trend = 'NEUTRAL'

        symbol_prices = sector_prices_data.get(symbol)
        if not symbol_prices or len(symbol_prices) < 2:
            return {
                'sector': sector,
                'sector_trend': sector_trend,
                'symbol_vs_sector': 'UNKNOWN',
                'confidence_adjustment': 0,
                'reasoning': f'Sector {sector} trend: {sector_trend}'
            }

        symbol_change = ((symbol_prices[-1] - symbol_prices[0]) / symbol_prices[0]) * 100

        if symbol_change > avg_sector_change + 0.5:
            position = 'LEADING'
            adjustment = +8
            reasoning = f'{symbol} {symbol_change:+.1f}% outperforming {sector} {avg_sector_change:+.1f}%'
        elif symbol_change < avg_sector_change - 0.5:
            position = 'LAGGING'
            adjustment = -8
            reasoning = f'{symbol} {symbol_change:+.1f}% underperforming {sector} {avg_sector_change:+.1f}%'
        else:
            position = 'INLINE'
            adjustment = 0
            reasoning = f'{symbol} {symbol_change:+.1f}% inline with {sector} {avg_sector_change:+.1f}%'

        return {
            'sector': sector,
            'sector_trend': sector_trend,
            'symbol_vs_sector': position,
            'confidence_adjustment': adjustment,
            'reasoning': reasoning
        }

    @staticmethod
    def check_btc_alignment(
            symbol: str,
            signal_direction: str,
            btc_trend: str,
            correlation: float
    ) -> Dict:
        """
        ИСПРАВЛЕНО: Теперь статический метод
        Блокировка только при STRONG correlation + CLEAR conflict
        """
        # Слабая корреляция - пропускаем
        if abs(correlation) < 0.5:
            return {
                'aligned': True,
                'should_block': False,
                'confidence_adjustment': 0,
                'reasoning': f'Weak BTC correlation {correlation:.2f}'
            }

        # Блокируем только при ОЧЕНЬ сильной корреляции
        if abs(correlation) > 0.8:
            # Положительная корреляция
            if correlation > 0.8:
                if signal_direction == 'LONG' and btc_trend == 'UP':
                    return {
                        'aligned': True,
                        'should_block': False,
                        'confidence_adjustment': +8,
                        'reasoning': 'LONG aligned with BTC uptrend (strong correlation)'
                    }
                elif signal_direction == 'SHORT' and btc_trend == 'DOWN':
                    return {
                        'aligned': True,
                        'should_block': False,
                        'confidence_adjustment': +8,
                        'reasoning': 'SHORT aligned with BTC downtrend (strong correlation)'
                    }
                else:
                    # Не блокируем, только снижаем confidence
                    return {
                        'aligned': False,
                        'should_block': False,
                        'confidence_adjustment': -12,
                        'reasoning': f'{signal_direction} misaligned with BTC {btc_trend}, correlation {correlation:.2f}, WARNING (not blocking)'
                    }

            # Отрицательная корреляция
            elif correlation < -0.8:
                if signal_direction == 'LONG' and btc_trend == 'DOWN':
                    return {
                        'aligned': True,
                        'should_block': False,
                        'confidence_adjustment': +8,
                        'reasoning': 'LONG with strong negative BTC correlation during BTC down'
                    }
                elif signal_direction == 'SHORT' and btc_trend == 'UP':
                    return {
                        'aligned': True,
                        'should_block': False,
                        'confidence_adjustment': +8,
                        'reasoning': 'SHORT with strong negative BTC correlation during BTC up'
                    }
                else:
                    return {
                        'aligned': False,
                        'should_block': False,
                        'confidence_adjustment': -12,
                        'reasoning': f'{signal_direction} misaligned with negative BTC correlation, WARNING (not blocking)'
                    }

        # Умеренная корреляция (0.5-0.8) - только adjustment
        return {
            'aligned': True,
            'should_block': False,
            'confidence_adjustment': 0,
            'reasoning': f'Moderate correlation {correlation:.2f}, monitoring'
        }


def extract_prices_from_candles(candles: List[List]) -> List[float]:
    """Extract close prices from candles"""
    try:
        return [float(candle[4]) for candle in candles]
    except (IndexError, ValueError, TypeError):
        return []


def calculate_price_change(prices: List[float], window: int = 24) -> float:
    """Calculate price change percentage"""
    if len(prices) < window:
        window = len(prices)

    if window < 2:
        return 0.0

    try:
        old_price = prices[-window]
        new_price = prices[-1]

        if old_price == 0:
            return 0.0

        change = ((new_price - old_price) / old_price) * 100
        return round(change, 2)
    except (IndexError, ZeroDivisionError):
        return 0.0


def determine_trend(prices: List[float], window: int = 20) -> str:
    """Determine price trend"""
    if len(prices) < window:
        window = len(prices)

    if window < 5:
        return 'FLAT'

    try:
        recent_prices = prices[-window:]

        first_third = np.mean(recent_prices[:window // 3])
        last_third = np.mean(recent_prices[-window // 3:])

        change_pct = ((last_third - first_third) / first_third) * 100

        if change_pct > 1.0:
            return 'UP'
        elif change_pct < -1.0:
            return 'DOWN'
        else:
            return 'FLAT'
    except Exception:
        return 'FLAT'


async def get_comprehensive_correlation_analysis(
        symbol: str,
        symbol_candles: List[List],
        btc_candles: List[List],
        signal_direction: str,
        sector_candles: Dict[str, List[List]] = None
) -> Dict:
    """Comprehensive correlation analysis"""
    analyzer = CorrelationAnalyzer()

    symbol_prices = extract_prices_from_candles(symbol_candles)
    btc_prices = extract_prices_from_candles(btc_candles)

    if len(symbol_prices) < 24 or len(btc_prices) < 24:
        return {
            'symbol': symbol,
            'error': 'Insufficient data',
            'total_confidence_adjustment': 0,
            'should_block_signal': False
        }

    btc_corr_result = analyzer.calculate_btc_correlation(
        symbol, symbol_prices, btc_prices, window=24
    )

    symbol_change_1h = calculate_price_change(symbol_prices, window=1)
    btc_change_1h = calculate_price_change(btc_prices, window=1)

    anomaly_result = analyzer.detect_correlation_anomaly(
        symbol,
        symbol_change_1h,
        btc_change_1h,
        btc_corr_result['correlation']
    )

    btc_trend = determine_trend(btc_prices, window=20)

    # ИСПРАВЛЕНО: Теперь вызываем как статический метод
    alignment_result = CorrelationAnalyzer.check_btc_alignment(
        symbol,
        signal_direction,
        btc_trend,
        btc_corr_result['correlation']
    )

    sector_result = {'confidence_adjustment': 0, 'reasoning': 'No sector data'}

    if sector_candles:
        sector_prices_data = {
            s: extract_prices_from_candles(candles)
            for s, candles in sector_candles.items()
        }
        sector_prices_data[symbol] = symbol_prices

        sector_result = analyzer.analyze_sector_performance(
            symbol,
            sector_prices_data
        )

    total_adjustment = (
            anomaly_result['confidence_adjustment'] +
            alignment_result['confidence_adjustment'] +
            sector_result['confidence_adjustment']
    )

    should_block = alignment_result['should_block']

    return {
        'symbol': symbol,
        'should_block_signal': should_block,
        'total_confidence_adjustment': total_adjustment,
        'btc_correlation': btc_corr_result,
        'correlation_anomaly': anomaly_result,
        'btc_alignment': alignment_result,
        'sector_analysis': sector_result,
        'price_changes': {
            'symbol_1h': symbol_change_1h,
            'btc_1h': btc_change_1h
        },
        'btc_trend': btc_trend
    }