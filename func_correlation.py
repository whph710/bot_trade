"""
Модуль корреляционного анализа
BTC correlation, sector analysis, anomaly detection
"""

import numpy as np
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Анализ корреляций между криптовалютами"""

    # Определение секторов
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
        self.price_cache = {}  # {symbol: [prices]}

    def get_sector(self, symbol: str) -> Optional[str]:
        """Определить сектор монеты"""
        for sector, symbols in self.SECTORS.items():
            if symbol in symbols:
                return sector
        return None

    def get_sector_peers(self, symbol: str, max_peers: int = 5) -> List[str]:
        """Получить пары из того же сектора"""
        sector = self.get_sector(symbol)
        if not sector:
            return []

        peers = [s for s in self.SECTORS[sector] if s != symbol]
        return peers[:max_peers]

    def cache_prices(self, symbol: str, prices: List[float]):
        """Кешировать ценовые данные для корреляционного анализа"""
        self.price_cache[symbol] = prices

    def calculate_correlation(self, prices1: List[float], prices2: List[float]) -> float:
        """
        Рассчитать корреляцию Пирсона между двумя ценовыми рядами

        Returns:
            correlation coefficient (-1 to 1)
        """
        if len(prices1) != len(prices2) or len(prices1) < 10:
            return 0.0

        try:
            # Используем numpy для точности
            arr1 = np.array(prices1, dtype=np.float64)
            arr2 = np.array(prices2, dtype=np.float64)

            # Убираем NaN и inf
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
            logger.debug(f"Ошибка расчета корреляции: {e}")
            return 0.0

    def calculate_btc_correlation(
            self,
            symbol: str,
            symbol_prices: List[float],
            btc_prices: List[float],
            window: int = 24
    ) -> Dict:
        """
        Рассчитать корреляцию с BTC

        Args:
            symbol: торговая пара
            symbol_prices: цены символа (закрытия свечей)
            btc_prices: цены BTC (закрытия свечей)
            window: окно для корреляции (количество свечей)

        Returns:
            {
                'correlation': 0.85,
                'correlation_strength': 'STRONG' / 'MODERATE' / 'WEAK',
                'is_correlated': True / False,
                'reasoning': '...'
            }
        """
        if len(symbol_prices) < window or len(btc_prices) < window:
            return {
                'correlation': 0.0,
                'correlation_strength': 'UNKNOWN',
                'is_correlated': False,
                'reasoning': 'Insufficient data for correlation'
            }

        # Берем последние N свечей
        recent_symbol = symbol_prices[-window:]
        recent_btc = btc_prices[-window:]

        corr = self.calculate_correlation(recent_symbol, recent_btc)

        # Определяем силу корреляции
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
        """
        Определить аномалию в корреляции (расхождение движений)

        Args:
            symbol: торговая пара
            symbol_change_pct: изменение цены символа за период (%)
            btc_change_pct: изменение цены BTC за период (%)
            correlation: текущая корреляция

        Returns:
            {
                'anomaly_detected': True / False,
                'anomaly_type': 'DECOUPLING_STRENGTH' / 'DECOUPLING_WEAKNESS' / 'NONE',
                'expected_direction': 'UP' / 'DOWN' / 'NEUTRAL',
                'confidence_adjustment': +10 / -10 / 0,
                'reasoning': '...'
            }
        """
        # Если корреляция слабая, аномалий не отслеживаем
        if abs(correlation) < 0.5:
            return {
                'anomaly_detected': False,
                'anomaly_type': 'NONE',
                'expected_direction': 'NEUTRAL',
                'confidence_adjustment': 0,
                'reasoning': f'Weak correlation {correlation:.2f}, no anomaly tracking'
            }

        # Определяем ожидаемое направление на основе BTC
        if correlation > 0.5:  # Положительная корреляция
            expected_move_sign = np.sign(btc_change_pct)
        elif correlation < -0.5:  # Отрицательная корреляция
            expected_move_sign = -np.sign(btc_change_pct)
        else:
            expected_move_sign = 0

        actual_move_sign = np.sign(symbol_change_pct)

        # Проверяем расхождение
        if expected_move_sign != 0 and actual_move_sign != 0:
            # Если знаки совпадают - нормально
            if expected_move_sign == actual_move_sign:
                # Проверяем силу движения
                expected_magnitude = abs(btc_change_pct) * abs(correlation)
                actual_magnitude = abs(symbol_change_pct)

                if actual_magnitude > expected_magnitude * 1.5:
                    # Символ двигается сильнее чем ожидалось
                    return {
                        'anomaly_detected': True,
                        'anomaly_type': 'DECOUPLING_STRENGTH',
                        'expected_direction': 'UP' if actual_move_sign > 0 else 'DOWN',
                        'confidence_adjustment': +10,
                        'reasoning': f'{symbol} moving {actual_magnitude:.1f}% vs expected {expected_magnitude:.1f}% (stronger than BTC)'
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
                # Знаки НЕ совпадают - аномалия!
                return {
                    'anomaly_detected': True,
                    'anomaly_type': 'DECOUPLING_WEAKNESS',
                    'expected_direction': 'NEUTRAL',
                    'confidence_adjustment': -15,
                    'reasoning': f'{symbol} {symbol_change_pct:+.1f}% vs BTC {btc_change_pct:+.1f}% - divergence, risky'
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
        """
        Анализ производительности сектора

        Args:
            symbol: анализируемая пара
            sector_prices_data: {symbol: [prices]} для всех пар сектора

        Returns:
            {
                'sector': 'LAYER1',
                'sector_trend': 'UP' / 'DOWN' / 'NEUTRAL',
                'symbol_vs_sector': 'LEADING' / 'LAGGING' / 'INLINE',
                'confidence_adjustment': +8 / -8 / 0,
                'reasoning': '...'
            }
        """
        sector = self.get_sector(symbol)

        if not sector or not sector_prices_data:
            return {
                'sector': 'UNKNOWN',
                'sector_trend': 'UNKNOWN',
                'symbol_vs_sector': 'UNKNOWN',
                'confidence_adjustment': 0,
                'reasoning': 'Sector data not available'
            }

        # Рассчитываем средний % изменения по сектору
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
                'reasoning': 'Insufficient sector peer data'
            }

        avg_sector_change = np.mean(sector_changes)

        # Определяем тренд сектора
        if avg_sector_change > 1.0:
            sector_trend = 'UP'
        elif avg_sector_change < -1.0:
            sector_trend = 'DOWN'
        else:
            sector_trend = 'NEUTRAL'

        # Определяем позицию символа относительно сектора
        symbol_prices = sector_prices_data.get(symbol)
        if not symbol_prices or len(symbol_prices) < 2:
            return {
                'sector': sector,
                'sector_trend': sector_trend,
                'symbol_vs_sector': 'UNKNOWN',
                'confidence_adjustment': 0,
                'reasoning': f'Sector {sector} trend: {sector_trend}, but no symbol data'
            }

        symbol_change = ((symbol_prices[-1] - symbol_prices[0]) / symbol_prices[0]) * 100

        # Сравниваем с сектором
        if symbol_change > avg_sector_change + 0.5:
            position = 'LEADING'
            adjustment = +8
            reasoning = f'{symbol} {symbol_change:+.1f}% outperforming {sector} sector avg {avg_sector_change:+.1f}%'
        elif symbol_change < avg_sector_change - 0.5:
            position = 'LAGGING'
            adjustment = -8
            reasoning = f'{symbol} {symbol_change:+.1f}% underperforming {sector} sector avg {avg_sector_change:+.1f}%'
        else:
            position = 'INLINE'
            adjustment = 0
            reasoning = f'{symbol} {symbol_change:+.1f}% moving inline with {sector} sector {avg_sector_change:+.1f}%'

        return {
            'sector': sector,
            'sector_trend': sector_trend,
            'symbol_vs_sector': position,
            'confidence_adjustment': adjustment,
            'reasoning': reasoning
        }

    def check_btc_alignment(
            self,
            symbol: str,
            signal_direction: str,
            btc_trend: str,
            correlation: float
    ) -> Dict:
        """
        Проверить выравнивание сигнала с трендом BTC

        Args:
            symbol: торговая пара
            signal_direction: 'LONG' / 'SHORT'
            btc_trend: 'UP' / 'DOWN' / 'FLAT'
            correlation: корреляция с BTC

        Returns:
            {
                'aligned': True / False,
                'should_block': True / False,
                'confidence_adjustment': -20 / 0,
                'reasoning': '...'
            }
        """
        # Если корреляция слабая, BTC не критичен
        if abs(correlation) < 0.5:
            return {
                'aligned': True,
                'should_block': False,
                'confidence_adjustment': 0,
                'reasoning': f'Weak BTC correlation {correlation:.2f}, independent movement ok'
            }

        # Сильная корреляция - проверяем alignment
        if correlation > 0.5:  # Положительная корреляция
            if signal_direction == 'LONG' and btc_trend == 'UP':
                return {
                    'aligned': True,
                    'should_block': False,
                    'confidence_adjustment': +5,
                    'reasoning': 'LONG aligned with BTC uptrend, good'
                }
            elif signal_direction == 'SHORT' and btc_trend == 'DOWN':
                return {
                    'aligned': True,
                    'should_block': False,
                    'confidence_adjustment': +5,
                    'reasoning': 'SHORT aligned with BTC downtrend, good'
                }
            else:
                return {
                    'aligned': False,
                    'should_block': True,
                    'confidence_adjustment': -20,
                    'reasoning': f'{signal_direction} against BTC {btc_trend} trend, high correlation {correlation:.2f}, BLOCK'
                }

        elif correlation < -0.5:  # Отрицательная корреляция
            if signal_direction == 'LONG' and btc_trend == 'DOWN':
                return {
                    'aligned': True,
                    'should_block': False,
                    'confidence_adjustment': +5,
                    'reasoning': 'LONG with negative BTC correlation during BTC down, aligned'
                }
            elif signal_direction == 'SHORT' and btc_trend == 'UP':
                return {
                    'aligned': True,
                    'should_block': False,
                    'confidence_adjustment': +5,
                    'reasoning': 'SHORT with negative BTC correlation during BTC up, aligned'
                }
            else:
                return {
                    'aligned': False,
                    'should_block': True,
                    'confidence_adjustment': -20,
                    'reasoning': f'{signal_direction} misaligned with negative BTC correlation, BLOCK'
                }

        return {
            'aligned': True,
            'should_block': False,
            'confidence_adjustment': 0,
            'reasoning': 'Moderate correlation, acceptable'
        }


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def extract_prices_from_candles(candles: List[List]) -> List[float]:
    """Извлечь цены закрытия из свечей"""
    try:
        return [float(candle[4]) for candle in candles]
    except (IndexError, ValueError, TypeError):
        return []


def calculate_price_change(prices: List[float], window: int = 24) -> float:
    """
    Рассчитать % изменения цены за окно

    Returns:
        percentage change (e.g., 2.5 for +2.5%)
    """
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
    """
    Определить тренд по ценам

    Returns:
        'UP' / 'DOWN' / 'FLAT'
    """
    if len(prices) < window:
        window = len(prices)

    if window < 5:
        return 'FLAT'

    try:
        recent_prices = prices[-window:]

        # Простой метод: сравниваем первую и последнюю треть
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
    """
    Полный корреляционный анализ для одной пары

    Args:
        symbol: анализируемая пара
        symbol_candles: свечи символа
        btc_candles: свечи BTC
        signal_direction: направление сигнала ('LONG'/'SHORT')
        sector_candles: опционально - свечи пар из сектора

    Returns:
        Полный анализ корреляций с рекомендациями
    """
    analyzer = CorrelationAnalyzer()

    # Извлекаем цены
    symbol_prices = extract_prices_from_candles(symbol_candles)
    btc_prices = extract_prices_from_candles(btc_candles)

    if len(symbol_prices) < 24 or len(btc_prices) < 24:
        return {
            'symbol': symbol,
            'error': 'Insufficient price data',
            'total_confidence_adjustment': 0,
            'should_block_signal': False
        }

    # 1. BTC Correlation
    btc_corr_result = analyzer.calculate_btc_correlation(
        symbol, symbol_prices, btc_prices, window=24
    )

    # 2. Price changes
    symbol_change_1h = calculate_price_change(symbol_prices, window=1)
    btc_change_1h = calculate_price_change(btc_prices, window=1)

    # 3. Correlation anomaly
    anomaly_result = analyzer.detect_correlation_anomaly(
        symbol,
        symbol_change_1h,
        btc_change_1h,
        btc_corr_result['correlation']
    )

    # 4. BTC trend
    btc_trend = determine_trend(btc_prices, window=20)

    # 5. BTC alignment check
    alignment_result = analyzer.check_btc_alignment(
        symbol,
        signal_direction,
        btc_trend,
        btc_corr_result['correlation']
    )

    # 6. Sector analysis (если данные есть)
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

    # Суммируем adjustments
    total_adjustment = (
            anomaly_result['confidence_adjustment'] +
            alignment_result['confidence_adjustment'] +
            sector_result['confidence_adjustment']
    )

    # Определяем блокировку
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