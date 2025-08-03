import asyncio
import json
import logging
import time
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import re
import datetime
from decimal import Decimal, ROUND_HALF_UP

# Импорты функций (предполагается, что они существуют)
from func_async import get_klines_async, get_usdt_trading_pairs, get_orderbook_async, get_24h_stats_async
from deepseek import deep_seek_selection, deep_seek_analysis, cleanup_http_client
from func_trade import (
    calculate_advanced_scalping_indicators,
    analyze_market_microstructure,
    detect_liquidity_grab_pattern,
    calculate_dynamic_levels
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scalping_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# КОНФИГУРАЦИЯ СКАЛЬПИНГА (согласно чек-листу)
SCALPING_CONFIG = {
    # Основные параметры
    'primary_timeframe': '5m',  # Основной таймфрейм
    'confirmation_timeframe': '15m',  # Подтверждающий таймфрейм
    'candles_for_scan': 100,  # Свечи для быстрого сканирования
    'candles_for_analysis': 50,  # Свечи для ИИ анализа
    'candles_for_detailed': 200,  # Свечи для детального анализа

    # Фильтрация по ликвидности (чек-лист)
    'min_24h_volume': 100_000_000,  # > 100 млн $
    'min_orderbook_depth': 100_000,  # ≥ $100,000 в топ-10
    'max_spread_percent': 0.05,  # ≤ 0.05% цены
    'min_atr_percent': 0.4,  # ≥ 0.4% достаточно тиков
    'max_correlation': 0.9,  # < 0.9 корреляция с другими парами

    # Торговые часы (EU/US overlap)
    'active_hours': list(range(14, 21)),  # 14:00-20:00 UTC
    'forbidden_hours': [22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8],

    # Обработка данных
    'batch_size': 30,  # Размер батча для параллельной обработки
    'max_pairs_for_ai_selection': 15,  # Максимум пар для первого ИИ
    'max_pairs_for_detailed': 5,  # Максимум пар для детального анализа
    'min_confidence_threshold': 75,  # Минимальная уверенность сигнала

    # Риск-менеджмент
    'max_position_risk': 0.02,  # Максимум 2% депозита на сделку
    'min_risk_reward': 1.5,  # Минимальное соотношение риск/прибыль
    'adaptive_stop_multiplier': 1.2,  # Множитель для адаптивного стопа
}


@dataclass
class LiquidityMetrics:
    """Метрики ликвидности для фильтрации пар"""
    symbol: str
    volume_24h: float
    orderbook_depth_bid: float
    orderbook_depth_ask: float
    spread_percent: float
    atr_percent: float
    funding_rate: float

    def passes_liquidity_filter(self) -> bool:
        """Проверка соответствия критериям ликвидности"""
        return (
                self.volume_24h >= SCALPING_CONFIG['min_24h_volume'] and
                min(self.orderbook_depth_bid, self.orderbook_depth_ask) >= SCALPING_CONFIG['min_orderbook_depth'] and
                self.spread_percent <= SCALPING_CONFIG['max_spread_percent'] and
                self.atr_percent >= SCALPING_CONFIG['min_atr_percent']
        )


@dataclass
class ScalpingSignal:
    """Улучшенный торговый сигнал для скальпинга"""
    symbol: str
    direction: str  # 'LONG', 'SHORT', 'NO_SIGNAL'
    confidence: int
    entry_price: float
    timestamp: int

    # Уровни и риски
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    position_size_percent: float

    # Качественные метрики
    liquidity_score: int
    microstructure_score: int
    volume_confirmation: bool
    pattern_strength: int

    # Причины входа
    entry_reasons: List[str]
    risk_factors: List[str]

    # Данные для ИИ анализа
    primary_candles: List = None
    confirmation_candles: List = None
    indicators_data: Dict = None
    microstructure_data: Dict = None


class AdvancedMarketScanner:
    """Продвинутый сканер рынков с фильтрацией по ликвидности"""

    def __init__(self):
        self.session_start = time.time()
        self.processed_pairs_cache = {}
        logger.info("🚀 Продвинутый сканер рынков инициализирован")

    def is_optimal_trading_time(self) -> bool:
        """Проверка оптимального времени для торговли"""
        current_hour = datetime.datetime.utcnow().hour
        return current_hour in SCALPING_CONFIG['active_hours']

    async def get_liquidity_metrics(self, symbol: str) -> Optional[LiquidityMetrics]:
        """Получение метрик ликвидности для пары"""
        try:
            # Получаем 24h статистику
            stats_24h = await get_24h_stats_async(symbol)
            if not stats_24h:
                return None

            # Получаем стакан заявок
            orderbook = await get_orderbook_async(symbol, limit=10)
            if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
                return None

            # Рассчитываем глубину стакана (топ-10 уровней)
            bid_depth = sum(float(bid[0]) * float(bid[1]) for bid in orderbook['bids'][:10])
            ask_depth = sum(float(ask[0]) * float(ask[1]) for ask in orderbook['asks'][:10])

            # Рассчитываем спред
            best_bid = float(orderbook['bids'][0][0])
            best_ask = float(orderbook['asks'][0][0])
            spread_percent = ((best_ask - best_bid) / best_bid) * 100

            # Получаем ATR для волатильности
            candles = await get_klines_async(symbol, SCALPING_CONFIG['primary_timeframe'], limit=20)
            if not candles:
                return None

            # Рассчитываем ATR
            atr_values = []
            for i in range(1, len(candles)):
                high = float(candles[i][2])
                low = float(candles[i][3])
                prev_close = float(candles[i - 1][4])

                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                atr_values.append(tr)

            current_price = float(candles[-1][4])
            atr_percent = (np.mean(atr_values[-14:]) / current_price) * 100 if atr_values else 0

            return LiquidityMetrics(
                symbol=symbol,
                volume_24h=float(stats_24h.get('volume', 0)) * current_price,  # В USDT
                orderbook_depth_bid=bid_depth,
                orderbook_depth_ask=ask_depth,
                spread_percent=spread_percent,
                atr_percent=atr_percent,
                funding_rate=float(stats_24h.get('fundingRate', 0)) * 100
            )

        except Exception as e:
            logger.error(f"❌ Ошибка получения метрик ликвидности {symbol}: {e}")
            return None

    async def filter_liquid_pairs(self, pairs: List[str]) -> List[str]:
        """Фильтрация пар по критериям ликвидности"""
        logger.info(f"🔍 Фильтрация {len(pairs)} пар по ликвидности")

        liquid_pairs = []

        # Обрабатываем батчами для оптимизации
        for i in range(0, len(pairs), SCALPING_CONFIG['batch_size']):
            batch = pairs[i:i + SCALPING_CONFIG['batch_size']]

            # Получаем метрики параллельно
            tasks = [self.get_liquidity_metrics(pair) for pair in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Фильтруем результаты
            for pair, metrics in zip(batch, results):
                if isinstance(metrics, LiquidityMetrics) and metrics.passes_liquidity_filter():
                    liquid_pairs.append(pair)
                    logger.debug(
                        f"✅ {pair}: V={metrics.volume_24h / 1e6:.1f}M, S={metrics.spread_percent:.3f}%, ATR={metrics.atr_percent:.2f}%")
                elif isinstance(metrics, Exception):
                    logger.error(f"❌ {pair}: {metrics}")

            # Небольшая пауза между батчами
            await asyncio.sleep(0.1)

        logger.info(f"✅ Отфильтровано {len(liquid_pairs)} ликвидных пар из {len(pairs)}")
        return liquid_pairs

    async def scan_pair_for_signals(self, symbol: str) -> Optional[ScalpingSignal]:
        """Сканирование пары на предмет скальпинговых сигналов"""
        try:
            # Получаем данные по основному таймфрейму
            primary_candles = await get_klines_async(
                symbol,
                SCALPING_CONFIG['primary_timeframe'],
                limit=SCALPING_CONFIG['candles_for_scan']
            )

            # Получаем данные по подтверждающему таймфрейму
            confirmation_candles = await get_klines_async(
                symbol,
                SCALPING_CONFIG['confirmation_timeframe'],
                limit=50
            )

            if not primary_candles or not confirmation_candles:
                return None

            # Рассчитываем продвинутые индикаторы
            indicators = calculate_advanced_scalping_indicators(primary_candles)

            # Анализируем микроструктуру
            microstructure = await analyze_market_microstructure(symbol)

            # Определяем сигнал
            signal_data = self._evaluate_scalping_signal(
                symbol, primary_candles, confirmation_candles,
                indicators, microstructure
            )

            if signal_data['direction'] == 'NO_SIGNAL':
                return None

            return ScalpingSignal(
                symbol=symbol,
                direction=signal_data['direction'],
                confidence=signal_data['confidence'],
                entry_price=signal_data['entry_price'],
                timestamp=int(time.time()),

                stop_loss=signal_data['stop_loss'],
                take_profit=signal_data['take_profit'],
                risk_reward_ratio=signal_data['risk_reward_ratio'],
                position_size_percent=signal_data['position_size_percent'],

                liquidity_score=signal_data['liquidity_score'],
                microstructure_score=signal_data['microstructure_score'],
                volume_confirmation=signal_data['volume_confirmation'],
                pattern_strength=signal_data['pattern_strength'],

                entry_reasons=signal_data['entry_reasons'],
                risk_factors=signal_data['risk_factors'],

                # Данные для ИИ
                primary_candles=primary_candles[-SCALPING_CONFIG['candles_for_analysis']:],
                confirmation_candles=confirmation_candles[-25:],
                indicators_data=self._clean_indicators_for_json(indicators),
                microstructure_data=microstructure
            )

        except Exception as e:
            logger.error(f"❌ Ошибка сканирования {symbol}: {e}")
            return None

    def _evaluate_scalping_signal(self, symbol: str, primary_candles: List,
                                  confirmation_candles: List, indicators: Dict,
                                  microstructure: Dict) -> Dict:
        """Комплексная оценка скальпингового сигнала"""

        current_price = float(primary_candles[-1][4])
        current_volume = float(primary_candles[-1][5])
        avg_volume = np.mean([float(c[5]) for c in primary_candles[-20:]])

        # Базовые условия
        entry_reasons = []
        risk_factors = []
        confidence = 0
        direction = 'NO_SIGNAL'

        # 1. ОБЪЕМНОЕ ПОДТВЕРЖДЕНИЕ
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        volume_confirmation = volume_ratio >= 1.5

        if volume_confirmation:
            entry_reasons.append(f"Volume spike: {volume_ratio:.1f}x")
            confidence += 15
        else:
            risk_factors.append(f"Low volume: {volume_ratio:.1f}x")

        # 2. ТЕХНИЧЕСКИЙ АНАЛИЗ
        rsi = indicators.get('rsi_current', 50)
        tema_alignment = indicators.get('tema_alignment', False)
        macd_signal = indicators.get('macd_signal', 'NEUTRAL')

        # Условия для LONG
        if (rsi < 35 and tema_alignment and macd_signal in ['BULLISH', 'CROSS_UP'] and
                indicators.get('near_support', False)):
            direction = 'LONG'
            confidence += 25
            entry_reasons.append("LONG: RSI oversold + TEMA aligned + MACD bullish + near support")

        # Условия для SHORT
        elif (rsi > 65 and not tema_alignment and macd_signal in ['BEARISH', 'CROSS_DOWN'] and
              indicators.get('near_resistance', False)):
            direction = 'SHORT'
            confidence += 25
            entry_reasons.append("SHORT: RSI overbought + TEMA bearish + MACD bearish + near resistance")

        # 3. МИКРОСТРУКТУРНЫЙ АНАЛИЗ
        if microstructure:
            book_imbalance = microstructure.get('book_imbalance', 0)
            spread_quality = microstructure.get('spread_stability', 0)

            if abs(book_imbalance) > 0.6:
                entry_reasons.append(f"Book imbalance: {book_imbalance:.2f}")
                confidence += 10

            if spread_quality > 0.7:
                confidence += 5
            else:
                risk_factors.append("Unstable spread")

        # 4. ПАТТЕРН СТОП-ОХОТЫ
        liquidity_grab = detect_liquidity_grab_pattern(primary_candles)
        if liquidity_grab['detected']:
            confidence += 20
            entry_reasons.append(f"Liquidity grab: {liquidity_grab['pattern']}")

        # 5. РАСЧЕТ УРОВНЕЙ
        atr = indicators.get('atr_current', current_price * 0.005)

        # Адаптивный стоп-лосс (согласно чек-листу)
        base_stop_distance = atr * SCALPING_CONFIG['adaptive_stop_multiplier']
        spread_adjustment = current_price * 0.0005  # 0.05% на спред
        commission_adjustment = current_price * 0.001  # 0.1% на комиссию
        slippage_adjustment = current_price * 0.0005  # 0.05% на проскальзывание

        stop_distance = base_stop_distance + spread_adjustment + commission_adjustment + slippage_adjustment
        stop_distance = max(stop_distance, current_price * 0.004)  # Минимум 0.4%
        stop_distance = min(stop_distance, current_price * 0.008)  # Максимум 0.8%

        if direction == 'LONG':
            stop_loss = current_price - stop_distance
            take_profit = current_price + (stop_distance * 2.0)  # R:R = 1:2
        elif direction == 'SHORT':
            stop_loss = current_price + stop_distance
            take_profit = current_price - (stop_distance * 2.0)
        else:
            stop_loss = take_profit = current_price

        # Расчет риска
        risk_percent = stop_distance / current_price
        risk_reward_ratio = abs(take_profit - current_price) / abs(
            stop_loss - current_price) if stop_loss != current_price else 0

        # Проверка минимальных требований
        if confidence < SCALPING_CONFIG['min_confidence_threshold']:
            direction = 'NO_SIGNAL'
            risk_factors.append(f"Low confidence: {confidence}")

        if risk_reward_ratio < SCALPING_CONFIG['min_risk_reward']:
            direction = 'NO_SIGNAL'
            risk_factors.append(f"Poor R:R: {risk_reward_ratio:.2f}")

        return {
            'direction': direction,
            'confidence': min(confidence, 100),
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': risk_reward_ratio,
            'position_size_percent': min(SCALPING_CONFIG['max_position_risk'],
                                         0.01 / risk_percent) if risk_percent > 0 else 0,

            'liquidity_score': min(confidence, 100),
            'microstructure_score': int(microstructure.get('overall_score', 50)) if microstructure else 50,
            'volume_confirmation': volume_confirmation,
            'pattern_strength': min(confidence, 100),

            'entry_reasons': entry_reasons,
            'risk_factors': risk_factors
        }

    def _clean_indicators_for_json(self, indicators: Dict) -> Dict:
        """Очистка индикаторов для JSON сериализации"""
        cleaned = {}
        for key, value in indicators.items():
            if isinstance(value, (np.ndarray, list)):
                # Берем только последние 20 значений и очищаем от NaN
                cleaned_values = []
                for v in (value[-20:] if len(value) > 20 else value):
                    if isinstance(v, (np.integer, np.floating)):
                        v = float(v)
                    if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                        cleaned_values.append(v)
                cleaned[key] = cleaned_values
            elif isinstance(value, (np.integer, np.floating)):
                v = float(value)
                if not (math.isnan(v) or math.isinf(v)):
                    cleaned[key] = v
                else:
                    cleaned[key] = 0.0
            else:
                cleaned[key] = value

        return cleaned


class AIScalpingOrchestrator:
    """ИИ оркестратор для двухэтапного анализа"""

    def __init__(self):
        self.selection_prompt = self._load_prompt('prompt2.txt')
        self.detailed_prompt = self._load_prompt('prompt.txt')
        logger.info("🤖 ИИ оркестратор инициализирован")

    def _load_prompt(self, filename: str) -> str:
        """Загрузка промпта из файла"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error(f"❌ Файл промпта {filename} не найден")
            return ""

    async def first_stage_selection(self, signals: List[ScalpingSignal]) -> List[str]:
        """ЭТАП 1: Быстрый отбор лучших сигналов через ИИ"""
        if not self.selection_prompt or not signals:
            return []

        logger.info(f"🤖 ЭТАП 1 ИИ: Анализ {len(signals)} сигналов для отбора")

        try:
            # Ограничиваем количество сигналов для ИИ
            top_signals = signals[:SCALPING_CONFIG['max_pairs_for_ai_selection']]

            # Подготавливаем данные для ИИ (компактный формат)
            ai_data = {
                'timestamp': int(time.time()),
                'market_session': 'EU_US_OVERLAP' if self._is_overlap_session() else 'OTHER',
                'total_signals': len(top_signals),
                'strategy': 'scalping_5m_15m_confirmation',

                'signals': []
            }

            for signal in top_signals:
                signal_data = {
                    'symbol': signal.symbol,
                    'direction': signal.direction,
                    'confidence': signal.confidence,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'risk_reward': round(signal.risk_reward_ratio, 2),

                    'scores': {
                        'liquidity': signal.liquidity_score,
                        'microstructure': signal.microstructure_score,
                        'pattern_strength': signal.pattern_strength
                    },

                    'confirmations': {
                        'volume_spike': signal.volume_confirmation,
                        'entry_reasons': signal.entry_reasons[:3],  # Топ-3 причины
                        'risk_factors': signal.risk_factors[:2]  # Топ-2 риска
                    },

                    # Ключевые технические показатели (последние 5 значений)
                    'technicals': {
                        'rsi_current': signal.indicators_data.get('rsi_current', 50),
                        'tema_aligned': signal.indicators_data.get('tema_alignment', False),
                        'macd_signal': signal.indicators_data.get('macd_signal', 'NEUTRAL'),
                        'atr_percent': signal.indicators_data.get('atr_percent', 0.5),
                        'volume_ratio': signal.indicators_data.get('volume_ratio', 1.0)
                    }
                }

                ai_data['signals'].append(signal_data)

            # Формируем запрос к ИИ
            message = f"""{self.selection_prompt}

=== СКАЛЬПИНГ 5M/15M: ПЕРВИЧНЫЙ ОТБОР ===
Время: {datetime.datetime.utcnow().strftime('%H:%M UTC')}
Сессия: {ai_data['market_session']}
Стратегия: Удержание 3-5 свечей, цель 0.4-0.6% чистой прибыли

{json.dumps(ai_data, indent=2, ensure_ascii=False)}

ЗАДАЧА: Выбери 3-5 лучших пар для скальпинга, учитывая:
1. Качество ликвидности и микроструктуры
2. Силу технических сигналов
3. Оптимальное соотношение риск/прибыль
4. Минимизацию корреляционных рисков

ФОРМАТ ОТВЕТА: {{"selected_pairs": ["BTCUSDT", "ETHUSDT"], "reasoning": "краткое обоснование"}}"""

            # Отправляем запрос
            ai_response = await deep_seek_selection(message)

            if not ai_response:
                logger.error("❌ ИИ не ответил на первичный отбор")
                return []

            # Парсим ответ
            selected_pairs = self._parse_selection_response(ai_response)

            logger.info(f"✅ ЭТАП 1 ИИ завершен: выбрано {len(selected_pairs)} пар")
            return selected_pairs

        except Exception as e:
            logger.error(f"❌ Ошибка первичного ИИ отбора: {e}")
            return []

    async def second_stage_analysis(self, symbol: str) -> Optional[Dict]:
        """ЭТАП 2: Детальный анализ выбранной пары"""
        if not self.detailed_prompt:
            return None

        logger.info(f"🔬 ЭТАП 2 ИИ: Детальный анализ {symbol}")

        try:
            # Получаем расширенные данные
            primary_candles = await get_klines_async(
                symbol,
                SCALPING_CONFIG['primary_timeframe'],
                limit=SCALPING_CONFIG['candles_for_detailed']
            )

            confirmation_candles = await get_klines_async(
                symbol,
                SCALPING_CONFIG['confirmation_timeframe'],
                limit=100
            )

            if not primary_candles or not confirmation_candles:
                logger.error(f"❌ Недостаточно данных для {symbol}")
                return None

            # Полный технический анализ
            full_indicators = calculate_advanced_scalping_indicators(primary_candles)
            microstructure = await analyze_market_microstructure(symbol)
            dynamic_levels = calculate_dynamic_levels(primary_candles)

            # Подготавливаем данные для детального анализа
            analysis_data = {
                'symbol': symbol,
                'timestamp': int(time.time()),
                'current_price': float(primary_candles[-1][4]),
                'market_session': 'EU_US_OVERLAP' if self._is_overlap_session() else 'OTHER',

                # Свечные данные (последние 30 свечей 5M + 15 свечей 15M)
                'price_action': {
                    'primary_5m': [
                        {
                            'timestamp': int(c[0]),
                            'open': float(c[1]),
                            'high': float(c[2]),
                            'low': float(c[3]),
                            'close': float(c[4]),
                            'volume': float(c[5])
                        } for c in primary_candles[-30:]
                    ],
                    'confirmation_15m': [
                        {
                            'timestamp': int(c[0]),
                            'open': float(c[1]),
                            'high': float(c[2]),
                            'low': float(c[3]),
                            'close': float(c[4]),
                            'volume': float(c[5])
                        } for c in confirmation_candles[-15:]
                    ]
                },

                # Технический анализ
                'technical_analysis': {
                    'trend': {
                        'tema_alignment': full_indicators.get('tema_alignment', False),
                        'tema_slope': full_indicators.get('tema_slope', 0),
                        'trend_strength': full_indicators.get('trend_strength', 0),
                        'trend_direction': full_indicators.get('trend_direction', 'SIDEWAYS')
                    },
                    'momentum': {
                        'rsi_current': full_indicators.get('rsi_current', 50),
                        'rsi_trend': full_indicators.get('rsi_trend', 'NEUTRAL'),
                        'stoch_signal': full_indicators.get('stoch_signal', 'NEUTRAL'),
                        'macd_signal': full_indicators.get('macd_signal', 'NEUTRAL'),
                        'momentum_divergence': full_indicators.get('momentum_divergence', False)
                    },
                    'volatility': {
                        'atr_current': full_indicators.get('atr_current', 0),
                        'atr_percent': full_indicators.get('atr_percent', 0.5),
                        'volatility_regime': full_indicators.get('volatility_regime', 'MEDIUM'),
                        'price_velocity': full_indicators.get('price_velocity', 0),
                        'volatility_breakout': full_indicators.get('volatility_breakout', False)
                    },
                    'volume': {
                        'volume_spike': full_indicators.get('volume_spike', False),
                        'volume_ratio': full_indicators.get('volume_ratio', 1.0),
                        'volume_trend': full_indicators.get('volume_trend', 'NEUTRAL'),
                        'institutional_flow': full_indicators.get('institutional_flow', 'NEUTRAL')
                    }
                },

                # Микроструктура рынка
                'microstructure': microstructure if microstructure else {},

                # Динамические уровни
                'levels': {
                    'support_levels': dynamic_levels.get('support_levels', [])[:5],
                    'resistance_levels': dynamic_levels.get('resistance_levels', [])[:5],
                    'pivot_points': dynamic_levels.get('pivot_points', {}),
                    'volume_profile': dynamic_levels.get('volume_profile', {}),
                    'liquidity_zones': dynamic_levels.get('liquidity_zones', [])
                },

                # Паттерны и сигналы
                'patterns': {
                    'liquidity_grab': detect_liquidity_grab_pattern(primary_candles),
                    'reversal_patterns': full_indicators.get('reversal_patterns', []),
                    'continuation_patterns': full_indicators.get('continuation_patterns', []),
                    'breakout_potential': full_indicators.get('breakout_potential', 0)
                }
            }

            # Формируем запрос для детального анализа
            message = f"""{self.detailed_prompt}

=== ДЕТАЛЬНЫЙ СКАЛЬПИНГОВЫЙ АНАЛИЗ ===
ПАРА: {symbol}
ВРЕМЯ: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
ТЕКУЩАЯ ЦЕНА: {analysis_data['current_price']}
СЕССИЯ: {analysis_data['market_session']}

СТРАТЕГИЯ: Скальпинг 5M с подтверждением 15M
ЦЕЛЬ: 0.4-0.6% чистой прибыли за 15-45 минут
УДЕРЖАНИЕ: 3-5 свечей

{json.dumps(analysis_data, indent=2, ensure_ascii=False)}

ТРЕБУЕТСЯ АНАЛИЗ:
1. Качество точки входа (спреды, ликвидность, timing)
2. Адаптивные уровни стоп-лосса и тейк-профита
3. Оценка рисков микроструктуры
4. План исполнения (тип ордеров, размер позиции)
5. Сценарии выхода и управления позицией

ОТВЕТЬ В ФОРМАТЕ JSON согласно промпту."""

            # Отправляем запрос
            analysis_result = await deep_seek_analysis(message)

            if analysis_result:
                # Сохраняем результат
                self._save_detailed_analysis(symbol, analysis_result, analysis_data)
                logger.info(f"✅ Детальный анализ {symbol} завершен")

                # Парсим JSON ответ из анализа
                parsed_result = self._parse_analysis_response(analysis_result)
                return parsed_result

            return None

        except Exception as e:
            logger.error(f"❌ Ошибка детального анализа {symbol}: {e}")
            return None

    def _is_overlap_session(self) -> bool:
        """Проверка пересечения EU/US сессий"""
        current_hour = datetime.datetime.utcnow().hour
        return 14 <= current_hour <= 18  # 14:00-18:00 UTC

    def _parse_selection_response(self, response: str) -> List[str]:
        """Парсинг ответа первичного отбора"""
        try:
            # Ищем JSON с выбранными парами
            json_pattern = r'\{[^}]*"selected_pairs"[^}]*\}'
            match = re.search(json_pattern, response)
            if match:
                data = json.loads(match.group())
                return data.get('selected_pairs', [])

            # Альтернативный поиск списка пар
            pair_pattern = r'([A-Z]{3,10}USDT)'
            pairs = re.findall(pair_pattern, response)
            return list(set(pairs))[:5]  # Уникальные пары, максимум 5

        except Exception as e:
            logger.error(f"❌ Ошибка парсинга отбора: {e}")
            return []

    def _parse_analysis_response(self, response: str) -> Optional[Dict]:
        """Парсинг детального анализа"""
        try:
            # Ищем JSON в ответе
            json_pattern = r'\{[^{}]*"coin"[^{}]*\}'
            match = re.search(json_pattern, response)
            if match:
                return json.loads(match.group())
            return None
        except Exception as e:
            logger.error(f"❌ Ошибка парсинга анализа: {e}")
            return None

    def _save_detailed_analysis(self, symbol: str, analysis: str, data: Dict):
        """Сохранение детального анализа"""
        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'analysis_{symbol}_{timestamp}.json'

            save_data = {
                'symbol': symbol,
                'timestamp': timestamp,
                'analysis_text': analysis,
                'technical_data': data
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            logger.info(f"📁 Анализ сохранен: {filename}")

        except Exception as e:
            logger.error(f"❌ Ошибка сохранения: {e}")


async def main():
    """Главная функция оптимизированного скальпингового бота"""
    logger.info("🚀 ПРОФЕССИОНАЛЬНЫЙ СКАЛЬПИНГОВЫЙ БОТ - ЗАПУСК")
    logger.info("🎯 Стратегия: 5M основной + 15M подтверждение")
    logger.info("💎 Цель: 0.4-0.6% чистой прибыли за 15-45 минут")
    logger.info("⚡ Двухэтапный ИИ анализ + фильтрация ликвидности")

    # Проверка торгового времени
    scanner = AdvancedMarketScanner()
    ai_orchestrator = AIScalpingOrchestrator()

    if not scanner.is_optimal_trading_time():
        logger.warning("⏰ Неоптимальное время для скальпинга (низкая ликвидность)")
        logger.info("🕐 Оптимальное время: 14:00-20:00 UTC (EU/US overlap)")
        return

    try:
        start_time = time.time()

        # ===== ЭТАП 1: ПОЛУЧЕНИЕ И ФИЛЬТРАЦИЯ ПАР =====
        logger.info("\n" + "=" * 60)
        logger.info("📊 ЭТАП 1: ФИЛЬТРАЦИЯ ПАР ПО ЛИКВИДНОСТИ")
        logger.info("=" * 60)

        # Получаем все USDT пары
        all_pairs = await get_usdt_trading_pairs()
        if not all_pairs:
            logger.error("❌ Не удалось получить список торговых пар")
            return

        logger.info(f"🔍 Получено {len(all_pairs)} USDT пар")

        # Фильтруем по ликвидности
        liquid_pairs = await scanner.filter_liquid_pairs(all_pairs)

        if not liquid_pairs:
            logger.error("❌ Нет пар, соответствующих критериям ликвидности")
            return

        logger.info(f"✅ Отфильтровано {len(liquid_pairs)} ликвидных пар")

        # ===== ЭТАП 2: СКАНИРОВАНИЕ НА СИГНАЛЫ =====
        logger.info("\n" + "=" * 60)
        logger.info("🔍 ЭТАП 2: СКАНИРОВАНИЕ СКАЛЬПИНГОВЫХ СИГНАЛОВ")
        logger.info("=" * 60)

        signals = []

        # Сканируем батчами для эффективности
        for i in range(0, len(liquid_pairs), SCALPING_CONFIG['batch_size']):
            batch = liquid_pairs[i:i + SCALPING_CONFIG['batch_size']]

            tasks = [scanner.scan_pair_for_signals(pair) for pair in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Собираем результаты
            for result in results:
                if isinstance(result, ScalpingSignal):
                    signals.append(result)

            # Логируем прогресс
            processed = min(i + SCALPING_CONFIG['batch_size'], len(liquid_pairs))
            logger.info(f"⏳ Просканировано: {processed}/{len(liquid_pairs)} пар")

            await asyncio.sleep(0.1)  # Небольшая пауза

        # Сортируем по уверенности
        signals.sort(key=lambda x: x.confidence, reverse=True)

        if not signals:
            logger.info("ℹ️ Качественных скальпинговых сигналов не найдено")
            return

        logger.info(f"📈 Найдено {len(signals)} перспективных сигналов")

        # Выводим топ-5 сигналов
        logger.info("\n🏆 ТОП-5 СИГНАЛОВ:")
        for i, signal in enumerate(signals[:5], 1):
            logger.info(f"{i}. {signal.symbol}: {signal.direction} "
                        f"(conf: {signal.confidence}%, R:R: {signal.risk_reward_ratio:.2f})")

        # ===== ЭТАП 3: ПЕРВИЧНЫЙ ИИ ОТБОР =====
        logger.info("\n" + "=" * 60)
        logger.info("🤖 ЭТАП 3: ПЕРВИЧНЫЙ ИИ ОТБОР")
        logger.info("=" * 60)

        selected_pairs = await ai_orchestrator.first_stage_selection(signals)

        if not selected_pairs:
            logger.info("ℹ️ ИИ не выбрал ни одной пары для детального анализа")
            return

        logger.info(f"✅ ИИ отобрал {len(selected_pairs)} пар: {selected_pairs}")

        # ===== ЭТАП 4: ДЕТАЛЬНЫЙ ИИ АНАЛИЗ =====
        logger.info("\n" + "=" * 60)
        logger.info("🔬 ЭТАП 4: ДЕТАЛЬНЫЙ ИИ АНАЛИЗ")
        logger.info("=" * 60)

        final_recommendations = []

        for pair in selected_pairs[:SCALPING_CONFIG['max_pairs_for_detailed']]:
            logger.info(f"🔬 Анализируем {pair}...")

            analysis_result = await ai_orchestrator.second_stage_analysis(pair)

            if analysis_result and analysis_result.get('coin'):
                final_recommendations.append({
                    'pair': pair,
                    'recommendation': analysis_result
                })
                logger.info(f"✅ {pair}: {analysis_result.get('direction', 'N/A')} "
                            f"@ {analysis_result.get('entry_price', 'N/A')}")
            else:
                logger.warning(f"⚠️ {pair}: анализ не дал торговых рекомендаций")

            # Пауза между запросами к ИИ
            await asyncio.sleep(2)

        # ===== ИТОГОВЫЙ ОТЧЕТ =====
        total_time = time.time() - start_time

        logger.info("\n" + "=" * 60)
        logger.info("🎉 АНАЛИЗ ЗАВЕРШЕН - ИТОГОВЫЙ ОТЧЕТ")
        logger.info("=" * 60)
        logger.info(f"⏱️  Время выполнения: {total_time:.1f} секунд")
        logger.info(f"📊 Всего пар проанализировано: {len(all_pairs)}")
        logger.info(f"💧 Ликвидных пар: {len(liquid_pairs)}")
        logger.info(f"📈 Найдено сигналов: {len(signals)}")
        logger.info(f"🤖 ИИ отобрал для детального анализа: {len(selected_pairs)}")
        logger.info(f"🎯 Финальных торговых рекомендаций: {len(final_recommendations)}")

        if final_recommendations:
            logger.info("\n🏆 ФИНАЛЬНЫЕ ТОРГОВЫЕ РЕКОМЕНДАЦИИ:")
            for i, rec in enumerate(final_recommendations, 1):
                r = rec['recommendation']
                logger.info(f"{i}. {rec['pair']}: {r.get('direction', 'N/A')} "
                            f"@ {r.get('entry_price', 'N/A')} "
                            f"(SL: {r.get('stop_loss', 'N/A')}, "
                            f"TP: {r.get('take_profit', 'N/A')})")

        logger.info(f"\n📁 Детальные результаты сохранены в analysis_*.json")

        # Очистка ресурсов
        await cleanup_http_client()

    except KeyboardInterrupt:
        logger.info("⏹️ Остановка по запросу пользователя")
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        logger.info("🔚 Сессия завершена")


if __name__ == "__main__":
    # Заголовок приложения
    print("=" * 80)
    print("🎯 ПРОФЕССИОНАЛЬНЫЙ СКАЛЬПИНГОВЫЙ БОТ BYBIT")
    print("📊 Стратегия: 5M основной + 15M подтверждение")
    print("💎 Цель: 0.4-0.6% чистой прибыли за 15-45 минут")
    print("⚡ Двухэтапный ИИ анализ + строгая фильтрация")
    print("🕐 Оптимальное время: 14:00-20:00 UTC")
    print("=" * 80)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Программа остановлена пользователем")
    except Exception as e:
        logger.error(f"💥 Фатальная ошибка: {e}")
    finally:
        logger.info("🔚 Работа завершена")