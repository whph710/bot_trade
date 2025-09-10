import asyncio
import json
import logging
import time
import math
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import re
import datetime

# ИМПОРТЫ
from func_async import get_klines_async, get_usdt_trading_pairs
from deepseek import deep_seek_selection, deep_seek_analysis, cleanup_http_client

# Импорт с МАКСИМАЛЬНЫМИ индикаторами
from func_trade import (detect_instruction_based_signals_enhanced,
                        calculate_all_indicators_comprehensive,
                        analyze_higher_timeframe_trend_comprehensive)

# Импорт обновленной конфигурации
from config import config

# Настройка логирования без эмоджи
logging.basicConfig(
    level=getattr(logging, config.system.LOG_LEVEL),
    format=config.system.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.system.LOG_FILE, encoding=config.system.ENCODING),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def clean_value(value):
    """Очистка значений от NaN и Infinity"""
    if isinstance(value, (np.integer, np.floating)):
        value = float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, np.ndarray):
        return [clean_value(x) for x in value.tolist()]

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value
    elif isinstance(value, dict):
        return {k: clean_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [clean_value(item) for item in value]
    else:
        return value


def safe_json_serialize(obj: Any) -> Any:
    """Безопасная JSON сериализация"""
    return clean_value(obj)


def filter_safe_pairs(pairs: List[str]) -> List[str]:
    """Фильтрация пар по критериям безопасности"""
    safe_pairs = []

    for pair in pairs:
        # Исключаем пары из blacklist
        if any(excluded in pair for excluded in config.safety.EXCLUDED_PAIRS):
            continue

        # Помечаем высокорисковые пары (но не исключаем полностью)
        is_high_risk = any(risky in pair for risky in config.safety.HIGH_RISK_PAIRS)

        # Основные фильтры
        if (pair.endswith('USDT') and
                not pair.startswith('USDT') and
                '-' not in pair and
                not any(char.isdigit() for char in pair)):
            safe_pairs.append(pair)

    logger.info(f"Отфильтровано {len(safe_pairs)} безопасных пар из {len(pairs)}")
    return safe_pairs


@dataclass
class EnhancedSignal:
    """Расширенный сигнал с максимальными данными для ИИ"""
    pair: str
    signal_type: str  # 'LONG', 'SHORT', 'NO_SIGNAL'
    confidence: int
    entry_price: float
    timestamp: int

    # Данные согласно консервативной стратегии
    pattern_type: str
    higher_tf_trend: str
    validation_score: str
    atr_current: float
    volume_ratio: float

    # МАКСИМАЛЬНЫЕ данные индикаторов для ИИ
    comprehensive_indicators: Dict = None
    higher_tf_analysis: Dict = None
    risk_metrics: Dict = None

    # Свечные данные для ИИ (увеличенные объемы)
    candles_5m_extended: List = None  # 300 свечей для полного анализа
    candles_15m_extended: List = None  # 100 свечей для контекста


class ConservativeAnalyzer:
    """Консервативный анализатор с фокусом на прибыльность"""

    def __init__(self):
        self.session_start = time.time()
        self.analyzed_pairs_count = 0
        self.promising_signals_count = 0
        logger.info("Консервативный анализатор запущен (макс. данные + строгие фильтры)")

    def passes_enhanced_liquidity_filter(self, symbol: str, candles: List) -> bool:
        """Усиленный фильтр ликвидности"""
        if not candles or len(candles) < 50:
            return False

        # Расчет среднего объема за последние 24 часа (288 свечей 5m)
        recent_candles = candles[-min(288, len(candles)):]
        total_volume_usd = sum(float(c[5]) * float(c[4]) for c in recent_candles)
        hours_covered = len(recent_candles) / 12  # 12 свечей в час на 5m

        if hours_covered > 0:
            daily_volume_estimate = (total_volume_usd / hours_covered) * 24
        else:
            daily_volume_estimate = 0

        # СТРОГИЕ требования к ликвидности
        liquidity_ok = daily_volume_estimate > config.trading.MIN_LIQUIDITY_VOLUME

        if not liquidity_ok:
            logger.debug(
                f"{symbol}: низкая ликвидность ${daily_volume_estimate:,.0f} < ${config.trading.MIN_LIQUIDITY_VOLUME:,.0f}")

        return liquidity_ok

    def check_enhanced_spread_quality(self, candles: List) -> bool:
        """Улучшенная проверка качества спреда"""
        if len(candles) < 20:
            return False

        # Анализ последних 20 свечей
        recent_spreads = []
        for candle in candles[-20:]:
            high_price = float(candle[2])
            low_price = float(candle[3])
            avg_price = (high_price + low_price) / 2

            if avg_price > 0:
                spread_estimate = ((high_price - low_price) / avg_price) * 100
                recent_spreads.append(spread_estimate)

        if not recent_spreads:
            return False

        avg_spread = sum(recent_spreads) / len(recent_spreads)
        spread_volatility = np.std(recent_spreads) if len(recent_spreads) > 1 else 0

        # Проверяем и средний спред, и его стабильность
        return (avg_spread < config.trading.MAX_SPREAD_PERCENT and
                spread_volatility < config.trading.MAX_SPREAD_PERCENT * 0.5)

    def check_time_filter(self) -> bool:
        """Проверка временных фильтров"""
        current_hour_utc = datetime.datetime.utcnow().hour

        # Избегаем волатильные часы
        if current_hour_utc in config.safety.AVOID_HOURS_UTC:
            logger.debug(f"Пропускаем час {current_hour_utc} UTC (высокая волатильность)")
            return False

        return True

    async def comprehensive_pair_analysis(self, symbol: str) -> Optional[EnhancedSignal]:
        """Максимально полный анализ пары с консервативными фильтрами"""
        try:
            self.analyzed_pairs_count += 1

            # Получаем МАКСИМУМ данных для анализа
            candles_5m_full = await get_klines_async(
                symbol,
                config.timeframe.ENTRY_TF,
                limit=config.timeframe.CANDLES_5M  # 500 свечей
            )

            candles_15m_full = await get_klines_async(
                symbol,
                config.timeframe.CONTEXT_TF,
                limit=config.timeframe.CANDLES_15M  # 200 свечей
            )

            if not candles_5m_full or not candles_15m_full:
                return None

            # СТРОГИЕ предварительные фильтры
            if not self.passes_enhanced_liquidity_filter(symbol, candles_5m_full):
                return None

            if not self.check_enhanced_spread_quality(candles_5m_full):
                return None

            if not self.check_time_filter():
                return None

            # Полный технический анализ с МАКСИМАЛЬНЫМИ данными
            signal_result = detect_instruction_based_signals_enhanced(
                candles_5m_full,
                candles_15m_full
            )

            if signal_result['signal'] == 'NO_SIGNAL':
                return None

            # Дополнительные консервативные фильтры
            confidence = int(signal_result['confidence'])
            if confidence < config.trading.MIN_CONFIDENCE:
                return None

            # Проверяем валидацию (4 из 5)
            validation_score = signal_result.get('validation_score', '0/5')
            validation_parts = validation_score.split('/')
            if len(validation_parts) == 2:
                passed = int(validation_parts[0])
                if passed < config.trading.VALIDATION_CHECKS_REQUIRED:
                    return None

            entry_price = float(candles_5m_full[-1][4])
            if math.isnan(entry_price):
                return None

            self.promising_signals_count += 1

            # Создаем расширенный сигнал с МАКСИМАЛЬНЫМИ данными
            enhanced_signal = EnhancedSignal(
                pair=symbol,
                signal_type=signal_result['signal'],
                confidence=confidence,
                entry_price=entry_price,
                timestamp=int(time.time()),

                # Основные данные
                pattern_type=signal_result.get('pattern_type', 'UNKNOWN'),
                higher_tf_trend=signal_result.get('higher_tf_analysis', {}).get('trend', 'UNKNOWN'),
                validation_score=validation_score,
                atr_current=signal_result.get('risk_metrics', {}).get('atr_based_stop', 0.0),
                volume_ratio=signal_result.get('comprehensive_data', {}).get('volume_analysis', {}).get('volume_ratio',
                                                                                                        1.0),

                # МАКСИМАЛЬНЫЕ данные для ИИ
                comprehensive_indicators=signal_result.get('comprehensive_data', {}),
                higher_tf_analysis=signal_result.get('higher_tf_analysis', {}),
                risk_metrics=signal_result.get('risk_metrics', {}),

                # Расширенные свечные данные для ИИ
                candles_5m_extended=candles_5m_full[-config.timeframe.CANDLES_FOR_FULL_ANALYSIS:],
                candles_15m_extended=candles_15m_full[-config.timeframe.CANDLES_FOR_CONTEXT:]
            )

            logger.debug(f"{symbol}: сигнал {enhanced_signal.signal_type} ({confidence}%, {validation_score})")
            return enhanced_signal

        except Exception as e:
            logger.error(f"Ошибка анализа {symbol}: {e}")
            return None

    async def mass_scan_markets_conservative(self) -> List[EnhancedSignal]:
        """Массовое сканирование с консервативными фильтрами"""
        start_time = time.time()
        logger.info("ЭТАП 1: Консервативное сканирование с максимальными данными")

        try:
            # Получаем и фильтруем пары по безопасности
            all_pairs = await get_usdt_trading_pairs()
            safe_pairs = filter_safe_pairs(all_pairs)

            if not safe_pairs:
                logger.error("Не найдено безопасных торговых пар")
                return []

            logger.info(f"Сканируем {len(safe_pairs)} безопасных пар")

            promising_signals = []

            # Обрабатываем консервативными батчами
            batch_size = config.processing.BATCH_SIZE
            for i in range(0, len(safe_pairs), batch_size):
                batch = safe_pairs[i:i + batch_size]

                tasks = [self.comprehensive_pair_analysis(pair) for pair in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, EnhancedSignal):
                        promising_signals.append(result)
                    elif isinstance(result, Exception):
                        logger.warning(f"Исключение при анализе: {result}")

                processed = min(i + batch_size, len(safe_pairs))
                logger.info(f"Обработано: {processed}/{len(safe_pairs)} (найдено {len(promising_signals)} сигналов)")

                # Пауза между батчами для снижения нагрузки
                if i + batch_size < len(safe_pairs):
                    await asyncio.sleep(config.processing.BATCH_DELAY)

            # Сортируем по уверенности (самые надежные первыми)
            promising_signals.sort(key=lambda x: (x.confidence, x.validation_score), reverse=True)

            execution_time = time.time() - start_time
            success_rate = (
                        len(promising_signals) / self.analyzed_pairs_count * 100) if self.analyzed_pairs_count > 0 else 0

            logger.info(f"ЭТАП 1 завершен: {len(promising_signals)} сигналов за {execution_time:.2f}сек")
            logger.info(
                f"Коэффициент успеха: {success_rate:.1f}% ({len(promising_signals)}/{self.analyzed_pairs_count})")

            return promising_signals

        except Exception as e:
            logger.error(f"Ошибка консервативного сканирования: {e}")
            return []


class MaxDataAISelector:
    """ИИ селектор с передачей максимальных данных"""

    def __init__(self):
        self.selection_prompt = self._load_prompt(config.ai.SELECTION_PROMPT_FILE)
        self.analysis_prompt = self._load_prompt(config.ai.ANALYSIS_PROMPT_FILE)

    def _load_prompt(self, filename: str) -> str:
        """Загрузка промпта"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                logger.info(f"Промпт загружен из {filename} ({len(content)} символов)")
                return content
        except FileNotFoundError:
            logger.error(f"Файл {filename} не найден")
            return ""

    def _prepare_maximum_data_for_ai(self, signals: List[EnhancedSignal]) -> Dict[str, Any]:
        """Подготовка МАКСИМАЛЬНЫХ данных для ИИ"""
        prepared_data = []

        for signal in signals:
            # Берем максимум свечей для анализа
            recent_5m_extended = signal.candles_5m_extended[
                                 -config.timeframe.CANDLES_FOR_AI_ANALYSIS:] if signal.candles_5m_extended else []
            recent_15m_extended = signal.candles_15m_extended[
                                  -config.timeframe.CANDLES_FOR_CONTEXT:] if signal.candles_15m_extended else []

            # МАКСИМАЛЬНЫЙ набор данных для ИИ
            signal_data = {
                'pair': signal.pair,
                'signal_type': signal.signal_type,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'pattern_type': signal.pattern_type,
                'higher_tf_trend': signal.higher_tf_trend,
                'validation_score': signal.validation_score,

                # ПОЛНЫЕ мультитаймфреймные данные
                'timeframes': {
                    '5m_extended_candles': [
                        {
                            'timestamp': int(c[0]),
                            'open': float(c[1]),
                            'high': float(c[2]),
                            'low': float(c[3]),
                            'close': float(c[4]),
                            'volume': float(c[5])
                        } for c in recent_5m_extended
                    ],
                    '15m_extended_context': [
                        {
                            'timestamp': int(c[0]),
                            'open': float(c[1]),
                            'high': float(c[2]),
                            'low': float(c[3]),
                            'close': float(c[4]),
                            'volume': float(c[5])
                        } for c in recent_15m_extended
                    ]
                },

                # ВСЕ ИНДИКАТОРЫ (максимальная история)
                'comprehensive_technical_analysis': safe_json_serialize({
                    # Полная система трендовых индикаторов
                    'trend_system': signal.comprehensive_indicators.get('trend_following', {}),

                    # Все осцилляторы с историей
                    'oscillators_full': signal.comprehensive_indicators.get('oscillators', {}),

                    # Полный MACD анализ
                    'macd_comprehensive': signal.comprehensive_indicators.get('macd_analysis', {}),

                    # Расширенный анализ волатильности
                    'volatility_extended': signal.comprehensive_indicators.get('volatility', {}),

                    # Максимальный объемный анализ
                    'volume_comprehensive': signal.comprehensive_indicators.get('volume_analysis', {}),

                    # Анализ моментума и тренда
                    'momentum_trend_full': signal.comprehensive_indicators.get('momentum_trend', {}),

                    # Уровни поддержки и сопротивления
                    'support_resistance_analysis': signal.comprehensive_indicators.get('support_resistance', {}),

                    # Свечной анализ
                    'candlestick_patterns': signal.comprehensive_indicators.get('candlestick_analysis', {}),

                    # Ценовое действие
                    'price_action_extended': signal.comprehensive_indicators.get('price_action', {}),

                    # Статистический анализ
                    'statistical_metrics': signal.comprehensive_indicators.get('statistics', {})
                }),

                # Расширенный анализ старшего таймфрейма
                'higher_timeframe_comprehensive': safe_json_serialize({
                    'trend_analysis': signal.higher_tf_analysis.get('trend', 'UNKNOWN'),
                    'strength': signal.higher_tf_analysis.get('strength', 0),
                    'confidence': signal.higher_tf_analysis.get('confidence', 0),
                    'trend_signals': signal.higher_tf_analysis.get('trend_signals', {}),
                    'ema_system_15m': signal.higher_tf_analysis.get('ema_values', {}),
                    'rsi_15m': signal.higher_tf_analysis.get('rsi_15m', 50),
                    'macd_15m': signal.higher_tf_analysis.get('macd_15m', {}),
                    'trend_quality': signal.higher_tf_analysis.get('trend_quality', {})
                }),

                # Полные метрики риска
                'risk_management': safe_json_serialize({
                    'stop_loss_percent': signal.risk_metrics.get('stop_loss_percent', 0.6),
                    'take_profit_percent': signal.risk_metrics.get('take_profit_percent', 1.2),
                    'risk_reward_ratio': signal.risk_metrics.get('risk_reward_ratio', 1.8),
                    'position_risk_score': signal.risk_metrics.get('position_risk_score', 5),
                    'recommended_position_size': signal.risk_metrics.get('recommended_position_size', 1.5),
                    'atr_based_stop': signal.risk_metrics.get('atr_based_stop', 0)
                }),

                # Дополнительные консервативные метрики
                'conservative_filters': {
                    'liquidity_score': 'HIGH' if signal.volume_ratio > 2.0 else 'MEDIUM' if signal.volume_ratio > 1.5 else 'LOW',
                    'volatility_regime': signal.comprehensive_indicators.get('volatility', {}).get('volatility_regime',
                                                                                                   'UNKNOWN'),
                    'trend_strength_15m': signal.higher_tf_analysis.get('trend_quality', {}).get('trend_strength',
                                                                                                 'WEAK'),
                    'validation_strict': signal.validation_score,
                    'signal_age_minutes': (int(time.time()) - signal.timestamp) // 60
                }
            }

            prepared_data.append(signal_data)

        return {
            'analysis_method': 'CONSERVATIVE_MAXIMUM_DATA_MULTI_TIMEFRAME',
            'strategy_focus': 'PROFIT_MAXIMIZATION_RISK_MINIMIZATION',
            'context_tf': config.timeframe.CONTEXT_TF + 'm',
            'entry_tf': config.timeframe.ENTRY_TF + 'm',
            'min_confidence_threshold': config.trading.MIN_CONFIDENCE,
            'validation_requirement': f"{config.trading.VALIDATION_CHECKS_REQUIRED}/{config.trading.VALIDATION_CHECKS_TOTAL}",
            'signals_count': len(prepared_data),
            'max_selections': config.ai.MAX_SELECTED_PAIRS,
            'timestamp': int(time.time()),
            'risk_management_profile': 'CONSERVATIVE',
            'data_depth': {
                '5m_candles_analyzed': config.timeframe.CANDLES_FOR_AI_ANALYSIS,
                '15m_candles_context': config.timeframe.CANDLES_FOR_CONTEXT,
                'indicators_total': 'ALL_COMPREHENSIVE'
            },
            'signals': prepared_data
        }

    async def select_best_pairs_with_max_data(self, signals: List[EnhancedSignal]) -> List[str]:
        """ИИ отбор с максимальными данными"""
        if not self.selection_prompt or not signals:
            logger.warning("Нет промпта или сигналов для ИИ отбора")
            return []

        logger.info(f"ЭТАП 2: ИИ отбор из {len(signals)} сигналов (максимальные данные)")

        try:
            # Берем топ сигналов для ИИ
            top_signals = signals[:config.ai.MAX_PAIRS_TO_AI]
            ai_data = self._prepare_maximum_data_for_ai(top_signals)

            # Расширенный промпт с максимальными данными
            message = f"""{self.selection_prompt}

=== КОНСЕРВАТИВНЫЙ АНАЛИЗ С МАКСИМАЛЬНЫМИ ДАННЫМИ ===
СТРАТЕГИЯ: {ai_data['strategy_focus']}
МЕТОД: {ai_data['analysis_method']}
ТАЙМФРЕЙМЫ: {ai_data['context_tf']} контекст + {ai_data['entry_tf']} точный вход
ГЛУБИНА ДАННЫХ: {ai_data['data_depth']['5m_candles_analyzed']} свечей 5m + {ai_data['data_depth']['15m_candles_context']} свечей 15m
ИНДИКАТОРЫ: ВСЕ ДОСТУПНЫЕ (EMA, SMA, RSI, MACD, Stochastic, Williams%R, CCI, ATR, Bollinger, Volume, Momentum, Support/Resistance)
МИНИМАЛЬНАЯ УВЕРЕННОСТЬ: {ai_data['min_confidence_threshold']}%
ВАЛИДАЦИЯ: {ai_data['validation_requirement']} обязательных проверок
ПРОФИЛЬ РИСКА: {ai_data['risk_management_profile']}
КОЛИЧЕСТВО СИГНАЛОВ: {len(top_signals)}

{json.dumps(ai_data, indent=2, ensure_ascii=False)}

ЗАДАЧА: Выбери максимум {config.ai.MAX_SELECTED_PAIRS} ЛУЧШИХ пар для консервативной торговли.
Приоритет: ПРИБЫЛЬНОСТЬ > БЕЗОПАСНОСТЬ > СКОРОСТЬ

ОБЯЗАТЕЛЬНЫЕ КРИТЕРИИ:
- Уверенность ≥{config.trading.MIN_CONFIDENCE}%
- Валидация ≥{config.trading.VALIDATION_CHECKS_REQUIRED}/5 проверок
- Ликвидность HIGH/MEDIUM
- Волатильность не HIGH_VOLATILITY
- Старший ТФ поддерживает направление
- Нет экстремальных RSI (<25 или >75)

Верни JSON: {{"pairs": ["SYMBOL1", "SYMBOL2", "SYMBOL3"], "reasoning": "краткое обоснование выбора"}}"""

            ai_response = await deep_seek_selection(message)

            if not ai_response:
                logger.error("Пустой ответ от ИИ селектора")
                return []

            selected_pairs = self._parse_ai_selection_response(ai_response)
            logger.info(f"ЭТАП 2: ИИ выбрал {len(selected_pairs)} пар из {len(top_signals)}")
            return selected_pairs

        except Exception as e:
            logger.error(f"Ошибка ИИ отбора: {e}")
            return []

    def _parse_ai_selection_response(self, response: str) -> List[str]:
        """Парсинг ответа ИИ с улучшенной обработкой"""
        try:
            # Ищем JSON в ответе
            json_match = re.search(r'\{[^}]*"pairs"[^}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                pairs = data.get('pairs', [])
                reasoning = data.get('reasoning', 'Не указано')

                if pairs:
                    logger.info(f"ИИ обоснование: {reasoning}")

                return pairs

            # Альтернативный поиск списка пар
            pairs_match = re.findall(r'[A-Z]{3,10}USDT', response)
            if pairs_match:
                unique_pairs = list(set(pairs_match))[:config.ai.MAX_SELECTED_PAIRS]
                logger.info(f"Извлечено {len(unique_pairs)} пар из текста ответа")
                return unique_pairs

            return []

        except Exception as e:
            logger.error(f"Ошибка парсинга ответа ИИ: {e}")
            return []

    async def comprehensive_analysis_with_max_data(self, pair: str) -> Optional[str]:
        """Максимально детальный анализ пары с полными данными"""
        if not self.analysis_prompt:
            logger.error("Отсутствует промпт для анализа")
            return None

        logger.info(f"ЭТАП 3: Максимально детальный анализ {pair}")

        try:
            # Получаем МАКСИМАЛЬНЫЕ данные для финального анализа
            full_candles_5m = await get_klines_async(
                pair,
                config.timeframe.ENTRY_TF,
                limit=config.timeframe.CANDLES_FOR_FULL_ANALYSIS  # 200 свечей
            )

            full_candles_15m = await get_klines_async(
                pair,
                config.timeframe.CONTEXT_TF,
                limit=100  # 100 свечей 15m для полного контекста
            )

            if not full_candles_5m or not full_candles_15m:
                logger.error(f"{pair}: не удалось получить полные данные")
                return None

            # Рассчитываем ВСЕ индикаторы с максимальной историей
            comprehensive_indicators = calculate_all_indicators_comprehensive(full_candles_5m)
            comprehensive_htf_analysis = analyze_higher_timeframe_trend_comprehensive(full_candles_15m)
            enhanced_signal_analysis = detect_instruction_based_signals_enhanced(full_candles_5m, full_candles_15m)

            # Подготавливаем МАКСИМАЛЬНЫЙ набор данных для финального анализа
            comprehensive_analysis_data = {
                'pair': pair,
                'timestamp': int(time.time()),
                'current_price': float(full_candles_5m[-1][4]),
                'analysis_method': 'MAXIMUM_DATA_COMPREHENSIVE_MULTI_TIMEFRAME',
                'strategy_type': 'CONSERVATIVE_PROFIT_MAXIMIZATION',

                # ПОЛНЫЕ рыночные данные (максимальная история)
                'market_context_comprehensive': {
                    '15m_trend_analysis': comprehensive_htf_analysis,
                    '5m_last_50_candles': [
                        {
                            'timestamp': int(c[0]),
                            'open': float(c[1]),
                            'high': float(c[2]),
                            'low': float(c[3]),
                            'close': float(c[4]),
                            'volume': float(c[5]),
                            'volume_usd': float(c[5]) * float(c[4])
                        } for c in full_candles_5m[-50:]
                    ],
                    '15m_last_20_candles': [
                        {
                            'timestamp': int(c[0]),
                            'open': float(c[1]),
                            'high': float(c[2]),
                            'low': float(c[3]),
                            'close': float(c[4]),
                            'volume': float(c[5]),
                            'volume_usd': float(c[5]) * float(c[4])
                        } for c in full_candles_15m[-20:]
                    ]
                },

                # ПОЛНЫЙ сигнальный анализ
                'signal_analysis_comprehensive': safe_json_serialize({
                    'primary_signal': enhanced_signal_analysis.get('signal', 'NO_SIGNAL'),
                    'pattern_type': enhanced_signal_analysis.get('pattern_type', 'NONE'),
                    'confidence': enhanced_signal_analysis.get('confidence', 0),
                    'validation_score': enhanced_signal_analysis.get('validation_score', '0/5'),
                    'entry_reasons': enhanced_signal_analysis.get('entry_reasons', []),
                    'validation_reasons': enhanced_signal_analysis.get('validation_reasons', []),
                    'risk_metrics': enhanced_signal_analysis.get('risk_metrics', {})
                }),

                # ВСЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (максимальная глубина)
                'technical_indicators_complete': safe_json_serialize(comprehensive_indicators),

                # ДОПОЛНИТЕЛЬНЫЕ аналитические слои
                'advanced_analytics': {
                    # Корреляционный анализ
                    'correlations': {
                        'price_volume_correlation': comprehensive_indicators.get('statistics', {}).get(
                            'correlation_volume_price', 0),
                        'rsi_price_divergence': comprehensive_indicators.get('oscillators', {}).get('rsi_divergence',
                                                                                                    False),
                        'macd_price_divergence': comprehensive_indicators.get('macd_analysis', {}).get(
                            'macd_divergence', False)
                    },

                    # Статистические метрики
                    'statistical_analysis': comprehensive_indicators.get('statistics', {}),

                    # Режимы рынка
                    'market_regimes': {
                        'volatility_regime': comprehensive_indicators.get('volatility', {}).get('volatility_regime',
                                                                                                'UNKNOWN'),
                        'trend_regime': comprehensive_htf_analysis.get('trend', 'UNKNOWN'),
                        'volume_regime': comprehensive_indicators.get('volume_analysis', {}).get('volume_trend',
                                                                                                 'UNKNOWN')
                    },

                    # Качество сигнала
                    'signal_quality_metrics': {
                        'multi_timeframe_sync': comprehensive_htf_analysis.get('confidence', 0) > 70,
                        'volume_confirmation_strong': comprehensive_indicators.get('volume_analysis', {}).get(
                            'volume_ratio', 1) > 1.5,
                        'trend_strength_adequate': comprehensive_htf_analysis.get('trend_quality', {}).get(
                            'trend_strength', 'WEAK') != 'WEAK',
                        'volatility_suitable': comprehensive_indicators.get('volatility', {}).get('volatility_regime',
                                                                                                  'HIGH') != 'HIGH_VOLATILITY',
                        'oscillators_not_extreme': (
                                30 < comprehensive_indicators.get('oscillators', {}).get('rsi14_current', 50) < 70
                        )
                    }
                },

                # КОНСЕРВАТИВНАЯ оценка рисков
                'risk_assessment_comprehensive': {
                    'overall_risk_score': enhanced_signal_analysis.get('risk_metrics', {}).get('position_risk_score',
                                                                                               5),
                    'stop_loss_analysis': {
                        'atr_based_stop': enhanced_signal_analysis.get('risk_metrics', {}).get('atr_based_stop', 0),
                        'percentage_stop': enhanced_signal_analysis.get('risk_metrics', {}).get('stop_loss_percent',
                                                                                                0.6),
                        'support_resistance_stop': comprehensive_indicators.get('support_resistance', {}).get(
                            'nearest_support', 0)
                    },
                    'profit_targets': {
                        'conservative_target': enhanced_signal_analysis.get('risk_metrics', {}).get(
                            'take_profit_percent', 1.2),
                        'resistance_target': comprehensive_indicators.get('support_resistance', {}).get(
                            'nearest_resistance', 0),
                        'risk_reward_ratio': enhanced_signal_analysis.get('risk_metrics', {}).get('risk_reward_ratio',
                                                                                                  1.8)
                    },
                    'position_sizing': {
                        'recommended_size': enhanced_signal_analysis.get('risk_metrics', {}).get(
                            'recommended_position_size', 1.5),
                        'max_size_allowed': config.trading.MAX_POSITION_SIZE_PERCENT,
                        'volatility_adjustment': comprehensive_indicators.get('volatility', {}).get('atr_normalized', 0)
                    }
                }
            }

            # Создаем максимально детальный промпт
            detailed_message = f"""{self.analysis_prompt}

=== МАКСИМАЛЬНО ДЕТАЛЬНЫЙ АНАЛИЗ ПО КОНСЕРВАТИВНОЙ СТРАТЕГИИ ===
ПАРА: {pair}
ЦЕНА: {comprehensive_analysis_data['current_price']}
МЕТОД: {comprehensive_analysis_data['analysis_method']}
СТРАТЕГИЯ: {comprehensive_analysis_data['strategy_type']}

ГЛУБИНА АНАЛИЗА:
- 5m свечей проанализировано: {len(full_candles_5m)}
- 15m свечей для контекста: {len(full_candles_15m)}  
- Индикаторов рассчитано: ВСЕ ДОСТУПНЫЕ
- История данных: МАКСИМАЛЬНАЯ

КОНСЕРВАТИВНЫЕ ТРЕБОВАНИЯ:
- Минимальная уверенность: {config.trading.MIN_CONFIDENCE}%
- Обязательная валидация: {config.trading.VALIDATION_CHECKS_REQUIRED}/5
- Максимальный риск позиции: {config.trading.MAX_POSITION_SIZE_PERCENT}%
- Минимальное R/R: {config.trading.MIN_RISK_REWARD}:1
- Стоп-лосс не менее: {config.trading.MIN_STOP_LOSS_PERCENT}%

{json.dumps(comprehensive_analysis_data, indent=2, ensure_ascii=False)}

ЗАДАЧА: Проведи максимально детальный анализ и дай окончательное решение:

1. ПОДТВЕРЖДЕНИЕ/ОТКЛОНЕНИЕ сигнала на основе всех данных
2. ТОЧНЫЕ уровни входа, стоп-лосса и тейк-профита
3. КОНСЕРВАТИВНЫЙ размер позиции 
4. ДЕТАЛЬНОЕ управление рисками
5. ВРЕМЕННЫЕ рамки удержания
6. ПЛАН действий при различных сценариях

Сфокусируйся на МАКСИМИЗАЦИИ ПРИБЫЛИ при МИНИМИЗАЦИИ РИСКОВ."""

            analysis_result = await deep_seek_analysis(detailed_message)

            if analysis_result:
                self._save_comprehensive_analysis(pair, analysis_result, comprehensive_analysis_data)
                logger.info(f"Максимально детальный анализ {pair} завершен")
                return analysis_result
            else:
                logger.error(f"Пустой результат анализа для {pair}")

            return None

        except Exception as e:
            logger.error(f"Ошибка максимального анализа {pair}: {e}")
            import traceback
            logger.error(f"Трейсбек: {traceback.format_exc()}")
            return None

    def _save_comprehensive_analysis(self, pair: str, analysis: str, raw_data: Dict):
        """Сохранение максимально детального анализа"""
        try:
            with open(config.system.ANALYSIS_LOG_FILE, 'a', encoding=config.system.ENCODING) as f:
                f.write(f"\n{'=' * 100}\n")
                f.write(f"МАКСИМАЛЬНО ДЕТАЛЬНЫЙ АНАЛИЗ\n")
                f.write(f"ПАРА: {pair}\n")
                f.write(f"ВРЕМЯ: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"МЕТОД: {raw_data.get('analysis_method', 'UNKNOWN')}\n")
                f.write(f"СТРАТЕГИЯ: {raw_data.get('strategy_type', 'UNKNOWN')}\n")
                f.write(f"ЦЕНА НА МОМЕНТ АНАЛИЗА: {raw_data.get('current_price', 0)}\n")
                f.write(f"ГЛУБИНА ДАННЫХ: МАКСИМАЛЬНАЯ\n")
                f.write(f"\nРЕЗУЛЬТАТ ИИ АНАЛИЗА:\n{analysis}\n")
                f.write(f"\nТЕХНИЧЕСКИЕ ДАННЫЕ (JSON):\n")
                f.write(
                    f"{json.dumps(raw_data.get('signal_analysis_comprehensive', {}), indent=2, ensure_ascii=False)}\n")
                f.write(f"{'=' * 100}\n")

        except Exception as e:
            logger.error(f"Ошибка сохранения анализа: {e}")


async def main():
    """Главная функция консервативного бота с максимальными данными"""
    logger.info("=" * 100)
    logger.info("КОНСЕРВАТИВНЫЙ СКАЛЬПИНГОВЫЙ БОТ - МАКСИМАЛЬНЫЕ ДАННЫЕ + СТРОГИЕ ФИЛЬТРЫ")
    logger.info("=" * 100)
    logger.info(f"Стратегия: МАКСИМИЗАЦИЯ ПРИБЫЛИ при МИНИМИЗАЦИИ РИСКОВ")
    logger.info(f"Таймфреймы: {config.timeframe.CONTEXT_TF}m контекст + {config.timeframe.ENTRY_TF}m точный вход")
    logger.info(f"Данные: {config.timeframe.CANDLES_5M} свечей 5m + {config.timeframe.CANDLES_15M} свечей 15m")
    logger.info(f"Минимальная уверенность: {config.trading.MIN_CONFIDENCE}%")
    logger.info(
        f"Валидация: {config.trading.VALIDATION_CHECKS_REQUIRED}/{config.trading.VALIDATION_CHECKS_TOTAL} обязательных проверок")
    logger.info(f"Максимальный размер позиции: {config.trading.MAX_POSITION_SIZE_PERCENT}%")
    logger.info("=" * 100)

    # Проверяем конфигурацию
    # if not config.validate():
    #     logger.error("Конфигурация не прошла валидацию!")
    #     return

    analyzer = ConservativeAnalyzer()
    ai_selector = MaxDataAISelector()

    try:
        # ЭТАП 1: Консервативное сканирование с максимальными данными
        logger.info("ЭТАП 1: Запуск консервативного сканирования...")

        promising_signals = await analyzer.mass_scan_markets_conservative()

        if not promising_signals:
            logger.info("Консервативные сигналы не найдены в текущих рыночных условиях")
            logger.info("Рекомендация: Дождитесь более благоприятных условий")
            return

        logger.info(f"\nНАЙДЕНО {len(promising_signals)} КОНСЕРВАТИВНЫХ СИГНАЛОВ:")
        for i, signal in enumerate(promising_signals[:10], 1):  # Показываем топ-10
            logger.info(f"  {i:2d}. {signal.pair}: {signal.pattern_type} {signal.signal_type} "
                        f"({signal.confidence}%, {signal.validation_score}, Vol:{signal.volume_ratio:.1f}x)")

        # ЭТАП 2: ИИ отбор с максимальными данными
        logger.info(f"\nЭТАП 2: ИИ отбор из топ-{min(len(promising_signals), config.ai.MAX_PAIRS_TO_AI)} сигналов...")

        selected_pairs = await ai_selector.select_best_pairs_with_max_data(promising_signals)

        if not selected_pairs:
            logger.info("ИИ не выбрал пары согласно консервативным критериям")
            logger.info("Возможные причины:")
            logger.info("- Недостаточная уверенность сигналов")
            logger.info("- Высокие риски в текущих рыночных условиях")
            logger.info("- Несоответствие критериям валидации")
            return

        logger.info(f"\nИИ ВЫБРАЛ {len(selected_pairs)} ПАР ДЛЯ ТОРГОВЛИ:")
        for i, pair in enumerate(selected_pairs, 1):
            logger.info(f"  {i}. {pair}")

        # ЭТАП 3: Максимально детальный анализ каждой выбранной пары
        logger.info(f"\nЭТАП 3: Максимально детальный анализ выбранных пар...")

        successful_analyses = 0
        total_time_start = time.time()

        for i, pair in enumerate(selected_pairs, 1):
            logger.info(f"\nАнализ {i}/{len(selected_pairs)}: {pair}")

            analysis_start = time.time()
            analysis = await ai_selector.comprehensive_analysis_with_max_data(pair)
            analysis_time = time.time() - analysis_start

            if analysis:
                successful_analyses += 1
                logger.info(f"✅ {pair} - детальный анализ завершен за {analysis_time:.1f}сек")
            else:
                logger.error(f"❌ {pair} - ошибка анализа")

            # Пауза между анализами для снижения нагрузки на API
            if i < len(selected_pairs):
                await asyncio.sleep(2)

        total_time = time.time() - total_time_start

        # ИТОГОВАЯ СТАТИСТИКА
        logger.info("\n" + "=" * 100)
        logger.info("КОНСЕРВАТИВНЫЙ АНАЛИЗ ЗАВЕРШЕН!")
        logger.info("=" * 100)
        logger.info(f"Стратегия: Максимизация прибыли + Минимизация рисков")
        logger.info(
            f"Глубина данных: МАКСИМАЛЬНАЯ ({config.timeframe.CANDLES_5M} + {config.timeframe.CANDLES_15M} свечей)")
        logger.info(f"Индикаторов: ВСЕ ДОСТУПНЫЕ")
        logger.info(f"Таймфреймы: {config.timeframe.CONTEXT_TF}m + {config.timeframe.ENTRY_TF}m")
        logger.info(f"")
        logger.info(f"РЕЗУЛЬТАТЫ:")
        logger.info(f"- Пар проанализировано: {analyzer.analyzed_pairs_count}")
        logger.info(f"- Консервативных сигналов: {len(promising_signals)}")
        logger.info(f"- ИИ отобрано: {len(selected_pairs)}")
        logger.info(f"- Успешных детальных анализов: {successful_analyses}")
        logger.info(f"- Время выполнения: {total_time:.1f} секунд")
        logger.info(f"- Коэффициент успеха: {(successful_analyses / len(selected_pairs) * 100):.1f}%")
        logger.info(f"")
        logger.info(f"Детальные результаты: {config.system.ANALYSIS_LOG_FILE}")
        logger.info("=" * 100)

        await cleanup_http_client()

    except KeyboardInterrupt:
        logger.info("Остановка по запросу пользователя")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        import traceback
        logger.error(f"Трейсбек: {traceback.format_exc()}")
    finally:
        # Финальная очистка
        try:
            await cleanup_http_client()
        except:
            pass


if __name__ == "__main__":
    logger.info("Запуск консервативного скальпингового бота...")
    logger.info("Конфигурация: Максимальные данные + Строгие фильтры")
    logger.info("Цель: Максимизация прибыли при минимизации рисков")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Программа остановлена пользователем")
    except Exception as e:
        logger.error(f"Фатальная ошибка: {e}")
    finally:
        logger.info("Работа завершена")