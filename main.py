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

# Импорт с индикаторами по инструкции
from func_trade import detect_instruction_based_signals, calculate_indicators_by_instruction

# Импорт оптимизированной конфигурации
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


def validate_candles_order(candles: List, symbol: str = "UNKNOWN") -> bool:
    """Проверка правильного порядка свечей (от старых к новым)"""
    if not candles or len(candles) < 2:
        return False

    # Проверяем что временные метки идут по возрастанию
    for i in range(1, len(candles)):
        if int(candles[i][0]) <= int(candles[i - 1][0]):
            logger.error(f"ОШИБКА ПОРЯДКА СВЕЧЕЙ {symbol}: {candles[i - 1][0]} -> {candles[i][0]}")
            return False

    logger.debug(f"Порядок свечей {symbol} корректен: {candles[0][0]} -> {candles[-1][0]}")
    return True


@dataclass
class InstructionBasedSignal:
    """Сигнал согласно инструкции с мультитаймфреймным анализом"""
    pair: str
    signal_type: str  # 'LONG', 'SHORT', 'NO_SIGNAL'
    confidence: int
    entry_price: float
    timestamp: int

    # Данные по инструкции
    pattern_type: str  # 'MOMENTUM_BREAKOUT', 'PULLBACK_ENTRY', etc.
    higher_tf_trend: str  # Тренд 15m
    validation_score: str  # "5/5" чек-пунктов
    atr_current: float  # Текущий ATR для волатильности
    volume_ratio: float  # Соотношение объемов

    # Для ИИ (ИСПРАВЛЕНО - НЕ отрезаем дополнительно)
    candles_5m: List = None  # 5m свечи для входа
    candles_15m: List = None  # 15m свечи для контекста
    indicators_data: Dict = None


class InstructionBasedAnalyzer:
    """Анализатор согласно инструкции: 15m контекст + 5m вход"""

    def __init__(self):
        self.session_start = time.time()
        logger.info("Оптимизированный анализатор запущен (15m+5m, адаптивные уровни)")

    def passes_liquidity_filter(self, symbol: str, candles: List) -> bool:
        """Фильтр ликвидности согласно оптимизированной конфигурации"""
        if not candles:
            return False

        # Примерная оценка объема (последние 24 свечи 5m = 2 часа)
        recent_volumes = [float(c[5]) * float(c[4]) for c in candles[-24:]]  # Объем в USD
        avg_hourly_volume = sum(recent_volumes) * 12  # Приблизительно за 24ч

        # МЕНЕЕ СТРОГИЕ требования
        return avg_hourly_volume > config.trading.MIN_LIQUIDITY_VOLUME

    def check_spread_quality(self, candles: List) -> bool:
        """Проверка стабильности спреда (упрощенно через ATR)"""
        if len(candles) < 5:
            return False

        # Используем ATR как прокси для спреда
        highs = [float(c[2]) for c in candles[-5:]]
        lows = [float(c[3]) for c in candles[-5:]]
        closes = [float(c[4]) for c in candles[-5:]]

        avg_price = sum(closes) / len(closes)
        avg_range = sum(h - l for h, l in zip(highs, lows)) / len(highs)

        spread_estimate = (avg_range / avg_price) * 100
        # МЕНЕЕ СТРОГИЕ требования к спреду
        return spread_estimate < config.trading.MAX_SPREAD_PERCENT

    async def quick_scan_pair(self, symbol: str) -> Optional[InstructionBasedSignal]:
        """Быстрое сканирование пары согласно инструкции"""
        try:
            # Получаем данные для мультитаймфреймного анализа
            # ВАЖНО: get_klines_async уже отрезает последнюю незакрытую свечу
            candles_5m = await get_klines_async(symbol, config.timeframe.ENTRY_TF,
                                                limit=config.timeframe.CANDLES_5M)
            candles_15m = await get_klines_async(symbol, config.timeframe.CONTEXT_TF,
                                                 limit=config.timeframe.CANDLES_15M)

            if not candles_5m or not candles_15m:
                return None

            # ПРОВЕРЯЕМ правильность порядка свечей
            if not validate_candles_order(candles_5m, f"{symbol}_5m"):
                logger.error(f"Неправильный порядок 5m свечей для {symbol}")
                return None

            if not validate_candles_order(candles_15m, f"{symbol}_15m"):
                logger.error(f"Неправильный порядок 15m свечей для {symbol}")
                return None

            # Фильтры согласно оптимизированной конфигурации
            if not self.passes_liquidity_filter(symbol, candles_5m):
                return None

            if not self.check_spread_quality(candles_5m):
                return None

            # Определяем сигнал согласно инструкции (мультитаймфрейм)
            signal_result = detect_instruction_based_signals(candles_5m, candles_15m)

            if signal_result['signal'] == 'NO_SIGNAL':
                return None

            # Создаем сигнал
            entry_price = float(candles_5m[-1][4])  # Последняя закрытая свеча
            confidence = int(signal_result['confidence'])

            if math.isnan(entry_price) or confidence < config.trading.MIN_CONFIDENCE:
                return None

            return InstructionBasedSignal(
                pair=symbol,
                signal_type=signal_result['signal'],
                confidence=confidence,
                entry_price=entry_price,
                timestamp=int(time.time()),

                # Данные согласно инструкции
                pattern_type=signal_result.get('pattern_type', 'UNKNOWN'),
                higher_tf_trend=signal_result.get('higher_tf_trend', 'UNKNOWN'),
                validation_score=signal_result.get('validation_score', '0/5'),
                atr_current=signal_result.get('atr_current', 0.0),
                volume_ratio=signal_result.get('volume_ratio', 1.0),

                # Данные для ИИ (ИСПРАВЛЕНО - берем нужное количество БЕЗ дополнительного отрезания)
                candles_5m=candles_5m[-config.timeframe.CANDLES_FOR_AI_SELECTION:] if len(
                    candles_5m) > config.timeframe.CANDLES_FOR_AI_SELECTION else candles_5m,
                candles_15m=candles_15m[-config.timeframe.CANDLES_FOR_CONTEXT:] if len(
                    candles_15m) > config.timeframe.CANDLES_FOR_CONTEXT else candles_15m,
                indicators_data=clean_value(signal_result.get('indicators', {}))
            )

        except Exception as e:
            logger.error(f"Ошибка сканирования {symbol}: {e}")
            return None

    async def mass_scan_markets(self) -> List[InstructionBasedSignal]:
        """Массовое сканирование с оптимизированными фильтрами"""
        start_time = time.time()
        logger.info("ЭТАП 1: Оптимизированное сканирование (адаптивные уровни)")

        try:
            pairs = await get_usdt_trading_pairs()
            if not pairs:
                return []

            logger.info(f"Сканируем {len(pairs)} пар (15m контекст + 5m вход, оптимизировано)")

            promising_signals = []

            # Обрабатываем батчами (оптимизированный размер)
            for i in range(0, len(pairs), config.processing.BATCH_SIZE):
                batch = pairs[i:i + config.processing.BATCH_SIZE]
                tasks = [self.quick_scan_pair(pair) for pair in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, InstructionBasedSignal):
                        promising_signals.append(result)

                processed = min(i + config.processing.BATCH_SIZE, len(pairs))
                logger.info(f"Обработано: {processed}/{len(pairs)} (найдено {len(promising_signals)} сигналов)")

                if i + config.processing.BATCH_SIZE < len(pairs):
                    await asyncio.sleep(config.processing.BATCH_DELAY)

            # Сортируем по уверенности
            promising_signals.sort(key=lambda x: x.confidence, reverse=True)

            execution_time = time.time() - start_time
            logger.info(f"ЭТАП 1: {len(promising_signals)} сигналов за {execution_time:.2f}сек (оптимизировано)")

            return promising_signals

        except Exception as e:
            logger.error(f"Ошибка сканирования: {e}")
            return []


class OptimizedAISelector:
    """Оптимизированный ИИ селектор для максимального профита"""

    def __init__(self):
        self.selection_prompt = self._load_prompt(config.ai.SELECTION_PROMPT_FILE)
        self.analysis_prompt = self._load_prompt('prompt_optimized.txt')  # Новый оптимизированный промпт

    def _load_prompt(self, filename: str) -> str:
        """Загрузка промпта"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                logger.info(f"Загружен промпт {filename} ({len(content)} символов)")
                return content
        except FileNotFoundError:
            logger.error(f"Файл {filename} не найден, используется базовый промпт")
            return self._get_default_prompt(filename)

    def _get_default_prompt(self, filename: str) -> str:
        """Базовые промпты если файлы не найдены"""
        if 'selection' in filename or 'prompt2' in filename:
            return "Ты эксперт по скальпингу. Выбери лучшие пары для торговли. Верни JSON: {'pairs': ['BTCUSDT']}"
        else:
            return "Ты эксперт-трейдер. Проанализируй данные и дай торговые рекомендации в JSON формате."

    def _prepare_signals_for_ai(self, signals: List[InstructionBasedSignal]) -> Dict[str, Any]:
        """Подготовка оптимизированных данных для ИИ"""
        prepared_data = []

        for signal in signals:
            # Больше свечей для лучшего анализа уровней
            recent_5m = signal.candles_5m[-50:] if signal.candles_5m and len(
                signal.candles_5m) >= 50 else signal.candles_5m
            recent_15m = signal.candles_15m[-30:] if signal.candles_15m and len(
                signal.candles_15m) >= 30 else signal.candles_15m

            # ПРОВЕРЯЕМ что свечи в правильном порядке
            if recent_5m:
                validate_candles_order(recent_5m, f"{signal.pair}_5m_for_AI")
            if recent_15m:
                validate_candles_order(recent_15m, f"{signal.pair}_15m_for_AI")

            signal_data = {
                'pair': signal.pair,
                'signal_type': signal.signal_type,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,

                # Данные согласно инструкции
                'pattern_type': signal.pattern_type,
                'higher_tf_trend': signal.higher_tf_trend,
                'validation_score': signal.validation_score,
                'atr_current': signal.atr_current,
                'volume_ratio': signal.volume_ratio,

                # Мультитаймфреймные данные (больше контекста)
                'timeframes': {
                    '5m_candles': [
                        {
                            'timestamp': int(c[0]),
                            'open': float(c[1]),
                            'high': float(c[2]),
                            'low': float(c[3]),
                            'close': float(c[4]),
                            'volume': float(c[5])
                        } for c in recent_5m
                    ] if recent_5m else [],
                    '15m_context': [
                        {
                            'timestamp': int(c[0]),
                            'open': float(c[1]),
                            'high': float(c[2]),
                            'low': float(c[3]),
                            'close': float(c[4]),
                            'volume': float(c[5])
                        } for c in recent_15m
                    ] if recent_15m else []
                },

                # Расширенные индикаторы для анализа уровней
                'technical_indicators': safe_json_serialize({
                    'trend_system': {
                        'ema5': signal.indicators_data.get('ema5', [])[-30:],
                        'ema8': signal.indicators_data.get('ema8', [])[-30:],
                        'ema21': signal.indicators_data.get('ema20', [])[-30:]  # Переименовали для ясности
                    },
                    'momentum_oscillators': {
                        'rsi14': signal.indicators_data.get('rsi', [])[-30:],
                        'rsi_current': signal.indicators_data.get('rsi_current', 50),
                        'macd_line': signal.indicators_data.get('macd_line', [])[-30:],
                        'macd_signal': signal.indicators_data.get('macd_signal', [])[-30:],
                        'macd_histogram': signal.indicators_data.get('macd_histogram', [])[-30:]
                    },
                    'volatility_analysis': {
                        'atr14': signal.indicators_data.get('atr', [])[-30:],
                        'atr_current': signal.indicators_data.get('atr_current', 0),
                        'atr_mean': signal.indicators_data.get('atr_mean', 0),
                        'atr_percentile': self._calculate_atr_percentile(signal.indicators_data)
                    },
                    'price_action': {
                        'bollinger_upper': signal.indicators_data.get('bb_upper', [])[-30:],
                        'bollinger_middle': signal.indicators_data.get('bb_middle', [])[-30:],
                        'bollinger_lower': signal.indicators_data.get('bb_lower', [])[-30:],
                        'bb_squeeze_detected': self._detect_bb_squeeze_simple(signal.indicators_data)
                    },
                    'volume_profile': {
                        'volume_sma20': signal.indicators_data.get('volume_sma', [])[-30:],
                        'current_volume': signal.indicators_data.get('volume_current', 0),
                        'volume_ratio': signal.indicators_data.get('volume_ratio', 1.0),
                        'volume_trend': self._calculate_volume_trend(signal.indicators_data)
                    }
                }),

                # Данные для анализа уровней
                'level_analysis_data': {
                    'recent_highs': self._extract_swing_points(recent_5m, 'high') if recent_5m else [],
                    'recent_lows': self._extract_swing_points(recent_5m, 'low') if recent_5m else [],
                    'price_clusters': self._find_price_clusters(recent_5m) if recent_5m else [],
                    'current_price_position': self._analyze_price_position(recent_5m) if recent_5m else {}
                }
            }

            prepared_data.append(signal_data)

        return {
            'analysis_method': 'optimized_multi_timeframe_with_adaptive_levels',
            'context_tf': config.timeframe.CONTEXT_TF + 'm',
            'entry_tf': config.timeframe.ENTRY_TF + 'm',
            'signals_count': len(prepared_data),
            'timestamp': int(time.time()),
            'optimization_focus': 'maximum_profit_adaptive_risk',
            'signals': prepared_data
        }

    def _calculate_atr_percentile(self, indicators: Dict) -> float:
        """Расчет процентиля текущего ATR"""
        atr_values = indicators.get('atr', [])
        current_atr = indicators.get('atr_current', 0)

        if not atr_values or current_atr == 0:
            return 50.0

        recent_atr = [x for x in atr_values[-50:] if x > 0]
        if len(recent_atr) < 10:
            return 50.0

        sorted_atr = sorted(recent_atr)
        position = sum(1 for x in sorted_atr if x <= current_atr)
        return (position / len(sorted_atr)) * 100

    def _detect_bb_squeeze_simple(self, indicators: Dict) -> bool:
        """Простое определение сжатия Bollinger Bands"""
        upper = indicators.get('bb_upper', [])
        lower = indicators.get('bb_lower', [])

        if len(upper) < 10 or len(lower) < 10:
            return False

        current_width = upper[-1] - lower[-1]
        avg_width = sum(upper[i] - lower[i] for i in range(-10, 0)) / 10

        return current_width < avg_width * 0.85

    def _calculate_volume_trend(self, indicators: Dict) -> str:
        """Определение тренда объема"""
        volume_sma = indicators.get('volume_sma', [])
        current_vol = indicators.get('volume_current', 0)

        if not volume_sma or current_vol == 0:
            return 'neutral'

        recent_avg = sum(volume_sma[-5:]) / min(5, len(volume_sma))

        if current_vol > recent_avg * 1.5:
            return 'spike'
        elif current_vol > recent_avg * 1.1:
            return 'above_average'
        elif current_vol < recent_avg * 0.8:
            return 'below_average'
        else:
            return 'normal'

    def _extract_swing_points(self, candles: List, point_type: str) -> List[float]:
        """Извлечение swing highs/lows для анализа уровней"""
        if not candles or len(candles) < 10:
            return []

        points = []
        price_idx = 2 if point_type == 'high' else 3  # high или low

        for i in range(5, len(candles) - 5):
            current_price = float(candles[i][price_idx])

            # Проверяем является ли точка локальным экстремумом
            if point_type == 'high':
                is_swing = all(current_price >= float(candles[j][price_idx]) for j in range(i - 5, i + 6))
            else:
                is_swing = all(current_price <= float(candles[j][price_idx]) for j in range(i - 5, i + 6))

            if is_swing:
                points.append(current_price)

        return sorted(set(points), reverse=(point_type == 'high'))[:10]  # Топ-10 уровней

    def _find_price_clusters(self, candles: List) -> List[Dict]:
        """Поиск кластеров цен (зоны поддержки/сопротивления)"""
        if not candles or len(candles) < 20:
            return []

        # Собираем все цены
        all_prices = []
        for candle in candles:
            all_prices.extend([float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4])])

        all_prices.sort()
        current_price = float(candles[-1][4])

        # Группируем цены в кластеры (в пределах 0.3% друг от друга)
        clusters = []
        tolerance = current_price * 0.003

        i = 0
        while i < len(all_prices):
            cluster_prices = [all_prices[i]]
            j = i + 1

            while j < len(all_prices) and all_prices[j] - all_prices[i] <= tolerance:
                cluster_prices.append(all_prices[j])
                j += 1

            if len(cluster_prices) >= 3:  # Минимум 3 касания
                cluster_center = sum(cluster_prices) / len(cluster_prices)
                clusters.append({
                    'level': cluster_center,
                    'strength': len(cluster_prices),
                    'distance_percent': abs(cluster_center - current_price) / current_price * 100
                })

            i = j if j > i + 1 else i + 1

        return sorted(clusters, key=lambda x: x['strength'], reverse=True)[:5]

    def _analyze_price_position(self, candles: List) -> Dict:
        """Анализ позиции текущей цены"""
        if not candles or len(candles) < 20:
            return {}

        current_price = float(candles[-1][4])

        # Анализируем за последние 50 свечей
        recent_candles = candles[-50:] if len(candles) >= 50 else candles

        highs = [float(c[2]) for c in recent_candles]
        lows = [float(c[3]) for c in recent_candles]

        max_high = max(highs)
        min_low = min(lows)
        range_size = max_high - min_low

        return {
            'current_price': current_price,
            'range_high': max_high,
            'range_low': min_low,
            'position_in_range_percent': ((current_price - min_low) / range_size * 100) if range_size > 0 else 50,
            'distance_to_high_percent': ((max_high - current_price) / current_price * 100),
            'distance_to_low_percent': ((current_price - min_low) / current_price * 100),
            'range_size_percent': (range_size / current_price * 100)
        }

    async def select_best_pairs(self, signals: List[InstructionBasedSignal]) -> List[str]:
        """Оптимизированный ИИ отбор для максимального профита"""
        if not self.selection_prompt or not signals:
            return []

        logger.info(f"ЭТАП 2: Оптимизированный ИИ отбор из {len(signals)} сигналов")

        try:
            top_signals = signals[:config.ai.MAX_PAIRS_TO_AI]
            ai_data = self._prepare_signals_for_ai(top_signals)

            message = f"""{self.selection_prompt}

=== ОПТИМИЗИРОВАННЫЙ МУЛЬТИТАЙМФРЕЙМНЫЙ АНАЛИЗ ===
ЦЕЛЬ: МАКСИМАЛЬНЫЙ ПРОФИТ С АДАПТИВНЫМИ УРОВНЯМИ
МЕТОД: {config.timeframe.CONTEXT_TF}m контекст + {config.timeframe.ENTRY_TF}m точный вход
ИНДИКАТОРЫ: EMA({config.indicators.EMA_FAST}/{config.indicators.EMA_MEDIUM}/{config.indicators.EMA_SLOW}), RSI({config.indicators.RSI_PERIOD}), MACD({config.indicators.MACD_FAST},{config.indicators.MACD_SLOW},{config.indicators.MACD_SIGNAL}), ATR({config.indicators.ATR_PERIOD})
ФОКУС: Адаптивные стоп-лоссы, максимальный R:R (2.0-4.0), анализ уровней
КОЛИЧЕСТВО СИГНАЛОВ: {len(top_signals)}

{json.dumps(ai_data, indent=2, ensure_ascii=False)}

ЗАДАЧА: Выбери максимум {config.ai.MAX_SELECTED_PAIRS} лучших пар для МАКСИМАЛЬНОГО ПРОФИТА.
Приоритет: высокий R:R потенциал, четкие уровни, сильные паттерны.

Верни JSON: {{"pairs": ["BTCUSDT", "ETHUSDT"]}}"""

            ai_response = await deep_seek_selection(message)

            if not ai_response:
                return []

            selected_pairs = self._parse_ai_response(ai_response)
            logger.info(f"ЭТАП 2: ИИ выбрал {len(selected_pairs)} пар для максимального профита")
            return selected_pairs

        except Exception as e:
            logger.error(f"Ошибка оптимизированного ИИ отбора: {e}")
            return []

    def _parse_ai_response(self, response: str) -> List[str]:
        """Парсинг ответа ИИ"""
        try:
            json_match = re.search(r'\{[^}]*"pairs"[^}]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                pairs = data.get('pairs', [])
                logger.info(f"ИИ ответ распарсен: {pairs}")
                return pairs

            logger.warning("JSON не найден в ответе ИИ")
            return []
        except Exception as e:
            logger.error(f"Ошибка парсинга ответа ИИ: {e}")
            return []

    async def detailed_analysis(self, pair: str) -> Optional[str]:
        """Детальный анализ для максимального профита с адаптивными уровнями"""
        if not self.analysis_prompt:
            logger.error("Промпт для анализа не загружен")
            return None

        logger.info(f"ЭТАП 3: Детальный анализ {pair} (адаптивные уровни, макс профит)")

        try:
            # Получаем БОЛЬШЕ данных для лучшего анализа уровней
            full_candles_5m = await get_klines_async(pair, config.timeframe.ENTRY_TF,
                                                     limit=config.timeframe.CANDLES_FOR_AI_ANALYSIS)
            full_candles_15m = await get_klines_async(pair, config.timeframe.CONTEXT_TF,
                                                      limit=80)  # Больше контекста

            if not full_candles_5m or not full_candles_15m:
                logger.error(f"Не удалось получить данные для {pair}")
                return None

            # ПРОВЕРЯЕМ порядок свечей
            if not validate_candles_order(full_candles_5m, f"{pair}_5m_detailed"):
                logger.error(f"Неправильный порядок 5m свечей для детального анализа {pair}")
                return None

            if not validate_candles_order(full_candles_15m, f"{pair}_15m_detailed"):
                logger.error(f"Неправильный порядок 15m свечей для детального анализа {pair}")
                return None

            # Полный расчет индикаторов
            full_indicators = calculate_indicators_by_instruction(full_candles_5m)
            signal_analysis = detect_instruction_based_signals(full_candles_5m, full_candles_15m)

            # Расширенный анализ уровней
            level_analysis = self._comprehensive_level_analysis(full_candles_5m, full_candles_15m)

            # Подготавливаем данные для детального анализа с фокусом на профит
            analysis_data = {
                'pair': pair,
                'timestamp': int(time.time()),
                'current_price': float(full_candles_5m[-1][4]),
                'analysis_method': 'optimized_profit_focused_multi_timeframe',

                # Контекст рынка с расширенными данными
                'market_context': {
                    '15m_trend': signal_analysis.get('higher_tf_trend', 'UNKNOWN'),
                    '15m_strength': self._calculate_trend_strength(full_candles_15m),
                    '5m_last_30_candles': [
                        {
                            'timestamp': int(c[0]),
                            'open': float(c[1]),
                            'high': float(c[2]),
                            'low': float(c[3]),
                            'close': float(c[4]),
                            'volume': float(c[5])
                        } for c in full_candles_5m[-30:]
                    ],
                    '15m_last_15_candles': [
                        {
                            'timestamp': int(c[0]),
                            'open': float(c[1]),
                            'high': float(c[2]),
                            'low': float(c[3]),
                            'close': float(c[4]),
                            'volume': float(c[5])
                        } for c in full_candles_15m[-15:]
                    ]
                },

                # Сигнал по инструкции
                'signal_analysis': {
                    'signal_detected': signal_analysis.get('signal', 'NO_SIGNAL'),
                    'pattern_type': signal_analysis.get('pattern_type', 'NONE'),
                    'confidence': signal_analysis.get('confidence', 0),
                    'validation_score': signal_analysis.get('validation_score', '0/5'),
                    'entry_reasons': signal_analysis.get('entry_reasons', []),
                    'validation_reasons': signal_analysis.get('validation_reasons', [])
                },

                # РАСШИРЕННЫЙ технический анализ
                'technical_analysis': safe_json_serialize({
                    'trend_analysis': {
                        'ema5_current': full_indicators.get('ema5', [])[-1] if full_indicators.get('ema5') else 0,
                        'ema8_current': full_indicators.get('ema8', [])[-1] if full_indicators.get('ema8') else 0,
                        'ema21_current': full_indicators.get('ema20', [])[-1] if full_indicators.get('ema20') else 0,
                        'ema_sequence': self._analyze_ema_sequence(full_indicators),
                        'ema_slope': self._calculate_ema_slope(full_indicators),
                        'price_vs_emas': self._analyze_price_vs_emas(float(full_candles_5m[-1][4]), full_indicators)
                    },
                    'momentum_analysis': {
                        'rsi14_current': full_indicators.get('rsi_current', 50),
                        'rsi_divergence': self._detect_rsi_divergence(full_candles_5m, full_indicators),
                        'rsi_trend': self._analyze_rsi_trend(full_indicators),
                        'macd_current': full_indicators.get('macd_line', [])[-1] if full_indicators.get(
                            'macd_line') else 0,
                        'macd_signal_current': full_indicators.get('macd_signal', [])[-1] if full_indicators.get(
                            'macd_signal') else 0,
                        'macd_histogram_current': full_indicators.get('macd_histogram', [])[-1] if full_indicators.get(
                            'macd_histogram') else 0,
                        'macd_crossover_recent': self._detect_macd_crossover(full_indicators),
                        'momentum_strength': self._calculate_momentum_strength(full_indicators)
                    },
                    'volatility_analysis': {
                        'atr14_current': full_indicators.get('atr_current', 0),
                        'atr_percentile': self._calculate_atr_percentile(full_indicators),
                        'volatility_regime': self._determine_volatility_regime(full_indicators),
                        'atr_trend': self._analyze_atr_trend(full_indicators),
                        'optimal_position_size': self._calculate_optimal_position_size(full_indicators,
                                                                                       float(full_candles_5m[-1][4]))
                    },
                    'volume_analysis': {
                        'volume_current': full_indicators.get('volume_current', 0),
                        'volume_sma20': full_indicators.get('volume_sma', [])[-1] if full_indicators.get(
                            'volume_sma') else 0,
                        'volume_ratio': full_indicators.get('volume_ratio', 1.0),
                        'volume_trend': self._calculate_volume_trend(full_indicators),
                        'volume_breakout_potential': self._assess_volume_breakout_potential(full_indicators),
                        'accumulation_distribution': self._calculate_basic_ad(full_candles_5m[-20:])
                    }
                }),

                # КРИТИЧЕСКИЙ блок - анализ уровней для адаптивного управления рисками
                'comprehensive_level_analysis': level_analysis,

                # Оценка риска и потенциала прибыли
                'profit_risk_assessment': {
                    'current_atr_percent': (full_indicators.get('atr_current', 0) / float(
                        full_candles_5m[-1][4])) * 100,
                    'expected_move_range': self._calculate_expected_move_range(full_indicators, level_analysis),
                    'risk_reward_scenarios': self._calculate_rr_scenarios(float(full_candles_5m[-1][4]),
                                                                          level_analysis),
                    'optimal_holding_time': self._estimate_optimal_holding_time(full_indicators, signal_analysis),
                    'market_efficiency': self._assess_market_efficiency(full_candles_5m, full_indicators)
                }
            }

            message = f"""{self.analysis_prompt}

=== ДЕТАЛЬНЫЙ АНАЛИЗ ДЛЯ МАКСИМАЛЬНОГО ПРОФИТА ===
ПАРА: {pair}
МЕТОД: Адаптивные уровни + мультитаймфрейм ({config.timeframe.CONTEXT_TF}m+{config.timeframe.ENTRY_TF}m)
ЦЕЛЬ: R:R 2.0-4.0, оптимальные стоп-лоссы, максимизация прибыли
ТЕКУЩАЯ ЦЕНА: {analysis_data['current_price']}

{json.dumps(analysis_data, indent=2, ensure_ascii=False)}

ЗАДАЧА: Проанализируй и дай МАКСИМАЛЬНО ПРИБЫЛЬНЫЕ торговые рекомендации:

1. АДАПТИВНЫЕ УРОВНИ: Определи точные уровни поддержки/сопротивления
2. УМНЫЙ СТОП-ЛОСС: За ключевые уровни, не по фиксированному ATR
3. МАКСИМАЛЬНЫЙ ТЕЙК-ПРОФИТ: К следующим значимым уровням, R:R 2.0-4.0
4. КОНТЕКСТУАЛЬНАЯ ОЦЕНКА: Учти все препятствия и возможности
5. КОНКРЕТНЫЕ УРОВНИ: Точные цифры с обоснованием

ВЕРНИ РЕЗУЛЬТАТ В СТРОГОМ JSON ФОРМАТЕ!"""

            analysis_result = await deep_seek_analysis(message)

            if analysis_result:
                self._save_optimized_analysis(pair, analysis_result, analysis_data)
                logger.info(f"Оптимизированный анализ {pair} завершен ({len(analysis_result)} символов)")
                return analysis_result

            return None

        except Exception as e:
            logger.error(f"Ошибка детального анализа {pair}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _comprehensive_level_analysis(self, candles_5m: List, candles_15m: List) -> Dict:
        """Комплексный анализ уровней для адаптивного управления рисками"""
        try:
            current_price = float(candles_5m[-1][4])

            # Анализируем 5m свечи для точных уровней
            swing_highs_5m = self._extract_swing_points(candles_5m[-100:] if len(candles_5m) >= 100 else candles_5m,
                                                        'high')
            swing_lows_5m = self._extract_swing_points(candles_5m[-100:] if len(candles_5m) >= 100 else candles_5m,
                                                       'low')

            # Анализируем 15m свечи для ключевых уровней
            swing_highs_15m = self._extract_swing_points(candles_15m[-50:] if len(candles_15m) >= 50 else candles_15m,
                                                         'high')
            swing_lows_15m = self._extract_swing_points(candles_15m[-50:] if len(candles_15m) >= 50 else candles_15m,
                                                        'low')

            # Объединяем и ранжируем уровни
            all_resistance_levels = list(set(swing_highs_5m + swing_highs_15m))
            all_support_levels = list(set(swing_lows_5m + swing_lows_15m))

            # Фильтруем уровни по близости к текущей цене
            nearby_resistance = [level for level in all_resistance_levels
                                 if level > current_price and (
                                             level - current_price) / current_price < 0.05]  # В пределах 5%
            nearby_support = [level for level in all_support_levels
                              if
                              level < current_price and (current_price - level) / current_price < 0.05]  # В пределах 5%

            nearby_resistance.sort()
            nearby_support.sort(reverse=True)

            return {
                'current_price': current_price,
                'immediate_resistance': nearby_resistance[:3],  # 3 ближайших уровня сопротивления
                'immediate_support': nearby_support[:3],  # 3 ближайших уровня поддержки
                'key_resistance_15m': swing_highs_15m[:5],  # Ключевые уровни с 15m
                'key_support_15m': swing_lows_15m[:5],  # Ключевые уровни с 15m
                'price_clusters': self._find_price_clusters(
                    candles_5m[-100:] if len(candles_5m) >= 100 else candles_5m),
                'level_confluence': self._find_level_confluence(all_resistance_levels + all_support_levels,
                                                                current_price),
                'breakout_levels': {
                    'upside_target': nearby_resistance[0] if nearby_resistance else current_price * 1.02,
                    'downside_target': nearby_support[0] if nearby_support else current_price * 0.98,
                    'major_resistance': max(swing_highs_15m[:3]) if swing_highs_15m else current_price * 1.05,
                    'major_support': min(swing_lows_15m[:3]) if swing_lows_15m else current_price * 0.95
                }
            }

        except Exception as e:
            logger.error(f"Ошибка анализа уровней: {e}")
            return {}

    def _find_level_confluence(self, all_levels: List[float], current_price: float) -> List[Dict]:
        """Поиск уровней с высокой концентрацией (confluence)"""
        if not all_levels:
            return []

        confluence_zones = []
        tolerance = current_price * 0.002  # 0.2% толерантность

        sorted_levels = sorted(all_levels)

        i = 0
        while i < len(sorted_levels):
            zone_levels = [sorted_levels[i]]
            j = i + 1

            # Группируем уровни в зоне
            while j < len(sorted_levels) and sorted_levels[j] - sorted_levels[i] <= tolerance:
                zone_levels.append(sorted_levels[j])
                j += 1

            if len(zone_levels) >= 2:  # Минимум 2 уровня для confluence
                zone_center = sum(zone_levels) / len(zone_levels)
                distance_percent = abs(zone_center - current_price) / current_price * 100

                confluence_zones.append({
                    'level': zone_center,
                    'strength': len(zone_levels),
                    'distance_percent': distance_percent,
                    'type': 'resistance' if zone_center > current_price else 'support'
                })

            i = j if j > i + 1 else i + 1

        return sorted(confluence_zones, key=lambda x: x['strength'], reverse=True)[:5]

    def _calculate_trend_strength(self, candles_15m: List) -> int:
        """Расчет силы тренда на 15m таймфрейме"""
        if len(candles_15m) < 10:
            return 0

        closes = [float(c[4]) for c in candles_15m[-10:]]

        # Простая оценка: сколько свечей идут в одном направлении
        up_moves = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])
        strength = (up_moves / (len(closes) - 1)) * 100

        return int(strength)

    def _analyze_ema_sequence(self, indicators: Dict) -> str:
        """Анализ последовательности EMA"""
        ema5 = indicators.get('ema5', [])
        ema8 = indicators.get('ema8', [])
        ema20 = indicators.get('ema20', [])

        if not all([ema5, ema8, ema20]):
            return 'insufficient_data'

        current_5 = ema5[-1]
        current_8 = ema8[-1]
        current_20 = ema20[-1]

        if current_5 > current_8 > current_20:
            return 'bullish_aligned'
        elif current_5 < current_8 < current_20:
            return 'bearish_aligned'
        else:
            return 'mixed'

    def _calculate_ema_slope(self, indicators: Dict) -> Dict:
        """Расчет наклона EMA"""
        result = {}

        for ema_name in ['ema5', 'ema8', 'ema20']:
            ema_values = indicators.get(ema_name, [])
            if len(ema_values) >= 5:
                recent_slope = (ema_values[-1] - ema_values[-5]) / ema_values[-5] * 100
                result[ema_name + '_slope'] = round(recent_slope, 4)
            else:
                result[ema_name + '_slope'] = 0

        return result

    def _analyze_price_vs_emas(self, current_price: float, indicators: Dict) -> Dict:
        """Анализ позиции цены относительно EMA"""
        result = {}

        for ema_name in ['ema5', 'ema8', 'ema20']:
            ema_values = indicators.get(ema_name, [])
            if ema_values:
                distance_percent = ((current_price - ema_values[-1]) / ema_values[-1]) * 100
                result[ema_name + '_distance'] = round(distance_percent, 3)
            else:
                result[ema_name + '_distance'] = 0

        return result

    def _detect_rsi_divergence(self, candles: List, indicators: Dict) -> str:
        """Простое определение дивергенции RSI"""
        if len(candles) < 20:
            return 'insufficient_data'

        rsi_values = indicators.get('rsi', [])
        if len(rsi_values) < 20:
            return 'insufficient_data'

        # Упрощенная проверка: сравниваем последние 10 и предыдущие 10 периодов
        recent_rsi = rsi_values[-10:]
        recent_prices = [float(c[4]) for c in candles[-10:]]

        rsi_trend = 'rising' if recent_rsi[-1] > recent_rsi[0] else 'falling'
        price_trend = 'rising' if recent_prices[-1] > recent_prices[0] else 'falling'

        if rsi_trend != price_trend:
            return f'bearish_divergence' if price_trend == 'rising' else 'bullish_divergence'
        else:
            return 'no_divergence'

    def _analyze_rsi_trend(self, indicators: Dict) -> str:
        """Анализ тренда RSI"""
        rsi_values = indicators.get('rsi', [])
        if len(rsi_values) < 5:
            return 'insufficient_data'

        recent_rsi = rsi_values[-5:]
        if recent_rsi[-1] > recent_rsi[0]:
            return 'strengthening'
        else:
            return 'weakening'

    def _detect_macd_crossover(self, indicators: Dict) -> Dict:
        """Определение недавних пересечений MACD"""
        macd_line = indicators.get('macd_line', [])
        macd_signal = indicators.get('macd_signal', [])

        if len(macd_line) < 3 or len(macd_signal) < 3:
            return {'recent_crossover': False, 'direction': 'none'}

        # Проверяем последние 3 периода на пересечение
        for i in range(len(macd_line) - 3, len(macd_line) - 1):
            if i < 1:
                continue

            prev_diff = macd_line[i - 1] - macd_signal[i - 1]
            curr_diff = macd_line[i] - macd_signal[i]

            if prev_diff <= 0 < curr_diff:
                return {'recent_crossover': True, 'direction': 'bullish', 'periods_ago': len(macd_line) - i - 1}
            elif prev_diff >= 0 > curr_diff:
                return {'recent_crossover': True, 'direction': 'bearish', 'periods_ago': len(macd_line) - i - 1}

        return {'recent_crossover': False, 'direction': 'none'}

    def _calculate_momentum_strength(self, indicators: Dict) -> int:
        """Расчет общей силы моментума (0-100)"""
        rsi_current = indicators.get('rsi_current', 50)
        macd_histogram = indicators.get('macd_histogram', [])

        score = 0

        # RSI компонент
        if 40 <= rsi_current <= 60:
            score += 20  # Нейтральная зона
        elif 30 <= rsi_current < 40 or 60 < rsi_current <= 70:
            score += 40  # Умеренный моментум
        elif rsi_current < 30 or rsi_current > 70:
            score += 60  # Сильный моментум

        # MACD компонент
        if macd_histogram and len(macd_histogram) >= 3:
            if abs(macd_histogram[-1]) > abs(macd_histogram[-3]):
                score += 40  # Растущая гистограмма

        return min(100, score)

    def _determine_volatility_regime(self, indicators: Dict) -> str:
        """Определение режима волатильности"""
        atr_values = indicators.get('atr', [])
        if len(atr_values) < 20:
            return 'unknown'

        current_atr = indicators.get('atr_current', 0)
        avg_atr = sum(atr_values[-20:]) / len(atr_values[-20:])

        ratio = current_atr / avg_atr if avg_atr > 0 else 1

        if ratio > 1.5:
            return 'high_volatility'
        elif ratio > 1.2:
            return 'elevated_volatility'
        elif ratio < 0.7:
            return 'low_volatility'
        else:
            return 'normal_volatility'

    def _analyze_atr_trend(self, indicators: Dict) -> str:
        """Анализ тренда ATR"""
        atr_values = indicators.get('atr', [])
        if len(atr_values) < 10:
            return 'insufficient_data'

        recent_avg = sum(atr_values[-5:]) / 5
        older_avg = sum(atr_values[-10:-5]) / 5

        if recent_avg > older_avg * 1.1:
            return 'expanding'
        elif recent_avg < older_avg * 0.9:
            return 'contracting'
        else:
            return 'stable'

    def _calculate_optimal_position_size(self, indicators: Dict, current_price: float) -> float:
        """Расчет оптимального размера позиции на основе ATR"""
        atr_current = indicators.get('atr_current', 0)
        if atr_current == 0 or current_price == 0:
            return config.trading.DEFAULT_POSITION_SIZE_PERCENT

        # Адаптируем размер позиции к волатильности
        atr_percent = (atr_current / current_price) * 100

        if atr_percent > 2.0:  # Высокая волатильность
            return max(1.0, config.trading.DEFAULT_POSITION_SIZE_PERCENT * 0.5)
        elif atr_percent < 0.5:  # Низкая волатильность
            return min(config.trading.MAX_POSITION_SIZE_PERCENT, config.trading.DEFAULT_POSITION_SIZE_PERCENT * 1.5)
        else:
            return config.trading.DEFAULT_POSITION_SIZE_PERCENT

    def _assess_volume_breakout_potential(self, indicators: Dict) -> str:
        """Оценка потенциала объемного пробоя"""
        volume_ratio = indicators.get('volume_ratio', 1.0)
        volume_sma = indicators.get('volume_sma', [])

        if not volume_sma or len(volume_sma) < 10:
            return 'insufficient_data'

        # Анализируем тренд объема
        recent_avg = sum(volume_sma[-5:]) / 5
        older_avg = sum(volume_sma[-10:-5]) / 5

        if volume_ratio > 2.0:
            return 'high_breakout_potential'
        elif volume_ratio > 1.5 and recent_avg > older_avg * 1.2:
            return 'moderate_breakout_potential'
        elif volume_ratio < 0.8:
            return 'low_breakout_potential'
        else:
            return 'normal_volume_activity'

    def _calculate_basic_ad(self, candles: List) -> float:
        """Упрощенный расчет Accumulation/Distribution"""
        if len(candles) < 10:
            return 0.0

        ad_sum = 0.0

        for candle in candles:
            high = float(candle[2])
            low = float(candle[3])
            close = float(candle[4])
            volume = float(candle[5])

            if high != low:
                money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
                money_flow_volume = money_flow_multiplier * volume
                ad_sum += money_flow_volume

        return round(ad_sum, 2)

    def _calculate_expected_move_range(self, indicators: Dict, level_analysis: Dict) -> Dict:
        """Расчет ожидаемого диапазона движения"""
        atr_current = indicators.get('atr_current', 0)
        current_price = level_analysis.get('current_price', 0)

        if atr_current == 0 or current_price == 0:
            return {}

        # Базовый диапазон на основе ATR
        atr_range_up = current_price + (atr_current * 2)
        atr_range_down = current_price - (atr_current * 2)

        # Корректировка на основе уровней
        immediate_resistance = level_analysis.get('immediate_resistance', [])
        immediate_support = level_analysis.get('immediate_support', [])

        realistic_upside = immediate_resistance[0] if immediate_resistance else atr_range_up
        realistic_downside = immediate_support[0] if immediate_support else atr_range_down

        return {
            'atr_based_range': {
                'upside': atr_range_up,
                'downside': atr_range_down,
                'range_percent': (atr_current * 4 / current_price) * 100
            },
            'level_adjusted_range': {
                'upside_target': realistic_upside,
                'downside_target': realistic_downside,
                'upside_percent': ((realistic_upside - current_price) / current_price) * 100,
                'downside_percent': ((current_price - realistic_downside) / current_price) * 100
            }
        }

    def _calculate_rr_scenarios(self, current_price: float, level_analysis: Dict) -> Dict:
        """Расчет сценариев риск/доходность"""
        immediate_resistance = level_analysis.get('immediate_resistance', [])
        immediate_support = level_analysis.get('immediate_support', [])

        scenarios = {}

        # Консервативный сценарий
        if immediate_resistance and immediate_support:
            conservative_target = immediate_resistance[0]
            conservative_stop = immediate_support[0]

            profit_potential = (conservative_target - current_price) / current_price * 100
            risk_potential = (current_price - conservative_stop) / current_price * 100

            scenarios['conservative'] = {
                'target': conservative_target,
                'stop': conservative_stop,
                'profit_percent': profit_potential,
                'risk_percent': risk_potential,
                'risk_reward_ratio': profit_potential / risk_potential if risk_potential > 0 else 0
            }

        # Агрессивный сценарий
        key_resistance = level_analysis.get('key_resistance_15m', [])
        key_support = level_analysis.get('key_support_15m', [])

        if key_resistance and key_support:
            aggressive_target = key_resistance[0]
            aggressive_stop = key_support[-1]  # Дальний уровень поддержки

            profit_potential = (aggressive_target - current_price) / current_price * 100
            risk_potential = (current_price - aggressive_stop) / current_price * 100

            scenarios['aggressive'] = {
                'target': aggressive_target,
                'stop': aggressive_stop,
                'profit_percent': profit_potential,
                'risk_percent': risk_potential,
                'risk_reward_ratio': profit_potential / risk_potential if risk_potential > 0 else 0
            }

        return scenarios

    def _estimate_optimal_holding_time(self, indicators: Dict, signal_analysis: Dict) -> str:
        """Оценка оптимального времени удержания позиции"""
        pattern_type = signal_analysis.get('pattern_type', 'UNKNOWN')
        atr_current = indicators.get('atr_current', 0)
        volume_ratio = indicators.get('volume_ratio', 1.0)

        # Базовое время в зависимости от паттерна
        base_time = {
            'MOMENTUM_BREAKOUT': 30,
            'SQUEEZE_BREAKOUT': 45,
            'PULLBACK_ENTRY': 40,
            'RANGE_SCALP': 20
        }.get(pattern_type, 35)

        # Корректировка на основе волатильности
        if atr_current > 0:
            # Высокая волатильность = быстрее достигаем целей
            volatility_adjustment = max(0.7, min(1.3, 1.0 / (atr_current * 1000)))
            base_time = int(base_time * volatility_adjustment)

        # Корректировка на основе объема
        if volume_ratio > 1.5:
            base_time = int(base_time * 0.8)  # Высокий объем = быстрее движение
        elif volume_ratio < 0.8:
            base_time = int(base_time * 1.2)  # Низкий объем = медленнее движение

        return f"{base_time}-{base_time + 15} minutes"

    def _assess_market_efficiency(self, candles: List, indicators: Dict) -> str:
        """Оценка эффективности рынка для скальпинга"""
        if len(candles) < 20:
            return 'insufficient_data'

        # Анализ последних 20 свечей
        recent_candles = candles[-20:]

        # Считаем количество gap'ов и резких движений
        gaps = 0
        big_moves = 0

        for i in range(1, len(recent_candles)):
            prev_close = float(recent_candles[i - 1][4])
            curr_open = float(recent_candles[i][1])
            curr_high = float(recent_candles[i][2])
            curr_low = float(recent_candles[i][3])
            curr_close = float(recent_candles[i][4])

            # Gap detection
            if abs(curr_open - prev_close) / prev_close > 0.002:  # Gap > 0.2%
                gaps += 1

            # Big move detection
            if (curr_high - curr_low) / curr_open > 0.01:  # Range > 1%
                big_moves += 1

        volume_ratio = indicators.get('volume_ratio', 1.0)

        # Общая оценка
        if gaps <= 2 and big_moves <= 5 and 0.8 <= volume_ratio <= 2.0:
            return 'high_efficiency'
        elif gaps <= 4 and big_moves <= 8:
            return 'moderate_efficiency'
        else:
            return 'low_efficiency'

    def _save_optimized_analysis(self, pair: str, analysis: str, analysis_data: Dict):
        """Сохранение оптимизированного анализа"""
        try:
            with open(config.system.ANALYSIS_LOG_FILE, 'a', encoding=config.system.ENCODING) as f:
                f.write(f"\n{'=' * 100}\n")
                f.write(f"ОПТИМИЗИРОВАННЫЙ АНАЛИЗ ДЛЯ МАКСИМАЛЬНОГО ПРОФИТА\n")
                f.write(f"ПАРА: {pair}\n")
                f.write(f"ВРЕМЯ: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(
                    f"МЕТОД: Адаптивные уровни + мультитаймфрейм ({config.timeframe.CONTEXT_TF}m+{config.timeframe.ENTRY_TF}m)\n")
                f.write(f"ЦЕНА: {analysis_data.get('current_price', 0)}\n")
                f.write(f"СИГНАЛ: {analysis_data.get('signal_analysis', {}).get('signal_detected', 'N/A')}\n")
                f.write(f"ПАТТЕРН: {analysis_data.get('signal_analysis', {}).get('pattern_type', 'N/A')}\n")
                f.write(f"УВЕРЕННОСТЬ: {analysis_data.get('signal_analysis', {}).get('confidence', 0)}%\n")
                f.write(f"{'=' * 50}\n")
                f.write(f"АНАЛИЗ ИИ:\n{analysis}\n")
                f.write(f"{'=' * 50}\n")
                f.write(f"ТЕХНИЧЕСКИЕ ДАННЫЕ:\n")
                f.write(
                    f"Уровни сопротивления: {analysis_data.get('comprehensive_level_analysis', {}).get('immediate_resistance', [])}\n")
                f.write(
                    f"Уровни поддержки: {analysis_data.get('comprehensive_level_analysis', {}).get('immediate_support', [])}\n")
                f.write(
                    f"ATR: {analysis_data.get('technical_analysis', {}).get('volatility_analysis', {}).get('atr14_current', 0)}\n")
                f.write(
                    f"Объем: {analysis_data.get('technical_analysis', {}).get('volume_analysis', {}).get('volume_ratio', 1.0)}\n")
                f.write(f"{'=' * 100}\n")
        except Exception as e:
            logger.error(f"Ошибка сохранения оптимизированного анализа: {e}")


async def main():
    """Главная функция оптимизированного бота для максимального профита"""
    logger.info("🚀 ОПТИМИЗИРОВАННЫЙ СКАЛЬПИНГОВЫЙ БОТ - МАКСИМАЛЬНЫЙ ПРОФИТ")
    logger.info(f"📊 Метод: {config.timeframe.CONTEXT_TF}m контекст + {config.timeframe.ENTRY_TF}m точный вход")
    logger.info(f"🎯 Цель: R:R {config.trading.MIN_RISK_REWARD}-{config.trading.DEFAULT_RISK_REWARD}, адаптивные уровни")
    logger.info(
        f"⚙️ Индикаторы: EMA({config.indicators.EMA_FAST}/{config.indicators.EMA_MEDIUM}/{config.indicators.EMA_SLOW}), RSI({config.indicators.RSI_PERIOD}), MACD({config.indicators.MACD_FAST},{config.indicators.MACD_SLOW},{config.indicators.MACD_SIGNAL}), ATR({config.indicators.ATR_PERIOD})")

    analyzer = InstructionBasedAnalyzer()
    ai_selector = OptimizedAISelector()

    try:
        # ЭТАП 1: Оптимизированное сканирование с менее строгими фильтрами
        promising_signals = await analyzer.mass_scan_markets()

        if not promising_signals:
            logger.info("❌ Оптимизированные сигналы не найдены")
            return

        logger.info(f"✅ Найдено {len(promising_signals)} оптимизированных сигналов")
        for signal in promising_signals[:8]:  # Показываем топ-8
            logger.info(
                f"   📈 {signal.pair}: {signal.pattern_type} ({signal.confidence}%, {signal.validation_score}, vol {signal.volume_ratio:.2f})")

        # ЭТАП 2: Оптимизированный ИИ отбор для максимального профита
        selected_pairs = await ai_selector.select_best_pairs(promising_signals)

        if not selected_pairs:
            logger.info("❌ ИИ не выбрал пары по оптимизированным критериям")
            return

        logger.info(f"🎯 ИИ выбрал {len(selected_pairs)} пар для МАКСИМАЛЬНОГО ПРОФИТА: {selected_pairs}")

        # ЭТАП 3: Детальный анализ с адаптивными уровнями
        successful_analyses = 0

        for pair in selected_pairs:
            logger.info(f"🔍 Детальный анализ {pair} (адаптивные уровни)...")
            analysis = await ai_selector.detailed_analysis(pair)

            if analysis:
                successful_analyses += 1
                logger.info(f"✅ {pair} - анализ завершен (оптимизирован для профита)")
            else:
                logger.error(f"❌ {pair} - ошибка анализа")

            await asyncio.sleep(2)  # Увеличенная пауза для качества

        # ИТОГИ ОПТИМИЗИРОВАННОЙ РАБОТЫ
        logger.info(f"\n{'=' * 80}")
        logger.info(f"🏆 ОПТИМИЗИРОВАННЫЙ АНАЛИЗ ЗАВЕРШЕН!")
        logger.info(
            f"📊 Метод: Адаптивные уровни + мультитаймфрейм ({config.timeframe.CONTEXT_TF}m+{config.timeframe.ENTRY_TF}m)")
        logger.info(f"🎯 Цель: R:R {config.trading.DEFAULT_RISK_REWARD}:1, максимизация прибыли")
        logger.info(f"📈 Найдено сигналов: {len(promising_signals)}")
        logger.info(f"🤖 ИИ отобрал: {len(selected_pairs)}")
        logger.info(f"✅ Успешных анализов: {successful_analyses}")
        logger.info(f"💾 Результаты: {config.system.ANALYSIS_LOG_FILE}")
        logger.info(f"⚡ Конфигурация: менее строгие фильтры, больше возможностей")
        logger.info(f"{'=' * 80}")

        await cleanup_http_client()

    except KeyboardInterrupt:
        logger.info("⏹️ Остановка по запросу пользователя")
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    logger.info("=" * 100)
    logger.info("🚀 ОПТИМИЗИРОВАННЫЙ СКАЛЬПИНГОВЫЙ БОТ ДЛЯ МАКСИМАЛЬНОГО ПРОФИТА")
    logger.info(f"📊 {config.timeframe.CONTEXT_TF}m контекст + {config.timeframe.ENTRY_TF}m точный вход")
    logger.info(f"🎯 R:R {config.trading.DEFAULT_RISK_REWARD}:1 | Адаптивные стоп-лоссы | Умные тейк-профиты")
    logger.info(
        f"⚙️ EMA({config.indicators.EMA_FAST}/{config.indicators.EMA_MEDIUM}/{config.indicators.EMA_SLOW}), RSI({config.indicators.RSI_PERIOD}), MACD({config.indicators.MACD_FAST},{config.indicators.MACD_SLOW},{config.indicators.MACD_SIGNAL}), ATR({config.indicators.ATR_PERIOD})")
    logger.info("🎨 Шаблоны: Momentum, Pullback, Squeeze, Range (оптимизированы)")
    logger.info("🔧 Фильтры: ослаблены для больше возможностей")
    logger.info("=" * 100)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹️ Программа остановлена пользователем")
    except Exception as e:
        logger.error(f"💥 Фатальная ошибка: {e}")
    finally:
        logger.info("🏁 Работа завершена")