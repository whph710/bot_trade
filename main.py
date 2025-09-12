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

# Импорт конфигурации
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

    # Для ИИ (только краткие данные для отбора)
    candles_5m: List = None  # 5m свечи для входа
    candles_15m: List = None  # 15m свечи для контекста
    indicators_data: Dict = None


class InstructionBasedAnalyzer:
    """Анализатор согласно инструкции: 15m контекст + 5m вход"""

    def __init__(self):
        self.session_start = time.time()
        logger.info("Анализатор по инструкции запущен (15m+5m)")

    def passes_liquidity_filter(self, symbol: str, candles: List) -> bool:
        """Фильтр ликвидности согласно инструкции"""
        if not candles:
            return False

        # Примерная оценка объема (последние 24 свечи 5m = 2 часа)
        recent_volumes = [float(c[5]) * float(c[4]) for c in candles[-24:]]  # Объем в USD
        avg_hourly_volume = sum(recent_volumes) * 12  # Приблизительно за 24ч

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
        return spread_estimate < config.trading.MAX_SPREAD_PERCENT

    async def quick_scan_pair(self, symbol: str) -> Optional[InstructionBasedSignal]:
        """Быстрое сканирование пары согласно инструкции"""
        try:
            # Получаем данные для быстрого сканирования (меньше данных)
            candles_5m = await get_klines_async(symbol, config.timeframe.ENTRY_TF,
                                                limit=config.timeframe.CANDLES_5M_QUICK)
            candles_15m = await get_klines_async(symbol, config.timeframe.CONTEXT_TF,
                                                 limit=config.timeframe.CANDLES_15M_QUICK)

            if not candles_5m or not candles_15m:
                return None

            # Фильтры согласно инструкции
            if not self.passes_liquidity_filter(symbol, candles_5m):
                return None

            if not self.check_spread_quality(candles_5m):
                return None

            # Определяем сигнал согласно инструкции (мультитаймфрейм)
            signal_result = detect_instruction_based_signals(candles_5m, candles_15m)

            if signal_result['signal'] == 'NO_SIGNAL':
                return None

            # Создаем сигнал
            entry_price = float(candles_5m[-1][4])
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

                # Данные для ИИ отбора (краткие)
                candles_5m=candles_5m[-config.timeframe.CANDLES_FOR_AI_SELECTION:],
                candles_15m=candles_15m[-config.timeframe.CANDLES_FOR_CONTEXT:],
                indicators_data=clean_value(signal_result.get('indicators', {}))
            )

        except Exception as e:
            logger.error(f"Ошибка сканирования {symbol}: {e}")
            return None

    async def mass_scan_markets(self) -> List[InstructionBasedSignal]:
        """Массовое сканирование с фильтрацией согласно инструкции"""
        start_time = time.time()
        logger.info("ЭТАП 1: Сканирование с фильтрами по инструкции")

        try:
            pairs = await get_usdt_trading_pairs()
            if not pairs:
                return []

            logger.info(f"Сканируем {len(pairs)} пар (15m контекст + 5m вход)")

            promising_signals = []

            # Обрабатываем батчами
            for i in range(0, len(pairs), config.processing.BATCH_SIZE):
                batch = pairs[i:i + config.processing.BATCH_SIZE]
                tasks = [self.quick_scan_pair(pair) for pair in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, InstructionBasedSignal):
                        promising_signals.append(result)

                processed = min(i + config.processing.BATCH_SIZE, len(pairs))
                logger.info(f"Обработано: {processed}/{len(pairs)}")

                if i + config.processing.BATCH_SIZE < len(pairs):
                    await asyncio.sleep(config.processing.BATCH_DELAY)

            # Сортируем по уверенности
            promising_signals.sort(key=lambda x: x.confidence, reverse=True)

            execution_time = time.time() - start_time
            logger.info(f"ЭТАП 1: {len(promising_signals)} сигналов за {execution_time:.2f}сек")

            return promising_signals

        except Exception as e:
            logger.error(f"Ошибка сканирования: {e}")
            return []


class InstructionBasedAISelector:
    """ИИ селектор согласно инструкции"""

    def __init__(self):
        self.selection_prompt = self._load_prompt(config.ai.SELECTION_PROMPT_FILE)
        self.analysis_prompt = self._load_prompt(config.ai.ANALYSIS_PROMPT_FILE)

    def _load_prompt(self, filename: str) -> str:
        """Загрузка промпта"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error(f"Файл {filename} не найден")
            return ""

    def _prepare_signals_for_ai(self, signals: List[InstructionBasedSignal]) -> Dict[str, Any]:
        """Подготовка данных для ИИ отбора (краткие данные)"""
        prepared_data = []

        for signal in signals:
            # Свечи 5m для анализа входа (краткие)
            recent_5m = signal.candles_5m[-30:] if signal.candles_5m else []
            # Свечи 15m для контекста (краткие)
            recent_15m = signal.candles_15m[-20:] if signal.candles_15m else []

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

                # Краткие мультитаймфреймные данные для отбора
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
                    ],
                    '15m_context': [
                        {
                            'timestamp': int(c[0]),
                            'open': float(c[1]),
                            'high': float(c[2]),
                            'low': float(c[3]),
                            'close': float(c[4]),
                            'volume': float(c[5])
                        } for c in recent_15m
                    ]
                },

                # Краткие индикаторы для отбора
                'technical_indicators': safe_json_serialize({
                    'ema_system': {
                        'ema5_current': signal.indicators_data.get('ema5', [])[-1] if signal.indicators_data.get('ema5') else 0,
                        'ema8_current': signal.indicators_data.get('ema8', [])[-1] if signal.indicators_data.get('ema8') else 0,
                        'ema20_current': signal.indicators_data.get('ema20', [])[-1] if signal.indicators_data.get('ema20') else 0
                    },
                    'rsi': {
                        'current': signal.indicators_data.get('rsi_current', 50)
                    },
                    'macd': {
                        'line_current': signal.indicators_data.get('macd_line', [])[-1] if signal.indicators_data.get('macd_line') else 0,
                        'signal_current': signal.indicators_data.get('macd_signal', [])[-1] if signal.indicators_data.get('macd_signal') else 0,
                        'histogram_current': signal.indicators_data.get('macd_histogram', [])[-1] if signal.indicators_data.get('macd_histogram') else 0
                    },
                    'volume': {
                        'ratio': signal.volume_ratio,
                        'current': signal.indicators_data.get('volume_current', 0)
                    }
                })
            }

            prepared_data.append(clean_value(signal_data))

        return {
            'timestamp': int(time.time()),
            'total_pairs': len(signals),
            'signals': prepared_data
        }

    async def ai_select_top_pairs(self, signals: List[InstructionBasedSignal]) -> List[str]:
        """ИИ отбор топ пар для детального анализа"""
        if not signals:
            return []

        logger.info(f"ЭТАП 2: ИИ отбор из {len(signals)} сигналов")

        # Берем максимум пар для ИИ согласно конфигу
        signals_for_ai = signals[:config.ai.MAX_PAIRS_TO_AI]

        # Подготавливаем краткие данные для быстрого отбора
        ai_data = self._prepare_signals_for_ai(signals_for_ai)

        try:
            # Быстрый отбор ИИ
            ai_response = await deep_seek_selection(
                data=json.dumps(ai_data, ensure_ascii=False, indent=2)
            )

            # Парсим ответ ИИ
            selected_pairs = self._parse_ai_selection(ai_response)

            logger.info(f"ЭТАП 2: ИИ выбрал {len(selected_pairs)} пар: {selected_pairs}")
            return selected_pairs[:config.ai.MAX_SELECTED_PAIRS]

        except Exception as e:
            logger.error(f"Ошибка ИИ отбора: {e}")
            # Fallback: берем топ по уверенности
            return [s.pair for s in signals_for_ai[:config.ai.MAX_SELECTED_PAIRS]]

    def _parse_ai_selection(self, ai_response: str) -> List[str]:
        """Парсинг ответа ИИ для извлечения выбранных пар"""
        try:
            # Ищем JSON в ответе
            json_match = re.search(r'\{[^{}]*"pairs"[^{}]*\}', ai_response)
            if json_match:
                json_data = json.loads(json_match.group())
                return json_data.get('pairs', [])

            # Если JSON не найден, ищем пары в тексте
            pairs = re.findall(r'[A-Z]{2,10}USDT', ai_response)
            return list(dict.fromkeys(pairs))  # Убираем дубликаты
        except:
            return []

    async def detailed_analysis_with_full_data(self, selected_pairs: List[str]) -> List[Dict]:
        """ФИНАЛЬНЫЙ АНАЛИЗ с полными данными для ИИ"""
        if not selected_pairs:
            return []

        logger.info(f"ЭТАП 3: Детальный анализ {len(selected_pairs)} пар с полными данными")

        results = []

        for pair in selected_pairs:
            try:
                # Загружаем БОЛЬШИЕ объемы данных для финального анализа
                candles_5m = await get_klines_async(
                    pair,
                    config.timeframe.ENTRY_TF,
                    limit=config.timeframe.DETAILED_CANDLES_5M
                )
                candles_15m = await get_klines_async(
                    pair,
                    config.timeframe.CONTEXT_TF,
                    limit=config.timeframe.DETAILED_CANDLES_15M
                )

                if not candles_5m or not candles_15m:
                    continue

                # Рассчитываем полные индикаторы для ОБОИХ таймфреймов
                indicators_5m = calculate_indicators_by_instruction(candles_5m)
                indicators_15m = calculate_indicators_by_instruction(candles_15m)

                # Определяем базовый сигнал
                signal_result = detect_instruction_based_signals(candles_5m, candles_15m)

                if signal_result['signal'] == 'NO_SIGNAL':
                    continue

                # Подготавливаем ПОЛНЫЕ данные для финального ИИ анализа
                full_analysis_data = self._prepare_full_data_for_final_analysis(
                    pair, candles_5m, candles_15m, indicators_5m, indicators_15m, signal_result
                )

                # Отправляем на детальный анализ ИИ
                ai_analysis = await deep_seek_analysis(
                    data=json.dumps(full_analysis_data, ensure_ascii=False, indent=2)
                )

                # Парсим финальную рекомендацию ИИ
                final_recommendation = self._parse_final_ai_analysis(ai_analysis, pair, signal_result)

                if final_recommendation:
                    results.append(final_recommendation)
                    logger.info(f"Финальная рекомендация для {pair}: {final_recommendation.get('signal', 'NO_SIGNAL')}")

            except Exception as e:
                logger.error(f"Ошибка детального анализа {pair}: {e}")
                continue

        logger.info(f"ЭТАП 3: Получено {len(results)} финальных рекомендаций")
        return results

    def _prepare_full_data_for_final_analysis(self, pair: str, candles_5m: List,
                                              candles_15m: List, indicators_5m: Dict,
                                              indicators_15m: Dict, signal_result: Dict) -> Dict:
        """Подготовка ПОЛНЫХ данных для финального анализа ИИ"""

        # Полные свечи для анализа
        full_5m_candles = [{
            'timestamp': int(c[0]),
            'open': float(c[1]),
            'high': float(c[2]),
            'low': float(c[3]),
            'close': float(c[4]),
            'volume': float(c[5])
        } for c in candles_5m]

        full_15m_candles = [{
            'timestamp': int(c[0]),
            'open': float(c[1]),
            'high': float(c[2]),
            'low': float(c[3]),
            'close': float(c[4]),
            'volume': float(c[5])
        } for c in candles_15m]

        # ПОЛНЫЕ исторические индикаторы
        def prepare_indicator_history(indicators: Dict, history_points: int) -> Dict:
            prepared = {}
            for key, value in indicators.items():
                if isinstance(value, list) and len(value) > 0:
                    prepared[key] = value[-history_points:] if len(value) >= history_points else value
                else:
                    prepared[key] = value
            return prepared

        full_indicators_5m = prepare_indicator_history(indicators_5m, config.timeframe.INDICATORS_HISTORY_POINTS)
        full_indicators_15m = prepare_indicator_history(indicators_15m, config.timeframe.INDICATORS_HISTORY_POINTS)

        return clean_value({
            'pair_info': {
                'symbol': pair,
                'current_price': float(candles_5m[-1][4]),
                'analysis_timestamp': int(time.time()),
                'signal_detected': signal_result['signal'],
                'pattern_type': signal_result.get('pattern_type', 'NONE'),
                'initial_confidence': signal_result.get('confidence', 0)
            },

            # ПОЛНЫЕ мультитаймфреймные данные
            'market_data': {
                'timeframe_5m': {
                    'candles_count': len(full_5m_candles),
                    'candles': full_5m_candles,
                    'period_hours': len(full_5m_candles) * 5 / 60
                },
                'timeframe_15m': {
                    'candles_count': len(full_15m_candles),
                    'candles': full_15m_candles,
                    'period_hours': len(full_15m_candles) * 15 / 60
                }
            },

            # ПОЛНЫЕ исторические индикаторы для ОБОИХ таймфреймов
            'technical_analysis': {
                'indicators_5m': {
                    'description': '5-минутные индикаторы для точного входа',
                    'data_points': len(full_indicators_5m.get('ema5', [])),
                    'ema_system': {
                        'ema5_history': full_indicators_5m.get('ema5', []),
                        'ema8_history': full_indicators_5m.get('ema8', []),
                        'ema20_history': full_indicators_5m.get('ema20', [])
                    },
                    'momentum_indicators': {
                        'rsi_history': full_indicators_5m.get('rsi', []),
                        'rsi_current': full_indicators_5m.get('rsi_current', 50),
                        'macd_line_history': full_indicators_5m.get('macd_line', []),
                        'macd_signal_history': full_indicators_5m.get('macd_signal', []),
                        'macd_histogram_history': full_indicators_5m.get('macd_histogram', [])
                    },
                    'volatility_indicators': {
                        'atr_history': full_indicators_5m.get('atr', []),
                        'atr_current': full_indicators_5m.get('atr_current', 0),
                        'atr_mean': full_indicators_5m.get('atr_mean', 0),
                        'bb_upper_history': full_indicators_5m.get('bb_upper', []),
                        'bb_middle_history': full_indicators_5m.get('bb_middle', []),
                        'bb_lower_history': full_indicators_5m.get('bb_lower', [])
                    },
                    'volume_analysis': {
                        'volume_sma_history': full_indicators_5m.get('volume_sma', []),
                        'volume_current': full_indicators_5m.get('volume_current', 0),
                        'volume_ratio': full_indicators_5m.get('volume_ratio', 1.0)
                    }
                },

                'indicators_15m': {
                    'description': '15-минутные индикаторы для контекста и тренда',
                    'data_points': len(full_indicators_15m.get('ema5', [])),
                    'ema_system': {
                        'ema5_history': full_indicators_15m.get('ema5', []),
                        'ema8_history': full_indicators_15m.get('ema8', []),
                        'ema20_history': full_indicators_15m.get('ema20', [])
                    },
                    'momentum_indicators': {
                        'rsi_history': full_indicators_15m.get('rsi', []),
                        'rsi_current': full_indicators_15m.get('rsi_current', 50),
                        'macd_line_history': full_indicators_15m.get('macd_line', []),
                        'macd_signal_history': full_indicators_15m.get('macd_signal', []),
                        'macd_histogram_history': full_indicators_15m.get('macd_histogram', [])
                    },
                    'volatility_indicators': {
                        'atr_history': full_indicators_15m.get('atr', []),
                        'atr_current': full_indicators_15m.get('atr_current', 0),
                        'atr_mean': full_indicators_15m.get('atr_mean', 0),
                        'bb_upper_history': full_indicators_15m.get('bb_upper', []),
                        'bb_middle_history': full_indicators_15m.get('bb_middle', []),
                        'bb_lower_history': full_indicators_15m.get('bb_lower', [])
                    },
                    'volume_analysis': {
                        'volume_sma_history': full_indicators_15m.get('volume_sma', []),
                        'volume_current': full_indicators_15m.get('volume_current', 0),
                        'volume_ratio': full_indicators_15m.get('volume_ratio', 1.0)
                    }
                }
            },

            # Результаты предварительного анализа
            'preliminary_analysis': {
                'signal_type': signal_result['signal'],
                'pattern_detected': signal_result.get('pattern_type', 'NONE'),
                'higher_tf_trend': signal_result.get('higher_tf_trend', 'UNKNOWN'),
                'validation_score': signal_result.get('validation_score', '0/5'),
                'entry_reasons': signal_result.get('entry_reasons', []),
                'validation_reasons': signal_result.get('validation_reasons', [])
            }
        })

    def _parse_final_ai_analysis(self, ai_response: str, pair: str, base_signal: Dict) -> Optional[Dict]:
        """Парсинг финального анализа ИИ"""
        try:
            # Ищем JSON в ответе ИИ
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                try:
                    analysis = json.loads(json_match.group())

                    # Проверяем обязательные поля
                    if analysis.get('signal') not in ['LONG', 'SHORT', 'NO_SIGNAL']:
                        return None

                    # Обогащаем данными
                    analysis['pair'] = pair
                    analysis['timestamp'] = int(time.time())
                    analysis['raw_ai_response'] = ai_response

                    return analysis
                except json.JSONDecodeError:
                    pass

            # Если JSON не распарсился, пробуем извлечь основные поля
            signal_match = re.search(r'"signal":\s*"(LONG|SHORT|NO_SIGNAL)"', ai_response)
            confidence_match = re.search(r'"confidence":\s*(\d+)', ai_response)

            if signal_match:
                return {
                    'pair': pair,
                    'signal': signal_match.group(1),
                    'confidence': int(confidence_match.group(1)) if confidence_match else 0,
                    'analysis': 'Partial parsing from AI response',
                    'timestamp': int(time.time()),
                    'raw_ai_response': ai_response
                }

            return None

        except Exception as e:
            logger.error(f"Ошибка парсинга ИИ анализа для {pair}: {e}")
            return None


class InstructionBasedBot:
    """Главный класс бота согласно инструкции"""

    def __init__(self):
        self.analyzer = InstructionBasedAnalyzer()
        self.ai_selector = InstructionBasedAISelector()
        self.session_stats = {
            'start_time': time.time(),
            'total_pairs_scanned': 0,
            'signals_found': 0,
            'ai_selections': 0,
            'final_recommendations': 0
        }

    async def run_full_analysis_cycle(self) -> List[Dict]:
        """Полный цикл анализа согласно инструкции"""
        cycle_start = time.time()

        logger.info("=== ЗАПУСК ПОЛНОГО ЦИКЛА АНАЛИЗА ===")
        logger.info("Методология: 15m контекст + 5m вход → ИИ отбор → детальный анализ")

        try:
            # ЭТАП 1: Массовое сканирование с фильтрацией
            signals = await self.analyzer.mass_scan_markets()
            self.session_stats['signals_found'] = len(signals)

            if not signals:
                logger.warning("Сигналы не найдены")
                return []

            # ЭТАП 2: ИИ отбор топ пар
            selected_pairs = await self.ai_selector.ai_select_top_pairs(signals)
            self.session_stats['ai_selections'] = len(selected_pairs)

            if not selected_pairs:
                logger.warning("ИИ не отобрал пары")
                return []

            # ЭТАП 3: Детальный анализ с полными данными
            final_results = await self.ai_selector.detailed_analysis_with_full_data(selected_pairs)
            self.session_stats['final_recommendations'] = len(final_results)

            cycle_time = time.time() - cycle_start

            # Финальная статистика
            logger.info("=== ЗАВЕРШЕНИЕ ЦИКЛА АНАЛИЗА ===")
            logger.info(f"Время выполнения: {cycle_time:.2f}сек")
            logger.info(f"Найдено сигналов: {self.session_stats['signals_found']}")
            logger.info(f"ИИ отобрал: {self.session_stats['ai_selections']}")
            logger.info(f"Финальных рекомендаций: {self.session_stats['final_recommendations']}")

            return final_results

        except Exception as e:
            logger.error(f"Критическая ошибка цикла: {e}")
            return []

    async def continuous_monitoring(self, interval_minutes: int = 5):
        """Непрерывный мониторинг рынка"""
        logger.info(f"Запуск непрерывного мониторинга (интервал {interval_minutes}мин)")

        while True:
            try:
                results = await self.run_full_analysis_cycle()

                if results:
                    self.display_trading_signals(results)

                logger.info(f"Следующий цикл через {interval_minutes} минут")
                await asyncio.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                logger.info("Остановка по запросу пользователя")
                break
            except Exception as e:
                logger.error(f"Ошибка в цикле мониторинга: {e}")
                await asyncio.sleep(60)  # Пауза при ошибке

    def display_trading_signals(self, results: List[Dict]):
        """Отображение торговых сигналов"""
        print("\n" + "="*80)
        print("🎯 ТОРГОВЫЕ СИГНАЛЫ (ПО ИНСТРУКЦИИ)")
        print("="*80)

        for i, result in enumerate(results, 1):
            signal = result.get('signal', 'NO_SIGNAL')
            pair = result.get('pair', 'UNKNOWN')
            confidence = result.get('confidence', 0)
            pattern = result.get('pattern_used', 'NONE')

            print(f"\n{i}. {pair}")
            print(f"   Сигнал: {signal} ({confidence}%)")
            print(f"   Паттерн: {pattern}")

            if 'entry_price' in result:
                print(f"   Вход: {result['entry_price']}")
            if 'stop_loss' in result:
                print(f"   Стоп: {result['stop_loss']}")
            if 'take_profit' in result:
                print(f"   Тейк: {result['take_profit']}")
            if 'validation_checklist' in result:
                score = result['validation_checklist'].get('score', '0/5')
                print(f"   Валидация: {score}")

        print("\n" + "="*80)


async def main():
    """Главная функция"""
    try:
        # Проверка конфигурации
        if not config.validate():
            logger.error("Ошибка валидации конфигурации")
            return

        # Создание бота
        bot = InstructionBasedBot()

        logger.info("Скальпинг-бот по инструкции запущен")
        logger.info("Конфигурация: 15m контекст + 5m вход")

        # Запуск анализа
        choice = input("Выберите режим:\n1 - Одиночный анализ\n2 - Непрерывный мониторинг\nВвод: ")

        if choice == '2':
            await bot.continuous_monitoring()
        else:
            results = await bot.run_full_analysis_cycle()
            if results:
                bot.display_trading_signals(results)
            else:
                print("Торговых возможностей не найдено")

    except KeyboardInterrupt:
        logger.info("Остановка по запросу пользователя")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
    finally:
        # Очистка ресурсов
        await cleanup_http_client()
        logger.info("Ресурсы очищены. Завершение работы.")


if __name__ == "__main__":
    asyncio.run(main())