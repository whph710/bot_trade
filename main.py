import asyncio
import json
import logging
import time
import math
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

# ИМПОРТЫ
from func_async import get_klines_async, get_usdt_trading_pairs
from deepseek import deep_seek_selection, deep_seek_analysis, cleanup_http_client
from func_trade import detect_instruction_based_signals, calculate_indicators_by_instruction
from config import config

# Настройка логирования
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
class TradingSignal:
    """Расширенный торговый сигнал с полными данными"""
    pair: str
    signal_type: str  # 'LONG', 'SHORT', 'NO_SIGNAL'
    confidence: int
    entry_price: float
    pattern_type: str
    higher_tf_trend: str
    validation_score: str
    atr_current: float
    volume_ratio: float
    timestamp: int

    # Полные данные для ИИ
    candles_5m: List = None
    candles_15m: List = None
    indicators: Dict = None


class MarketAnalyzer:
    """Анализатор рынка с полными данными"""

    def __init__(self):
        self.session_start = time.time()
        logger.info("Скальпинговый анализатор запущен")

    def passes_basic_filters(self, symbol: str, candles: List) -> bool:
        """Базовые фильтры ликвидности"""
        if not candles or len(candles) < 50:
            return False

        # Оценка объема (упрощенная)
        recent_volumes = [float(c[5]) * float(c[4]) for c in candles[-24:]]
        avg_hourly_volume = sum(recent_volumes) * 12

        return avg_hourly_volume > config.trading.MIN_LIQUIDITY_VOLUME

    async def scan_pair(self, symbol: str) -> Optional[TradingSignal]:
        """Сканирование одной пары с сохранением полных данных"""
        try:
            # Получаем данные
            candles_5m = await get_klines_async(symbol, config.timeframe.ENTRY_TF,
                                                limit=config.timeframe.CANDLES_5M)
            candles_15m = await get_klines_async(symbol, config.timeframe.CONTEXT_TF,
                                                 limit=config.timeframe.CANDLES_15M)

            if not candles_5m or not candles_15m:
                return None

            # Проверяем порядок свечей
            if not validate_candles_order(candles_5m, f"{symbol}_5m"):
                return None
            if not validate_candles_order(candles_15m, f"{symbol}_15m"):
                return None

            # Базовые фильтры
            if not self.passes_basic_filters(symbol, candles_5m):
                return None

            # Анализ по инструкции
            signal_result = detect_instruction_based_signals(candles_5m, candles_15m)

            if signal_result['signal'] == 'NO_SIGNAL':
                return None

            entry_price = float(candles_5m[-1][4])
            confidence = int(signal_result['confidence'])

            if confidence < config.trading.MIN_CONFIDENCE:
                return None

            return TradingSignal(
                pair=symbol,
                signal_type=signal_result['signal'],
                confidence=confidence,
                entry_price=entry_price,
                pattern_type=signal_result.get('pattern_type', 'UNKNOWN'),
                higher_tf_trend=signal_result.get('higher_tf_trend', 'UNKNOWN'),
                validation_score=signal_result.get('validation_score', '0/5'),
                atr_current=signal_result.get('atr_current', 0.0),
                volume_ratio=signal_result.get('volume_ratio', 1.0),
                timestamp=int(time.time()),

                # СОХРАНЯЕМ ПОЛНЫЕ ДАННЫЕ
                candles_5m=candles_5m,
                candles_15m=candles_15m,
                indicators=clean_value(signal_result.get('indicators', {}))
            )

        except Exception as e:
            logger.error(f"Ошибка сканирования {symbol}: {e}")
            return None

    async def mass_scan(self) -> List[TradingSignal]:
        """Массовое сканирование"""
        start_time = time.time()
        logger.info("ЭТАП 1: Сканирование рынка")

        try:
            pairs = await get_usdt_trading_pairs()
            if not pairs:
                return []

            logger.info(f"Сканируем {len(pairs)} пар")
            signals = []

            # Обрабатываем батчами
            for i in range(0, len(pairs), config.processing.BATCH_SIZE):
                batch = pairs[i:i + config.processing.BATCH_SIZE]
                tasks = [self.scan_pair(pair) for pair in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, TradingSignal):
                        signals.append(result)

                processed = min(i + config.processing.BATCH_SIZE, len(pairs))
                logger.info(f"Обработано: {processed}/{len(pairs)} (найдено {len(signals)} сигналов)")

                if i + config.processing.BATCH_SIZE < len(pairs):
                    await asyncio.sleep(config.processing.BATCH_DELAY)

            # Сортируем по уверенности
            signals.sort(key=lambda x: x.confidence, reverse=True)

            execution_time = time.time() - start_time
            logger.info(f"ЭТАП 1: {len(signals)} сигналов за {execution_time:.2f}сек")

            return signals

        except Exception as e:
            logger.error(f"Ошибка сканирования: {e}")
            return []


class AISelector:
    """ИИ селектор с расширенными данными"""

    def __init__(self):
        self.selection_prompt = self._load_prompt(config.ai.SELECTION_PROMPT_FILE)
        self.analysis_prompt = self._load_prompt(config.ai.ANALYSIS_PROMPT_FILE)

    def _load_prompt(self, filename: str) -> str:
        """Загрузка промпта"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                logger.info(f"Загружен промпт {filename}")
                return content
        except FileNotFoundError:
            logger.error(f"Файл {filename} не найден")
            return "Ты эксперт-трейдер. Анализируй данные и давай рекомендации в JSON."

    def prepare_signals_for_ai_selection(self, signals: List[TradingSignal]) -> Dict[str, Any]:
        """Подготовка данных для ИИ отбора (ограниченные данные)"""
        prepared_signals = []

        for signal in signals:
            # Обрезаем данные согласно конфигу для отбора
            recent_5m = signal.candles_5m[-config.timeframe.CANDLES_FOR_AI_SELECTION_5M:] if signal.candles_5m else []
            recent_15m = signal.candles_15m[
                         -config.timeframe.CANDLES_FOR_AI_SELECTION_15M:] if signal.candles_15m else []

            # Обрезаем индикаторы
            indicators = signal.indicators or {}
            trimmed_indicators = {}

            for key, value in indicators.items():
                if isinstance(value, list) and len(value) > config.timeframe.CANDLES_FOR_AI_SELECTION_INDICATORS:
                    trimmed_indicators[key] = value[-config.timeframe.CANDLES_FOR_AI_SELECTION_INDICATORS:]
                else:
                    trimmed_indicators[key] = value

            signal_data = {
                'pair': signal.pair,
                'signal_type': signal.signal_type,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'pattern_type': signal.pattern_type,
                'higher_tf_trend': signal.higher_tf_trend,
                'validation_score': signal.validation_score,
                'atr_current': signal.atr_current,
                'volume_ratio': signal.volume_ratio,

                # Ограниченные свечные данные
                'recent_5m_candles': [
                    {
                        'timestamp': int(c[0]),
                        'open': float(c[1]),
                        'high': float(c[2]),
                        'low': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5])
                    } for c in recent_5m
                ],

                'context_15m_candles': [
                    {
                        'timestamp': int(c[0]),
                        'open': float(c[1]),
                        'high': float(c[2]),
                        'low': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5])
                    } for c in recent_15m
                ],

                # Расширенные индикаторы (последние 45 значений)
                'indicators': {
                    'rsi_current': trimmed_indicators.get('rsi', [])[-45:] if trimmed_indicators.get('rsi') else [],
                    'ema_alignment': self._get_ema_alignment(trimmed_indicators),
                    'ema5': trimmed_indicators.get('ema5', [])[-45:] if trimmed_indicators.get('ema5') else [],
                    'ema8': trimmed_indicators.get('ema8', [])[-45:] if trimmed_indicators.get('ema8') else [],
                    'ema20': trimmed_indicators.get('ema20', [])[-45:] if trimmed_indicators.get('ema20') else [],
                    'macd_signal': self._get_macd_status(trimmed_indicators),
                    'macd_line': trimmed_indicators.get('macd_line', [])[-45:] if trimmed_indicators.get(
                        'macd_line') else [],
                    'macd_histogram': trimmed_indicators.get('macd_histogram', [])[-45:] if trimmed_indicators.get(
                        'macd_histogram') else [],
                    'atr': trimmed_indicators.get('atr', [])[-45:] if trimmed_indicators.get('atr') else [],
                    'volume_sma': trimmed_indicators.get('volume_sma', [])[-45:] if trimmed_indicators.get(
                        'volume_sma') else [],
                    'volume_status': 'high' if signal.volume_ratio > 1.5 else 'normal'
                }
            }

            prepared_signals.append(signal_data)

        return {
            'method': 'multi_timeframe_scalping',
            'signals_count': len(prepared_signals),
            'timestamp': int(time.time()),
            'signals': prepared_signals
        }

    def _get_ema_alignment(self, indicators: Dict) -> str:
        """Определение выравнивания EMA"""
        ema5 = indicators.get('ema5', [])
        ema8 = indicators.get('ema8', [])
        ema20 = indicators.get('ema20', [])

        if not all([ema5, ema8, ema20]):
            return 'unknown'

        current_5 = ema5[-1]
        current_8 = ema8[-1]
        current_20 = ema20[-1]

        if current_5 > current_8 > current_20:
            return 'bullish'
        elif current_5 < current_8 < current_20:
            return 'bearish'
        else:
            return 'mixed'

    def _get_macd_status(self, indicators: Dict) -> str:
        """Получение статуса MACD"""
        macd_histogram = indicators.get('macd_histogram', [])
        if not macd_histogram:
            return 'unknown'

        current = macd_histogram[-1]
        if current > 0:
            return 'bullish'
        elif current < 0:
            return 'bearish'
        else:
            return 'neutral'

    async def select_best_pairs(self, signals: List[TradingSignal]) -> List[str]:
        """ИИ отбор лучших пар с расширенными данными"""
        if not self.selection_prompt or not signals:
            return []

        logger.info(f"ЭТАП 2: ИИ отбор из {len(signals)} сигналов")

        try:
            top_signals = signals[:config.ai.MAX_PAIRS_TO_AI]
            ai_data = self.prepare_signals_for_ai_selection(top_signals)

            message = f"""{self.selection_prompt}

=== ДАННЫЕ ДЛЯ АНАЛИЗА ===
МЕТОД: {config.timeframe.CONTEXT_TF}m контекст + {config.timeframe.ENTRY_TF}m вход
КОЛИЧЕСТВО СИГНАЛОВ: {len(top_signals)}
ДАННЫЕ ПО ИНДИКАТОРАМ: последние {config.timeframe.CANDLES_FOR_AI_SELECTION_INDICATORS} значений
СВЕЧИ 5M: последние {config.timeframe.CANDLES_FOR_AI_SELECTION_5M} свечей
СВЕЧИ 15M: последние {config.timeframe.CANDLES_FOR_AI_SELECTION_15M} свечей

{json.dumps(ai_data, indent=2, ensure_ascii=False)}

Выбери максимум {config.ai.MAX_SELECTED_PAIRS} лучших пар.
Верни JSON: {{"pairs": ["BTCUSDT", "ETHUSDT"]}}"""

            ai_response = await deep_seek_selection(message)
            if not ai_response:
                return []

            selected_pairs = self._parse_ai_response(ai_response)
            logger.info(f"ЭТАП 2: ИИ выбрал {len(selected_pairs)} пар")
            return selected_pairs

        except Exception as e:
            logger.error(f"Ошибка ИИ отбора: {e}")
            return []

    def _parse_ai_response(self, response: str) -> List[str]:
        """Парсинг ответа ИИ"""
        try:
            json_match = re.search(r'\{[^}]*"pairs"[^}]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                pairs = data.get('pairs', [])
                return pairs
            return []
        except Exception as e:
            logger.error(f"Ошибка парсинга ответа ИИ: {e}")
            return []

    async def detailed_analysis(self, pair: str) -> Optional[str]:
        """Детальный анализ пары с полными данными"""
        logger.info(f"ЭТАП 3: Детальный анализ {pair}")

        try:
            # Получаем свежие данные для анализа (МАКСИМАЛЬНЫЕ объемы)
            candles_5m = await get_klines_async(pair, config.timeframe.ENTRY_TF,
                                                limit=config.timeframe.CANDLES_FOR_AI_ANALYSIS_5M + 50)
            candles_15m = await get_klines_async(pair, config.timeframe.CONTEXT_TF,
                                                 limit=config.timeframe.CANDLES_FOR_AI_ANALYSIS_15M + 30)

            if not candles_5m or not candles_15m:
                return None

            # Проверяем порядок
            if not validate_candles_order(candles_5m, f"{pair}_5m_analysis"):
                return None
            if not validate_candles_order(candles_15m, f"{pair}_15m_analysis"):
                return None

            # Полный анализ с большим количеством данных
            indicators = calculate_indicators_by_instruction(candles_5m)
            signal_analysis = detect_instruction_based_signals(candles_5m, candles_15m)

            current_price = float(candles_5m[-1][4])

            # Обрезаем данные до нужного размера для анализа
            analysis_candles_5m = candles_5m[-config.timeframe.CANDLES_FOR_AI_ANALYSIS_5M:]
            analysis_candles_15m = candles_15m[-config.timeframe.CANDLES_FOR_AI_ANALYSIS_15M:]

            # Обрезаем индикаторы до нужного размера
            analysis_indicators = {}
            for key, value in indicators.items():
                if isinstance(value, list) and len(value) > config.timeframe.CANDLES_FOR_AI_ANALYSIS_INDICATORS:
                    analysis_indicators[key] = value[-config.timeframe.CANDLES_FOR_AI_ANALYSIS_INDICATORS:]
                else:
                    analysis_indicators[key] = value

            # Подготавливаем ПОЛНЫЕ данные для детального анализа
            analysis_data = {
                'pair': pair,
                'current_price': current_price,
                'timestamp': int(time.time()),

                'signal_data': {
                    'signal': signal_analysis.get('signal', 'NO_SIGNAL'),
                    'pattern_type': signal_analysis.get('pattern_type', 'NONE'),
                    'confidence': signal_analysis.get('confidence', 0),
                    'validation_score': signal_analysis.get('validation_score', '0/5'),
                    'higher_tf_trend': signal_analysis.get('higher_tf_trend', 'UNKNOWN')
                },

                # ПОЛНЫЕ свечные данные (200 и 100 свечей)
                'recent_candles_5m': [
                    {
                        'timestamp': int(c[0]),
                        'open': float(c[1]),
                        'high': float(c[2]),
                        'low': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5])
                    } for c in analysis_candles_5m
                ],

                'context_candles_15m': [
                    {
                        'timestamp': int(c[0]),
                        'open': float(c[1]),
                        'high': float(c[2]),
                        'low': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5])
                    } for c in analysis_candles_15m
                ],

                # ПОЛНЫЕ массивы индикаторов (последние 200 значений)
                'key_indicators': {
                    'rsi': analysis_indicators.get('rsi', []),
                    'atr': analysis_indicators.get('atr', []),
                    'volume_ratio': [analysis_indicators.get('volume_ratio', 1.0)] * len(
                        analysis_indicators.get('rsi', [1])),
                    'ema5': analysis_indicators.get('ema5', []),
                    'ema8': analysis_indicators.get('ema8', []),
                    'ema20': analysis_indicators.get('ema20', []),
                    'macd_line': analysis_indicators.get('macd_line', []),
                    'macd_signal_line': analysis_indicators.get('macd_signal', []),
                    'macd_histogram': analysis_indicators.get('macd_histogram', []),
                    'bb_upper': analysis_indicators.get('bb_upper', []),
                    'bb_middle': analysis_indicators.get('bb_middle', []),
                    'bb_lower': analysis_indicators.get('bb_lower', []),
                    'volume_sma': analysis_indicators.get('volume_sma', [])
                },

                'levels': self._extract_key_levels(analysis_candles_5m, analysis_candles_15m, current_price)
            }

            message = f"""{self.analysis_prompt}

=== ДЕТАЛЬНЫЙ АНАЛИЗ ===
ПАРА: {pair}
ЦЕНА: {current_price}
МЕТОД: {config.timeframe.CONTEXT_TF}m контекст + {config.timeframe.ENTRY_TF}m вход
ДАННЫЕ: {len(analysis_candles_5m)} свечей 5m, {len(analysis_candles_15m)} свечей 15m
ИНДИКАТОРЫ: {len(analysis_indicators.get('rsi', []))} значений по каждому индикатору

{json.dumps(analysis_data, indent=2, ensure_ascii=False)}

Проанализируй ПОЛНЫЕ данные и дай точные торговые рекомендации в JSON формате.
Определи точные уровни входа, стоп-лосса и тейк-профита на основе всех предоставленных данных."""

            analysis_result = await deep_seek_analysis(message)

            if analysis_result:
                self._save_analysis(pair, analysis_result, analysis_data)
                logger.info(f"Анализ {pair} завершен (обработано {len(analysis_candles_5m)} свечей 5m)")
                return analysis_result

        except Exception as e:
            logger.error(f"Ошибка анализа {pair}: {e}")
            return None

    def _extract_key_levels(self, candles_5m: List, candles_15m: List, current_price: float) -> Dict:
        """Извлечение ключевых уровней с расширенным анализом"""
        try:
            # Используем больше данных для анализа уровней
            extended_5m = candles_5m[-100:] if len(candles_5m) >= 100 else candles_5m
            extended_15m = candles_15m[-50:] if len(candles_15m) >= 50 else candles_15m

            # Находим swing highs/lows на 5m
            swing_highs_5m = []
            swing_lows_5m = []

            for i in range(3, len(extended_5m) - 3):
                high = float(extended_5m[i][2])
                low = float(extended_5m[i][3])

                # Swing high (более строгие условия)
                if (high >= float(extended_5m[i - 1][2]) and high >= float(extended_5m[i - 2][2]) and
                        high >= float(extended_5m[i - 3][2]) and high >= float(extended_5m[i + 1][2]) and
                        high >= float(extended_5m[i + 2][2]) and high >= float(extended_5m[i + 3][2])):
                    swing_highs_5m.append(high)

                # Swing low (более строгие условия)
                if (low <= float(extended_5m[i - 1][3]) and low <= float(extended_5m[i - 2][3]) and
                        low <= float(extended_5m[i - 3][3]) and low <= float(extended_5m[i + 1][3]) and
                        low <= float(extended_5m[i + 2][3]) and low <= float(extended_5m[i + 3][3])):
                    swing_lows_5m.append(low)

            # Находим swing highs/lows на 15m (более сильные уровни)
            swing_highs_15m = []
            swing_lows_15m = []

            for i in range(2, len(extended_15m) - 2):
                high = float(extended_15m[i][2])
                low = float(extended_15m[i][3])

                # Swing high на 15m
                if (high >= float(extended_15m[i - 1][2]) and high >= float(extended_15m[i - 2][2]) and
                        high >= float(extended_15m[i + 1][2]) and high >= float(extended_15m[i + 2][2])):
                    swing_highs_15m.append(high)

                # Swing low на 15m
                if (low <= float(extended_15m[i - 1][3]) and low <= float(extended_15m[i - 2][3]) and
                        low <= float(extended_15m[i + 1][3]) and low <= float(extended_15m[i + 2][3])):
                    swing_lows_15m.append(low)

            # Объединяем уровни с разными весами
            all_resistance = swing_highs_5m + [h * 1.1 for h in swing_highs_15m]  # 15m уровни важнее
            all_support = swing_lows_5m + [l * 0.9 for l in swing_lows_15m]  # 15m уровни важнее

            # Фильтруем близкие к текущей цене
            nearby_resistance = [level for level in all_resistance
                                 if level > current_price and (level - current_price) / current_price < 0.05]
            nearby_support = [level for level in all_support
                              if level < current_price and (current_price - level) / current_price < 0.05]

            # Добавляем психологические уровни
            price_str = str(int(current_price))
            if len(price_str) >= 3:
                # Круглые числа
                round_levels = []
                base = int(price_str[:-2]) * 100
                for offset in [-200, -100, 0, 100, 200]:
                    round_level = base + offset
                    if abs(round_level - current_price) / current_price < 0.03:
                        if round_level > current_price:
                            nearby_resistance.append(float(round_level))
                        else:
                            nearby_support.append(float(round_level))

            return {
                'resistance_levels': sorted(set(nearby_resistance))[:5],  # Топ 5
                'support_levels': sorted(set(nearby_support), reverse=True)[:5],  # Топ 5
                'range_high': max([float(c[2]) for c in extended_5m]),
                'range_low': min([float(c[3]) for c in extended_5m]),
                'strong_resistance_15m': sorted(set(swing_highs_15m))[-3:] if swing_highs_15m else [],
                'strong_support_15m': sorted(set(swing_lows_15m))[:3] if swing_lows_15m else []
            }

        except Exception as e:
            logger.error(f"Ошибка извлечения уровней: {e}")
            return {}

    def _save_analysis(self, pair: str, analysis: str, data: Dict):
        """Сохранение анализа с дополнительной информацией"""
        try:
            with open(config.system.ANALYSIS_LOG_FILE, 'a', encoding=config.system.ENCODING) as f:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"АНАЛИЗ: {pair}\n")
                f.write(f"ВРЕМЯ: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"ЦЕНА: {data.get('current_price', 0)}\n")
                f.write(f"СИГНАЛ: {data.get('signal_data', {}).get('signal', 'N/A')}\n")
                f.write(f"ПАТТЕРН: {data.get('signal_data', {}).get('pattern_type', 'N/A')}\n")
                f.write(f"УВЕРЕННОСТЬ: {data.get('signal_data', {}).get('confidence', 0)}%\n")
                f.write(f"СВЕЧИ 5M: {len(data.get('recent_candles_5m', []))}\n")
                f.write(f"СВЕЧИ 15M: {len(data.get('context_candles_15m', []))}\n")
                f.write(f"ИНДИКАТОРЫ: {len(data.get('key_indicators', {}).get('rsi', []))} значений\n")
                f.write(f"{'=' * 40}\n")
                f.write(f"{analysis}\n")
                f.write(f"{'=' * 80}\n")
        except Exception as e:
            logger.error(f"Ошибка сохранения анализа: {e}")


async def main():
    """Главная функция"""
    logger.info("🚀 СКАЛЬПИНГОВЫЙ БОТ ЗАПУЩЕН (РАСШИРЕННАЯ ВЕРСИЯ)")
    logger.info(f"📊 Метод: {config.timeframe.CONTEXT_TF}m + {config.timeframe.ENTRY_TF}m")
    logger.info(
        f"📈 Данные для отбора: {config.timeframe.CANDLES_FOR_AI_SELECTION_5M}+{config.timeframe.CANDLES_FOR_AI_SELECTION_15M} свечей")
    logger.info(
        f"📊 Данные для анализа: {config.timeframe.CANDLES_FOR_AI_ANALYSIS_5M}+{config.timeframe.CANDLES_FOR_AI_ANALYSIS_15M} свечей")

    analyzer = MarketAnalyzer()
    ai_selector = AISelector()

    try:
        # ЭТАП 1: Сканирование с сохранением полных данных
        signals = await analyzer.mass_scan()

        if not signals:
            logger.info("❌ Сигналы не найдены")
            return

        logger.info(f"✅ Найдено {len(signals)} сигналов")
        for signal in signals[:5]:
            logger.info(f"   📈 {signal.pair}: {signal.pattern_type} ({signal.confidence}%)")

        # ЭТАП 2: ИИ отбор с расширенными данными
        selected_pairs = await ai_selector.select_best_pairs(signals)

        if not selected_pairs:
            logger.info("❌ ИИ не выбрал пары")
            return

        logger.info(f"🎯 Выбрано: {selected_pairs}")

        # ЭТАП 3: Детальный анализ с полными данными
        successful_analyses = 0

        for pair in selected_pairs:
            analysis = await ai_selector.detailed_analysis(pair)
            if analysis:
                successful_analyses += 1
                logger.info(f"✅ {pair} - полный анализ завершен")
            else:
                logger.error(f"❌ {pair} - ошибка анализа")

            await asyncio.sleep(1)

        # ИТОГИ
        logger.info(f"\n{'=' * 60}")
        logger.info(f"🏆 АНАЛИЗ ЗАВЕРШЕН (РАСШИРЕННАЯ ВЕРСИЯ)")
        logger.info(f"📈 Найдено сигналов: {len(signals)}")
        logger.info(f"🤖 ИИ отобрал: {len(selected_pairs)}")
        logger.info(f"✅ Успешных анализов: {successful_analyses}")
        logger.info(f"📊 Данные отбора: {config.timeframe.CANDLES_FOR_AI_SELECTION_INDICATORS} индикаторов")
        logger.info(f"📈 Данные анализа: {config.timeframe.CANDLES_FOR_AI_ANALYSIS_INDICATORS} индикаторов")
        logger.info(f"💾 Результаты: {config.system.ANALYSIS_LOG_FILE}")
        logger.info(f"{'=' * 60}")

        await cleanup_http_client()

    except KeyboardInterrupt:
        logger.info("⏹️ Остановка по запросу пользователя")
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 СКАЛЬПИНГОВЫЙ БОТ (РАСШИРЕННАЯ ВЕРСИЯ)")
    logger.info(f"📊 {config.timeframe.CONTEXT_TF}m + {config.timeframe.ENTRY_TF}m")
    logger.info(f"🎯 R:R {config.trading.DEFAULT_RISK_REWARD}:1")
    logger.info(f"📈 Отбор: {config.timeframe.CANDLES_FOR_AI_SELECTION_INDICATORS} значений")
    logger.info(f"📊 Анализ: {config.timeframe.CANDLES_FOR_AI_ANALYSIS_INDICATORS} значений")
    logger.info("=" * 60)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹️ Программа остановлена")
    except Exception as e:
        logger.error(f"💥 Фатальная ошибка: {e}")
    finally:
        logger.info("🏁 Работа завершена")