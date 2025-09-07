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
    """Упрощенный торговый сигнал"""
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

    # Данные для ИИ (упрощенные)
    candles_5m: List = None
    candles_15m: List = None
    indicators: Dict = None


class MarketAnalyzer:
    """Упрощенный анализатор рынка"""

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
        """Сканирование одной пары"""
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

                # Упрощенные данные для ИИ
                candles_5m=candles_5m[-config.timeframe.CANDLES_FOR_AI_SELECTION:],
                candles_15m=candles_15m[-config.timeframe.CANDLES_FOR_CONTEXT:],
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
    """Упрощенный ИИ селектор"""

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

    def prepare_signals_for_ai(self, signals: List[TradingSignal]) -> Dict[str, Any]:
        """Подготовка данных для ИИ (упрощенная)"""
        prepared_signals = []

        for signal in signals:
            # Берем только необходимые данные
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

                # Упрощенные свечные данные (только последние)
                'recent_5m_candles': [
                    {
                        'timestamp': int(c[0]),
                        'open': float(c[1]),
                        'high': float(c[2]),
                        'low': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5])
                    } for c in signal.candles_5m[-20:] if signal.candles_5m
                ],

                'context_15m_candles': [
                    {
                        'timestamp': int(c[0]),
                        'open': float(c[1]),
                        'high': float(c[2]),
                        'low': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5])
                    } for c in signal.candles_15m[-15:] if signal.candles_15m
                ],

                # Основные индикаторы
                'indicators': {
                    'rsi_current': signal.indicators.get('rsi_current', 50),
                    'ema_alignment': self._get_ema_alignment(signal.indicators),
                    'macd_signal': self._get_macd_status(signal.indicators),
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
        """ИИ отбор лучших пар"""
        if not self.selection_prompt or not signals:
            return []

        logger.info(f"ЭТАП 2: ИИ отбор из {len(signals)} сигналов")

        try:
            top_signals = signals[:config.ai.MAX_PAIRS_TO_AI]
            ai_data = self.prepare_signals_for_ai(top_signals)

            message = f"""{self.selection_prompt}

=== ДАННЫЕ ДЛЯ АНАЛИЗА ===
МЕТОД: {config.timeframe.CONTEXT_TF}m контекст + {config.timeframe.ENTRY_TF}m вход
КОЛИЧЕСТВО СИГНАЛОВ: {len(top_signals)}

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
        """Детальный анализ пары"""
        logger.info(f"ЭТАП 3: Детальный анализ {pair}")

        try:
            # Получаем свежие данные для анализа
            candles_5m = await get_klines_async(pair, config.timeframe.ENTRY_TF,
                                                limit=config.timeframe.CANDLES_FOR_AI_ANALYSIS)
            candles_15m = await get_klines_async(pair, config.timeframe.CONTEXT_TF, limit=80)

            if not candles_5m or not candles_15m:
                return None

            # Проверяем порядок
            if not validate_candles_order(candles_5m, f"{pair}_5m_analysis"):
                return None
            if not validate_candles_order(candles_15m, f"{pair}_15m_analysis"):
                return None

            # Полный анализ
            indicators = calculate_indicators_by_instruction(candles_5m)
            signal_analysis = detect_instruction_based_signals(candles_5m, candles_15m)

            current_price = float(candles_5m[-1][4])

            # Подготавливаем упрощенные данные
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

                'recent_candles_5m': [
                    {
                        'timestamp': int(c[0]),
                        'open': float(c[1]),
                        'high': float(c[2]),
                        'low': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5])
                    } for c in candles_5m[-30:]
                ],

                'context_candles_15m': [
                    {
                        'timestamp': int(c[0]),
                        'open': float(c[1]),
                        'high': float(c[2]),
                        'low': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5])
                    } for c in candles_15m[-20:]
                ],

                'key_indicators': {
                    'rsi_current': indicators.get('rsi_current', 50),
                    'atr_current': indicators.get('atr_current', 0),
                    'volume_ratio': indicators.get('volume_ratio', 1.0),
                    'ema5': indicators.get('ema5', [])[-1] if indicators.get('ema5') else 0,
                    'ema8': indicators.get('ema8', [])[-1] if indicators.get('ema8') else 0,
                    'ema20': indicators.get('ema20', [])[-1] if indicators.get('ema20') else 0
                },

                'levels': self._extract_key_levels(candles_5m, candles_15m, current_price)
            }

            message = f"""{self.analysis_prompt}

=== ДЕТАЛЬНЫЙ АНАЛИЗ ===
ПАРА: {pair}
ЦЕНА: {current_price}
МЕТОД: {config.timeframe.CONTEXT_TF}m контекст + {config.timeframe.ENTRY_TF}m вход

{json.dumps(analysis_data, indent=2, ensure_ascii=False)}

Проанализируй и дай торговые рекомендации в JSON формате.
Определи точные уровни входа, стоп-лосса и тейк-профита."""

            analysis_result = await deep_seek_analysis(message)

            if analysis_result:
                self._save_analysis(pair, analysis_result, analysis_data)
                logger.info(f"Анализ {pair} завершен")
                return analysis_result

        except Exception as e:
            logger.error(f"Ошибка анализа {pair}: {e}")
            return None

    def _extract_key_levels(self, candles_5m: List, candles_15m: List, current_price: float) -> Dict:
        """Извлечение ключевых уровней (упрощенное)"""
        try:
            # Последние 50 свечей 5m для локальных уровней
            recent_5m = candles_5m[-50:] if len(candles_5m) >= 50 else candles_5m
            recent_15m = candles_15m[-30:] if len(candles_15m) >= 30 else candles_15m

            # Находим swing highs/lows
            swing_highs = []
            swing_lows = []

            for i in range(2, len(recent_5m) - 2):
                high = float(recent_5m[i][2])
                low = float(recent_5m[i][3])

                # Swing high
                if (high >= float(recent_5m[i - 1][2]) and high >= float(recent_5m[i - 2][2]) and
                        high >= float(recent_5m[i + 1][2]) and high >= float(recent_5m[i + 2][2])):
                    swing_highs.append(high)

                # Swing low
                if (low <= float(recent_5m[i - 1][3]) and low <= float(recent_5m[i - 2][3]) and
                        low <= float(recent_5m[i + 1][3]) and low <= float(recent_5m[i + 2][3])):
                    swing_lows.append(low)

            # Фильтруем близкие к текущей цене
            nearby_resistance = [level for level in swing_highs
                                 if level > current_price and (level - current_price) / current_price < 0.03]
            nearby_support = [level for level in swing_lows
                              if level < current_price and (current_price - level) / current_price < 0.03]

            return {
                'resistance_levels': sorted(nearby_resistance)[:3],
                'support_levels': sorted(nearby_support, reverse=True)[:3],
                'range_high': max([float(c[2]) for c in recent_5m]),
                'range_low': min([float(c[3]) for c in recent_5m])
            }

        except Exception as e:
            logger.error(f"Ошибка извлечения уровней: {e}")
            return {}

    def _save_analysis(self, pair: str, analysis: str, data: Dict):
        """Сохранение анализа"""
        try:
            with open(config.system.ANALYSIS_LOG_FILE, 'a', encoding=config.system.ENCODING) as f:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"АНАЛИЗ: {pair}\n")
                f.write(f"ВРЕМЯ: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"ЦЕНА: {data.get('current_price', 0)}\n")
                f.write(f"СИГНАЛ: {data.get('signal_data', {}).get('signal', 'N/A')}\n")
                f.write(f"{'=' * 40}\n")
                f.write(f"{analysis}\n")
                f.write(f"{'=' * 80}\n")
        except Exception as e:
            logger.error(f"Ошибка сохранения анализа: {e}")


async def main():
    """Главная функция"""
    logger.info("🚀 СКАЛЬПИНГОВЫЙ БОТ ЗАПУЩЕН")
    logger.info(f"📊 Метод: {config.timeframe.CONTEXT_TF}m + {config.timeframe.ENTRY_TF}m")

    analyzer = MarketAnalyzer()
    ai_selector = AISelector()

    try:
        # ЭТАП 1: Сканирование
        signals = await analyzer.mass_scan()

        if not signals:
            logger.info("❌ Сигналы не найдены")
            return

        logger.info(f"✅ Найдено {len(signals)} сигналов")
        for signal in signals[:5]:
            logger.info(f"   📈 {signal.pair}: {signal.pattern_type} ({signal.confidence}%)")

        # ЭТАП 2: ИИ отбор
        selected_pairs = await ai_selector.select_best_pairs(signals)

        if not selected_pairs:
            logger.info("❌ ИИ не выбрал пары")
            return

        logger.info(f"🎯 Выбрано: {selected_pairs}")

        # ЭТАП 3: Детальный анализ
        successful_analyses = 0

        for pair in selected_pairs:
            analysis = await ai_selector.detailed_analysis(pair)
            if analysis:
                successful_analyses += 1
                logger.info(f"✅ {pair} - анализ завершен")
            else:
                logger.error(f"❌ {pair} - ошибка анализа")

            await asyncio.sleep(1)

        # ИТОГИ
        logger.info(f"\n{'=' * 60}")
        logger.info(f"🏆 АНАЛИЗ ЗАВЕРШЕН")
        logger.info(f"📈 Найдено сигналов: {len(signals)}")
        logger.info(f"🤖 ИИ отобрал: {len(selected_pairs)}")
        logger.info(f"✅ Успешных анализов: {successful_analyses}")
        logger.info(f"💾 Результаты: {config.system.ANALYSIS_LOG_FILE}")
        logger.info(f"{'=' * 60}")

        await cleanup_http_client()

    except KeyboardInterrupt:
        logger.info("⏹️ Остановка по запросу пользователя")
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 СКАЛЬПИНГОВЫЙ БОТ")
    logger.info(f"📊 {config.timeframe.CONTEXT_TF}m + {config.timeframe.ENTRY_TF}m")
    logger.info(f"🎯 R:R {config.trading.DEFAULT_RISK_REWARD}:1")
    logger.info("=" * 60)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹️ Программа остановлена")
    except Exception as e:
        logger.error(f"💥 Фатальная ошибка: {e}")
    finally:
        logger.info("🏁 Работа завершена")