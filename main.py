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

# НОВЫЙ ИМПОРТ с индикаторами по инструкции
from func_trade import detect_instruction_based_signals, calculate_indicators_by_instruction

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

# КОНФИГУРАЦИЯ ПО ИНСТРУКЦИИ
SCALPING_CONFIG = {
    # Таймфреймы согласно иерархии
    'context_tf': '15',  # 15m для контекста/фильтра
    'entry_tf': '5',  # 5m для точного входа

    # Количество свечей
    'candles_15m': 100,  # Для контекста
    'candles_5m': 200,  # Для анализа входа
    'candles_for_ai_selection': 30,  # Для первичного отбора ИИ
    'candles_for_ai_analysis': 100,  # Для детального анализа

    # Параметры обработки
    'batch_size': 40,
    'min_confidence': 70,
    'max_pairs_to_ai': 8,

    # Фильтры по инструкции
    'min_liquidity_volume': 50_000_000,  # Минимум $50M объем
    'max_spread_percent': 0.15,  # Максимум 0.15% спред
}


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
    validation_score: str  # "3/4" чек-пунктов
    atr_current: float  # Текущий ATR для волатильности
    volume_ratio: float  # Соотношение объемов

    # Для ИИ
    candles_5m: List = None  # 5m свечи для входа
    candles_15m: List = None  # 15m свечи для контекста
    indicators_data: Dict = None


class InstructionBasedAnalyzer:
    """Анализатор согласно инструкции: 15m контекст + 5m вход"""

    def __init__(self):
        self.session_start = time.time()
        logger.info("🎯 Анализатор по инструкции запущен (15m+5m)")

    def passes_liquidity_filter(self, symbol: str, candles: List) -> bool:
        """Фильтр ликвидности согласно инструкции"""
        if not candles:
            return False

        # Примерная оценка объема (последние 24 свечи 5m ≈ 2 часа)
        recent_volumes = [float(c[5]) * float(c[4]) for c in candles[-24:]]  # Объем в USD
        avg_hourly_volume = sum(recent_volumes) * 12  # Приблизительно за 24ч

        return avg_hourly_volume > SCALPING_CONFIG['min_liquidity_volume']

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
        return spread_estimate < SCALPING_CONFIG['max_spread_percent']

    async def quick_scan_pair(self, symbol: str) -> Optional[InstructionBasedSignal]:
        """Быстрое сканирование пары согласно инструкции"""
        try:
            # Получаем данные для мультитаймфреймного анализа
            candles_5m = await get_klines_async(symbol, SCALPING_CONFIG['entry_tf'],
                                                limit=SCALPING_CONFIG['candles_5m'])
            candles_15m = await get_klines_async(symbol, SCALPING_CONFIG['context_tf'],
                                                 limit=SCALPING_CONFIG['candles_15m'])

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

            if math.isnan(entry_price) or confidence < SCALPING_CONFIG['min_confidence']:
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
                validation_score=signal_result.get('validation_score', '0/4'),
                atr_current=signal_result.get('atr_current', 0.0),
                volume_ratio=signal_result.get('volume_ratio', 1.0),

                # Данные для ИИ
                candles_5m=candles_5m[-SCALPING_CONFIG['candles_for_ai_selection']:],
                candles_15m=candles_15m[-20:],  # Последние 20 свечей 15m для контекста
                indicators_data=clean_value(signal_result.get('indicators', {}))
            )

        except Exception as e:
            logger.error(f"❌ Ошибка сканирования {symbol}: {e}")
            return None

    async def mass_scan_markets(self) -> List[InstructionBasedSignal]:
        """Массовое сканирование с фильтрацией согласно инструкции"""
        start_time = time.time()
        logger.info("🔍 ЭТАП 1: Сканирование с фильтрами по инструкции")

        try:
            pairs = await get_usdt_trading_pairs()
            if not pairs:
                return []

            logger.info(f"📊 Сканируем {len(pairs)} пар (15m контекст + 5m вход)")

            promising_signals = []

            # Обрабатываем батчами
            for i in range(0, len(pairs), SCALPING_CONFIG['batch_size']):
                batch = pairs[i:i + SCALPING_CONFIG['batch_size']]
                tasks = [self.quick_scan_pair(pair) for pair in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, InstructionBasedSignal):
                        promising_signals.append(result)

                processed = min(i + SCALPING_CONFIG['batch_size'], len(pairs))
                logger.info(f"⏳ Обработано: {processed}/{len(pairs)}")

                if i + SCALPING_CONFIG['batch_size'] < len(pairs):
                    await asyncio.sleep(0.1)

            # Сортируем по уверенности
            promising_signals.sort(key=lambda x: x.confidence, reverse=True)

            execution_time = time.time() - start_time
            logger.info(f"✅ ЭТАП 1: {len(promising_signals)} сигналов за {execution_time:.2f}сек")

            return promising_signals

        except Exception as e:
            logger.error(f"❌ Ошибка сканирования: {e}")
            return []


class InstructionBasedAISelector:
    """ИИ селектор согласно инструкции"""

    def __init__(self):
        self.selection_prompt = self._load_prompt('prompt2.txt')
        self.analysis_prompt = self._load_prompt('prompt.txt')

    def _load_prompt(self, filename: str) -> str:
        """Загрузка промпта"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error(f"❌ Файл {filename} не найден")
            return ""

    def _prepare_signals_for_ai(self, signals: List[InstructionBasedSignal]) -> Dict[str, Any]:
        """Подготовка данных для ИИ согласно инструкции"""
        prepared_data = []

        for signal in signals:
            # Свечи 5m для анализа входа
            recent_5m = signal.candles_5m[-30:] if signal.candles_5m else []
            # Свечи 15m для контекста
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

                # Мультитаймфреймные данные
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

                # Индикаторы согласно инструкции
                'technical_indicators': safe_json_serialize({
                    'ema_system': {
                        'ema5': signal.indicators_data.get('ema5', [])[-20:],
                        'ema8': signal.indicators_data.get('ema8', [])[-20:],
                        'ema20': signal.indicators_data.get('ema20', [])[-20:]
                    },
                    'momentum': {
                        'rsi9': signal.indicators_data.get('rsi', [])[-20:],
                        'rsi_current': signal.indicators_data.get('rsi_current', 50),
                        'macd_line': signal.indicators_data.get('macd_line', [])[-20:],
                        'macd_signal': signal.indicators_data.get('macd_signal', [])[-20:],
                        'macd_histogram': signal.indicators_data.get('macd_histogram', [])[-20:]
                    },
                    'volatility': {
                        'atr14': signal.indicators_data.get('atr', [])[-20:],
                        'atr_current': signal.indicators_data.get('atr_current', 0),
                        'atr_mean': signal.indicators_data.get('atr_mean', 0)
                    },
                    'bollinger_bands': {
                        'upper': signal.indicators_data.get('bb_upper', [])[-20:],
                        'middle': signal.indicators_data.get('bb_middle', [])[-20:],
                        'lower': signal.indicators_data.get('bb_lower', [])[-20:]
                    },
                    'volume': {
                        'volume_sma20': signal.indicators_data.get('volume_sma', [])[-20:],
                        'current_volume': signal.indicators_data.get('volume_current', 0),
                        'volume_ratio': signal.indicators_data.get('volume_ratio', 1.0)
                    }
                })
            }

            prepared_data.append(signal_data)

        return {
            'analysis_method': 'multi_timeframe_instruction_based',
            'context_tf': '15m',
            'entry_tf': '5m',
            'signals_count': len(prepared_data),
            'timestamp': int(time.time()),
            'signals': prepared_data
        }

    async def select_best_pairs(self, signals: List[InstructionBasedSignal]) -> List[str]:
        """ИИ отбор лучших пар согласно инструкции"""
        if not self.selection_prompt or not signals:
            return []

        logger.info(f"🤖 ЭТАП 2: ИИ отбор из {len(signals)} сигналов")

        try:
            top_signals = signals[:SCALPING_CONFIG['max_pairs_to_ai']]
            ai_data = self._prepare_signals_for_ai(top_signals)

            message = f"""{self.selection_prompt}

=== МУЛЬТИТАЙМФРЕЙМНЫЙ АНАЛИЗ ПО ИНСТРУКЦИИ ===
МЕТОД: 15m контекст + 5m точный вход
ИНДИКАТОРЫ: EMA(5/8/20), RSI(9), MACD(12,26,9), ATR(14), Bollinger(20,2)
ШАБЛОНЫ: Momentum breakout, Pullback, Squeeze breakout, Range scalp
КОЛИЧЕСТВО СИГНАЛОВ: {len(top_signals)}

{json.dumps(ai_data, indent=2, ensure_ascii=False)}

ЗАДАЧА: Выбери максимум 3-5 лучших пар для торговли по инструкции.
Учти валидацию сигналов (3 из 4 чек-пунктов) и объемное подтверждение.

Верни JSON: {{"pairs": ["BTCUSDT", "ETHUSDT"]}}"""

            ai_response = await deep_seek_selection(message)

            if not ai_response:
                return []

            selected_pairs = self._parse_ai_response(ai_response)
            logger.info(f"✅ ЭТАП 2: ИИ выбрал {len(selected_pairs)} пар")
            return selected_pairs

        except Exception as e:
            logger.error(f"❌ Ошибка ИИ отбора: {e}")
            return []

    def _parse_ai_response(self, response: str) -> List[str]:
        """Парсинг ответа ИИ"""
        try:
            json_match = re.search(r'\{[^}]*"pairs"[^}]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return data.get('pairs', [])
            return []
        except:
            return []

    async def detailed_analysis(self, pair: str) -> Optional[str]:
        """Детальный анализ пары согласно инструкции"""
        if not self.analysis_prompt:
            return None

        logger.info(f"🔬 ЭТАП 3: Детальный анализ {pair}")

        try:
            # Получаем полные данные для детального анализа
            full_candles_5m = await get_klines_async(pair, '5', limit=SCALPING_CONFIG['candles_for_ai_analysis'])
            full_candles_15m = await get_klines_async(pair, '15', limit=60)

            if not full_candles_5m or not full_candles_15m:
                return None

            # Полный расчет индикаторов
            full_indicators = calculate_indicators_by_instruction(full_candles_5m)
            signal_analysis = detect_instruction_based_signals(full_candles_5m, full_candles_15m)

            # Подготавливаем данные для детального анализа
            analysis_data = {
                'pair': pair,
                'timestamp': int(time.time()),
                'current_price': float(full_candles_5m[-1][4]),
                'analysis_method': 'instruction_based_multi_timeframe',

                # Мультитаймфреймные данные
                'market_context': {
                    '15m_trend': signal_analysis.get('higher_tf_trend', 'UNKNOWN'),
                    '5m_last_20_candles': [
                        {
                            'open': float(c[1]),
                            'high': float(c[2]),
                            'low': float(c[3]),
                            'close': float(c[4]),
                            'volume': float(c[5])
                        } for c in full_candles_5m[-20:]
                    ],
                    '15m_last_10_candles': [
                        {
                            'open': float(c[1]),
                            'high': float(c[2]),
                            'low': float(c[3]),
                            'close': float(c[4]),
                            'volume': float(c[5])
                        } for c in full_candles_15m[-10:]
                    ]
                },

                # Полный технический анализ согласно инструкции
                'instruction_based_analysis': {
                    'signal_detected': signal_analysis.get('signal', 'NO_SIGNAL'),
                    'pattern_type': signal_analysis.get('pattern_type', 'NONE'),
                    'confidence': signal_analysis.get('confidence', 0),
                    'validation_score': signal_analysis.get('validation_score', '0/4'),
                    'entry_reasons': signal_analysis.get('entry_reasons', []),
                    'validation_reasons': signal_analysis.get('validation_reasons', [])
                },

                # Индикаторы согласно инструкции
                'technical_indicators': safe_json_serialize({
                    'trend_following': {
                        'ema5_current': full_indicators.get('ema5', [])[-1] if full_indicators.get('ema5') else 0,
                        'ema8_current': full_indicators.get('ema8', [])[-1] if full_indicators.get('ema8') else 0,
                        'ema20_current': full_indicators.get('ema20', [])[-1] if full_indicators.get('ema20') else 0,
                        'ema_alignment': (
                                len(full_indicators.get('ema5', [])) > 0 and
                                len(full_indicators.get('ema8', [])) > 0 and
                                len(full_indicators.get('ema20', [])) > 0 and
                                full_indicators['ema5'][-1] > full_indicators['ema8'][-1] > full_indicators['ema20'][-1]
                        )
                    },
                    'momentum_filter': {
                        'rsi9_current': full_indicators.get('rsi_current', 50),
                        'rsi_trend': 'bullish' if full_indicators.get('rsi_current', 50) > 50 else 'bearish',
                        'rsi_extreme': (
                                full_indicators.get('rsi_current', 50) < 30 or
                                full_indicators.get('rsi_current', 50) > 70
                        )
                    },
                    'macd_confirmation': {
                        'macd_current': full_indicators.get('macd_line', [])[-1] if full_indicators.get(
                            'macd_line') else 0,
                        'signal_current': full_indicators.get('macd_signal', [])[-1] if full_indicators.get(
                            'macd_signal') else 0,
                        'histogram_current': full_indicators.get('macd_histogram', [])[-1] if full_indicators.get(
                            'macd_histogram') else 0,
                        'bullish_crossover': (
                                len(full_indicators.get('macd_line', [])) >= 2 and
                                len(full_indicators.get('macd_signal', [])) >= 2 and
                                full_indicators['macd_line'][-2] <= full_indicators['macd_signal'][-2] and
                                full_indicators['macd_line'][-1] > full_indicators['macd_signal'][-1]
                        )
                    },
                    'volatility_control': {
                        'atr14_current': full_indicators.get('atr_current', 0),
                        'atr_mean': full_indicators.get('atr_mean', 0),
                        'volatility_suitable': full_indicators.get('atr_current', 0) >= full_indicators.get('atr_mean',
                                                                                                            0) * 0.9,
                        'atr_percent': (full_indicators.get('atr_current', 0) / float(full_candles_5m[-1][4])) * 100
                    },
                    'volume_confirmation': {
                        'volume_current': full_indicators.get('volume_current', 0),
                        'volume_sma20': full_indicators.get('volume_sma', [])[-1] if full_indicators.get(
                            'volume_sma') else 0,
                        'volume_ratio': full_indicators.get('volume_ratio', 1.0),
                        'volume_spike': full_indicators.get('volume_ratio', 1.0) > 1.5
                    },
                    'bollinger_analysis': {
                        'bb_upper': full_indicators.get('bb_upper', [])[-1] if full_indicators.get('bb_upper') else 0,
                        'bb_middle': full_indicators.get('bb_middle', [])[-1] if full_indicators.get(
                            'bb_middle') else 0,
                        'bb_lower': full_indicators.get('bb_lower', [])[-1] if full_indicators.get('bb_lower') else 0,
                        'price_position': self._get_bb_position(float(full_candles_5m[-1][4]), full_indicators),
                        'squeeze_potential': self._detect_bb_squeeze(full_indicators)
                    }
                }),

                'risk_metrics': {
                    'atr_based_stop': full_indicators.get('atr_current', 0) * 1.2,
                    'atr_percent_risk': (full_indicators.get('atr_current', 0) / float(full_candles_5m[-1][4])) * 100,
                    'volume_liquidity': 'high' if full_indicators.get('volume_ratio', 1.0) > 1.5 else 'normal'
                }
            }

            message = f"""{self.analysis_prompt}

=== ДЕТАЛЬНЫЙ АНАЛИЗ ПО ИНСТРУКЦИИ ===
ПАРА: {pair}
МЕТОД: Мультитаймфреймный анализ (15m контекст + 5m вход)
ИНДИКАТОРЫ: EMA(5/8/20), RSI(9), MACD(12,26,9), ATR(14), Bollinger(20,2)
ЦЕНА: {analysis_data['current_price']}

{json.dumps(analysis_data, indent=2, ensure_ascii=False)}

Проанализируй согласно инструкции и дай конкретные рекомендации:
1. Подтверждение сигнала по шаблонам
2. Расчет стоп-лосса и тейк-профита
3. Валидация объемом и волатильностью
4. Мультитаймфреймное подтверждение"""

            analysis_result = await deep_seek_analysis(message)

            if analysis_result:
                self._save_analysis(pair, analysis_result)
                logger.info(f"✅ Анализ {pair} завершен")
                return analysis_result

            return None

        except Exception as e:
            logger.error(f"❌ Ошибка анализа {pair}: {e}")
            return None

    def _get_bb_position(self, price: float, indicators: Dict) -> str:
        """Позиция цены относительно Bollinger Bands"""
        bb_upper = indicators.get('bb_upper', [])
        bb_lower = indicators.get('bb_lower', [])

        if not bb_upper or not bb_lower:
            return 'unknown'

        if price > bb_upper[-1]:
            return 'above_upper'
        elif price < bb_lower[-1]:
            return 'below_lower'
        else:
            return 'inside_bands'

    def _detect_bb_squeeze(self, indicators: Dict) -> bool:
        """Определение сжатия Bollinger Bands"""
        bb_upper = indicators.get('bb_upper', [])
        bb_lower = indicators.get('bb_lower', [])

        if len(bb_upper) < 10 or len(bb_lower) < 10:
            return False

        current_width = bb_upper[-1] - bb_lower[-1]
        avg_width = sum(bb_upper[i] - bb_lower[i] for i in range(-10, 0)) / 10

        return current_width < avg_width * 0.8

    def _save_analysis(self, pair: str, analysis: str):
        """Сохранение анализа"""
        try:
            with open('instruction_based_analysis.log', 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"ПАРА: {pair}\n")
                f.write(f"ВРЕМЯ: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"МЕТОД: Инструкция (15m+5m)\n")
                f.write(f"АНАЛИЗ:\n{analysis}\n")
                f.write(f"{'=' * 80}\n")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения: {e}")


async def main():
    """Главная функция бота согласно инструкции"""
    logger.info("🎯 СКАЛЬПИНГОВЫЙ БОТ ПО ИНСТРУКЦИИ - ЗАПУСК")
    logger.info("📊 Метод: 15m контекст + 5m точный вход")
    logger.info("🔧 Индикаторы: EMA(5/8/20), RSI(9), MACD(12,26,9), ATR(14), Bollinger(20,2)")

    analyzer = InstructionBasedAnalyzer()
    ai_selector = InstructionBasedAISelector()

    try:
        # ЭТАП 1: Сканирование с фильтрацией согласно инструкции
        promising_signals = await analyzer.mass_scan_markets()

        if not promising_signals:
            logger.info("ℹ️ Сигналы согласно инструкции не найдены")
            return

        logger.info(f"📈 Найдено {len(promising_signals)} сигналов по инструкции")
        for signal in promising_signals[:5]:  # Показываем топ-5
            logger.info(f"   {signal.pair}: {signal.pattern_type} ({signal.confidence}%, {signal.validation_score})")

        # ЭТАП 2: ИИ отбор согласно критериям
        selected_pairs = await ai_selector.select_best_pairs(promising_signals)

        if not selected_pairs:
            logger.info("ℹ️ ИИ не выбрал пары согласно критериям")
            return

        logger.info(f"🤖 ИИ выбрал {len(selected_pairs)} пар: {selected_pairs}")

        # ЭТАП 3: Детальный анализ каждой пары
        successful_analyses = 0

        for pair in selected_pairs:
            analysis = await ai_selector.detailed_analysis(pair)

            if analysis:
                successful_analyses += 1
                logger.info(f"✅ {pair} - детальный анализ завершен")
            else:
                logger.error(f"❌ {pair} - ошибка анализа")

            await asyncio.sleep(1)  # Пауза между запросами

        # ИТОГИ
        logger.info(f"\n🎉 АНАЛИЗ ПО ИНСТРУКЦИИ ЗАВЕРШЕН!")
        logger.info(f"📊 Метод: Мультитаймфреймный (15m+5m)")
        logger.info(f"🎯 Найдено сигналов: {len(promising_signals)}")
        logger.info(f"🤖 ИИ отобрал: {len(selected_pairs)}")
        logger.info(f"📋 Успешных анализов: {successful_analyses}")
        logger.info(f"📁 Результаты: instruction_based_analysis.log")

        await cleanup_http_client()

    except KeyboardInterrupt:
        logger.info("⏹️ Остановка по запросу")
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🎯 СКАЛЬПИНГОВЫЙ БОТ СОГЛАСНО ИНСТРУКЦИИ")
    logger.info("📊 15m контекст + 5m точный вход")
    logger.info("🔧 EMA(5/8/20), RSI(9), MACD(12,26,9), ATR(14), Bollinger(20,2)")
    logger.info("📋 Шаблоны: Momentum, Pullback, Squeeze, Range scalp")
    logger.info("=" * 80)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Программа остановлена")
    except Exception as e:
        logger.error(f"💥 Фатальная ошибка: {e}")
    finally:
        logger.info("🔚 Работа завершена")