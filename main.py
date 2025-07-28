import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import re

from func_trade import (
    get_signal_details,
    check_ema_tsi_signal,
    calculate_all_indicators_extended  # Используем новую расширенную функцию
)
from func_async import get_klines_async, get_usdt_trading_pairs
from deepseek import deep_seek

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class StrengthMetrics:
    """Метрики силы сигнала"""
    ema_spread: float
    tsi_momentum: float
    volume_spike: float
    price_move: float
    signal_age: int
    rsi_strength: float
    level_proximity: float
    mtf_confluence: bool


@dataclass
class KeyIndicators:
    """Ключевые индикаторы для этапа 1 (отбор)"""
    current_ema_values: List[float]
    current_tsi: List[float]
    trend_direction: str
    last_10_ema1: List[float]
    last_10_ema2: List[float]
    last_10_ema3: List[float]
    last_10_tsi: List[float]
    last_10_tsi_signal: List[float]
    # Новые индикаторы (последние 10 значений)
    last_10_rsi: List[float]
    last_10_volume_spikes: List[Dict]
    last_support_levels: List[float]
    last_resistance_levels: List[float]
    recent_divergences: List[Dict]
    h1_trend: str
    h4_trend: str
    current_atr: float


@dataclass
class PairAnalysisResult:
    """Результат анализа пары"""
    pair: str
    signal: str
    reason: Optional[str] = None
    strength_metrics: Optional[StrengthMetrics] = None
    recent_candles: Optional[List] = None
    key_indicators: Optional[KeyIndicators] = None
    full_candles: Optional[List] = None
    full_indicators: Optional[Dict] = None
    details: Optional[Dict] = None


class TradingSignalAnalyzer:
    """Анализатор торговых сигналов EMA+TSI с расширенными индикаторами"""

    def __init__(self,
                 ema1_period: int = 7,
                 ema2_period: int = 14,
                 ema3_period: int = 28,
                 tsi_long: int = 12,
                 tsi_short: int = 6,
                 tsi_signal: int = 6,
                 batch_size: int = 200,
                 candles_for_ai: int = 200):

        self.ema1_period = ema1_period
        self.ema2_period = ema2_period
        self.ema3_period = ema3_period
        self.tsi_long = tsi_long
        self.tsi_short = tsi_short
        self.tsi_signal = tsi_signal
        self.batch_size = batch_size
        self.candles_for_ai = candles_for_ai

        # Предвычисляем константы
        self.required_candles_for_analysis = max(self.ema3_period, self.tsi_long, 50) + 70

    def _safe_get_value(self, data: List, index: int, default=0):
        """Безопасное получение значения из списка"""
        try:
            return data[index] if 0 <= index < len(data) else default
        except (IndexError, TypeError):
            return default

    def _calculate_strength_metrics(self, candles: List, all_indicators: Dict, signal_type: str) -> StrengthMetrics:
        """Расчет расширенных метрик силы сигнала"""
        try:
            # Предвычисляем часто используемые значения
            current_price = float(candles[-1][4])
            prev_price = float(self._safe_get_value(candles, -10, candles[0])[4])

            # EMA spread
            ema1_current = all_indicators['ema1_values'][-1]
            ema3_current = all_indicators['ema3_values'][-1]
            ema_spread = abs((ema1_current - ema3_current) / ema3_current * 100) if ema3_current != 0 else 0

            # TSI momentum
            tsi_current = all_indicators['tsi_values'][-1]
            tsi_prev = self._safe_get_value(all_indicators['tsi_values'], -5, tsi_current)
            tsi_momentum = abs(tsi_current - tsi_prev)

            # Volume spike (используем данные из volume_spikes)
            volume_spikes = all_indicators.get('volume_spikes', [])
            volume_spike = volume_spikes[-1]['ratio'] if volume_spikes else 1.0

            # Price move
            price_move = abs((current_price - prev_price) / prev_price * 100) if prev_price != 0 else 0

            # RSI strength
            rsi_current = all_indicators.get('rsi_current', 50)
            rsi_strength = abs(50 - rsi_current)  # Чем дальше от 50, тем сильнее

            # Level proximity (близость к уровням)
            nearby_support = all_indicators.get('nearby_support', [])
            nearby_resistance = all_indicators.get('nearby_resistance', [])
            level_proximity = 0
            if nearby_support:
                level_proximity = max(level_proximity, 3 - min(s['distance_percent'] for s in nearby_support))
            if nearby_resistance:
                level_proximity = max(level_proximity, 3 - min(r['distance_percent'] for r in nearby_resistance))

            # MTF confluence
            mtf_confluence = all_indicators.get('mtf_confluence', False)

            return StrengthMetrics(
                ema_spread=round(ema_spread, 2),
                tsi_momentum=round(tsi_momentum, 2),
                volume_spike=round(volume_spike, 2),
                price_move=round(price_move, 2),
                signal_age=1,
                rsi_strength=round(rsi_strength, 2),
                level_proximity=round(level_proximity, 2),
                mtf_confluence=mtf_confluence
            )

        except Exception as e:
            logger.error(f"Ошибка расчета метрик силы: {e}")
            return StrengthMetrics(0.0, 0.0, 1.0, 0.0, 1, 0.0, 0.0, False)

    def _determine_trend_direction(self, all_indicators: Dict, signal_type: str) -> str:
        """Определение направления тренда с учетом новых индикаторов"""
        try:
            # Базовые EMA условия
            ema1_values = all_indicators['ema1_values'][-5:]
            ema2_values = all_indicators['ema2_values'][-5:]
            ema3_values = all_indicators['ema3_values'][-5:]
            tsi_values = all_indicators['tsi_values'][-5:]

            if not all([ema1_values, ema2_values, ema3_values, tsi_values]):
                return 'SIDEWAYS'

            # Базовые условия
            ema1_trend = ema1_values[-1] > ema1_values[0]
            ema_alignment = ema1_values[-1] > ema2_values[-1] > ema3_values[-1]
            tsi_strength = abs(tsi_values[-1])

            # Новые условия
            rsi_current = all_indicators.get('rsi_current', 50)
            h1_trend = all_indicators.get('h1_trend', 'UNKNOWN')
            h4_trend = all_indicators.get('h4_trend', 'UNKNOWN')
            mtf_confluence = all_indicators.get('mtf_confluence', False)
            rsi_divergences = all_indicators.get('rsi_divergences', [])

            # Усиленная логика с новыми индикаторами
            if signal_type == 'LONG':
                if (ema_alignment and ema1_trend and tsi_strength > 20 and
                        h1_trend == 'BULLISH' and h4_trend == 'BULLISH' and
                        rsi_current < 70):
                    return 'STRONG_UP'
                elif (ema_alignment and ema1_trend and
                      (h1_trend == 'BULLISH' or h4_trend == 'BULLISH')):
                    return 'WEAK_UP'
                elif ema1_trend:
                    return 'NEUTRAL_UP'

            elif signal_type == 'SHORT':
                if (not ema_alignment and not ema1_trend and tsi_strength > 20 and
                        h1_trend == 'BEARISH' and h4_trend == 'BEARISH' and
                        rsi_current > 30):
                    return 'STRONG_DOWN'
                elif (not ema_alignment and not ema1_trend and
                      (h1_trend == 'BEARISH' or h4_trend == 'BEARISH')):
                    return 'WEAK_DOWN'
                elif not ema1_trend:
                    return 'NEUTRAL_DOWN'

            return 'SIDEWAYS'

        except Exception as e:
            logger.error(f"Ошибка определения тренда: {e}")
            return 'SIDEWAYS'

    def _extract_key_indicators_for_selection(self, all_indicators: Dict, length: int = 10) -> Dict[str, Any]:
        """Извлечение ключевых индикаторов для этапа отбора"""
        try:
            return {
                # Базовые индикаторы (последние 10)
                f'last_{length}_ema1': [round(val, 6) for val in all_indicators['ema1_values'][-length:]],
                f'last_{length}_ema2': [round(val, 6) for val in all_indicators['ema2_values'][-length:]],
                f'last_{length}_ema3': [round(val, 6) for val in all_indicators['ema3_values'][-length:]],
                f'last_{length}_tsi': [round(val, 2) for val in all_indicators['tsi_values'][-length:]],
                f'last_{length}_tsi_signal': [round(val, 2) for val in all_indicators['tsi_signal_values'][-length:]],

                # Новые индикаторы для отбора
                f'last_{length}_rsi': [round(val, 2) for val in all_indicators['rsi_values'][-length:]],
                f'last_{length}_volume_spikes': all_indicators.get('volume_spikes', [])[-length:],
                'last_support_levels': all_indicators.get('support_levels', [])[-3:],  # 3 ближайших
                'last_resistance_levels': all_indicators.get('resistance_levels', [])[-3:],  # 3 ближайших
                'recent_divergences': all_indicators.get('rsi_divergences', []),
                'h1_trend': all_indicators.get('h1_trend', 'UNKNOWN'),
                'h4_trend': all_indicators.get('h4_trend', 'UNKNOWN'),
                'current_atr': all_indicators.get('current_atr', 0)
            }
        except Exception as e:
            logger.error(f"Ошибка извлечения ключевых индикаторов: {e}")
            return {}

    async def analyze_pair(self, symbol: str) -> PairAnalysisResult:
        """Анализ одной торговой пары с расширенными индикаторами"""
        try:
            # Получаем данные свечей
            candles = await get_klines_async(
                symbol,
                interval="15",
                limit=self.required_candles_for_analysis
            )

            if not candles or len(candles) < self.required_candles_for_analysis:
                return PairAnalysisResult(
                    pair=symbol,
                    signal='NO_SIGNAL',
                    reason='INSUFFICIENT_DATA'
                )

            # Анализируем сигнал
            signal = check_ema_tsi_signal(
                candles, self.ema1_period, self.ema2_period, self.ema3_period,
                self.tsi_long, self.tsi_short, self.tsi_signal
            )

            if signal not in ['LONG', 'SHORT']:
                return PairAnalysisResult(
                    pair=symbol,
                    signal=signal,
                    reason='NO_SIGNAL_DETECTED'
                )

            # Получаем ВСЕ индикаторы (базовые + новые)
            all_indicators = calculate_all_indicators_extended(
                candles, self.ema1_period, self.ema2_period, self.ema3_period,
                self.tsi_long, self.tsi_short, self.tsi_signal
            )

            if not all_indicators:
                return PairAnalysisResult(
                    pair=symbol,
                    signal='ERROR',
                    reason='INDICATOR_CALCULATION_FAILED'
                )

            # Получаем детальную информацию
            details = get_signal_details(
                candles, self.ema1_period, self.ema2_period, self.ema3_period,
                self.tsi_long, self.tsi_short, self.tsi_signal
            )

            # Рассчитываем расширенные метрики силы
            strength_metrics = self._calculate_strength_metrics(candles, all_indicators, signal)
            trend_direction = self._determine_trend_direction(all_indicators, signal)

            # Извлекаем ключевые индикаторы для отбора
            key_indicators_data = self._extract_key_indicators_for_selection(all_indicators, 10)

            # Формируем расширенные ключевые индикаторы
            key_indicators = KeyIndicators(
                current_ema_values=[
                    round(all_indicators['ema1_values'][-1], 6),
                    round(all_indicators['ema2_values'][-1], 6),
                    round(all_indicators['ema3_values'][-1], 6)
                ],
                current_tsi=[
                    round(all_indicators['tsi_values'][-1], 2),
                    round(all_indicators['tsi_signal_values'][-1], 2)
                ],
                trend_direction=trend_direction,
                **key_indicators_data
            )

            # Подготавливаем полные данные для этапа 2
            full_indicators = {
                key: values[-self.candles_for_ai:]
                for key, values in all_indicators.items()
                if isinstance(values, list)
            }

            # Добавляем скалярные значения
            full_indicators.update({
                key: value for key, value in all_indicators.items()
                if not isinstance(value, list)
            })

            return PairAnalysisResult(
                pair=symbol,
                signal=signal,
                strength_metrics=strength_metrics,
                recent_candles=candles[-10:],
                key_indicators=key_indicators,
                full_candles=candles[-self.candles_for_ai:],
                full_indicators=full_indicators,
                details=details
            )

        except Exception as e:
            return PairAnalysisResult(
                pair=symbol,
                signal='ERROR',
                reason=str(e)
            )

    async def analyze_all_pairs(self) -> Dict[str, Any]:
        """Анализ всех торговых пар (оптимизированная версия)"""
        start_time = time.time()
        logger.info("🔍 ЭТАП: Массовый анализ торговых пар с расширенными индикаторами")

        try:
            pairs = await get_usdt_trading_pairs()
        except Exception as e:
            logger.error(f"❌ ЭТАП ПРОВАЛЕН: Не удалось получить торговые пары - {e}")
            return self._create_failed_result(f'Не удалось получить список торговых пар: {e}', 0)

        if not pairs:
            logger.error("❌ ЭТАП ПРОВАЛЕН: Список торговых пар пуст")
            return self._create_failed_result('Список торговых пар пуст', 0)

        logger.info(f"📊 ЭТАП: Анализ {len(pairs)} пар на сигналы (5 критических улучшений)")

        # Параллельная обработка батчами
        all_results = await self._process_pairs_in_batches(pairs)

        # Фильтруем результаты
        pairs_with_signals = [
            asdict(result) for result in all_results
            if result.signal in ['LONG', 'SHORT']
        ]

        # Подсчитываем статистику
        signal_counts = self._calculate_signal_statistics(all_results)
        execution_time = time.time() - start_time

        logger.info(f"✅ ЭТАП ЗАВЕРШЕН: Найдено {len(pairs_with_signals)} сигналов за {execution_time:.1f}сек")

        return {
            'success': True,
            'pairs_data': pairs_with_signals,
            'all_pairs_data': [asdict(result) for result in all_results],
            'signal_counts': signal_counts,
            'total_pairs_checked': len(all_results),
            'execution_time': execution_time
        }

    def _create_failed_result(self, message: str, execution_time: float) -> Dict[str, Any]:
        """Создание результата с ошибкой"""
        return {
            'success': False,
            'message': message,
            'pairs_data': [],
            'signal_counts': {'LONG': 0, 'SHORT': 0, 'NO_SIGNAL': 0},
            'execution_time': execution_time
        }

    async def _process_pairs_in_batches(self, pairs: List[str]) -> List[PairAnalysisResult]:
        """Обработка пар батчами"""
        all_results = []

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            tasks = [self.analyze_pair(pair) for pair in batch]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"❌ Исключение при анализе пары: {result}")
                else:
                    all_results.append(result)

            # Логирование прогресса
            progress = min(i + self.batch_size, len(pairs))
            logger.info(f"⏳ ПРОГРЕСС: {progress}/{len(pairs)} пар проанализировано")

            await asyncio.sleep(0.2)  # Пауза между батчами

        return all_results

    def _calculate_signal_statistics(self, results: List[PairAnalysisResult]) -> Dict[str, int]:
        """Подсчет статистики сигналов"""
        signal_counts = {'LONG': 0, 'SHORT': 0, 'NO_SIGNAL': 0}

        for result in results:
            signal = result.signal
            if signal in signal_counts:
                signal_counts[signal] += 1
            else:
                signal_counts['NO_SIGNAL'] += 1

        return signal_counts


class AIProcessor:
    """Класс для работы с нейросетью"""

    @staticmethod
    def load_prompt(filename: str = 'prompt2.txt') -> str:
        """Загрузка промпта из файла"""
        try:
            logger.info(f"📄 ЭТАП: Загрузка промпта из {filename}")
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            logger.info("✅ ЭТАП ЗАВЕРШЕН: Промпт успешно загружен")
            return content
        except FileNotFoundError:
            logger.error(f"❌ ЭТАП ПРОВАЛЕН: Файл {filename} не найден")
            return ""
        except Exception as e:
            logger.error(f"❌ ЭТАП ПРОВАЛЕН: Ошибка чтения {filename} - {str(e)}")
            return ""

    @staticmethod
    def create_pairs_selection_message(base_prompt: str, pairs_data: List[Dict[str, Any]]) -> str:
        """Создание сообщения для первичного отбора пар с расширенными индикаторами"""
        # Создаем сокращенную версию данных для отбора (только последние 10 значений)
        pairs_summary = [
            {
                'pair': pair_data['pair'],
                'signal': pair_data['signal'],
                'strength_metrics': pair_data.get('strength_metrics', {}),

                # СУЩЕСТВУЮЩИЕ ИНДИКАТОРЫ (последние 10)
                'ema_indicators': {
                    'ema7_last10': pair_data.get('key_indicators', {}).get('last_10_ema1', []),
                    'ema14_last10': pair_data.get('key_indicators', {}).get('last_10_ema2', []),
                    'ema28_last10': pair_data.get('key_indicators', {}).get('last_10_ema3', []),
                    'tsi_last10': pair_data.get('key_indicators', {}).get('last_10_tsi', []),
                    'tsi_signal_last10': pair_data.get('key_indicators', {}).get('last_10_tsi_signal', [])
                },

                # НОВЫЕ ИНДИКАТОРЫ (последние 10)
                'additional_indicators': {
                    'support_levels': pair_data.get('key_indicators', {}).get('last_support_levels', []),
                    'resistance_levels': pair_data.get('key_indicators', {}).get('last_resistance_levels', []),
                    'rsi_last10': pair_data.get('key_indicators', {}).get('last_10_rsi', []),
                    'rsi_divergences': pair_data.get('key_indicators', {}).get('recent_divergences', []),
                    'volume_spikes_last10': pair_data.get('key_indicators', {}).get('last_10_volume_spikes', []),
                    'h1_trend': pair_data.get('key_indicators', {}).get('h1_trend', 'UNKNOWN'),
                    'h4_trend': pair_data.get('key_indicators', {}).get('h4_trend', 'UNKNOWN'),
                    'atr_current': pair_data.get('key_indicators', {}).get('current_atr', 0)
                },

                'recent_candles': pair_data.get('recent_candles', [])[-10:]  # Только последние 10 свечей
            }
            for pair_data in pairs_data
        ]

        return f"""{base_prompt}

=== ДАННЫЕ ДЛЯ АНАЛИЗА ===
ВСЕГО ПАР С СИГНАЛАМИ: {len(pairs_data)}
ТАЙМФРЕЙМ: 15 минут
СТРАТЕГИЯ: EMA+TSI + 5 критических улучшений

=== РАСШИРЕННАЯ СВОДКА ПО ПАРАМ ===
{json.dumps(pairs_summary, indent=2, ensure_ascii=False)}

Пожалуйста, проанализируй данные с учетом всех индикаторов и верни JSON в формате: {{"pairs": ["BTCUSDT", "ETHUSDT"]}} или {{"pairs": []}}
"""

    @staticmethod
    def create_detailed_analysis_message(base_prompt: str, pair_info: Dict[str, Any]) -> str:
        """Создание сообщения для детального анализа с полными данными"""
        details = pair_info.get('details', {})
        strength_metrics = pair_info.get('strength_metrics', {})
        key_indicators = pair_info.get('key_indicators', {})
        full_indicators = pair_info.get('full_indicators', {})

        analysis_header = f"""=== ДЕТАЛЬНЫЙ АНАЛИЗ ПАРЫ ===
ТОРГОВАЯ ПАРА: {pair_info['pair']}
ТИП СИГНАЛА: {pair_info['signal']}
ТЕКУЩАЯ ЦЕНА: {details.get('last_price', 0):.6f}
НАПРАВЛЕНИЕ ТРЕНДА: {key_indicators.get('trend_direction', 'UNKNOWN')}

РАСШИРЕННЫЕ МЕТРИКИ СИЛЫ СИГНАЛА:
- EMA Spread: {strength_metrics.get('ema_spread', 0)}%
- TSI Momentum: {strength_metrics.get('tsi_momentum', 0)}
- Volume Spike: {strength_metrics.get('volume_spike', 1)}x
- Price Move: {strength_metrics.get('price_move', 0)}%
- Signal Age: {strength_metrics.get('signal_age', 1)} свечей
- RSI Strength: {strength_metrics.get('rsi_strength', 0)}
- Level Proximity: {strength_metrics.get('level_proximity', 0)}
- MTF Confluence: {strength_metrics.get('mtf_confluence', False)}
"""

        # Полный набор индикаторов для детального анализа
        extended_indicators_section = f"""=== ПОЛНЫЙ НАБОР ИНДИКАТОРОВ ===

БАЗОВЫЕ ИНДИКАТОРЫ (EMA + TSI):
EMA7 FULL (200): {full_indicators.get('ema1_values', [])}
EMA14 FULL (200): {full_indicators.get('ema2_values', [])}
EMA28 FULL (200): {full_indicators.get('ema3_values', [])}
TSI FULL (200): {full_indicators.get('tsi_values', [])}
TSI SIGNAL FULL (200): {full_indicators.get('tsi_signal_values', [])}

УРОВНЕВЫЙ АНАЛИЗ:
SUPPORT LEVELS: {full_indicators.get('support_levels', [])}
RESISTANCE LEVELS: {full_indicators.get('resistance_levels', [])}
NEARBY SUPPORT: {full_indicators.get('nearby_support', [])}
NEARBY RESISTANCE: {full_indicators.get('nearby_resistance', [])}

RSI + ДИВЕРГЕНЦИИ:
RSI FULL (200): {full_indicators.get('rsi_values', [])}
RSI CURRENT: {full_indicators.get('rsi_current', 50)}
RSI DIVERGENCES: {full_indicators.get('rsi_divergences', [])}

ОБЪЕМНЫЙ АНАЛИЗ:
VOLUME SPIKES: {full_indicators.get('volume_spikes', [])}
VOLUME PROFILE: {full_indicators.get('volume_profile', [])}

МУЛЬТИТАЙМФРЕЙМ:
1H TREND: {full_indicators.get('h1_trend', 'UNKNOWN')}
4H TREND: {full_indicators.get('h4_trend', 'UNKNOWN')}
MTF CONFLUENCE: {full_indicators.get('mtf_confluence', False)}

ДИНАМИЧЕСКИЕ СТОПЫ:
ATR VALUES (200): {full_indicators.get('atr_values', [])}
CURRENT ATR: {full_indicators.get('current_atr', 0)}
STOP LEVELS: {full_indicators.get('dynamic_stops', {})}
"""

        candles_section = f"""=== СВЕЧНОЙ ГРАФИК ===
ПОСЛЕДНИЕ 10 СВЕЧЕЙ:
{json.dumps(pair_info.get('recent_candles', []), indent=2)}

ПОЛНЫЕ ДАННЫЕ СВЕЧЕЙ (последние {len(pair_info.get('full_candles', []))} свечей):
{json.dumps(pair_info.get('full_candles', []), indent=2)}
"""

        return f"{base_prompt}\n\n{analysis_header}\n\n{extended_indicators_section}\n\n{candles_section}"

    @staticmethod
    def parse_ai_response(ai_response: str) -> List[str]:
        """Парсинг ответа нейросети"""
        try:
            json_match = re.search(r'\{[^}]*"pairs"[^}]*\}', ai_response)
            if json_match:
                response_data = json.loads(json_match.group())
                return response_data.get('pairs', [])
            return []
        except json.JSONDecodeError as e:
            logger.error(f"❌ Ошибка парсинга JSON: {e}")
            return []

    @staticmethod
    @staticmethod
    def write_ai_response_to_file(pair_info: Dict[str, Any], ai_response: str):
        """Запись ответа нейросети в файл с расширенными метриками"""
        try:
            strength_metrics = pair_info.get('strength_metrics', {})
            key_indicators = pair_info.get('key_indicators', {})

            with open('ai_responses.log', 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 80}\n")
                f.write(
                    f"ПАРА: {pair_info['pair']} | СИГНАЛ: {pair_info['signal']} | ВРЕМЯ: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"ЦЕНА: {pair_info.get('details', {}).get('last_price', 0):.6f}\n")
                f.write(f"НАПРАВЛЕНИЕ ТРЕНДА: {key_indicators.get('trend_direction', 'UNKNOWN')}\n")

                # Записываем расширенные метрики
                f.write(f"\nРАСШИРЕННЫЕ МЕТРИКИ СИЛЫ:\n")
                f.write(f"- EMA Spread: {strength_metrics.get('ema_spread', 0)}%\n")
                f.write(f"- TSI Momentum: {strength_metrics.get('tsi_momentum', 0)}\n")
                f.write(f"- Volume Spike: {strength_metrics.get('volume_spike', 1)}x\n")
                f.write(f"- Price Move: {strength_metrics.get('price_move', 0)}%\n")
                f.write(f"- RSI Strength: {strength_metrics.get('rsi_strength', 0)}\n")
                f.write(f"- Level Proximity: {strength_metrics.get('level_proximity', 0)}\n")
                f.write(f"- MTF Confluence: {strength_metrics.get('mtf_confluence', False)}\n")

                # Добавляем новые индикаторы
                f.write(f"\nДОПОЛНИТЕЛЬНЫЕ ИНДИКАТОРЫ:\n")
                f.write(f"- 1H Trend: {key_indicators.get('h1_trend', 'UNKNOWN')}\n")
                f.write(f"- 4H Trend: {key_indicators.get('h4_trend', 'UNKNOWN')}\n")
                f.write(f"- Current ATR: {key_indicators.get('current_atr', 0):.6f}\n")
                f.write(f"- RSI Divergences: {len(key_indicators.get('recent_divergences', []))}\n")
                f.write(f"- Volume Spikes: {len(key_indicators.get('last_10_volume_spikes', []))}\n")

                f.write(f"\nОТВЕТ ИИ:\n{ai_response}\n")
                f.write("=" * 80 + "\n")

        except Exception as e:
            logger.error(f"❌ Ошибка записи в файл: {e}")


async def main():
    """Главная функция с двухэтапным анализом и расширенными индикаторами"""
    logger.info("🚀 ЗАПУСК: Торговый бот с расширенной стратегией EMA+TSI")

    # Инициализация анализатора с оптимизированными параметрами
    analyzer = TradingSignalAnalyzer(
        ema1_period=7,
        ema2_period=14,
        ema3_period=28,
        tsi_long=12,  # Ускоренный TSI для 15M
        tsi_short=6,  # Ускоренный TSI для 15M
        tsi_signal=6,  # Ускоренный TSI для 15M
        batch_size=150,  # Оптимизированный размер батча
        candles_for_ai=200
    )

    try:
        # ===== ЭТАП 1: МАССОВЫЙ АНАЛИЗ С РАСШИРЕННЫМИ ИНДИКАТОРАМИ =====
        logger.info("🔍 ЭТАП 1: Массовый анализ всех торговых пар")
        analysis_result = await analyzer.analyze_all_pairs()

        if not analysis_result['success']:
            logger.error(f"❌ ЭТАП 1 ПРОВАЛЕН: {analysis_result.get('message', 'Неизвестная ошибка')}")
            return

        pairs_with_signals = analysis_result['pairs_data']
        signal_counts = analysis_result['signal_counts']

        logger.info(f"✅ ЭТАП 1 ЗАВЕРШЕН: Найдено {len(pairs_with_signals)} пар с сигналами")
        logger.info(
            f"📊 СТАТИСТИКА: LONG={signal_counts['LONG']}, SHORT={signal_counts['SHORT']}, NO_SIGNAL={signal_counts['NO_SIGNAL']}")

        if not pairs_with_signals:
            logger.info("ℹ️  Нет пар с торговыми сигналами. Завершение работы.")
            return

        # ===== ЭТАП 2: ИИ ОТБОР ЛУЧШИХ ПАР =====
        logger.info("🤖 ЭТАП 2: ИИ отбор лучших пар с расширенным анализом")

        # Загружаем промпт для отбора
        selection_prompt = AIProcessor.load_prompt('prompt2.txt')
        if not selection_prompt:
            logger.error("❌ ЭТАП 2 ПРОВАЛЕН: Не удалось загрузить промпт")
            return

        # Создаем сообщение для ИИ с расширенными данными
        selection_message = AIProcessor.create_pairs_selection_message(selection_prompt, pairs_with_signals)

        # Отправляем запрос к ИИ
        logger.info("⏳ Отправка данных на анализ ИИ...")
        ai_selection_response = await deep_seek(selection_message)

        if not ai_selection_response:
            logger.error("❌ ЭТАП 2 ПРОВАЛЕН: ИИ не вернул ответ")
            return

        # Парсим ответ ИИ
        selected_pairs = AIProcessor.parse_ai_response(ai_selection_response)

        if not selected_pairs:
            logger.info("ℹ️  ИИ не выбрал ни одной пары для торговли")
            logger.info(f"📄 ОТВЕТ ИИ: {ai_selection_response[:500]}...")
            return

        logger.info(f"✅ ЭТАП 2 ЗАВЕРШЕН: ИИ выбрал {len(selected_pairs)} пар: {', '.join(selected_pairs)}")

        # ===== ЭТАП 3: ДЕТАЛЬНЫЙ АНАЛИЗ ВЫБРАННЫХ ПАР =====
        logger.info("📊 ЭТАП 3: Детальный анализ выбранных пар")

        detailed_prompt = AIProcessor.load_prompt('prompt.txt')
        if not detailed_prompt:
            logger.error("❌ ЭТАП 3 ПРОВАЛЕН: Не удалось загрузить детальный промпт")
            return

        # Анализируем каждую выбранную пару детально
        for pair_name in selected_pairs:
            logger.info(f"🔬 Детальный анализ пары: {pair_name}")

            # Находим данные пары
            pair_data = None
            for pair_info in pairs_with_signals:
                if pair_info['pair'] == pair_name:
                    pair_data = pair_info
                    break

            if not pair_data:
                logger.error(f"❌ Данные для пары {pair_name} не найдены")
                continue

            # Создаем детальное сообщение с полными данными
            detailed_message = AIProcessor.create_detailed_analysis_message(detailed_prompt, pair_data)

            # Отправляем на детальный анализ
            logger.info(f"⏳ Отправка {pair_name} на детальный анализ...")
            detailed_response = await deep_seek(detailed_message)

            if detailed_response:
                logger.info(f"✅ Получен детальный анализ для {pair_name}")
                logger.info(f"📋 ПЛАН ТОРГОВЛИ:\n{detailed_response[:1000]}...")

                # Записываем в файл
                AIProcessor.write_ai_response_to_file(pair_data, detailed_response)
            else:
                logger.error(f"❌ Не удалось получить детальный анализ для {pair_name}")

            # Пауза между запросами
            await asyncio.sleep(2)

        logger.info("🎉 ВСЕ ЭТАПЫ ЗАВЕРШЕНЫ УСПЕШНО!")
        logger.info(f"📁 Результаты сохранены в ai_responses.log")

    except Exception as e:
        logger.error(f"💥 КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    """Точка входа в программу"""
    logger.info("=" * 80)
    logger.info("🎯 ТОРГОВЫЙ БОТ - РАСШИРЕННАЯ СТРАТЕГИЯ EMA+TSI")
    logger.info("📈 Версия: 2.0 с 5 критическими улучшениями")
    logger.info("⏰ Таймфрейм: 15 минут")
    logger.info("🔧 Индикаторы: EMA(7,14,28) + TSI(12,6,6) + Support/Resistance + RSI + Volume + MTF + ATR")
    logger.info("=" * 80)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹️  Остановка по запросу пользователя")
    except Exception as e:
        logger.error(f"💥 ФАТАЛЬНАЯ ОШИБКА: {e}")
    finally:
        logger.info("👋 Работа завершена")