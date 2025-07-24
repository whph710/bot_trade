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
    calculate_indicators_for_candles
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


@dataclass
class KeyIndicators:
    """Ключевые индикаторы"""
    current_ema_values: List[float]
    current_tsi: List[float]
    trend_direction: str
    last_10_ema1: List[float]
    last_10_ema2: List[float]
    last_10_ema3: List[float]
    last_10_tsi: List[float]
    last_10_tsi_signal: List[float]


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
    """Анализатор торговых сигналов EMA+TSI с двухэтапной обработкой"""

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

    def _calculate_strength_metrics(self, candles: List, indicators: Dict, signal_type: str) -> StrengthMetrics:
        """Расчет метрик силы сигнала (оптимизированная версия)"""
        try:
            # Предвычисляем часто используемые значения
            current_price = float(candles[-1][4])
            prev_price = float(self._safe_get_value(candles, -10, candles[0])[4])

            # EMA spread
            ema1_current = indicators['ema1_values'][-1]
            ema3_current = indicators['ema3_values'][-1]
            ema_spread = abs((ema1_current - ema3_current) / ema3_current * 100) if ema3_current != 0 else 0

            # TSI momentum
            tsi_current = indicators['tsi_values'][-1]
            tsi_prev = self._safe_get_value(indicators['tsi_values'], -5, tsi_current)
            tsi_momentum = abs(tsi_current - tsi_prev)

            # Volume spike (оптимизированный расчет)
            volumes = [float(candle[5]) for candle in candles[-20:]]
            if len(volumes) > 1:
                avg_volume = sum(volumes[:-1]) / len(volumes[:-1])
                volume_spike = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
            else:
                volume_spike = 1.0

            # Price move
            price_move = abs((current_price - prev_price) / prev_price * 100) if prev_price != 0 else 0

            return StrengthMetrics(
                ema_spread=round(ema_spread, 2),
                tsi_momentum=round(tsi_momentum, 2),
                volume_spike=round(volume_spike, 2),
                price_move=round(price_move, 2),
                signal_age=1
            )

        except Exception as e:
            logger.error(f"Ошибка расчета метрик силы: {e}")
            return StrengthMetrics(0.0, 0.0, 1.0, 0.0, 1)

    def _determine_trend_direction(self, indicators: Dict, signal_type: str) -> str:
        """Определение направления тренда (оптимизированная версия)"""
        try:
            # Получаем последние значения одним обращением
            ema1_values = indicators['ema1_values'][-5:]
            ema2_values = indicators['ema2_values'][-5:]
            ema3_values = indicators['ema3_values'][-5:]
            tsi_values = indicators['tsi_values'][-5:]

            if not all([ema1_values, ema2_values, ema3_values, tsi_values]):
                return 'SIDEWAYS'

            # Предвычисляем условия
            ema1_trend = ema1_values[-1] > ema1_values[0]
            ema_alignment = ema1_values[-1] > ema2_values[-1] > ema3_values[-1]
            tsi_strength = abs(tsi_values[-1])

            # Упрощенная логика определения тренда
            trend_conditions = {
                'LONG': {
                    'STRONG_UP': ema_alignment and ema1_trend and tsi_strength > 20,
                    'WEAK_UP': ema_alignment and ema1_trend,
                    'NEUTRAL_UP': ema1_trend
                },
                'SHORT': {
                    'STRONG_DOWN': not ema_alignment and not ema1_trend and tsi_strength > 20,
                    'WEAK_DOWN': not ema_alignment and not ema1_trend,
                    'NEUTRAL_DOWN': not ema1_trend
                }
            }

            conditions = trend_conditions.get(signal_type, {})
            for trend, condition in conditions.items():
                if condition:
                    return trend

            return 'SIDEWAYS'

        except Exception as e:
            logger.error(f"Ошибка определения тренда: {e}")
            return 'SIDEWAYS'

    def _extract_indicator_values(self, indicators: Dict, length: int = 10) -> Dict[str, List[float]]:
        """Извлечение значений индикаторов за последние N периодов"""
        return {
            f'last_{length}_ema1': [round(val, 6) for val in indicators['ema1_values'][-length:]],
            f'last_{length}_ema2': [round(val, 6) for val in indicators['ema2_values'][-length:]],
            f'last_{length}_ema3': [round(val, 6) for val in indicators['ema3_values'][-length:]],
            f'last_{length}_tsi': [round(val, 2) for val in indicators['tsi_values'][-length:]],
            f'last_{length}_tsi_signal': [round(val, 2) for val in indicators['tsi_signal_values'][-length:]]
        }

    async def analyze_pair(self, symbol: str) -> PairAnalysisResult:
        """Анализ одной торговой пары (оптимизированная версия)"""
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

            # Получаем детальную информацию и индикаторы
            details = get_signal_details(
                candles, self.ema1_period, self.ema2_period, self.ema3_period,
                self.tsi_long, self.tsi_short, self.tsi_signal
            )

            indicators = calculate_indicators_for_candles(
                candles, self.ema1_period, self.ema2_period, self.ema3_period,
                self.tsi_long, self.tsi_short, self.tsi_signal
            )

            if not indicators:
                return PairAnalysisResult(
                    pair=symbol,
                    signal='ERROR',
                    reason='INDICATOR_CALCULATION_FAILED'
                )

            # Рассчитываем метрики и тренд
            strength_metrics = self._calculate_strength_metrics(candles, indicators, signal)
            trend_direction = self._determine_trend_direction(indicators, signal)

            # Извлекаем значения индикаторов
            last_10_indicators = self._extract_indicator_values(indicators, 10)

            # Формируем ключевые индикаторы
            key_indicators = KeyIndicators(
                current_ema_values=[
                    round(indicators['ema1_values'][-1], 6),
                    round(indicators['ema2_values'][-1], 6),
                    round(indicators['ema3_values'][-1], 6)
                ],
                current_tsi=[
                    round(indicators['tsi_values'][-1], 2),
                    round(indicators['tsi_signal_values'][-1], 2)
                ],
                trend_direction=trend_direction,
                **last_10_indicators
            )

            # Подготавливаем полные данные для AI
            full_indicators = {
                key: values[-self.candles_for_ai:]
                for key, values in indicators.items()
            }

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
        logger.info("🔍 ЭТАП: Массовый анализ торговых пар")

        try:
            pairs = await get_usdt_trading_pairs()
        except Exception as e:
            logger.error(f"❌ ЭТАП ПРОВАЛЕН: Не удалось получить торговые пары - {e}")
            return self._create_failed_result(f'Не удалось получить список торговых пар: {e}', 0)

        if not pairs:
            logger.error("❌ ЭТАП ПРОВАЛЕН: Список торговых пар пуст")
            return self._create_failed_result('Список торговых пар пуст', 0)

        logger.info(f"📊 ЭТАП: Анализ {len(pairs)} пар на сигналы")

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
        """Создание сообщения для первичного отбора пар"""
        # Создаем сокращенную версию данных для отбора
        pairs_summary = [
            {
                'pair': pair_data['pair'],
                'signal': pair_data['signal'],
                'strength_metrics': pair_data.get('strength_metrics', {}),
                'key_indicators': {
                    'current_ema_values': pair_data.get('key_indicators', {}).get('current_ema_values', []),
                    'current_tsi': pair_data.get('key_indicators', {}).get('current_tsi', []),
                    'trend_direction': pair_data.get('key_indicators', {}).get('trend_direction', ''),
                    'last_10_ema1': pair_data.get('key_indicators', {}).get('last_10_ema1', []),
                    'last_10_ema2': pair_data.get('key_indicators', {}).get('last_10_ema2', []),
                    'last_10_ema3': pair_data.get('key_indicators', {}).get('last_10_ema3', []),
                    'last_10_tsi': pair_data.get('key_indicators', {}).get('last_10_tsi', []),
                    'last_10_tsi_signal': pair_data.get('key_indicators', {}).get('last_10_tsi_signal', [])
                },
                'recent_candles': pair_data.get('recent_candles', [])
            }
            for pair_data in pairs_data
        ]

        return f"""{base_prompt}

=== ДАННЫЕ ДЛЯ АНАЛИЗА ===
ВСЕГО ПАР С СИГНАЛАМИ: {len(pairs_data)}
ТАЙМФРЕЙМ: 15 минут

=== СВОДКА ПО ПАРАМ ===
{json.dumps(pairs_summary, indent=2, ensure_ascii=False)}

Пожалуйста, проанализируй данные и верни JSON в формате: {{"pairs": ["BTCUSDT", "ETHUSDT"]}} или {{"pairs": []}}
"""

    @staticmethod
    def create_detailed_analysis_message(base_prompt: str, pair_info: Dict[str, Any]) -> str:
        """Создание сообщения для детального анализа"""
        details = pair_info.get('details', {})
        strength_metrics = pair_info.get('strength_metrics', {})
        key_indicators = pair_info.get('key_indicators', {})

        analysis_header = f"""=== ДЕТАЛЬНЫЙ АНАЛИЗ ПАРЫ ===
ТОРГОВАЯ ПАРА: {pair_info['pair']}
ТИП СИГНАЛА: {pair_info['signal']}
ТЕКУЩАЯ ЦЕНА: {details.get('last_price', 0):.6f}
НАПРАВЛЕНИЕ ТРЕНДА: {key_indicators.get('trend_direction', 'UNKNOWN')}

МЕТРИКИ СИЛЫ СИГНАЛА:
- EMA Spread: {strength_metrics.get('ema_spread', 0)}%
- TSI Momentum: {strength_metrics.get('tsi_momentum', 0)}
- Volume Spike: {strength_metrics.get('volume_spike', 1)}x
- Price Move: {strength_metrics.get('price_move', 0)}%
- Signal Age: {strength_metrics.get('signal_age', 1)} свечей
"""

        # Остальная часть сообщения (индикаторы и свечи)
        indicators_section = f"""=== ЗНАЧЕНИЯ ИНДИКАТОРОВ ===
ТЕКУЩИЕ ЗНАЧЕНИЯ:
EMA7: {key_indicators.get('current_ema_values', [0, 0, 0])[0]}
EMA14: {key_indicators.get('current_ema_values', [0, 0, 0])[1]}
EMA28: {key_indicators.get('current_ema_values', [0, 0, 0])[2]}
TSI: {key_indicators.get('current_tsi', [0, 0])[0]}
TSI Signal: {key_indicators.get('current_tsi', [0, 0])[1]}

ПОСЛЕДНИЕ 10 ЗНАЧЕНИЙ:
EMA7 VALUES: {key_indicators.get('last_10_ema1', [])}
EMA14 VALUES: {key_indicators.get('last_10_ema2', [])}
EMA28 VALUES: {key_indicators.get('last_10_ema3', [])}
TSI VALUES: {key_indicators.get('last_10_tsi', [])}
TSI SIGNAL VALUES: {key_indicators.get('last_10_tsi_signal', [])}

ПОЛНЫЕ МАССИВЫ ДЛЯ АНАЛИЗА:
EMA7 FULL: {pair_info.get('full_indicators', {}).get('ema1_values', [])}
EMA14 FULL: {pair_info.get('full_indicators', {}).get('ema2_values', [])}
EMA28 FULL: {pair_info.get('full_indicators', {}).get('ema3_values', [])}
TSI FULL: {pair_info.get('full_indicators', {}).get('tsi_values', [])}
TSI SIGNAL FULL: {pair_info.get('full_indicators', {}).get('tsi_signal_values', [])}
"""

        candles_section = f"""=== СВЕЧНОЙ ГРАФИК ===
ПОСЛЕДНИЕ 10 СВЕЧЕЙ:
{json.dumps(pair_info.get('recent_candles', []), indent=2)}

ПОЛНЫЕ ДАННЫЕ СВЕЧЕЙ (последние {len(pair_info.get('full_candles', []))} свечей):
{json.dumps(pair_info.get('full_candles', []), indent=2)}
"""

        return f"{base_prompt}\n\n{analysis_header}\n\n{indicators_section}\n\n{candles_section}"

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
    def write_ai_response_to_file(pair_info: Dict[str, Any], ai_response: str):
        """Запись ответа нейросети в файл"""
        try:
            strength_metrics = pair_info.get('strength_metrics', {})
            key_indicators = pair_info.get('key_indicators', {})

            with open('ai_responses.log', 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 80}\n")
                f.write(
                    f"ПАРА: {pair_info['pair']} | СИГНАЛ: {pair_info['signal']} | ВРЕМЯ: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"МЕТРИКИ: EMA_SPREAD={strength_metrics.get('ema_spread', 0)}% | ")
                f.write(f"TSI_MOMENTUM={strength_metrics.get('tsi_momentum', 0)} | ")
                f.write(f"VOLUME_SPIKE={strength_metrics.get('volume_spike', 1)}x | ")
                f.write(f"TREND={key_indicators.get('trend_direction', 'UNKNOWN')}\n")
                f.write(f"{'=' * 80}\n")
                f.write(f"{ai_response}\n")
                f.flush()

            logger.info(f"✅ ПОДЭТАП ЗАВЕРШЕН: {pair_info['pair']} - ответ записан в файл")
        except Exception as e:
            logger.error(f"❌ Ошибка записи в файл для {pair_info['pair']}: {str(e)}")


async def select_pairs_with_ai(pairs_data: List[Dict[str, Any]]) -> List[str]:
    """Первый этап: отбор пар с помощью нейросети"""
    try:
        logger.info("🤖 ЭТАП 1: Отбор пар нейросетью")

        selection_prompt = AIProcessor.load_prompt('prompt2.txt')
        if not selection_prompt:
            logger.error("❌ ЭТАП 1 ПРОВАЛЕН: Промпт для отбора не загружен")
            return []

        selection_message = AIProcessor.create_pairs_selection_message(selection_prompt, pairs_data)
        logger.info(f"📤 ОТПРАВКА: Данных по {len(pairs_data)} парам для отбора")

        ai_response = await deep_seek(selection_message)
        selected_pairs = AIProcessor.parse_ai_response(ai_response)

        logger.info(f"✅ ЭТАП 1 ЗАВЕРШЕН: Отобрано {len(selected_pairs)} пар из {len(pairs_data)}")
        if selected_pairs:
            logger.info(f"📋 ОТОБРАННЫЕ ПАРЫ: {', '.join(selected_pairs)}")

        return selected_pairs

    except Exception as e:
        logger.error(f"❌ ЭТАП 1 ПРОВАЛЕН: Критическая ошибка отбора - {str(e)}")
        return []


async def process_selected_pairs_with_ai(pairs_data: List[Dict[str, Any]], selected_pairs: List[str]):
    """Второй этап: детальный анализ отобранных пар"""
    try:
        logger.info("🤖 ЭТАП 2: Детальный анализ отобранных пар")

        analysis_prompt = AIProcessor.load_prompt('prompt.txt')
        if not analysis_prompt:
            logger.error("❌ ЭТАП 2 ПРОВАЛЕН: Промпт для анализа не загружен")
            return

        selected_pairs_data = [
            pair_data for pair_data in pairs_data
            if pair_data['pair'] in selected_pairs
        ]

        if not selected_pairs_data:
            logger.warning("⚠️ ЭТАП 2: Нет данных для отобранных пар")
            return

        logger.info(f"🔄 ЭТАП 2: Обработка {len(selected_pairs_data)} отобранных пар")

        for i, pair_info in enumerate(selected_pairs_data, 1):
            try:
                logger.info(
                    f"📤 ПОДЭТАП: Детальный анализ {i}/{len(selected_pairs_data)} - {pair_info['pair']} ({pair_info['signal']})")

                analysis_message = AIProcessor.create_detailed_analysis_message(analysis_prompt, pair_info)
                ai_response = await deep_seek(analysis_message)
                AIProcessor.write_ai_response_to_file(pair_info, ai_response)

                await asyncio.sleep(2)  # Пауза между запросами

            except Exception as e:
                logger.error(f"❌ ПОДЭТАП ПРОВАЛЕН: Ошибка анализа {pair_info['pair']} - {str(e)}")
                continue

        logger.info("✅ ЭТАП 2 ЗАВЕРШЕН: Все отобранные пары проанализированы")

    except Exception as e:
        logger.error(f"❌ ЭТАП 2 ПРОВАЛЕН: Критическая ошибка детального анализа - {str(e)}")


async def main():
    """Главная функция с двухэтапным анализом"""
    try:
        logger.info("🚀 СТАРТ: Запуск двухэтапного EMA+TSI анализа")

        analyzer = TradingSignalAnalyzer(
            ema1_period=7, ema2_period=14, ema3_period=28,
            tsi_long=12, tsi_short=6, tsi_signal=6
        )

        logger.info(
            f"⚙️ НАСТРОЙКИ: EMA({analyzer.ema1_period},{analyzer.ema2_period},{analyzer.ema3_period}) | TSI({analyzer.tsi_long},{analyzer.tsi_short},{analyzer.tsi_signal})")

        # ЭТАП 0: Анализ всех пар
        result = await analyzer.analyze_all_pairs()

        if not result['success']:
            logger.error(f"❌ ПРОВАЛ: {result['message']}")
            return

        pairs_with_signals = result['pairs_data']

        if not pairs_with_signals:
            logger.warning("⚠️ РЕЗУЛЬТАТ: Сигналы не найдены")
            logger.info(f"📊 СТАТИСТИКА: Проверено {result['total_pairs_checked']} пар за {result['execution_time']:.1f}сек")
            return

        logger.info(f"📊 СТАТИСТИКА: Найдено {len(pairs_with_signals)} сигналов из {result['total_pairs_checked']} пар")
        logger.info(f"📈 LONG: {result['signal_counts']['LONG']} | 📉 SHORT: {result['signal_counts']['SHORT']}")

        # ЭТАП 1: Отбор пар нейросетью
        selected_pairs = await select_pairs_with_ai(pairs_with_signals)

        if not selected_pairs:
            logger.warning("⚠️ ЭТАП 1: Нейросеть не отобрала ни одной пары")
            return

        # ЭТАП 2: Детальный анализ отобранных пар
        await process_selected_pairs_with_ai(pairs_with_signals, selected_pairs)

        logger.info("🎯 ФИНИШ: Двухэтапный анализ завершен успешно")

    except Exception as e:
        logger.error(f"💥 КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        import traceback
        logger.error(f"📋 TRACEBACK:\n{traceback.format_exc()}")

    finally:
        logger.info("🏁 ЗАВЕРШЕНИЕ: Программа завершила работу")


if __name__ == "__main__":
    asyncio.run(main())