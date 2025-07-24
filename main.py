import asyncio
import json
import logging
import time
from typing import List, Dict, Any

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


class TradingSignalAnalyzer:
    """
    Анализатор торговых сигналов EMA+TSI с двухэтапной обработкой
    """

    def __init__(self,
                 ema1_period: int = 7,
                 ema2_period: int = 14,
                 ema3_period: int = 28,
                 tsi_long: int = 12,
                 tsi_short: int = 6,
                 tsi_signal: int = 6):
        """
        Инициализация анализатора
        """
        self.ema1_period = ema1_period
        self.ema2_period = ema2_period
        self.ema3_period = ema3_period
        self.tsi_long = tsi_long
        self.tsi_short = tsi_short
        self.tsi_signal = tsi_signal

        # Рассчитываем необходимое количество свечей для анализа
        self.required_candles_for_analysis = max(self.ema3_period, self.tsi_long, 50) + 70
        # Для отправки в нейросеть берем последние 100 свечей
        self.candles_for_ai = 200

    def calculate_strength_metrics(self, candles: List, indicators: Dict, signal_type: str) -> Dict[str, float]:
        """
        Расчет метрик силы сигнала
        """
        try:
            # Получаем последние значения
            current_price = float(candles[-1][4])  # close price
            prev_price = float(candles[-10][4]) if len(candles) >= 10 else float(candles[0][4])

            # EMA spread - расстояние между EMA в %
            ema1_current = indicators['ema1_values'][-1]
            ema3_current = indicators['ema3_values'][-1]
            ema_spread = abs((ema1_current - ema3_current) / ema3_current * 100)

            # TSI momentum - скорость изменения TSI
            tsi_current = indicators['tsi_values'][-1]
            tsi_prev = indicators['tsi_values'][-5] if len(indicators['tsi_values']) >= 5 else indicators['tsi_values'][
                0]
            tsi_momentum = abs(tsi_current - tsi_prev)

            # Volume spike - превышение среднего объёма
            volumes = [float(candle[5]) for candle in candles[-20:]]  # последние 20 свечей
            avg_volume = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else volumes[0]
            current_volume = volumes[-1]
            volume_spike = current_volume / avg_volume if avg_volume > 0 else 1.0

            # Price move - движение цены за последние свечи в %
            price_move = abs((current_price - prev_price) / prev_price * 100)

            # Signal age - возраст сигнала (пока ставим 1, так как это новый сигнал)
            signal_age = 1

            return {
                'ema_spread': round(ema_spread, 2),
                'tsi_momentum': round(tsi_momentum, 2),
                'volume_spike': round(volume_spike, 2),
                'price_move': round(price_move, 2),
                'signal_age': signal_age
            }

        except Exception as e:
            logger.error(f"Ошибка расчета метрик силы: {e}")
            return {
                'ema_spread': 0.0,
                'tsi_momentum': 0.0,
                'volume_spike': 1.0,
                'price_move': 0.0,
                'signal_age': 1
            }

    def determine_trend_direction(self, indicators: Dict, signal_type: str) -> str:
        """
        Определение направления тренда
        """
        try:
            # Берем последние 5 значений EMA для анализа тренда
            ema1_values = indicators['ema1_values'][-5:]
            ema2_values = indicators['ema2_values'][-5:]
            ema3_values = indicators['ema3_values'][-5:]
            tsi_values = indicators['tsi_values'][-5:]

            # Проверяем направление EMA
            ema1_trend = ema1_values[-1] > ema1_values[0]  # растет ли EMA7
            ema_alignment = ema1_values[-1] > ema2_values[-1] > ema3_values[-1]  # правильное расположение для LONG

            # Проверяем силу TSI
            tsi_strength = abs(tsi_values[-1])

            if signal_type == 'LONG':
                if ema_alignment and ema1_trend and tsi_strength > 20:
                    return 'STRONG_UP'
                elif ema_alignment and ema1_trend:
                    return 'WEAK_UP'
                elif ema1_trend:
                    return 'NEUTRAL_UP'
                else:
                    return 'SIDEWAYS'
            else:  # SHORT
                if not ema_alignment and not ema1_trend and tsi_strength > 20:
                    return 'STRONG_DOWN'
                elif not ema_alignment and not ema1_trend:
                    return 'WEAK_DOWN'
                elif not ema1_trend:
                    return 'NEUTRAL_DOWN'
                else:
                    return 'SIDEWAYS'

        except Exception as e:
            logger.error(f"Ошибка определения тренда: {e}")
            return 'SIDEWAYS'

    async def analyze_pair(self, symbol: str) -> Dict[str, Any]:
        """
        Анализ одной торговой пары с новой структурой данных
        """
        try:
            # Получаем данные свечей для анализа
            candles = await get_klines_async(
                symbol,
                interval="15",
                limit=self.required_candles_for_analysis
            )

            if not candles or len(candles) < self.required_candles_for_analysis:
                return {
                    'pair': symbol,
                    'signal': 'NO_SIGNAL',
                    'reason': 'INSUFFICIENT_DATA'
                }

            # Анализируем сигнал
            signal = check_ema_tsi_signal(
                candles,
                self.ema1_period,
                self.ema2_period,
                self.ema3_period,
                self.tsi_long,
                self.tsi_short,
                self.tsi_signal
            )

            # Если нет сигнала, возвращаем минимальную информацию
            if signal not in ['LONG', 'SHORT']:
                return {
                    'pair': symbol,
                    'signal': signal,
                    'reason': 'NO_SIGNAL_DETECTED'
                }

            # Получаем детальную информацию
            details = get_signal_details(
                candles,
                self.ema1_period,
                self.ema2_period,
                self.ema3_period,
                self.tsi_long,
                self.tsi_short,
                self.tsi_signal
            )

            # Рассчитываем индикаторы для всех свечей
            indicators = calculate_indicators_for_candles(
                candles,
                self.ema1_period,
                self.ema2_period,
                self.ema3_period,
                self.tsi_long,
                self.tsi_short,
                self.tsi_signal
            )

            if not indicators:
                return {
                    'pair': symbol,
                    'signal': 'ERROR',
                    'reason': 'INDICATOR_CALCULATION_FAILED'
                }

            # Рассчитываем метрики силы сигнала
            strength_metrics = self.calculate_strength_metrics(candles, indicators, signal)

            # Определяем направление тренда
            trend_direction = self.determine_trend_direction(indicators, signal)

            # Берем последние 10 свечей
            recent_candles = candles[-10:]

            # Берем последние 10 значений индикаторов
            last_10_ema1 = indicators['ema1_values'][-10:] if len(indicators['ema1_values']) >= 10 else indicators[
                'ema1_values']
            last_10_ema2 = indicators['ema2_values'][-10:] if len(indicators['ema2_values']) >= 10 else indicators[
                'ema2_values']
            last_10_ema3 = indicators['ema3_values'][-10:] if len(indicators['ema3_values']) >= 10 else indicators[
                'ema3_values']
            last_10_tsi = indicators['tsi_values'][-10:] if len(indicators['tsi_values']) >= 10 else indicators[
                'tsi_values']
            last_10_tsi_signal = indicators['tsi_signal_values'][-10:] if len(
                indicators['tsi_signal_values']) >= 10 else indicators['tsi_signal_values']

            # Формируем новую структуру данных
            return {
                'pair': symbol,
                'signal': signal,
                'strength_metrics': strength_metrics,
                'recent_candles': recent_candles,
                'key_indicators': {
                    'current_ema_values': [
                        round(indicators['ema1_values'][-1], 6),
                        round(indicators['ema2_values'][-1], 6),
                        round(indicators['ema3_values'][-1], 6)
                    ],
                    'current_tsi': [
                        round(indicators['tsi_values'][-1], 2),
                        round(indicators['tsi_signal_values'][-1], 2)
                    ],
                    'trend_direction': trend_direction,
                    'last_10_ema1': [round(val, 6) for val in last_10_ema1],
                    'last_10_ema2': [round(val, 6) for val in last_10_ema2],
                    'last_10_ema3': [round(val, 6) for val in last_10_ema3],
                    'last_10_tsi': [round(val, 2) for val in last_10_tsi],
                    'last_10_tsi_signal': [round(val, 2) for val in last_10_tsi_signal]
                },
                # Сохраняем полные данные для детального анализа
                'full_candles': candles[-self.candles_for_ai:],
                'full_indicators': {
                    'ema1_values': indicators['ema1_values'][-self.candles_for_ai:],
                    'ema2_values': indicators['ema2_values'][-self.candles_for_ai:],
                    'ema3_values': indicators['ema3_values'][-self.candles_for_ai:],
                    'tsi_values': indicators['tsi_values'][-self.candles_for_ai:],
                    'tsi_signal_values': indicators['tsi_signal_values'][-self.candles_for_ai:]
                },
                'details': details
            }

        except Exception as e:
            return {
                'pair': symbol,
                'signal': 'ERROR',
                'reason': str(e)
            }

    async def analyze_all_pairs(self) -> Dict[str, Any]:
        """
        Анализ всех торговых пар на наличие сигналов
        """
        start_time = time.time()
        logger.info("🔍 ЭТАП: Массовый анализ торговых пар")

        # Получаем список торговых пар
        try:
            pairs = await get_usdt_trading_pairs()
        except Exception as e:
            logger.error(f"❌ ЭТАП ПРОВАЛЕН: Не удалось получить торговые пары - {e}")
            return {
                'success': False,
                'message': f'Не удалось получить список торговых пар: {e}',
                'pairs_data': [],
                'signal_counts': {'LONG': 0, 'SHORT': 0, 'NO_SIGNAL': 0},
                'execution_time': 0
            }

        if not pairs:
            logger.error("❌ ЭТАП ПРОВАЛЕН: Список торговых пар пуст")
            return {
                'success': False,
                'message': 'Список торговых пар пуст',
                'pairs_data': [],
                'signal_counts': {'LONG': 0, 'SHORT': 0, 'NO_SIGNAL': 0},
                'execution_time': 0
            }

        logger.info(f"📊 ЭТАП: Анализ {len(pairs)} пар на сигналы")

        # Анализируем все пары параллельно (батчами для избежания rate limit)
        batch_size = 200
        all_results = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]

            # Создаем задачи для батча
            tasks = [self.analyze_pair(pair) for pair in batch]

            # Выполняем батч
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Добавляем результаты
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"❌ Исключение при анализе пары: {result}")
                else:
                    all_results.append(result)

            # Прогресс
            progress = min(i + batch_size, len(pairs))
            logger.info(f"⏳ ПРОГРЕСС: {progress}/{len(pairs)} пар проанализировано")

            # Небольшая пауза между батчами
            await asyncio.sleep(0.2)

        # Фильтруем только пары с сигналами (LONG или SHORT)
        pairs_with_signals = [
            result for result in all_results
            if result['signal'] in ['LONG', 'SHORT']
        ]

        # Подсчитываем статистику
        signal_counts = {'LONG': 0, 'SHORT': 0, 'NO_SIGNAL': 0}
        for result in all_results:
            signal = result['signal']
            if signal in signal_counts:
                signal_counts[signal] += 1
            else:
                signal_counts['NO_SIGNAL'] += 1

        execution_time = time.time() - start_time

        logger.info(f"✅ ЭТАП ЗАВЕРШЕН: Найдено {len(pairs_with_signals)} сигналов за {execution_time:.1f}сек")

        return {
            'success': True,
            'pairs_data': pairs_with_signals,
            'all_pairs_data': all_results,
            'signal_counts': signal_counts,
            'total_pairs_checked': len(all_results),
            'execution_time': execution_time
        }


def load_prompt_from_file(filename: str = 'prompt2.txt') -> str:
    """
    Загрузка промпта из файла
    """
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


def create_pairs_selection_message(base_prompt: str, pairs_data: List[Dict[str, Any]]) -> str:
    """
    Создание сообщения для первичного отбора пар нейросетью
    """
    # Формируем сводную информацию по всем парам с сигналами
    pairs_summary = []

    for pair_data in pairs_data:
        summary = {
            'pair': pair_data['pair'],
            'signal': pair_data['signal'],
            'strength_metrics': pair_data['strength_metrics'],
            'key_indicators': {
                'current_ema_values': pair_data['key_indicators']['current_ema_values'],
                'current_tsi': pair_data['key_indicators']['current_tsi'],
                'trend_direction': pair_data['key_indicators']['trend_direction'],
                'last_10_ema1': pair_data['key_indicators']['last_10_ema1'],
                'last_10_ema2': pair_data['key_indicators']['last_10_ema2'],
                'last_10_ema3': pair_data['key_indicators']['last_10_ema3'],
                'last_10_tsi': pair_data['key_indicators']['last_10_tsi'],
                'last_10_tsi_signal': pair_data['key_indicators']['last_10_tsi_signal']
            },
            'recent_candles': pair_data['recent_candles']
        }
        pairs_summary.append(summary)

    # Формируем полное сообщение
    full_message = f"""{base_prompt}

=== ДАННЫЕ ДЛЯ АНАЛИЗА ===
ВСЕГО ПАР С СИГНАЛАМИ: {len(pairs_data)}
ТАЙМФРЕЙМ: 15 минут

=== СВОДКА ПО ПАРАМ ===
{json.dumps(pairs_summary, indent=2, ensure_ascii=False)}

Пожалуйста, проанализируй данные и верни JSON в формате: {{"pairs": ["BTCUSDT", "ETHUSDT"]}} или {{"pairs": []}}
"""

    return full_message


def create_detailed_analysis_message(base_prompt: str, pair_info: Dict[str, Any]) -> str:
    """
    Создание сообщения для детального анализа выбранной пары
    """
    # Извлекаем детали сигнала
    details = pair_info['details']

    # Формируем заголовок с ключевой информацией
    analysis_header = f"""=== ДЕТАЛЬНЫЙ АНАЛИЗ ПАРЫ ===
ТОРГОВАЯ ПАРА: {pair_info['pair']}
ТИП СИГНАЛА: {pair_info['signal']}
ТЕКУЩАЯ ЦЕНА: {details['last_price']:.6f}
НАПРАВЛЕНИЕ ТРЕНДА: {pair_info['key_indicators']['trend_direction']}

МЕТРИКИ СИЛЫ СИГНАЛА:
- EMA Spread: {pair_info['strength_metrics']['ema_spread']}%
- TSI Momentum: {pair_info['strength_metrics']['tsi_momentum']}
- Volume Spike: {pair_info['strength_metrics']['volume_spike']}x
- Price Move: {pair_info['strength_metrics']['price_move']}%
- Signal Age: {pair_info['strength_metrics']['signal_age']} свечей
"""

    # Формируем данные индикаторов
    indicators_section = f"""=== ЗНАЧЕНИЯ ИНДИКАТОРОВ ===
ТЕКУЩИЕ ЗНАЧЕНИЯ:
EMA7: {pair_info['key_indicators']['current_ema_values'][0]}
EMA14: {pair_info['key_indicators']['current_ema_values'][1]}
EMA28: {pair_info['key_indicators']['current_ema_values'][2]}
TSI: {pair_info['key_indicators']['current_tsi'][0]}
TSI Signal: {pair_info['key_indicators']['current_tsi'][1]}

ПОСЛЕДНИЕ 10 ЗНАЧЕНИЙ:
EMA7 VALUES: {pair_info['key_indicators']['last_10_ema1']}
EMA14 VALUES: {pair_info['key_indicators']['last_10_ema2']}
EMA28 VALUES: {pair_info['key_indicators']['last_10_ema3']}
TSI VALUES: {pair_info['key_indicators']['last_10_tsi']}
TSI SIGNAL VALUES: {pair_info['key_indicators']['last_10_tsi_signal']}

ПОЛНЫЕ МАССИВЫ ДЛЯ АНАЛИЗА:
EMA7 FULL: {pair_info['full_indicators']['ema1_values']}
EMA14 FULL: {pair_info['full_indicators']['ema2_values']}
EMA28 FULL: {pair_info['full_indicators']['ema3_values']}
TSI FULL: {pair_info['full_indicators']['tsi_values']}
TSI SIGNAL FULL: {pair_info['full_indicators']['tsi_signal_values']}
"""

    # Формируем свечные данные
    candles_section = f"""=== СВЕЧНОЙ ГРАФИК ===
ПОСЛЕДНИЕ 10 СВЕЧЕЙ:
{json.dumps(pair_info['recent_candles'], indent=2)}

ПОЛНЫЕ ДАННЫЕ СВЕЧЕЙ (последние {len(pair_info['full_candles'])} свечей):
{json.dumps(pair_info['full_candles'], indent=2)}
"""

    # Собираем полное сообщение
    full_message = f"""{base_prompt}

{analysis_header}

{indicators_section}

{candles_section}
"""

    return full_message


async def select_pairs_with_ai(pairs_data: List[Dict[str, Any]]) -> List[str]:
    """
    Первый этап: отбор пар с помощью нейросети
    """
    try:
        logger.info("🤖 ЭТАП 1: Отбор пар нейросетью")

        # Загружаем промпт для отбора
        selection_prompt = load_prompt_from_file('prompt2.txt')

        if not selection_prompt:
            logger.error("❌ ЭТАП 1 ПРОВАЛЕН: Промпт для отбора не загружен")
            return []

        # Создаем сообщение для отбора
        selection_message = create_pairs_selection_message(selection_prompt, pairs_data)

        logger.info(f"📤 ОТПРАВКА: Данных по {len(pairs_data)} парам для отбора")

        # Отправляем в нейросеть
        ai_response = await deep_seek(selection_message)

        # Парсим ответ нейросети
        try:
            # Ищем JSON в ответе
            import re
            json_match = re.search(r'\{[^}]*"pairs"[^}]*\}', ai_response)
            if json_match:
                response_data = json.loads(json_match.group())
                selected_pairs = response_data.get('pairs', [])

                logger.info(f"✅ ЭТАП 1 ЗАВЕРШЕН: Отобрано {len(selected_pairs)} пар из {len(pairs_data)}")
                if selected_pairs:
                    logger.info(f"📋 ОТОБРАННЫЕ ПАРЫ: {', '.join(selected_pairs)}")

                return selected_pairs
            else:
                logger.error("❌ ЭТАП 1 ПРОВАЛЕН: JSON не найден в ответе нейросети")
                return []

        except json.JSONDecodeError as e:
            logger.error(f"❌ ЭТАП 1 ПРОВАЛЕН: Ошибка парсинга JSON - {e}")
            logger.error(f"Ответ нейросети: {ai_response[:500]}...")
            return []

    except Exception as e:
        logger.error(f"❌ ЭТАП 1 ПРОВАЛЕН: Критическая ошибка отбора - {str(e)}")
        return []


def write_ai_response_to_file(pair_info: Dict[str, Any], ai_response: str):
    """
    Запись ответа нейросети в файл с немедленным сохранением
    """
    try:
        with open('ai_responses.log', 'a', encoding='utf-8') as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(
                f"ПАРА: {pair_info['pair']} | СИГНАЛ: {pair_info['signal']} | ВРЕМЯ: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"МЕТРИКИ: EMA_SPREAD={pair_info['strength_metrics']['ema_spread']}% | ")
            f.write(f"TSI_MOMENTUM={pair_info['strength_metrics']['tsi_momentum']} | ")
            f.write(f"VOLUME_SPIKE={pair_info['strength_metrics']['volume_spike']}x | ")
            f.write(f"TREND={pair_info['key_indicators']['trend_direction']}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"{ai_response}\n")

            # Принудительно сбрасываем буфер
            f.flush()

        logger.info(f"✅ ПОДЭТАП ЗАВЕРШЕН: {pair_info['pair']} - ответ записан в файл")

    except Exception as e:
        logger.error(f"❌ Ошибка записи в файл для {pair_info['pair']}: {str(e)}")


async def process_selected_pairs_with_ai(pairs_data: List[Dict[str, Any]], selected_pairs: List[str]):
    """
    Второй этап: детальный анализ отобранных пар
    """
    try:
        logger.info("🤖 ЭТАП 2: Детальный анализ отобранных пар")

        # Загружаем промпт для детального анализа (используем prompt.txt)
        analysis_prompt = load_prompt_from_file('prompt.txt')

        if not analysis_prompt:
            logger.error("❌ ЭТАП 2 ПРОВАЛЕН: Промпт для анализа не загружен")
            return

        # Фильтруем только отобранные пары
        selected_pairs_data = [
            pair_data for pair_data in pairs_data
            if pair_data['pair'] in selected_pairs
        ]

        if not selected_pairs_data:
            logger.warning("⚠️ ЭТАП 2: Нет данных для отобранных пар")
            return

        logger.info(f"🔄 ЭТАП 2: Обработка {len(selected_pairs_data)} отобранных пар")

        # Обрабатываем каждую отобранную пару
        for i, pair_info in enumerate(selected_pairs_data, 1):
            try:
                logger.info(
                    f"📤 ПОДЭТАП: Детальный анализ {i}/{len(selected_pairs_data)} - {pair_info['pair']} ({pair_info['signal']})")

                # Создаем сообщение для детального анализа
                analysis_message = create_detailed_analysis_message(analysis_prompt, pair_info)

                # Отправляем в нейросеть
                ai_response = await deep_seek(analysis_message)

                # Записываем ответ в файл
                write_ai_response_to_file(pair_info, ai_response)

                # Пауза между запросами
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"❌ ПОДЭТАП ПРОВАЛЕН: Ошибка анализа {pair_info['pair']} - {str(e)}")
                continue

        logger.info("✅ ЭТАП 2 ЗАВЕРШЕН: Все отобранные пары проанализированы")

    except Exception as e:
        logger.error(f"❌ ЭТАП 2 ПРОВАЛЕН: Критическая ошибка детального анализа - {str(e)}")


async def main():
    """
    Главная функция с двухэтапным анализом
    """
    try:
        logger.info("🚀 СТАРТ: Запуск двухэтапного EMA+TSI анализа")

        # Создаем анализатор с настройками индикаторов
        analyzer = TradingSignalAnalyzer(
            ema1_period=7,
            ema2_period=14,
            ema3_period=28,
            tsi_long=12,
            tsi_short=6,
            tsi_signal=6
        )

        logger.info(
            f"⚙️ НАСТРОЙКИ: EMA({analyzer.ema1_period},{analyzer.ema2_period},{analyzer.ema3_period}) | TSI({analyzer.tsi_long},{analyzer.tsi_short},{analyzer.tsi_signal})")

        # ЭТАП 0: Анализ всех пар на сигналы
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