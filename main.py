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
    Анализатор торговых сигналов EMA+TSI для всех торговых пар Bybit
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
        self.required_candles_for_analysis = max(self.ema3_period, self.tsi_long, 50) + 50
        # Для отправки в нейросеть берем последние 100 свечей
        self.candles_for_ai = 100

    async def analyze_pair(self, symbol: str) -> Dict[str, Any]:
        """
        Анализ одной торговой пары на наличие EMA+TSI сигнала
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
                    'reason': 'INSUFFICIENT_DATA',
                    'details': {},
                    'candles': [],
                    'indicators': {}
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

            # Если есть сигнал, подготавливаем данные для нейросети
            if signal in ['LONG', 'SHORT']:
                # Берем последние 100 свечей для отправки в нейросеть
                candles_for_ai = candles[-self.candles_for_ai:] if len(candles) >= self.candles_for_ai else candles

                # Рассчитываем индикаторы для этих свечей
                indicators = calculate_indicators_for_candles(
                    candles,
                    self.ema1_period,
                    self.ema2_period,
                    self.ema3_period,
                    self.tsi_long,
                    self.tsi_short,
                    self.tsi_signal
                )

                # Обрезаем индикаторы до соответствующего размера
                if indicators:
                    start_idx = len(candles) - len(candles_for_ai)
                    indicators_for_ai = {
                        'ema1_values': indicators['ema1_values'][start_idx:],
                        'ema2_values': indicators['ema2_values'][start_idx:],
                        'ema3_values': indicators['ema3_values'][start_idx:],
                        'tsi_values': indicators['tsi_values'][start_idx:],
                        'tsi_signal_values': indicators['tsi_signal_values'][start_idx:]
                    }
                else:
                    indicators_for_ai = {}

                return {
                    'pair': symbol,
                    'signal': signal,
                    'details': details,
                    'candles': candles_for_ai,
                    'indicators': indicators_for_ai
                }
            else:
                return {
                    'pair': symbol,
                    'signal': signal,
                    'details': details,
                    'candles': [],
                    'indicators': {}
                }

        except Exception as e:
            return {
                'pair': symbol,
                'signal': 'ERROR',
                'reason': str(e),
                'details': {},
                'candles': [],
                'indicators': {}
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
        batch_size = 50  # Анализируем по 50 пар одновременно
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
            'pairs_data': pairs_with_signals,  # Только пары с сигналами
            'all_pairs_data': all_results,  # Все проанализированные пары
            'signal_counts': signal_counts,
            'total_pairs_checked': len(all_results),
            'execution_time': execution_time
        }


def load_prompt_from_file(filename: str = 'prompt.txt') -> str:
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


def create_ai_message(base_prompt: str, pair_info: Dict[str, Any]) -> str:
    """
    Создание правильно структурированного сообщения для ИИ
    """
    # Извлекаем детали сигнала
    details = pair_info['details']

    # Формируем заголовок с ключевой информацией для анализа
    analysis_header = f"""=== ДАННЫЕ ДЛЯ АНАЛИЗА ===
ТОРГОВАЯ ПАРА: {pair_info['pair']}
ТИП СИГНАЛА: {pair_info['signal']} ({details['reason']})
ТЕКУЩАЯ ЦЕНА: {details['last_price']:.6f}
ТАЙМФРЕЙМ: 15 минут
КОЛИЧЕСТВО СВЕЧЕЙ: {len(pair_info['candles'])}
РАСПОЛОЖЕНИЕ EMA: {details['ema_alignment']}
НАПРАВЛЕНИЕ TSI: {details['tsi_crossover_direction']}
"""

    # Формируем данные индикаторов
    indicators_section = f"""=== ЗНАЧЕНИЯ ИНДИКАТОРОВ ===
EMA7 (текущее): {details['ema1']:.6f}
EMA14 (текущее): {details['ema2']:.6f} 
EMA28 (текущее): {details['ema3']:.6f}
TSI (текущее): {details['tsi_value']:.2f}
TSI Signal (текущее): {details['tsi_signal_value']:.2f}

EMA7 VALUES: {pair_info['indicators']['ema1_values']}
EMA14 VALUES: {pair_info['indicators']['ema2_values']}
EMA28 VALUES: {pair_info['indicators']['ema3_values']}
TSI VALUES: {pair_info['indicators']['tsi_values']}
TSI SIGNAL VALUES: {pair_info['indicators']['tsi_signal_values']}
"""

    # Формируем свечные данные
    candles_section = f"""=== СВЕЧНОЙ ГРАФИК (последние {len(pair_info['candles'])} свечей) ===
Формат: [timestamp, open, high, low, close, volume, turnover]
{json.dumps(pair_info['candles'], indent=2)}
"""

    # Собираем полное сообщение
    full_message = f"""{base_prompt}

{analysis_header}

{indicators_section}

{candles_section}
"""

    return full_message


def write_ai_response_to_file(pair_info: Dict[str, Any], ai_response: str):
    """
    Запись ответа нейросети в файл с немедленным сохранением
    """
    try:
        with open('ai_responses.log', 'a', encoding='utf-8') as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(
                f"ПАРА: {pair_info['pair']} | СИГНАЛ: {pair_info['signal']} | ВРЕМЯ: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"{ai_response}\n")

            # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Принудительно сбрасываем буфер
            f.flush()

        logger.info(f"✅ ПОДЭТАП ЗАВЕРШЕН: {pair_info['pair']} - ответ записан в файл")

    except Exception as e:
        logger.error(f"❌ Ошибка записи в файл для {pair_info['pair']}: {str(e)}")


async def process_pairs_with_ai(pairs_data: List[Dict[str, Any]]):
    """
    Обработка каждой пары отдельно с помощью нейросети
    """
    try:
        logger.info("🤖 ЭТАП: Подготовка к анализу нейросетью")

        # Загружаем промпт из файла
        base_prompt = load_prompt_from_file('prompt.txt')

        if not base_prompt:
            logger.error("❌ ЭТАП ПРОВАЛЕН: Промпт не загружен, прерываем анализ ИИ")
            return

        logger.info(f"🔄 ЭТАП: Обработка {len(pairs_data)} пар нейросетью")

        # Обрабатываем каждую пару отдельно
        for i, pair_info in enumerate(pairs_data, 1):
            try:
                logger.info(f"📤 ПОДЭТАП: Отправка {i}/{len(pairs_data)} - {pair_info['pair']} ({pair_info['signal']})")

                # Проверяем наличие данных
                if not pair_info.get('candles') or not pair_info.get('indicators'):
                    logger.warning(f"⚠️ ПОДЭТАП ПРОПУЩЕН: Недостаточно данных для {pair_info['pair']}")
                    continue

                # Создаем правильно структурированное сообщение
                ai_message = create_ai_message(base_prompt, pair_info)

                # Отправляем в нейросеть
                ai_response = await deep_seek(ai_message)

                # Используем отдельную функцию для записи с flush
                write_ai_response_to_file(pair_info, ai_response)

                # Пауза между запросами
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"❌ ПОДЭТАП ПРОВАЛЕН: Ошибка обработки {pair_info['pair']} - {str(e)}")
                continue

        logger.info("✅ ЭТАП ЗАВЕРШЕН: Все пары обработаны нейросетью")

    except Exception as e:
        logger.error(f"❌ ЭТАП ПРОВАЛЕН: Критическая ошибка анализа ИИ - {str(e)}")


async def main():
    """
    Главная функция - запуск EMA+TSI анализа торговых пар
    """
    try:
        logger.info("🚀 СТАРТ: Запуск EMA+TSI торгового бота")

        # Создаем анализатор с настройками индикаторов
        analyzer = TradingSignalAnalyzer(
            ema1_period=7,  # Быстрая EMA
            ema2_period=14,  # Средняя EMA
            ema3_period=28,  # Медленная EMA
            tsi_long=12,  # Длинный период TSI
            tsi_short=6,  # Короткий период TSI
            tsi_signal=6  # Период сигнальной линии TSI
        )

        logger.info(
            f"⚙️ НАСТРОЙКИ: EMA({analyzer.ema1_period},{analyzer.ema2_period},{analyzer.ema3_period}) | TSI({analyzer.tsi_long},{analyzer.tsi_short},{analyzer.tsi_signal})")

        # Запускаем полный анализ
        result = await analyzer.analyze_all_pairs()

        # Обрабатываем результат
        if result['success']:
            logger.info("🎯 РЕЗУЛЬТАТ: Анализ завершен успешно")

            # Основная статистика
            signal_counts = result['signal_counts']
            total_signals = signal_counts['LONG'] + signal_counts['SHORT']

            logger.info(
                f"📊 СТАТИСТИКА: {total_signals} сигналов из {result['total_pairs_checked']} пар за {result['execution_time']:.1f}сек")
            logger.info(f"📈 LONG: {signal_counts['LONG']} | 📉 SHORT: {signal_counts['SHORT']}")

            # Обрабатываем найденные сигналы
            if result['pairs_data']:
                logger.info("🎯 НАЙДЕННЫЕ СИГНАЛЫ:")
                for pair_data in result['pairs_data']:
                    signal_emoji = "📈" if pair_data['signal'] == 'LONG' else "📉"
                    logger.info(f"{signal_emoji} {pair_data['pair']}: {pair_data['details']['last_price']:.6f}")

                # Отправляем каждую пару в нейросеть
                await process_pairs_with_ai(result['pairs_data'])

            else:
                logger.info("🔍 РЕЗУЛЬТАТ: Сигналы не найдены")

        else:
            logger.error(f"❌ ФИНАЛ: Анализ завершился с ошибкой - {result.get('message', 'Unknown error')}")

    except KeyboardInterrupt:
        logger.info("🛑 ПРЕРЫВАНИЕ: Остановлено пользователем")
    except Exception as e:
        logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    """
    Точка входа в программу
    """
    try:
        # Запускаем асинхронную главную функцию
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Программа остановлена пользователем")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()