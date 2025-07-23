import asyncio
import json
import logging
import time
from typing import List, Dict, Any
import aiohttp
import numpy as np
from func_trade import get_signal_details, check_ema_tsi_signal, calculate_ema, calculate_tsi
from func_async import get_klines_async
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

        self.base_url = "https://api.bybit.com"
        self.session = None

    async def __aenter__(self):
        """Асинхронный контекст менеджер - вход"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Асинхронный контекст менеджер - выход"""
        if self.session:
            await self.session.close()

    async def get_trading_pairs(self) -> List[str]:
        """
        Получение списка всех торговых пар USDT
        """
        try:
            url = f"{self.base_url}/v5/market/instruments-info"
            params = {
                'category': 'linear'
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # Фильтруем только USDT пары и активные
                    pairs = []
                    for instrument in data['result']['list']:
                        symbol = instrument['symbol']
                        status = instrument['status']

                        # Берем только USDT пары и активные
                        if (symbol.endswith('USDT') and
                                status == 'Trading' and
                                not symbol.startswith('USDT')):  # Исключаем обратные пары
                            pairs.append(symbol)

                    logger.info(f"📊 Найдено {len(pairs)} торговых пар USDT")
                    return pairs

                else:
                    logger.error(f"❌ Ошибка получения торговых пар: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"❌ Ошибка при получении торговых пар: {e}")
            return []

    async def get_klines(self, symbol: str, interval: str = "15", limit: int = None) -> List[List[str]]:
        """
        Получение данных свечей для торговой пары
        """
        if limit is None:
            limit = self.required_candles_for_analysis

        try:
            url = f"{self.base_url}/v5/market/kline"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    klines = data['result']['list']

                    # Проверяем порядок и при необходимости разворачиваем
                    if klines and len(klines) > 1:
                        if int(klines[0][0]) > int(klines[-1][0]):
                            klines.reverse()

                    return klines
                else:
                    logger.warning(f"⚠️  Ошибка получения данных для {symbol}: {response.status}")
                    return []

        except Exception as e:
            logger.warning(f"⚠️  Ошибка при получении свечей {symbol}: {e}")
            return []

    def calculate_indicators_for_candles(self, candles: List[List[str]]) -> Dict[str, Any]:
        """
        Рассчитывает все индикаторы для свечей и возвращает их значения
        """
        if not candles or len(candles) < self.required_candles_for_analysis:
            return {}

        # Извлекаем цены закрытия
        prices = np.array([float(candle[4]) for candle in candles])

        # Рассчитываем EMA
        ema1_values = calculate_ema(prices, self.ema1_period)
        ema2_values = calculate_ema(prices, self.ema2_period)
        ema3_values = calculate_ema(prices, self.ema3_period)

        # Рассчитываем TSI
        tsi_data = calculate_tsi(candles, self.tsi_long, self.tsi_short, self.tsi_signal)

        return {
            'ema1_values': ema1_values.tolist(),
            'ema2_values': ema2_values.tolist(),
            'ema3_values': ema3_values.tolist(),
            'tsi_values': tsi_data['tsi'].tolist(),
            'tsi_signal_values': tsi_data['signal'].tolist()
        }

    async def analyze_pair(self, symbol: str) -> Dict[str, Any]:
        """
        Анализ одной торговой пары на наличие EMA+TSI сигнала
        """
        try:
            # Получаем данные свечей для анализа (больше данных)
            candles = await self.get_klines(symbol, limit=self.required_candles_for_analysis)

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

            # Если есть сигнал, рассчитываем индикаторы для последних 100 свечей
            if signal in ['LONG', 'SHORT']:
                # Берем последние 100 свечей для отправки в нейросеть
                candles_for_ai = candles[-self.candles_for_ai:] if len(candles) >= self.candles_for_ai else candles

                # Рассчитываем индикаторы для этих свечей
                indicators = self.calculate_indicators_for_candles(candles)

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
            logger.warning(f"⚠️  Ошибка анализа {symbol}: {e}")
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

        # Получаем список торговых пар
        pairs = await self.get_trading_pairs()

        if not pairs:
            return {
                'success': False,
                'message': 'Не удалось получить список торговых пар',
                'pairs_data': [],
                'signal_counts': {'LONG': 0, 'SHORT': 0, 'NO_SIGNAL': 0},
                'execution_time': 0
            }

        logger.info(f"🔍 Начинаем анализ {len(pairs)} торговых пар...")

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
                    logger.error(f"❌ Исключение при анализе: {result}")
                else:
                    all_results.append(result)

            # Прогресс
            logger.info(f"📈 Проанализировано: {min(i + batch_size, len(pairs))}/{len(pairs)} пар")

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
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.error(f"❌ Файл {filename} не найден")
        return ""
    except Exception as e:
        logger.error(f"❌ Ошибка чтения файла {filename}: {e}")
        return ""


def format_pair_data_for_ai(pair_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Форматирование данных пары для отправки в ИИ
    """
    # Формируем структурированные данные для нейросети
    formatted_data = {
        'pair': pair_info['pair'],
        'signal': pair_info['signal'],
        'signal_details': {
            'last_price': pair_info['details']['last_price'],
            'ema1': pair_info['details']['ema1'],
            'ema2': pair_info['details']['ema2'],
            'ema3': pair_info['details']['ema3'],
            'tsi_value': pair_info['details']['tsi_value'],
            'tsi_signal_value': pair_info['details']['tsi_signal_value'],
            'ema_alignment': pair_info['details']['ema_alignment'],
            'tsi_crossover_direction': pair_info['details']['tsi_crossover_direction'],
            'reason': pair_info['details']['reason']
        },
        'candles_data': {
            'count': len(pair_info['candles']),
            'candles': pair_info['candles']  # Полные данные свечей
        },
        'indicators_data': pair_info['indicators']  # Полные данные индикаторов
    }

    return formatted_data


def create_full_prompt(base_prompt: str, pair_data: Dict[str, Any]) -> str:
    """
    Создание полного промпта для отправки в ИИ
    """
    # Создаем сокращенную версию для лога (без полных массивов)
    pair_summary = {
        'pair': pair_data['pair'],
        'signal': pair_data['signal'],
        'signal_details': pair_data['signal_details'],
        'candles_count': pair_data['candles_data']['count'],
        'indicators_count': {
            'ema1_values': len(pair_data['indicators_data'].get('ema1_values', [])),
            'ema2_values': len(pair_data['indicators_data'].get('ema2_values', [])),
            'ema3_values': len(pair_data['indicators_data'].get('ema3_values', [])),
            'tsi_values': len(pair_data['indicators_data'].get('tsi_values', [])),
            'tsi_signal_values': len(pair_data['indicators_data'].get('tsi_signal_values', []))
        }
    }

    # Формируем полный промпт (с полными данными)
    full_prompt = f"""{base_prompt}

ДАННЫЕ ТОРГОВОЙ ПАРЫ:
{json.dumps(pair_data, ensure_ascii=False, indent=2)}"""

    logger.info(
        f"📊 Подготовлены данные для {pair_data['pair']}: {json.dumps(pair_summary, ensure_ascii=False, indent=2)}")

    return full_prompt


async def process_pairs_with_ai(pairs_data: List[Dict[str, Any]]):
    """
    Обработка каждой пары отдельно с помощью нейросети
    """
    try:
        # Загружаем промпт из файла
        base_prompt = load_prompt_from_file('prompt.txt')

        if not base_prompt:
            logger.error("❌ Не удалось загрузить промпт из файла prompt.txt")
            return

        logger.info(f"🤖 Начинаем обработку {len(pairs_data)} пар с помощью ИИ...")

        # Обрабатываем каждую пару отдельно
        for i, pair_info in enumerate(pairs_data, 1):
            try:
                logger.info(f"🔍 Анализ пары {i}/{len(pairs_data)}: {pair_info['pair']}")

                # Проверяем наличие данных
                if not pair_info.get('candles') or not pair_info.get('indicators'):
                    logger.warning(f"⚠️ Недостаточно данных для {pair_info['pair']}")
                    continue

                # Форматируем данные пары
                formatted_data = format_pair_data_for_ai(pair_info)

                # Создаем полный промпт
                full_prompt = create_full_prompt(base_prompt, formatted_data)

                # Отправляем в нейросеть
                logger.info(f"📤 Отправка в ИИ: {pair_info['pair']}")
                ai_response = await deep_seek(full_prompt)

                # Логируем результат
                logger.info(f"📥 Ответ ИИ для {pair_info['pair']}: {ai_response}")

                # Пауза между запросами
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"❌ Ошибка обработки пары {pair_info['pair']}: {e}")
                continue

        logger.info("✅ Обработка пар завершена!")

    except Exception as e:
        logger.error(f"❌ Ошибка в process_pairs_with_ai: {e}")


async def main():
    """
    Главная функция - запуск EMA+TSI анализа торговых пар
    """
    try:
        logger.info("🚀 ЗАПУСК EMA+TSI ТОРГОВОГО БОТА")
        logger.info("=" * 70)

        # Создаем анализатор с настройками индикаторов
        async with TradingSignalAnalyzer(
                ema1_period=7,  # Быстрая EMA
                ema2_period=14,  # Средняя EMA
                ema3_period=28,  # Медленная EMA
                tsi_long=12,  # Длинный период TSI
                tsi_short=6,  # Короткий период TSI
                tsi_signal=6  # Период сигнальной линии TSI
        ) as analyzer:

            logger.info(f"📊 Настройки анализатора:")
            logger.info(f"   - Свечей для анализа: {analyzer.required_candles_for_analysis}")
            logger.info(f"   - Свечей для ИИ: {analyzer.candles_for_ai}")
            logger.info(f"   - EMA периоды: {analyzer.ema1_period}, {analyzer.ema2_period}, {analyzer.ema3_period}")
            logger.info(f"   - TSI параметры: {analyzer.tsi_long}, {analyzer.tsi_short}, {analyzer.tsi_signal}")

            # Запускаем полный анализ
            result = await analyzer.analyze_all_pairs()

            # Обрабатываем результат
            if result['success']:
                logger.info("=" * 70)
                logger.info("✅ АНАЛИЗ УСПЕШНО ЗАВЕРШЕН")
                logger.info("=" * 70)

                # Выводим основную статистику
                logger.info(f"⏱️  Время выполнения: {result['execution_time']:.2f} секунд")
                logger.info(f"📊 Проверено пар: {result['total_pairs_checked']}")

                # Детальная статистика сигналов
                signal_counts = result['signal_counts']
                logger.info(f"📈 LONG сигналы: {signal_counts['LONG']}")
                logger.info(f"📉 SHORT сигналы: {signal_counts['SHORT']}")
                logger.info(f"⚪ Без сигналов: {signal_counts['NO_SIGNAL']}")

                total_signals = signal_counts['LONG'] + signal_counts['SHORT']
                logger.info(f"🎯 Всего сигналов найдено: {total_signals}")

                # Обрабатываем найденные сигналы
                if result['pairs_data']:
                    logger.info("=" * 70)
                    logger.info("🎯 НАЙДЕННЫЕ СИГНАЛЫ:")
                    logger.info("=" * 70)

                    for pair_data in result['pairs_data']:
                        signal_emoji = "📈" if pair_data['signal'] == 'LONG' else "📉"
                        candles_count = len(pair_data.get('candles', []))
                        indicators_ready = bool(pair_data.get('indicators'))

                        logger.info(f"{signal_emoji} {pair_data['pair']}: {pair_data['signal']} "
                                    f"(Цена: {pair_data['details']['last_price']:.6f}, "
                                    f"Свечей: {candles_count}, Индикаторы: {'✅' if indicators_ready else '❌'})")

                    # Отправляем каждую пару в нейросеть
                    logger.info("=" * 70)
                    logger.info("🤖 АНАЛИЗ НЕЙРОСЕТЬЮ")
                    logger.info("=" * 70)
                    await process_pairs_with_ai(result['pairs_data'])

                else:
                    logger.info("🔍 Сигналы не найдены на текущий момент")

            else:
                logger.error("❌ АНАЛИЗ ЗАВЕРШИЛСЯ С ОШИБКОЙ")
                logger.error(f"Причина: {result.get('message', 'Unknown error')}")

    except KeyboardInterrupt:
        logger.info("🛑 Анализ прерван пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка в main(): {e}")
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