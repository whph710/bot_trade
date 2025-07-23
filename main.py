import asyncio
import json
import logging
import time
from typing import List, Dict, Any
import aiohttp
from func_trade import get_signal_details, check_ema_tsi_signal

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

        Args:
            ema1_period: Период быстрой EMA
            ema2_period: Период средней EMA
            ema3_period: Период медленной EMA
            tsi_long: Длинный период TSI
            tsi_short: Короткий период TSI
            tsi_signal: Период сигнальной линии TSI
        """
        self.ema1_period = ema1_period
        self.ema2_period = ema2_period
        self.ema3_period = ema3_period
        self.tsi_long = tsi_long
        self.tsi_short = tsi_short
        self.tsi_signal = tsi_signal

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

        Returns:
            Список торговых пар
        """
        try:
            url = f"{self.base_url}/v5/market/instruments-info"
            params = {
                'category': 'spot'
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

    async def get_klines(self, symbol: str, interval: str = "15", limit: int = 200) -> List[List[str]]:
        """
        Получение данных свечей для торговой пары

        Args:
            symbol: Торговая пара
            interval: Интервал (по умолчанию 15 минут)
            limit: Количество свечей

        Returns:
            Список свечей в формате Bybit
        """
        try:
            url = f"{self.base_url}/v5/market/kline"
            params = {
                'category': 'spot',
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['result']['list']
                else:
                    logger.warning(f"⚠️  Ошибка получения данных для {symbol}: {response.status}")
                    return []

        except Exception as e:
            logger.warning(f"⚠️  Ошибка при получении свечей {symbol}: {e}")
            return []

    async def analyze_pair(self, symbol: str) -> Dict[str, Any]:
        """
        Анализ одной торговой пары на наличие EMA+TSI сигнала

        Args:
            symbol: Торговая пара

        Returns:
            Результат анализа пары
        """
        try:
            # Получаем данные свечей
            candles = await self.get_klines(symbol)

            if not candles or len(candles) < 50:
                return {
                    'pair': symbol,
                    'signal': 'NO_SIGNAL',
                    'reason': 'INSUFFICIENT_DATA',
                    'details': {}
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

            return {
                'pair': symbol,
                'signal': signal,
                'details': details
            }

        except Exception as e:
            logger.warning(f"⚠️  Ошибка анализа {symbol}: {e}")
            return {
                'pair': symbol,
                'signal': 'ERROR',
                'reason': str(e),
                'details': {}
            }

    async def analyze_all_pairs(self) -> Dict[str, Any]:
        """
        Анализ всех торговых пар на наличие сигналов

        Returns:
            Результат полного анализа
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
        batch_size = 10  # Анализируем по 10 пар одновременно
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
            await asyncio.sleep(0.1)

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

                # Выводим найденные сигналы
                if result['pairs_data']:
                    logger.info("=" * 70)
                    logger.info("🎯 НАЙДЕННЫЕ СИГНАЛЫ:")
                    logger.info("=" * 70)

                    for pair_data in result['pairs_data']:
                        signal_emoji = "📈" if pair_data['signal'] == 'LONG' else "📉"
                        logger.info(f"{signal_emoji} {pair_data['pair']}: {pair_data['signal']} "
                                    f"(Цена: {pair_data['details']['last_price']:.6f})")

                    # Сохраняем результаты
                    save_results_to_file(result)

                    # Отправляем каждую пару отдельно в нейросеть
                    await process_pairs_with_ai(result['pairs_data'], analyzer)

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


def save_results_to_file(result: Dict[str, Any]):
    """
    Сохранение результатов анализа в файл

    Args:
        result: Результат анализа
    """
    try:
        # Подготавливаем данные для сохранения
        save_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'execution_time': result['execution_time'],
            'total_pairs_checked': result['total_pairs_checked'],
            'signal_counts': result['signal_counts'],
            'signals_found': []
        }

        # Добавляем информацию о найденных сигналах
        for pair_info in result['pairs_data']:
            save_data['signals_found'].append({
                'pair': pair_info['pair'],
                'signal': pair_info['signal'],
                'last_price': pair_info['details']['last_price'],
                'ema1': pair_info['details']['ema1'],
                'ema2': pair_info['details']['ema2'],
                'ema3': pair_info['details']['ema3'],
                'tsi_value': pair_info['details']['tsi_value'],
                'tsi_signal_value': pair_info['details']['tsi_signal_value'],
                'ema_alignment': pair_info['details']['ema_alignment'],
                'tsi_crossover_direction': pair_info['details']['tsi_crossover_direction'],
                'reason': pair_info['details']['reason']
            })

        # Сохраняем в JSON файл
        with open('trading_analysis_result.json', 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info("💾 Результаты сохранены в файл 'trading_analysis_result.json'")

    except Exception as e:
        logger.error(f"❌ Ошибка сохранения результатов: {e}")


def load_prompt_from_file(filename: str) -> str:
    """
    Загрузка промпта из файла

    Args:
        filename: Имя файла с промптом

    Returns:
        Текст промпта или пустая строка при ошибке
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning(f"⚠️  Файл {filename} не найден")
        return ""
    except Exception as e:
        logger.error(f"❌ Ошибка чтения файла {filename}: {e}")
        return ""


def format_pair_data(pair_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Красивое форматирование данных пары для нейросети

    Args:
        pair_info: Информация о торговой паре

    Returns:
        Отформатированные данные
    """
    return {
        'pair': pair_info['pair'],
        'signal': pair_info['signal'],
        'last_price': pair_info['details']['last_price'],
        'ema1': pair_info['details']['ema1'],
        'ema2': pair_info['details']['ema2'],
        'ema3': pair_info['details']['ema3'],
        'tsi_value': pair_info['details']['tsi_value'],
        'tsi_signal_value': pair_info['details']['tsi_signal_value'],
        'ema_alignment': pair_info['details']['ema_alignment'],
        'tsi_crossover_direction': pair_info['details']['tsi_crossover_direction'],
        'reason': pair_info['details']['reason']
    }


async def process_pairs_with_ai(pairs_data: List[Dict[str, Any]], analyzer: TradingSignalAnalyzer):
    """
    Обработка каждой пары отдельно с помощью нейросети

    Args:
        pairs_data: Список пар с сигналами
        analyzer: Экземпляр анализатора для получения свечей
    """
    try:
        # Загружаем промпты из файлов
        main_prompt = load_prompt_from_file('prompt_main.txt')
        analysis_prompt = load_prompt_from_file('prompt_analysis.txt')

        if not main_prompt or not analysis_prompt:
            logger.warning("⚠️  Не удалось загрузить промпты из файлов, используем стандартные")
            main_prompt = "Проанализируй торговый сигнал для криптовалютной пары:"
            analysis_prompt = "Дай рекомендации по входу в позицию и риск-менеджменту."

        logger.info(f"🤖 Начинаем обработку {len(pairs_data)} пар с помощью ИИ...")

        # Создаем папку для результатов если её нет
        import os
        os.makedirs('ai_results', exist_ok=True)

        # Обрабатываем каждую пару отдельно
        for i, pair_info in enumerate(pairs_data, 1):
            try:
                logger.info(f"🔍 Анализ пары {i}/{len(pairs_data)}: {pair_info['pair']}")

                # Получаем свечи для этой пары
                candles = await analyzer.get_klines(pair_info['pair'], interval="15", limit=100)

                # Форматируем данные пары
                formatted_data = format_pair_data(pair_info)

                # Создаем полный промпт для этой пары
                full_prompt = create_single_pair_prompt(
                    main_prompt,
                    analysis_prompt,
                    formatted_data,
                    candles
                )

                # Сохраняем промпт для каждой пары
                prompt_filename = f'ai_results/prompt_{pair_info["pair"]}_{pair_info["signal"]}.txt'
                with open(prompt_filename, 'w', encoding='utf-8') as f:
                    f.write(full_prompt)

                logger.info(f"💾 Промпт для {pair_info['pair']} сохранен: {prompt_filename}")

                # Здесь можно добавить отправку в API нейросети
                # await send_to_openai_api(full_prompt, pair_info['pair'])

                # Небольшая пауза между запросами
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"❌ Ошибка обработки пары {pair_info['pair']}: {e}")
                continue

        logger.info("✅ Обработка пар завершена!")

    except Exception as e:
        logger.error(f"❌ Ошибка в process_pairs_with_ai: {e}")


def create_single_pair_prompt(main_prompt: str, analysis_prompt: str, pair_data: Dict[str, Any],
                              candles: List[List[str]]) -> str:
    """
    Создание полного промпта для одной торговой пары

    Args:
        main_prompt: Основной промпт из файла
        analysis_prompt: Промпт для анализа из файла
        pair_data: Отформатированные данные пары
        candles: Данные свечей

    Returns:
        Полный промпт для отправки в ИИ
    """
    # Основной промпт
    prompt = f"{main_prompt}\n\n"

    # Данные по паре
    prompt += "ДАННЫЕ ТОРГОВОЙ ПАРЫ:\n"
    prompt += "=" * 50 + "\n"
    prompt += f"Пара: {pair_data['pair']}\n"
    prompt += f"Сигнал: {pair_data['signal']}\n"
    prompt += f"Текущая цена: {pair_data['last_price']:.6f}\n"
    prompt += f"EMA7: {pair_data['ema1']:.6f}\n"
    prompt += f"EMA14: {pair_data['ema2']:.6f}\n"
    prompt += f"EMA28: {pair_data['ema3']:.6f}\n"
    prompt += f"TSI: {pair_data['tsi_value']:.2f}\n"
    prompt += f"TSI Signal: {pair_data['tsi_signal_value']:.2f}\n"
    prompt += f"Направление EMA: {pair_data['ema_alignment']}\n"
    prompt += f"Направление пересечения TSI: {pair_data['tsi_crossover_direction']}\n"
    prompt += f"Причина сигнала: {pair_data['reason']}\n\n"

    # Последние свечи (берем последние 20 для анализа)
    if candles:
        prompt += "ПОСЛЕДНИЕ СВЕЧИ (OHLCV):\n"
        prompt += "=" * 50 + "\n"
        prompt += "Timestamp | Open | High | Low | Close | Volume\n"
        prompt += "-" * 50 + "\n"

        # Берем последние 20 свечей и форматируем
        recent_candles = candles[:20]  # Bybit возвращает в обратном порядке
        for candle in recent_candles:
            timestamp = candle[0]
            open_price = float(candle[1])
            high_price = float(candle[2])
            low_price = float(candle[3])
            close_price = float(candle[4])
            volume = float(candle[5])

            prompt += f"{timestamp} | {open_price:.6f} | {high_price:.6f} | {low_price:.6f} | {close_price:.6f} | {volume:.2f}\n"

        prompt += "\n"

    # Промпт для анализа
    prompt += f"{analysis_prompt}\n"

    return prompt


async def send_to_ai_analysis(pairs_data: List[Dict[str, Any]]):
    """
    Отправка данных пар с сигналами для анализа нейросетью

    Args:
        pairs_data: Список пар с сигналами
    """
    try:
        # Формируем текст для анализа ИИ
        ai_prompt = create_ai_prompt(pairs_data)

        logger.info("🤖 Подготовлен промпт для анализа ИИ:")
        logger.info("=" * 50)
        logger.info(ai_prompt)
        logger.info("=" * 50)

        # Здесь можно добавить отправку в OpenAI API, Claude API и т.д.
        # Пока что просто сохраняем промпт в файл
        with open('ai_analysis_prompt.txt', 'w', encoding='utf-8') as f:
            f.write(ai_prompt)

        logger.info("💾 Промпт для ИИ сохранен в файл 'ai_analysis_prompt.txt'")

    except Exception as e:
        logger.error(f"❌ Ошибка подготовки анализа ИИ: {e}")


def create_ai_prompt(pairs_data: List[Dict[str, Any]]) -> str:
    """
    Создание промпта для анализа нейросетью

    Args:
        pairs_data: Данные пар с сигналами

    Returns:
        Текст промпта
    """
    prompt = """Проанализируй торговые сигналы по криптовалютным парам на основе индикатора EMA+TSI:

НАЙДЕННЫЕ СИГНАЛЫ:

"""

    for pair_data in pairs_data:
        details = pair_data['details']
        prompt += f"""
Пара: {pair_data['pair']}
Сигнал: {pair_data['signal']}
Текущая цена: {details['last_price']:.6f}
EMA7: {details['ema1']:.6f}
EMA14: {details['ema2']:.6f}  
EMA28: {details['ema3']:.6f}
TSI: {details['tsi_value']:.2f}
TSI Signal: {details['tsi_signal_value']:.2f}
Направление EMA: {details['ema_alignment']}
Направление пересечения TSI: {details['tsi_crossover_direction']}
Причина сигнала: {details['reason']}
---"""

    prompt += """

ЗАДАЧА:
1. Проанализируй качество найденных сигналов
2. Определи наиболее перспективные пары для торговли
3. Оцени общее состояние рынка на основе соотношения LONG/SHORT сигналов
4. Дай рекомендации по риск-менеджменту
5. Укажи на что обратить внимание при входе в позицию

Ответ дай структурированно с конкретными рекомендациями."""

    return prompt


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