import asyncio
import time
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from func_async import get_usdt_linear_symbols, get_klines_async, get_orderbook_async
from func_trade import (
    calculate_atr,
    analyze_last_candle,
    get_detailed_signal_info
)
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


class TradingAnalyzer:
    def __init__(self, atr_threshold: float = 0.01, min_pairs_per_direction: int = 14):
        """
        Инициализация анализатора торговли

        Args:
            atr_threshold: Минимальный порог ATR для фильтрации
            min_pairs_per_direction: Минимальное количество пар для анализа в каждом направлении
        """
        self.atr_threshold = atr_threshold
        self.min_pairs_per_direction = min_pairs_per_direction

    async def collect_and_filter_by_atr(self) -> List[str]:
        """
        Этап 1: Сбор всех пар и фильтрация по ATR

        Returns:
            Список пар, прошедших фильтрацию по ATR
        """
        logger.info("Этап 1: Сбор данных и фильтрация по ATR")

        # Получаем все USDT пары
        all_pairs = await get_usdt_linear_symbols()
        logger.info(f"Найдено {len(all_pairs)} торговых пар")

        # Фильтруем по ATR
        filtered_pairs = []
        semaphore = asyncio.Semaphore(20)

        async def check_atr_for_pair(pair: str) -> Optional[str]:
            async with semaphore:
                try:
                    candles = await get_klines_async(symbol=pair, interval=15, limit=50)
                    if not candles or len(candles) < 20:
                        return None

                    atr = calculate_atr(candles, period=14)
                    if atr >= self.atr_threshold:
                        return pair
                    return None
                except Exception as e:
                    logger.debug(f"Ошибка проверки ATR для {pair}: {e}")
                    return None

        # Обрабатываем батчами
        batch_size = 50
        for i in range(0, len(all_pairs), batch_size):
            batch = all_pairs[i:i + batch_size]
            tasks = [check_atr_for_pair(pair) for pair in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, str):
                    filtered_pairs.append(result)

            logger.info(f"Обработано {min(i + batch_size, len(all_pairs))} пар, отфильтровано: {len(filtered_pairs)}")
            await asyncio.sleep(0.1)

        logger.info(f"Фильтрация по ATR завершена: {len(filtered_pairs)} пар прошли фильтр")
        return filtered_pairs

    async def analyze_ema_signals(self, pairs: List[str]) -> Dict[str, List[str]]:
        """
        Этап 2: Анализ EMA сигналов и разделение на направления

        Args:
            pairs: Список пар для анализа

        Returns:
            Словарь с парами, разделенными по направлениям
        """
        logger.info("Этап 2: Анализ EMA сигналов")

        long_pairs = []
        short_pairs = []
        semaphore = asyncio.Semaphore(25)

        async def analyze_pair_ema(pair: str) -> Optional[Tuple[str, str]]:
            async with semaphore:
                try:
                    candles = await get_klines_async(symbol=pair, interval=15, limit=100)
                    if not candles or len(candles) < 50:
                        return None

                    signal = analyze_last_candle(candles)
                    if signal in ['LONG', 'SHORT']:
                        return pair, signal
                    return None
                except Exception as e:
                    logger.debug(f"Ошибка EMA анализа для {pair}: {e}")
                    return None

        # Анализируем EMA сигналы
        batch_size = 40
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            tasks = [analyze_pair_ema(pair) for pair in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, tuple):
                    pair, signal = result
                    if signal == 'LONG':
                        long_pairs.append(pair)
                    elif signal == 'SHORT':
                        short_pairs.append(pair)

            await asyncio.sleep(0.1)

        logger.info(f"EMA анализ завершен: LONG={len(long_pairs)}, SHORT={len(short_pairs)}")
        return {'LONG': long_pairs, 'SHORT': short_pairs}

    async def add_orderbook_data(self, pairs: List[str]) -> Dict[str, Dict]:
        """
        Этап 3: Добавление данных стакана к парам

        Args:
            pairs: Список пар для получения данных стакана

        Returns:
            Словарь с данными пар и их стаканами
        """
        logger.info(f"Этап 3: Получение данных стакана для {len(pairs)} пар")

        pairs_with_data = {}
        semaphore = asyncio.Semaphore(15)

        async def get_pair_full_data(pair: str) -> Optional[Tuple[str, Dict]]:
            async with semaphore:
                try:
                    # Получаем свечи и стакан параллельно
                    candles_task = get_klines_async(symbol=pair, interval=15, limit=100)
                    orderbook_task = get_orderbook_async(symbol=pair, limit=25)

                    candles, orderbook = await asyncio.gather(candles_task, orderbook_task)

                    if not candles or len(candles) < 50:
                        return None

                    # Получаем детальную информацию о EMA
                    ema_details = get_detailed_signal_info(candles)
                    atr = calculate_atr(candles, period=14)

                    return pair, {
                        'candles': candles,
                        'orderbook': orderbook,
                        'ema_details': ema_details,
                        'atr': atr
                    }
                except Exception as e:
                    logger.debug(f"Ошибка получения данных для {pair}: {e}")
                    return None

        # Получаем данные батчами
        batch_size = 20
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            tasks = [get_pair_full_data(pair) for pair in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, tuple):
                    pair, data = result
                    pairs_with_data[pair] = data

            await asyncio.sleep(0.1)

        logger.info(f"Данные стакана получены для {len(pairs_with_data)} пар")
        return pairs_with_data

    async def analyze_direction_with_ai(self, pairs_data: Dict[str, Dict], direction: str) -> Optional[str]:
        """
        Этап 4: Анализ направления с помощью ИИ

        Args:
            pairs_data: Данные пар для анализа
            direction: Направление анализа ('LONG' или 'SHORT')

        Returns:
            Выбранная ИИ пара или None
        """
        if not pairs_data:
            logger.warning(f"Нет данных для анализа направления {direction}")
            return None

        logger.info(f"Этап 4: Анализ направления {direction} с помощью ИИ ({len(pairs_data)} пар)")

        try:
            # Читаем промпт для первичного анализа
            try:
                with open("prompt2.txt", 'r', encoding='utf-8') as file:
                    prompt2 = file.read()
            except FileNotFoundError:
                prompt2 = """Проанализируй торговые данные и верни результат в виде Python словаря.
                           Формат: {'pairs': ['BTCUSDT', 'ETHUSDT']}. Выбери до 5 лучших пар для торговли.
                           Учитывай EMA сигналы, ATR и данные стакана."""

            # Подготавливаем данные для ИИ
            analysis_data = {}
            for pair, data in pairs_data.items():
                analysis_data[pair] = {
                    'candles_recent': data['candles'][-20:],  # Последние 20 свечей
                    'ema_signal': data['ema_details']['signal'],
                    'ema_alignment': data['ema_details']['ema_alignment'],
                    'atr': data['atr'],
                    'orderbook_bids': data['orderbook']['b'][:5],  # Топ 5 бидов
                    'orderbook_asks': data['orderbook']['a'][:5],  # Топ 5 асков
                    'last_price': data['ema_details']['last_price']
                }

            # Формируем промпт с указанием направления
            direction_prompt = f"""
            {prompt2}

            КРИТИЧЕСКИ ВАЖНО: Анализируй ТОЛЬКО {direction} позиции!

            Направление: {direction}
            EMA Сигналы: Все представленные пары имеют {direction} сигнал

            Данные включают:
            - Свечи (последние 20)
            - EMA сигналы и выравнивание
            - ATR для волатильности
            - Данные стакана (5 лучших бидов/асков)
            """

            # Первичный анализ
            ai_response = await deep_seek(
                data=str(analysis_data),
                prompt=direction_prompt,
                max_tokens=1500,
                timeout=30
            )

            # Парсим ответ
            selected_pairs = self.parse_ai_response(ai_response)
            if not selected_pairs:
                logger.warning(f"ИИ не смог выбрать пары для {direction}")
                return None

            logger.info(f"ИИ выбрал {len(selected_pairs)} пар для {direction}: {selected_pairs}")

            # Берем первую пару для финального анализа
            selected_pair = selected_pairs[0]
            return selected_pair

        except Exception as e:
            logger.error(f"Ошибка при анализе направления {direction}: {e}")
            return None

    async def final_analysis_with_ai(self, pair: str, pair_data: Dict, direction: str) -> Optional[Dict]:
        """
        Этап 5: Финальный анализ с получением точки входа, стопа и тейка

        Args:
            pair: Выбранная пара
            pair_data: Данные пары
            direction: Направление торговли

        Returns:
            Словарь с торговыми рекомендациями
        """
        logger.info(f"Этап 5: Финальный анализ для {pair} ({direction})")

        try:
            # Читаем промпт для финального анализа
            try:
                with open("prompt.txt", 'r', encoding='utf-8') as file:
                    main_prompt = file.read()
            except FileNotFoundError:
                main_prompt = """Ты опытный трейдер. Проанализируй данные и дай торговые рекомендации.
                               Укажи точку входа, стоп-лосс, тейк-профит и рекомендуемое плечо."""

            # Подготавливаем полные данные для финального анализа
            full_data = {
                'pair': pair,
                'direction': direction,
                'full_candles': pair_data['candles'],
                'ema_details': pair_data['ema_details'],
                'atr': pair_data['atr'],
                'orderbook': pair_data['orderbook'],
                'current_price': pair_data['ema_details']['last_price']
            }

            # Формируем финальный промпт
            final_prompt = f"""
            {main_prompt}

            ТОРГОВОЕ ЗАДАНИЕ:
            Пара: {pair}
            Направление: {direction}

            Требуется определить:
            1. Точку входа (Entry)
            2. Стоп-лосс (Stop Loss)
            3. Тейк-профит (Take Profit)
            4. Рекомендуемое плечо (Leverage)

            Данные включают:
            - Полные свечи для технического анализа
            - EMA сигналы (7, 14, 28)
            - ATR для определения волатильности
            - Текущий стакан заявок
            - Текущую цену

            ВАЖНО: Все рекомендации должны соответствовать направлению {direction}!
            """

            # Финальный анализ
            ai_response = await deep_seek(
                data=str(full_data),
                prompt=final_prompt,
                max_tokens=2000,
                timeout=45
            )

            return {
                'pair': pair,
                'direction': direction,
                'analysis': ai_response,
                'current_price': pair_data['ema_details']['last_price'],
                'atr': pair_data['atr'],
                'ema_signal': pair_data['ema_details']['signal']
            }

        except Exception as e:
            logger.error(f"Ошибка при финальном анализе {pair}: {e}")
            return None

    def parse_ai_response(self, ai_response: str) -> List[str]:
        """Парсинг ответа ИИ для извлечения пар"""
        if not ai_response:
            return []

        # Поиск JSON
        try:
            data = json.loads(ai_response.strip())
            if isinstance(data, dict) and 'pairs' in data:
                return data['pairs'][:5]  # Максимум 5 пар
        except json.JSONDecodeError:
            pass

        # Поиск пар в тексте
        pairs_pattern = r'["\']([A-Z]+USDT)["\']'
        found_pairs = re.findall(pairs_pattern, ai_response)

        if found_pairs:
            return list(dict.fromkeys(found_pairs))[:5]  # Убираем дубликаты, максимум 5

        return []

    async def run_full_analysis(self) -> Dict[str, Optional[Dict]]:
        """
        Основная функция - полный анализ для обоих направлений

        Returns:
            Словарь с результатами для LONG и SHORT
        """
        logger.info("🚀 ЗАПУСК ПОЛНОГО ТОРГОВОГО АНАЛИЗА")
        start_time = time.time()

        try:
            # Этап 1: Фильтрация по ATR
            atr_filtered_pairs = await self.collect_and_filter_by_atr()
            if len(atr_filtered_pairs) < 20:
                logger.warning("Недостаточно пар прошло фильтрацию по ATR")
                return {'LONG': None, 'SHORT': None}

            # Этап 2: Анализ EMA сигналов
            ema_signals = await self.analyze_ema_signals(atr_filtered_pairs)

            results = {}

            # Анализируем каждое направление
            for direction in ['LONG', 'SHORT']:
                direction_pairs = ema_signals.get(direction, [])

                if len(direction_pairs) < self.min_pairs_per_direction:
                    logger.warning(f"Недостаточно пар для {direction}: {len(direction_pairs)}")
                    results[direction] = None
                    continue

                # Этап 3: Добавляем данные стакана
                pairs_with_data = await self.add_orderbook_data(direction_pairs)

                if not pairs_with_data:
                    logger.warning(f"Нет данных для анализа {direction}")
                    results[direction] = None
                    continue

                # Этап 4: Первичный анализ с ИИ
                selected_pair = await self.analyze_direction_with_ai(pairs_with_data, direction)

                if not selected_pair or selected_pair not in pairs_with_data:
                    logger.warning(f"ИИ не выбрал пару для {direction}")
                    results[direction] = None
                    continue

                # Этап 5: Финальный анализ
                final_result = await self.final_analysis_with_ai(
                    selected_pair,
                    pairs_with_data[selected_pair],
                    direction
                )

                results[direction] = final_result

            elapsed_time = time.time() - start_time
            logger.info(f"✅ Полный анализ завершен за {elapsed_time:.2f} секунд")

            return results

        except Exception as e:
            logger.error(f"Критическая ошибка в полном анализе: {e}")
            return {'LONG': None, 'SHORT': None}


def print_results(results: Dict[str, Optional[Dict]]):
    """Вывод результатов анализа"""
    print("\n" + "=" * 80)
    print("🎯 РЕЗУЛЬТАТЫ ТОРГОВОГО АНАЛИЗА")
    print("=" * 80)

    for direction in ['LONG', 'SHORT']:
        result = results.get(direction)

        print(f"\n📊 {direction} ПОЗИЦИЯ:")
        print("-" * 40)

        if result:
            print(f"Пара: {result['pair']}")
            print(f"Направление: {result['direction']}")
            print(f"Текущая цена: {result['current_price']}")
            print(f"ATR: {result['atr']:.6f}")
            print(f"EMA сигнал: {result['ema_signal']}")
            print(f"\nАНАЛИЗ:")
            print("-" * 20)
            print(result['analysis'])
        else:
            print("❌ Не найдено подходящих возможностей")

    print("\n" + "=" * 80)


async def main():
    """Главная функция"""
    logger.info("🔥 ЗАПУСК ТОРГОВОГО БОТА")

    try:
        # Создаем анализатор
        analyzer = TradingAnalyzer(
            atr_threshold=0.01,  # Минимальный ATR
            min_pairs_per_direction=5  # Минимум пар для анализа
        )

        # Запускаем полный анализ
        results = await analyzer.run_full_analysis()

        # Выводим результаты
        print_results(results)

    except KeyboardInterrupt:
        logger.info("⏹️  Остановлено пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
    finally:
        logger.info("👋 Завершение работы")


if __name__ == "__main__":
    asyncio.run(main())