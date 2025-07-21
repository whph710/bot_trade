import asyncio
import time
import json
import logging
import re
import os
from typing import Dict, List, Optional, Tuple

from func_async import get_usdt_linear_symbols, get_klines_async
from func_trade import (
    calculate_atr,
    analyze_last_candle,
    get_detailed_signal_info,
    check_tsi_confirmation
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
    def __init__(self,
                 atr_threshold: float = 0.01,
                 min_pairs_per_direction: int = 0,  # Уменьшено с 5 до 3 для тестирования
                 use_tsi_filter: bool = True,
                 tsi_long: int = 25,
                 tsi_short: int = 13,
                 tsi_signal: int = 13):
        """
        Инициализация анализатора торговли

        Args:
            atr_threshold: Минимальный порог ATR для фильтрации
            min_pairs_per_direction: Минимальное количество пар для анализа в каждом направлении
            use_tsi_filter: Использовать TSI как дополнительный фильтр
            tsi_long: Длинный период TSI
            tsi_short: Короткий период TSI
            tsi_signal: Период сигнальной линии TSI
        """
        self.atr_threshold = atr_threshold
        self.min_pairs_per_direction = min_pairs_per_direction
        self.use_tsi_filter = use_tsi_filter
        self.tsi_long = tsi_long
        self.tsi_short = tsi_short
        self.tsi_signal = tsi_signal

        # Счетчики для диагностики
        self.ema_signal_counts = {'LONG': 0, 'SHORT': 0, 'NO_SIGNAL': 0}
        self.tsi_confirmation_counts = {'LONG': 0, 'SHORT': 0, 'REJECTED': 0}

        if use_tsi_filter:
            logger.info(f"🔍 TSI фильтр ВКЛЮЧЕН (периоды: {tsi_long}, {tsi_short}, {tsi_signal})")
        else:
            logger.info("⚠️  TSI фильтр ОТКЛЮЧЕН")

    def read_prompt_file(self, filename: str, default_content: str = "") -> str:
        """
        Чтение промпта из файла с обработкой ошибок

        Args:
            filename: Имя файла
            default_content: Содержимое по умолчанию

        Returns:
            Содержимое файла или значение по умолчанию
        """
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
                    if content:
                        logger.info(f"✅ Промпт загружен из {filename}")
                        return content
                    else:
                        logger.warning(f"⚠️  Файл {filename} пустой, используется промпт по умолчанию")
            else:
                logger.warning(f"⚠️  Файл {filename} не найден, используется промпт по умолчанию")
        except Exception as e:
            logger.error(f"❌ Ошибка чтения файла {filename}: {e}")

        return default_content

    async def collect_and_filter_by_atr(self) -> List[str]:
        """
        Этап 1: Сбор всех пар и фильтрация по ATR

        Returns:
            Список пар, прошедших фильтрацию по ATR
        """
        logger.info("=" * 60)
        logger.info("ЭТАП 1: Сбор данных и фильтрация по ATR")
        logger.info("=" * 60)

        # Получаем все USDT пары
        all_pairs = await get_usdt_linear_symbols()
        logger.info(f"📊 Найдено {len(all_pairs)} торговых пар")

        # Фильтруем по ATR
        filtered_pairs = []
        semaphore = asyncio.Semaphore(20)

        async def check_atr_for_pair(pair: str) -> Optional[Tuple[str, float]]:
            async with semaphore:
                try:
                    candles = await get_klines_async(symbol=pair, interval=15, limit=50)
                    if not candles or len(candles) < 20:
                        return None

                    atr = calculate_atr(candles, period=14)
                    if atr >= self.atr_threshold:
                        return pair, atr
                    return None
                except Exception as e:
                    logger.debug(f"Ошибка проверки ATR для {pair}: {e}")
                    return None

        # Обрабатываем батчами
        batch_size = 50
        atr_results = []

        for i in range(0, len(all_pairs), batch_size):
            batch = all_pairs[i:i + batch_size]
            tasks = [check_atr_for_pair(pair) for pair in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, tuple):
                    pair, atr_value = result
                    atr_results.append((pair, atr_value))
                    filtered_pairs.append(pair)

            logger.info(f"Обработано {min(i + batch_size, len(all_pairs))} пар, отфильтровано: {len(filtered_pairs)}")
            await asyncio.sleep(0.1)

        # Сортируем по ATR (по убыванию - более волатильные пары первыми)
        atr_results.sort(key=lambda x: x[1], reverse=True)
        filtered_pairs = [pair for pair, _ in atr_results]

        logger.info(f"✅ Фильтрация по ATR завершена: {len(filtered_pairs)} пар прошли фильтр")
        if atr_results:
            logger.info(f"📈 ТОП-5 по ATR: {[(pair, f'{atr:.6f}') for pair, atr in atr_results[:5]]}")

        return filtered_pairs

    async def analyze_ema_signals(self, pairs: List[str]) -> Dict[str, List[str]]:
        """
        Этап 2: Анализ EMA сигналов с TSI подтверждением и разделение на направления

        Args:
            pairs: Список пар для анализа

        Returns:
            Словарь с парами, разделенными по направлениям
        """
        logger.info("=" * 60)
        logger.info(f"ЭТАП 2: Анализ EMA сигналов {'с TSI подтверждением' if self.use_tsi_filter else 'без TSI'}")
        logger.info("=" * 60)

        long_pairs = []
        short_pairs = []
        semaphore = asyncio.Semaphore(25)

        # Сбрасываем счетчики диагностики
        self.ema_signal_counts = {'LONG': 0, 'SHORT': 0, 'NO_SIGNAL': 0}
        self.tsi_confirmation_counts = {'LONG': 0, 'SHORT': 0, 'REJECTED': 0}

        async def analyze_pair_ema(pair: str) -> Optional[Tuple[str, str, Dict]]:
            async with semaphore:
                try:
                    # Нужно больше данных для TSI
                    limit = 150 if self.use_tsi_filter else 100
                    candles = await get_klines_async(symbol=pair, interval=15, limit=limit)

                    if not candles or len(candles) < (100 if self.use_tsi_filter else 50):
                        return None

                    # Сначала получаем EMA сигнал без TSI
                    ema_signal = analyze_last_candle(
                        candles,
                        use_tsi_filter=False  # Получаем чистый EMA сигнал
                    )

                    # Увеличиваем счетчик EMA сигналов
                    self.ema_signal_counts[ema_signal] += 1

                    if ema_signal not in ['LONG', 'SHORT']:
                        return None

                    # Если TSI фильтр включен, проверяем подтверждение
                    if self.use_tsi_filter:
                        tsi_confirmed = check_tsi_confirmation(
                            candles,
                            ema_signal,
                            self.tsi_long,
                            self.tsi_short,
                            self.tsi_signal
                        )

                        if tsi_confirmed:
                            self.tsi_confirmation_counts[ema_signal] += 1
                        else:
                            self.tsi_confirmation_counts['REJECTED'] += 1
                            return None

                    # Получаем детальную информацию
                    signal_details = get_detailed_signal_info(
                        candles,
                        use_tsi_filter=self.use_tsi_filter,
                        tsi_long=self.tsi_long,
                        tsi_short=self.tsi_short,
                        tsi_signal=self.tsi_signal
                    )

                    return pair, ema_signal, signal_details

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
                    pair, signal, details = result
                    if signal == 'LONG':
                        long_pairs.append(pair)
                    elif signal == 'SHORT':
                        short_pairs.append(pair)

            logger.info(f"Обработано {min(i + batch_size, len(pairs))} пар")
            await asyncio.sleep(0.1)

        # Выводим диагностическую информацию
        logger.info("📊 ДИАГНОСТИКА EMA СИГНАЛОВ:")
        for signal_type, count in self.ema_signal_counts.items():
            logger.info(f"   {signal_type}: {count}")

        if self.use_tsi_filter:
            logger.info("🔍 ДИАГНОСТИКА TSI ПОДТВЕРЖДЕНИЙ:")
            for conf_type, count in self.tsi_confirmation_counts.items():
                logger.info(f"   {conf_type}: {count}")

        logger.info(f"✅ EMA анализ завершен: LONG={len(long_pairs)}, SHORT={len(short_pairs)}")
        return {'LONG': long_pairs, 'SHORT': short_pairs}

    async def prepare_pairs_data(self, pairs: List[str]) -> Dict[str, Dict]:
        """
        Этап 3: Подготовка данных пар для анализа

        Args:
            pairs: Список пар для получения данных

        Returns:
            Словарь с данными пар
        """
        logger.info("=" * 60)
        logger.info(f"ЭТАП 3: Подготовка данных для {len(pairs)} пар")
        logger.info("=" * 60)

        pairs_with_data = {}
        semaphore = asyncio.Semaphore(25)

        async def get_pair_data(pair: str) -> Optional[Tuple[str, Dict]]:
            async with semaphore:
                try:
                    # Получаем свечи (больше данных для TSI)
                    limit = 150 if self.use_tsi_filter else 100
                    candles = await get_klines_async(symbol=pair, interval=15, limit=limit)

                    if not candles or len(candles) < (100 if self.use_tsi_filter else 50):
                        return None

                    # Получаем детальную информацию о EMA с TSI
                    ema_details = get_detailed_signal_info(
                        candles,
                        use_tsi_filter=self.use_tsi_filter,
                        tsi_long=self.tsi_long,
                        tsi_short=self.tsi_short,
                        tsi_signal=self.tsi_signal
                    )
                    atr = calculate_atr(candles, period=14)

                    return pair, {
                        'candles': candles,
                        'ema_details': ema_details,
                        'atr': atr
                    }
                except Exception as e:
                    logger.debug(f"Ошибка получения данных для {pair}: {e}")
                    return None

        # Получаем данные батчами
        batch_size = 30
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            tasks = [get_pair_data(pair) for pair in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, tuple):
                    pair, data = result
                    pairs_with_data[pair] = data

            logger.info(f"Обработано {min(i + batch_size, len(pairs))} пар")
            await asyncio.sleep(0.1)

        logger.info(f"✅ Данные получены для {len(pairs_with_data)} пар")
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

        logger.info("=" * 60)
        logger.info(f"ЭТАП 4: Анализ направления {direction} с помощью ИИ ({len(pairs_data)} пар)")
        logger.info("=" * 60)

        try:
            # Читаем промпт для первичного анализа
            prompt2 = self.read_prompt_file(
                "prompt2.txt",
                """Проанализируй торговые данные и верни результат в виде Python словаря.
                Формат: {'pairs': ['BTCUSDT', 'ETHUSDT']}. Выбери до 5 лучших пар для торговли.
                Учитывай EMA сигналы, TSI подтверждение и ATR для определения лучших возможностей."""
            )

            # Подготавливаем данные для ИИ
            analysis_data = {}
            for pair, data in pairs_data.items():
                ema_details = data['ema_details']
                analysis_data[pair] = {
                    'candles_recent': data['candles'][-20:],  # Последние 20 свечей
                    'ema_signal': ema_details['signal'],
                    'ema_alignment': ema_details['ema_alignment'],
                    'atr': data['atr'],
                    'last_price': ema_details['last_price'],
                    'ema_fast_value': ema_details['ema_fast_value'],
                    'ema_medium_value': ema_details['ema_medium_value'],
                    'ema_slow_value': ema_details['ema_slow_value'],
                    'tsi_used': ema_details.get('tsi_used', self.use_tsi_filter),
                    'tsi_confirmed': ema_details.get('tsi_confirmed', False)
                }

                # Добавляем TSI данные если доступны
                if 'tsi_value' in ema_details:
                    analysis_data[pair]['tsi_value'] = ema_details['tsi_value']
                    analysis_data[pair]['tsi_signal_value'] = ema_details['tsi_signal_value']
                    analysis_data[pair]['tsi_histogram'] = ema_details['tsi_histogram']

            # Формируем промпт с указанием направления
            tsi_info = f" с TSI подтверждением (периоды: {self.tsi_long}, {self.tsi_short}, {self.tsi_signal})" if self.use_tsi_filter else ""
            direction_prompt = f"""
            {prompt2}

            КРИТИЧЕСКИ ВАЖНО: Анализируй ТОЛЬКО {direction} позиции!

            Направление: {direction}
            EMA Сигналы: Все представленные пары имеют {direction} сигнал{tsi_info}

            Данные включают:
            - Свечи (последние 20)
            - EMA сигналы и выравнивание (7, 14, 28 периоды)
            - TSI подтверждение (если включено): пересечение TSI линии с сигнальной линией
            - ATR для определения волатильности
            - Текущие значения EMA для анализа силы тренда

            {'TSI Логика: для LONG - TSI пересекает сигнальную линию снизу вверх, для SHORT - сверху вниз' if self.use_tsi_filter else ''}
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

            logger.info(f"🤖 ИИ выбрал {len(selected_pairs)} пар для {direction}: {selected_pairs}")

            # Берем первую пару для финального анализа
            selected_pair = selected_pairs[0]
            return selected_pair

        except Exception as e:
            logger.error(f"❌ Ошибка при анализе направления {direction}: {e}")
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
        logger.info("=" * 60)
        logger.info(f"ЭТАП 5: Финальный анализ для {pair} ({direction})")
        logger.info("=" * 60)

        try:
            # Читаем промпт для финального анализа
            main_prompt = self.read_prompt_file(
                "prompt.txt",
                """Ты опытный трейдер. Проанализируй данные и дай торговые рекомендации.
                Укажи точку входа, стоп-лосс, тейк-профит и рекомендуемое плечо.
                Используй технический анализ на основе EMA, TSI и ATR."""
            )

            # Подготавливаем полные данные для финального анализа
            full_data = {
                'pair': pair,
                'direction': direction,
                'full_candles': pair_data['candles'],
                'ema_details': pair_data['ema_details'],
                'atr': pair_data['atr'],
                'current_price': pair_data['ema_details']['last_price']
            }

            # Формируем финальный промпт
            tsi_confirmation = ""
            if self.use_tsi_filter and pair_data['ema_details'].get('tsi_confirmed'):
                tsi_confirmation = f"\n✅ TSI ПОДТВЕРЖДЕНИЕ: Сигнал подтвержден пересечением TSI (периоды: {self.tsi_long}, {self.tsi_short}, {self.tsi_signal})"

            final_prompt = f"""
            {main_prompt}

            ТОРГОВОЕ ЗАДАНИЕ:
            Пара: {pair}
            Направление: {direction}
            Фильтры: EMA выравнивание{'+ TSI подтверждение' if self.use_tsi_filter else ''}{tsi_confirmation}

            Требуется определить:
            1. Точку входа (Entry)
            2. Стоп-лосс (Stop Loss)
            3. Тейк-профит (Take Profit)
            4. Рекомендуемое плечо (Leverage)

            Данные включают:
            - Полные свечи для технического анализа
            - EMA сигналы (7, 14, 28)
            {'- TSI индикатор с подтверждением пересечения' if self.use_tsi_filter else ''}
            - ATR для определения волатильности и размера позиции
            - Текущую цену и EMA значения

            ВАЖНО: Все рекомендации должны соответствовать направлению {direction}!
            Используй ATR для определения оптимальных уровней стоп-лосса и тейк-профита.
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
                'ema_signal': pair_data['ema_details']['signal'],
                'tsi_used': pair_data['ema_details'].get('tsi_used', self.use_tsi_filter),
                'tsi_confirmed': pair_data['ema_details'].get('tsi_confirmed', False)
            }

        except Exception as e:
            logger.error(f"❌ Ошибка при финальном анализе {pair}: {e}")
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
                logger.warning("⚠️  Недостаточно пар прошло фильтрацию по ATR")
                return {'LONG': None, 'SHORT': None}

            # Этап 2: Анализ EMA сигналов с TSI подтверждением
            ema_signals = await self.analyze_ema_signals(atr_filtered_pairs)

            results = {}

            # Анализируем каждое направление
            for direction in ['LONG', 'SHORT']:
                direction_pairs = ema_signals.get(direction, [])

                if len(direction_pairs) < self.min_pairs_per_direction:
                    logger.warning(
                        f"⚠️  Недостаточно пар для {direction}: {len(direction_pairs)} (требуется минимум {self.min_pairs_per_direction})")
                    results[direction] = None
                    continue

                # Этап 3: Подготавливаем данные пар
                pairs_with_data = await self.prepare_pairs_data(direction_pairs)

                if not pairs_with_data:
                    logger.warning(f"⚠️  Нет данных для анализа {direction}")
                    results[direction] = None
                    continue

                # Этап 4: Первичный анализ с ИИ
                selected_pair = await self.analyze_direction_with_ai(pairs_with_data, direction)

                if not selected_pair or selected_pair not in pairs_with_data:
                    logger.warning(f"⚠️  ИИ не выбрал пару для {direction}")
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
            logger.error(f"❌ Критическая ошибка в полном анализе: {e}")
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

            # Показываем информацию о TSI
            if result.get('tsi_used', False):
                tsi_status = "✅ ПОДТВЕРЖДЕН" if result.get('tsi_confirmed', False) else "❌ НЕ ПОДТВЕРЖДЕН"
                print(f"TSI фильтр: {tsi_status}")
            else:
                print("TSI фильтр: ОТКЛЮЧЕН")

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
        # Создаем анализатор с TSI фильтром
        analyzer = TradingAnalyzer(
            atr_threshold=0.01,  # Минимальный ATR
            min_pairs_per_direction=3,  # Уменьшено для тестирования
            use_tsi_filter=True,  # Включаем TSI фильтр
            tsi_long=25,  # Длинный период TSI
            tsi_short=13,  # Короткий период TSI
            tsi_signal=13  # Период сигнальной линии TSI
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