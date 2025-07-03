import asyncio
import time
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from func_async import get_usdt_linear_symbols, get_klines_async
from func_trade import (
    CVDNadarayaWatsonEMAIndicator,
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


def calculate_atr(candles: List[List[str]], period: int = 14) -> float:
    """Простое вычисление ATR для фильтрации пар."""
    if len(candles) < period:
        return 0.0

    true_ranges = []
    for i in range(1, len(candles)):
        high = float(candles[i][2])
        low = float(candles[i][3])
        prev_close = float(candles[i - 1][4])

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)

    return sum(true_ranges[-period:]) / period if true_ranges else 0.0


def detect_candlestick_signals(pattern_data: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Простое обнаружение паттернов японских свечей."""
    signals = {'long': [], 'short': []}

    for pair, data in pattern_data.items():
        candles = data['candles']
        if len(candles) < 3:
            continue

        # Простые паттерны
        last_candle = candles[-1]
        prev_candle = candles[-2]

        close = float(last_candle[4])
        open_price = float(last_candle[1])
        high = float(last_candle[2])
        low = float(last_candle[3])

        prev_close = float(prev_candle[4])
        prev_open = float(prev_candle[1])

        # Бычий паттерн
        if (close > open_price and  # Зеленая свеча
                close > prev_close and  # Закрытие выше предыдущего
                (high - close) < (close - open_price) * 0.3):  # Небольшая верхняя тень
            signals['long'].append(pair)

        # Медвежий паттерн
        elif (close < open_price and  # Красная свеча
              close < prev_close and  # Закрытие ниже предыдущего
              (close - low) < (open_price - close) * 0.3):  # Небольшая нижняя тень
            signals['short'].append(pair)

    return signals


def get_user_direction_choice() -> str:
    """Получение выбора направления от пользователя."""
    while True:
        print("\n" + "=" * 50)
        print("ВЫБОР НАПРАВЛЕНИЯ ТОРГОВЛИ")
        print("=" * 50)
        print("Выберите направление для анализа:")
        print("1. long  - только длинные позиции")
        print("2. short - только короткие позиции")
        print("3. 0     - все направления (автономный выбор)")
        print("-" * 50)

        choice = input("Введите ваш выбор (long/short/0): ").strip().lower()

        if choice in ['long', 'short', '0']:
            if choice == '0':
                print(f"✓ Выбрано: АВТОНОМНЫЙ АНАЛИЗ (все направления)")
            else:
                print(f"✓ Выбрано направление: {choice.upper()}")
            return choice
        else:
            print("❌ Неверный выбор! Введите: long, short или 0")


async def process_single_pair_full(pair: str, limit: int = 100, interval: str = "15") -> Optional[Tuple[str, Dict]]:
    """Обработка одной торговой пары с CVD + Nadaraya-Watson + EMA анализом."""
    try:
        candles_raw = await get_klines_async(symbol=pair, interval=interval, limit=limit)

        if not candles_raw or len(candles_raw) < 4:
            return None

        # Быстрая фильтрация по ATR с адаптивным порогом
        atr_candles = candles_raw[-20:] if len(candles_raw) >= 20 else candles_raw
        atr = calculate_atr(atr_candles)

        # Адаптивный порог ATR на основе цены
        price = float(candles_raw[-1][4])  # Последняя цена закрытия
        min_atr = 0.01 if price < 1 else 0.02 if price < 100 else 0.05

        if atr <= min_atr:
            return None

        # CVD + Nadaraya-Watson + EMA анализ
        cvd_nw_ema_signal = None
        cvd_nw_ema_details = None

        # Проверяем достаточно ли данных для анализа
        if len(candles_raw) >= 150:  # Минимум для корректной работы
            try:
                # Получаем простой сигнал с EMA подтверждением
                cvd_nw_ema_signal = analyze_last_candle(candles_raw)

                # Получаем детальную информацию с EMA данными
                cvd_nw_ema_details = get_detailed_signal_info(candles_raw)

            except Exception as e:
                logger.warning(f"Ошибка CVD+NW+EMA анализа для {pair}: {e}")

        return pair, {
            "candles_full": candles_raw,
            "candles_3": candles_raw[-3:],
            "candles_20": candles_raw[-20:] if len(candles_raw) >= 20 else candles_raw,
            "atr": atr,
            "cvd_nw_ema_signal": cvd_nw_ema_signal,
            "cvd_nw_ema_details": cvd_nw_ema_details
        }

    except Exception as e:
        logger.error(f"Ошибка обработки {pair}: {e}")
        return None


async def collect_all_data() -> Dict[str, Dict]:
    """Сбор данных с улучшенным контролем параллелизма."""
    start_time = time.time()
    logger.info("Сбор данных по торговым парам...")

    try:
        usdt_pairs = await get_usdt_linear_symbols()
        semaphore = asyncio.Semaphore(30)

        async def process_with_semaphore(pair):
            async with semaphore:
                return await process_single_pair_full(pair, limit=200)

        # Батч-обработка для лучшей производительности
        batch_size = 50
        filtered_data = {}
        error_count = 0
        cvd_ema_signals_count = {'LONG': 0, 'SHORT': 0, 'NO_SIGNAL': 0}

        for i in range(0, len(usdt_pairs), batch_size):
            batch = usdt_pairs[i:i + batch_size]
            tasks = [process_with_semaphore(pair) for pair in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    error_count += 1
                    continue
                if result is not None:
                    pair, data = result
                    filtered_data[pair] = data

                    # Подсчитываем CVD+NW+EMA сигналы
                    if data.get('cvd_nw_ema_signal'):
                        cvd_ema_signals_count[data['cvd_nw_ema_signal']] += 1

            # Короткая пауза между батчами
            if i + batch_size < len(usdt_pairs):
                await asyncio.sleep(0.1)

        elapsed_time = time.time() - start_time
        logger.info(f"Загружено {len(filtered_data)} пар за {elapsed_time:.2f}с (ошибок: {error_count})")
        logger.info(
            f"CVD+NW+EMA сигналы: LONG={cvd_ema_signals_count['LONG']}, SHORT={cvd_ema_signals_count['SHORT']}, NO_SIGNAL={cvd_ema_signals_count['NO_SIGNAL']}")

        return filtered_data

    except Exception as e:
        logger.error(f"Критическая ошибка при сборе данных: {e}")
        return {}


def extract_data_for_patterns(all_data: Dict[str, Dict]) -> Dict[str, Dict]:
    """Извлечение данных для паттернов."""
    return {
        pair: {"candles": data["candles_3"]}
        for pair, data in all_data.items()
        if "candles_3" in data and len(data["candles_3"]) >= 3
    }


def extract_cvd_ema_signal_pairs(all_data: Dict[str, Dict], signal_type: str) -> List[str]:
    """Извлечение пар с определенным CVD+NW+EMA сигналом."""
    pairs = []
    for pair, data in all_data.items():
        if data.get('cvd_nw_ema_signal') == signal_type:
            pairs.append(pair)
    return pairs


def extract_data_subset(all_data: Dict[str, Dict], pairs: List[str], candle_key: str) -> Dict[str, List]:
    """Извлечение подмножества данных."""
    return {
        pair: all_data[pair][candle_key]
        for pair in pairs
        if pair in all_data and candle_key in all_data[pair]
    }


def parse_ai_response(ai_response: str) -> Optional[Dict]:
    """Парсинг ответа ИИ."""
    if not ai_response or ai_response.strip() == "":
        return None

    # Быстрая проверка на JSON
    try:
        return json.loads(ai_response.strip())
    except json.JSONDecodeError:
        pass

    # Поиск JSON блока
    json_pattern = r'\{[^{}]*"pairs"[^{}]*\[[^\]]*\][^{}]*\}'
    json_match = re.search(json_pattern, ai_response, re.DOTALL)

    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback - поиск пар в тексте
    pairs_pattern = r'["\']([A-Z]+USDT)["\']'
    found_pairs = re.findall(pairs_pattern, ai_response)

    if found_pairs:
        unique_pairs = list(dict.fromkeys(found_pairs))[:10]
        return {'pairs': unique_pairs}

    return None


def get_filtered_pairs_by_direction(candlestick_signals: Dict, cvd_ema_signals: Dict, direction: str) -> List[str]:
    """Получение пар, отфильтрованных по выбранному направлению с учетом CVD+NW+EMA сигналов."""
    if direction == '0':
        # Автономный режим - возвращаем все пары
        all_pairs = set()

        # Добавляем пары из паттернов японских свечей
        for dir_name in ['long', 'short']:
            if dir_name in candlestick_signals:
                all_pairs.update(candlestick_signals[dir_name])

        # Добавляем пары из CVD+NW+EMA сигналов
        all_pairs.update(cvd_ema_signals.get('LONG', []))
        all_pairs.update(cvd_ema_signals.get('SHORT', []))

        selected_pairs = list(all_pairs)
        logger.info(f"Автономный режим: найдено {len(selected_pairs)} пар с потенциальными паттернами")
        logger.info(
            f"  - Японские свечи: {len(candlestick_signals.get('long', []))} long, {len(candlestick_signals.get('short', []))} short")
        logger.info(
            f"  - CVD+NW+EMA: {len(cvd_ema_signals.get('LONG', []))} LONG, {len(cvd_ema_signals.get('SHORT', []))} SHORT")

    else:
        # Фильтруем по выбранному направлению
        selected_pairs = set()

        # Добавляем из японских свечей
        if direction in candlestick_signals:
            selected_pairs.update(candlestick_signals[direction])

        # Добавляем из CVD+NW+EMA сигналов
        cvd_direction = direction.upper()
        if cvd_direction in cvd_ema_signals:
            selected_pairs.update(cvd_ema_signals[cvd_direction])

        selected_pairs = list(selected_pairs)
        logger.info(f"Направление {direction.upper()}: найдено {len(selected_pairs)} пар")

        if not selected_pairs:
            logger.warning(f"Нет найденных паттернов для направления {direction.upper()}")

    return selected_pairs


def create_direction_system_prompt(base_prompt: str, direction: str) -> str:
    """Создание системного промпта с учетом направления."""
    if direction == '0':
        # Автономный режим
        direction_addition = """
        КРИТИЧЕСКИ ВАЖНО: 
        - Самостоятельно определи оптимальное направление (long/short) для КАЖДОЙ пары
        - Основывайся на ВСЕХ доступных технических индикаторах, включая CVD, Nadaraya-Watson и EMA
        - EMA выравнивание: Fast > Medium > Slow для LONG, Fast < Medium < Slow для SHORT
        - НЕ следуй предвзятым предположениям о направлении
        - Выбери ОДНУ наиболее перспективную возможность
        - Учитывай подтверждение сигналов всеми тремя индикаторами (CVD + NW + EMA)
        """
    else:
        # Конкретное направление
        direction_addition = f"""
        ВАЖНОЕ ОГРАНИЧЕНИЕ: Рассматривать сделки ТОЛЬКО {direction.upper()}

        Анализируй данные исключительно с точки зрения {direction.upper()} позиций.
        Игнорируй возможности для противоположного направления.
        Особое внимание уделяй:
        - CVD и Nadaraya-Watson сигналам для {direction.upper()} направления
        - EMA выравниванию для {direction.upper()} (Fast>Medium>Slow для LONG, Fast<Medium<Slow для SHORT)
        - Подтверждению всех трех систем индикаторов
        """

    return f"{base_prompt}\n{direction_addition}"


async def analyze_with_ai(data: Dict, direction: str, cvd_ema_data: Dict = None) -> Optional[Dict]:
    """Первичный анализ с ИИ с учетом выбранного направления и CVD+NW+EMA данных."""
    try:
        # Читаем промпт
        try:
            with open("prompt2.txt", 'r', encoding='utf-8') as file:
                prompt2 = file.read()
        except FileNotFoundError:
            prompt2 = """Проанализируй торговые данные и верни результат в виде Python словаря.
                       Формат: {'pairs': ['BTCUSDT', 'ETHUSDT']}. Выбери до 10 лучших пар для торговли.
                       Учитывай CVD (Cumulative Volume Delta), Nadaraya-Watson envelope и EMA сигналы."""

        base_system_prompt = f"""{prompt2}

        ДАННЫЕ: Свечи в хронологическом порядке (от старых к новым).
        Последний индекс = текущая свеча.

        ДОПОЛНИТЕЛЬНО: Доступны сигналы CVD + Nadaraya-Watson + EMA:
        - CVD показывает накопленную дельту объемов (бычий/медвежий sentiment)
        - Nadaraya-Watson envelope - адаптивный конверт для определения точек входа
        - EMA (9, 21, 50) - подтверждение направления тренда
        - Сигналы генерируются только при подтверждении ВСЕХ трех систем индикаторов
        - LONG: пересечение цены под нижней границей + бычий CVD + EMA 9>21>50
        - SHORT: пересечение над верхней границей + медвежий CVD + EMA 9<21<50"""

        # Создаем промпт с учетом направления
        if direction == '0':
            system_prompt = f"{base_system_prompt}\n\nВАЖНО: Анализируй ВСЕ возможности для Long И Short позиций. Не ограничивайся одним направлением - ищи лучшие торговые возможности в любом направлении. Особое внимание уделяй сигналам CVD+NW+EMA с полным подтверждением."
        else:
            system_prompt = f"{base_system_prompt}\n\nВАЖНОЕ ОГРАНИЧЕНИЕ: Рассматривать сделки ТОЛЬКО {direction.upper()}\nАнализируй данные исключительно с точки зрения {direction.upper()} позиций. Приоритет - сигналам CVD+NW+EMA с полным подтверждением для {direction.upper()}."

        # Добавляем информацию о CVD+EMA сигналах в контекст
        cvd_ema_context = ""
        if cvd_ema_data:
            cvd_ema_context = f"\n\nCVD+NW+EMA СИГНАЛЫ (С ПОЛНЫМ ПОДТВЕРЖДЕНИЕМ):\n"
            for signal_type, pairs in cvd_ema_data.items():
                if pairs:
                    cvd_ema_context += f"- {signal_type}: {', '.join(pairs[:10])}\n"

        full_prompt = system_prompt + cvd_ema_context

        logger.info(f"Первичный анализ ИИ: {len(data)} пар, направление: {direction if direction != '0' else 'AUTO'}")

        ai_response = await deep_seek(
            data=str(data),
            prompt=full_prompt,
            max_tokens=2000,
            timeout=45
        )

        parsed_data = parse_ai_response(ai_response)

        if parsed_data and isinstance(parsed_data, dict) and 'pairs' in parsed_data:
            return parsed_data

        # Fallback
        available_pairs = list(data.keys())[:5]
        logger.warning("Используем fallback для выбора пар")
        return {'pairs': available_pairs}

    except Exception as e:
        logger.error(f"Ошибка при анализе с ИИ: {e}")
        return None


async def final_ai_analysis(data: Dict, direction: str, all_data: Dict = None) -> Optional[str]:
    """Финальный анализ с учетом выбранного направления и CVD+NW+EMA данных."""
    try:
        try:
            with open('prompt.txt', 'r', encoding='utf-8') as file:
                main_prompt = file.read()
        except FileNotFoundError:
            main_prompt = """Ты опытный трейдер. Проанализируй данные и дай рекомендации.
                           Обрати особое внимание на сигналы CVD (Cumulative Volume Delta), Nadaraya-Watson envelope и EMA."""

        base_system_prompt = f"""
        {main_prompt}

        ДАННЫЕ: Свечи в хронологическом порядке (от старых к новым).
        Формат: [timestamp, open, high, low, close, volume, turnover]
        Последний индекс = текущая свеча.

        ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ:
        - CVD (Cumulative Volume Delta): Показывает накопленную дельту объемов
        - Nadaraya-Watson Envelope: Адаптивный конверт для определения точек входа
        - EMA (9, 21, 50): Подтверждение направления тренда
        - Японские свечи: Классические паттерны разворота и продолжения
        - ATR: Для определения волатильности

        ВАЖНО: Сигналы генерируются только при подтверждении ВСЕХ систем индикаторов.
        """

        # Добавляем информацию о CVD+NW+EMA сигналах для анализируемых пар
        cvd_ema_info = ""
        if all_data:
            cvd_ema_info = "\n\nCVD+NW+EMA АНАЛИЗ ДЛЯ ВЫБРАННЫХ ПАР:\n"
            for pair in data.keys():
                if pair in all_data and all_data[pair].get('cvd_nw_ema_details'):
                    details = all_data[pair]['cvd_nw_ema_details']
                    cvd_ema_info += f"- {pair}: Сигнал={details.get('signal', 'N/A')}, CVD={details.get('cvd_status', 'N/A')}, EMA={details.get('ema_alignment', 'N/A')}, Цена={details.get('last_price', 'N/A')}\n"

        # Создаем финальный промпт с учетом направления
        system_prompt = create_direction_system_prompt(base_system_prompt + cvd_ema_info, direction)

        direction_display = direction.upper() if direction != '0' else 'АВТОНОМНЫЙ'
        logger.info(f"Финальный анализ ИИ: {len(data)} пар, режим: {direction_display}")

        return await deep_seek(
            data=str(data),
            prompt=system_prompt,
            timeout=60
        )

    except Exception as e:
        logger.error(f"Ошибка при финальном анализе: {e}")
        return None


async def run_trading_analysis(direction: str) -> Optional[str]:
    """Основная функция анализа с учетом выбранного направления и CVD+NW+EMA."""
    try:
        direction_display = direction.upper() if direction != '0' else 'АВТОНОМНЫЙ'
        logger.info(f"АНАЛИЗ ТОРГОВЫХ ВОЗМОЖНОСТЕЙ - РЕЖИМ: {direction_display}")

        # Этап 1: Сбор данных
        all_data = await collect_all_data()
        if not all_data:
            return None

        # Этап 2: Поиск паттернов японских свечей
        pattern_data = extract_data_for_patterns(all_data)
        candlestick_signals = detect_candlestick_signals(pattern_data)

        # Этап 3: Извлечение CVD+NW+EMA сигналов
        cvd_ema_signals = {
            'LONG': extract_cvd_ema_signal_pairs(all_data, 'LONG'),
            'SHORT': extract_cvd_ema_signal_pairs(all_data, 'SHORT')
        }

        # Этап 4: Фильтрация по направлению
        selected_pairs = get_filtered_pairs_by_direction(candlestick_signals, cvd_ema_signals, direction)
        if not selected_pairs:
            direction_msg = f"для направления {direction.upper()}" if direction != '0' else ""
            return f"Нет найденных торговых паттернов {direction_msg}. Попробуйте позже или выберите другое направление."

        # Этап 5: Первичный анализ ИИ
        detailed_data = extract_data_subset(all_data, selected_pairs, "candles_20")
        if not detailed_data:
            return None

        ai_analysis = await analyze_with_ai(detailed_data, direction, cvd_ema_signals)
        if not ai_analysis or 'pairs' not in ai_analysis:
            return None

        final_pairs = ai_analysis['pairs']
        logger.info(f"ИИ выбрал: {len(final_pairs)} пар для детального анализа")

        # Этап 6: Финальный анализ
        if final_pairs:
            extended_data = extract_data_subset(all_data, final_pairs, "candles_full")
            if extended_data:
                return await final_ai_analysis(extended_data, direction, all_data)

        return None

    except Exception as e:
        logger.error(f"Ошибка в анализе: {e}")
        return None


async def main():
    """Главная функция с интерактивным выбором направления."""
    logger.info("СТАРТ ТОРГОВОГО БОТА С CVD + NADARAYA-WATSON + EMA")

    try:
        # Получаем выбор пользователя
        direction = get_user_direction_choice()

        print(f"\n🚀 Запуск анализа с CVD + Nadaraya-Watson + EMA...")
        print("-" * 50)

        # Запускаем анализ с выбранным направлением
        result = await run_trading_analysis(direction)

        if result:
            print(f"\n{'=' * 60}")
            print("РЕЗУЛЬТАТ АНАЛИЗА (CVD + NADARAYA-WATSON + EMA)")
            print("=" * 60)
            print(f"{result}")
            print("=" * 60)
        else:
            logger.error("Анализ не завершен")
            print("\n❌ Анализ не удался. Проверьте логи для подробностей.")

    except KeyboardInterrupt:
        logger.info("Остановлено пользователем")
        print("\n⏹️  Работа прервана пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        print(f"\n❌ Критическая ошибка: {e}")
    finally:
        logger.info("ЗАВЕРШЕНИЕ")
        print("\n👋 Завершение работы")


if __name__ == "__main__":
    asyncio.run(main())