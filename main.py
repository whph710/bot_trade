import asyncio
import time
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from func_async import get_usdt_linear_symbols, get_klines_async
from func_trade import calculate_atr, detect_candlestick_signals
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


def ensure_chronological_order(candles: List[List[str]]) -> List[List[str]]:
    """
    Гарантирует, что свечи расположены в хронологическом порядке (от старых к новым).

    Args:
        candles: Список свечных данных

    Returns:
        Список свечей от старых к новым
    """
    if not candles or len(candles) < 2:
        return candles

    # Проверяем порядок по timestamp (первый элемент каждой свечи)
    first_timestamp = int(candles[0][0])
    last_timestamp = int(candles[-1][0])

    # Если первая свеча новее последней, переворачиваем массив
    if first_timestamp > last_timestamp:
        logger.debug("Переворачиваем порядок свечей с убывающего на возрастающий")
        return candles[::-1]

    logger.debug("Свечи уже в правильном порядке (от старых к новым)")
    return candles


async def process_single_pair_full(pair: str, limit: int = 100, interval: str = "15") -> Optional[Tuple[str, Dict]]:
    """Асинхронная обработка одной торговой пары с полными данными."""
    try:
        # Получаем свечи от API
        candles_raw = await get_klines_async(symbol=pair, interval=interval, limit=limit)

        if not candles_raw or len(candles_raw) < 4:
            return None

        # ВАЖНО: Гарантируем правильный порядок данных (от старых к новым)
        candles_ordered = ensure_chronological_order(candles_raw)

        # Логируем для отладки
        if len(candles_ordered) >= 2:
            first_ts = int(candles_ordered[0][0])
            last_ts = int(candles_ordered[-1][0])
            logger.debug(f"{pair}: первая свеча {first_ts}, последняя {last_ts}")

        # Фильтруем по ATR (только пары с достаточной волатильностью)
        atr_candles = candles_ordered[-20:] if len(candles_ordered) >= 20 else candles_ordered
        atr = calculate_atr(atr_candles)
        if atr <= 0.02:
            return None

        return pair, {
            "candles_full": candles_ordered,  # Все свечи (до 100) от старых к новым
            "candles_3": candles_ordered[-3:],  # Последние 3 свечи для паттернов
            "candles_20": candles_ordered[-20:] if len(candles_ordered) >= 20 else candles_ordered,
            # Последние 20 для первичного анализа
            "atr": atr
        }

    except Exception as e:
        logger.error(f"Ошибка при обработке пары {pair}: {e}")
        return None


async def collect_all_data() -> Dict[str, Dict]:
    """Собирает данные по всем торговым парам."""
    start_time = time.time()
    logger.info("Сбор данных по торговым парам...")

    try:
        usdt_pairs = await get_usdt_linear_symbols()
        semaphore = asyncio.Semaphore(50)

        async def process_with_semaphore(pair):
            async with semaphore:
                return await process_single_pair_full(pair, limit=100)

        tasks = [process_with_semaphore(pair) for pair in usdt_pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        filtered_data = {}
        error_count = 0

        for result in results:
            if isinstance(result, Exception):
                error_count += 1
                continue
            if result is not None:
                pair, data = result
                filtered_data[pair] = data

        elapsed_time = time.time() - start_time
        logger.info(f"Загружено {len(filtered_data)} пар за {elapsed_time:.2f}с (ошибок: {error_count})")

        return filtered_data

    except Exception as e:
        logger.error(f"Критическая ошибка при сборе данных: {e}")
        return {}


def extract_data_for_patterns(all_data: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Извлекает данные для поиска свечных паттернов (последние 3 свечи).

    Важно: Данные уже в правильном порядке (от старых к новым),
    поэтому последние 3 свечи - это самые свежие данные.
    """
    pattern_data = {}
    for pair, data in all_data.items():
        if "candles_3" in data and len(data["candles_3"]) >= 3:
            # Данные уже от старых к новым, берем как есть
            pattern_data[pair] = {"candles": data["candles_3"]}

            # Логируем для проверки порядка
            candles = data["candles_3"]
            if len(candles) >= 2:
                first_ts = int(candles[0][0])
                last_ts = int(candles[-1][0])
                if first_ts > last_ts:
                    logger.warning(f"ВНИМАНИЕ: {pair} имеет неправильный порядок свечей в паттернах!")

    logger.info(f"Подготовлено {len(pattern_data)} пар для поиска паттернов")
    return pattern_data


def extract_data_subset(all_data: Dict[str, Dict], pairs: List[str], candle_key: str) -> Dict[str, List]:
    """
    Извлекает подмножество данных для указанных пар.

    Args:
        all_data: Полные данные по всем парам
        pairs: Список пар для извлечения
        candle_key: Ключ для извлечения данных ('candles_3', 'candles_20', 'candles_full')
    """
    subset_data = {}
    for pair in pairs:
        if pair in all_data and candle_key in all_data[pair]:
            candles = all_data[pair][candle_key]

            # Дополнительная проверка порядка
            if len(candles) >= 2:
                first_ts = int(candles[0][0])
                last_ts = int(candles[-1][0])
                if first_ts > last_ts:
                    logger.warning(f"ВНИМАНИЕ: {pair} имеет неправильный порядок в {candle_key}!")
                    # Исправляем порядок если нужно
                    candles = ensure_chronological_order(candles)

            subset_data[pair] = candles

    logger.info(f"Извлечено данных для {len(subset_data)} пар ({candle_key})")
    return subset_data


def parse_ai_response(ai_response: str) -> Optional[Dict]:
    """Парсит ответ ИИ и извлекает список торговых пар."""
    if not ai_response or ai_response.strip() == "":
        return None

    # Метод 1: Прямое преобразование в JSON
    try:
        return json.loads(ai_response.strip())
    except json.JSONDecodeError:
        pass

    # Метод 2: Поиск JSON блока в тексте
    json_pattern = r'\{[^{}]*"pairs"[^{}]*\[[^\]]*\][^{}]*\}'
    json_matches = re.findall(json_pattern, ai_response, re.DOTALL)

    for match in json_matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Метод 3: Поиск списка пар в тексте
    pairs_pattern = r'["\']([A-Z]+USDT)["\']'
    found_pairs = re.findall(pairs_pattern, ai_response)

    if found_pairs:
        unique_pairs = list(dict.fromkeys(found_pairs))[:10]
        return {'pairs': unique_pairs}

    return None


async def analyze_with_ai(data: Dict) -> Optional[Dict]:
    """Первичный анализ с ИИ (20 свечей) - БЕЗ указания направления."""
    try:
        # Читаем промпт из prompt2.txt
        try:
            with open("prompt2.txt", 'r', encoding='utf-8') as file:
                prompt2 = file.read()
        except FileNotFoundError:
            prompt2 = """Проанализируй торговые данные и верни результат в виде Python словаря с ключом 'pairs', 
                       содержащим список рекомендуемых торговых пар для анализа. 
                       Формат ответа: {'pairs': ['BTCUSDT', 'ETHUSDT']}

                       ВАЖНО: Данные свечей предоставлены в хронологическом порядке от старых к новым.
                       Последняя свеча в каждом массиве - самая свежая."""

        # УБИРАЕМ упоминание направления из системного промпта
        system_prompt = f"""{prompt2}

        ВАЖНАЯ ИНФОРМАЦИЯ О ДАННЫХ:
        - Все свечи расположены в хронологическом порядке (от старых к новым)
        - Индекс 0 = самая старая свеча, последний индекс = самая новая свеча
        - Для анализа трендов используй эту хронологию
        - Проанализируй данные и выбери наиболее перспективные торговые пары для дальнейшего анализа"""

        logger.info(f"Первичный анализ ИИ: {len(data)} пар")

        ai_response = await deep_seek(
            data=str(data),
            prompt=system_prompt
        )

        parsed_data = parse_ai_response(ai_response)

        if parsed_data and isinstance(parsed_data, dict) and 'pairs' in parsed_data:
            logger.info(f"ИИ рекомендует {len(parsed_data['pairs'])} пар для дальнейшего анализа")
            return parsed_data

        # Fallback: возвращаем первые 5 пар из исходных данных
        available_pairs = list(data.keys())[:5]
        logger.warning(f"Используем fallback для выбора пар")
        return {'pairs': available_pairs}

    except Exception as e:
        logger.error(f"Ошибка при анализе с ИИ: {e}")
        return None


async def final_ai_analysis(data: Dict) -> Optional[str]:
    """Финальный анализ с ИИ (100 свечей) - БЕЗ указания направления."""
    try:
        # Читаем основной промпт из prompt.txt
        try:
            with open('prompt.txt', 'r', encoding='utf-8') as file:
                main_prompt = file.read()
        except FileNotFoundError:
            main_prompt = """Ты опытный трейдер. Проанализируй данные и дай рекомендации.

            ВАЖНО: Данные свечей предоставлены в хронологическом порядке от старых к новым.
            Используй эту информацию для правильного анализа трендов и паттернов."""

        # УБИРАЕМ упоминание направления из системного промпта
        system_prompt = f"""КРИТИЧЕСКИ ВАЖНАЯ ИНФОРМАЦИЯ О СТРУКТУРЕ ДАННЫХ:
        - Каждая свеча представлена как [timestamp, open, high, low, close, volume, turnover]
        - Все массивы свечей отсортированы в хронологическом порядке (от старых к новым)
        - Индекс 0 = самая старая свеча, последний индекс = самая новая/текущая свеча
        - Для определения тренда: сравнивай значения от начала к концу массива
        - Последние несколько свечей показывают текущую рыночную ситуацию

        {main_prompt}"""

        logger.info(f"Финальный анализ ИИ: {len(data)} пар")

        final_response = await deep_seek(
            data=str(data),
            prompt=system_prompt
        )

        return final_response

    except Exception as e:
        logger.error(f"Ошибка при финальном анализе: {e}")
        return None


def get_user_direction() -> str:
    """Получает от пользователя направление торговли."""
    while True:
        direction = input('Выберите направление торговли (long/short): ').strip().lower()
        if direction in ['long', 'short']:
            return direction
        print("Некорректный ввод. Введите 'long' или 'short'")


def filter_pairs_by_direction(direction: str, candlestick_signals: Dict) -> List[str]:
    """Фильтрует пары по выбранному направлению локально."""
    selected_pairs = candlestick_signals.get(direction, [])

    if not selected_pairs:
        logger.warning(f"Нет найденных паттернов для направления {direction}")
        return []

    logger.info(f"Найдено паттернов {direction}: {len(selected_pairs)} пар")
    logger.info(f"Пары с паттернами: {', '.join(selected_pairs[:10])}{'...' if len(selected_pairs) > 10 else ''}")

    return selected_pairs


async def run_trading_analysis(direction: str) -> Optional[str]:
    """
    Основная функция анализа торговых сигналов.

    ВАЖНО: Все данные свечей обрабатываются в хронологическом порядке (от старых к новым).
    Направление торговли используется только для локальной фильтрации и НЕ передается в ИИ.

    Этапы:
    1. Загрузка 100 свечей для всех пар (от старых к новым)
    2. Поиск паттернов в последних 3 свечах
    3. Локальная фильтрация по направлению (long/short)
    4. Первичный анализ ИИ (20 свечей) для отбора лучших пар
    5. Финальный анализ ИИ (100 свечей) для окончательных рекомендаций
    """
    try:
        logger.info(f"ЗАПУСК АНАЛИЗА: {direction.upper()}")
        logger.info("=" * 60)

        # Шаг 1: Сбор данных
        logger.info("ЭТАП 1: Загрузка и нормализация данных")
        all_data = await collect_all_data()
        if not all_data:
            logger.error("Нет данных для анализа")
            return None

        # Проверяем правильность порядка данных
        sample_pair = next(iter(all_data.keys()))
        sample_candles = all_data[sample_pair]["candles_full"]
        if len(sample_candles) >= 2:
            first_ts = int(sample_candles[0][0])
            last_ts = int(sample_candles[-1][0])
            logger.info(f"Проверка данных ({sample_pair}): от {first_ts} до {last_ts}")
            if first_ts > last_ts:
                logger.error("КРИТИЧЕСКАЯ ОШИБКА: Данные в неправильном порядке!")
                return None

        # Шаг 2: Поиск паттернов
        logger.info("ЭТАП 2: Поиск свечных паттернов")
        pattern_data = extract_data_for_patterns(all_data)
        candlestick_signals = detect_candlestick_signals(pattern_data)

        # Шаг 3: Локальная фильтрация по направлению (БЕЗ отправки в ИИ)
        logger.info(f"ЭТАП 3: Фильтрация по направлению {direction.upper()}")
        selected_pairs = filter_pairs_by_direction(direction, candlestick_signals)

        if not selected_pairs:
            return f"К сожалению, не найдено свечных паттернов для направления {direction}. Попробуйте позже или выберите другое направление."

        # Шаг 4: Первичный анализ ИИ (20 свечей) - БЕЗ упоминания направления
        logger.info("ЭТАП 4: Первичный анализ ИИ (20 свечей)")
        detailed_data = extract_data_subset(all_data, selected_pairs, "candles_20")
        if not detailed_data:
            logger.error("Не удалось извлечь данные для первичного анализа")
            return None

        ai_analysis = await analyze_with_ai(detailed_data)
        if not ai_analysis or 'pairs' not in ai_analysis:
            logger.error("Не удалось получить первичный анализ от ИИ")
            return None

        final_pairs = ai_analysis['pairs']
        logger.info(f"ИИ отобрал для финального анализа: {len(final_pairs)} пар")
        logger.info(f"Финальные пары: {', '.join(final_pairs)}")

        # Шаг 5: Финальный анализ (100 свечей) - БЕЗ упоминания направления
        if final_pairs:
            logger.info("ЭТАП 5: Финальный анализ (100 свечей)")
            extended_data = extract_data_subset(all_data, final_pairs, "candles_full")

            if not extended_data:
                logger.error("Нет данных для финального анализа")
                return None

            # Последняя проверка данных перед отправкой в ИИ
            for pair, candles in list(extended_data.items())[:3]:  # Проверяем первые 3 пары
                if len(candles) >= 2:
                    first_ts = int(candles[0][0])
                    last_ts = int(candles[-1][0])
                    logger.debug(
                        f"Финальные данные {pair}: {first_ts} -> {last_ts} ({'✓' if first_ts < last_ts else '✗'})")

            final_result = await final_ai_analysis(extended_data)
            return final_result

        return None

    except Exception as e:
        logger.error(f"Критическая ошибка в run_trading_analysis: {e}")
        return None


async def main():
    """Главная функция программы."""
    logger.info("=" * 60)
    logger.info("ЗАПУСК ТОРГОВОГО БОТА")
    logger.info("=" * 60)

    try:
        direction = get_user_direction()
        logger.info(f"Выбрано направление: {direction.upper()}")

        result = await run_trading_analysis(direction)

        if result:
            logger.info("=" * 60)
            logger.info("РЕЗУЛЬТАТ АНАЛИЗА:")
            logger.info("=" * 60)
            print("\n" + result + "\n")
            logger.info("=" * 60)
        else:
            logger.error("Анализ не завершен успешно")

    except KeyboardInterrupt:
        logger.info("Программа прервана пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка в main: {e}")
    finally:
        logger.info("Программа завершена")
        logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())