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


async def process_single_pair_full(pair: str, limit: int = 100, interval: str = "15") -> Optional[Tuple[str, Dict]]:
    """Асинхронная обработка одной торговой пары с полными данными (100 свечей)."""
    try:
        candles = await get_klines_async(symbol=pair, interval=interval, limit=limit)
        candles = candles[::-1]  # От старых к новым

        if not candles or len(candles) < 4:
            return None

        # Фильтруем по ATR (только пары с достаточной волатильностью)
        # Используем последние 20 свечей для расчета ATR
        atr_candles = candles[-20:] if len(candles) >= 20 else candles
        atr = calculate_atr(atr_candles)
        if atr <= 0.02:
            return None

        # Возвращаем все данные сразу
        return pair, {
            "candles_full": candles,  # Полные данные (100 свечей)
            "candles_3": candles[-3:],  # Последние 3 свечи для паттернов
            "candles_20": candles[-20:],  # Последние 20 свечей для первичного анализа
            "atr": atr
        }

    except Exception as e:
        logger.error(f"Ошибка при обработке пары {pair}: {e}")
        return None


async def collect_all_data() -> Dict[str, Dict]:
    """Собирает ВСЕ данные по всем торговым парам за один раз."""
    start_time = time.time()
    logger.info("Сбор полных данных по торговым парам (100 свечей)...")

    try:
        usdt_pairs = await get_usdt_linear_symbols()

        # Ограничиваем количество одновременных запросов для стабильности
        semaphore = asyncio.Semaphore(100)  # Максимум 50 одновременных запросов

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
        logger.info(f"Сбор данных завершен: {len(filtered_data)} пар за {elapsed_time:.2f}с")
        if error_count > 0:
            logger.warning(f"Ошибок при загрузке: {error_count}")

        return filtered_data

    except Exception as e:
        logger.error(f"Критическая ошибка при сборе данных: {e}")
        return {}


def extract_data_for_patterns(all_data: Dict[str, Dict]) -> Dict[str, Dict]:
    """Извлекает данные для поиска свечных паттернов (3 свечи)."""
    pattern_data = {}
    for pair, data in all_data.items():
        if "candles_3" in data and len(data["candles_3"]) >= 3:
            pattern_data[pair] = {"candles": data["candles_3"]}
    return pattern_data


def extract_data_subset(all_data: Dict[str, Dict], pairs: List[str], candle_key: str) -> Dict[str, List]:
    """Извлекает подмножество данных для указанных пар."""
    subset_data = {}
    for pair in pairs:
        if pair in all_data and candle_key in all_data[pair]:
            subset_data[pair] = all_data[pair][candle_key]
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


async def analyze_with_ai(data: Dict, direction: str) -> Optional[Dict]:
    """Анализирует данные с помощью ИИ (первичный анализ с prompt2.txt)."""
    try:
        # Читаем промпт из prompt2.txt
        try:
            with open("prompt2.txt", 'r', encoding='utf-8') as file:
                prompt2 = file.read()
        except FileNotFoundError:
            prompt2 = """Проанализируй торговые данные и верни результат в виде Python словаря с ключом 'pairs', 
                       содержащим список рекомендуемых торговых пар для анализа. 
                       Формат ответа: {'pairs': ['BTCUSDT', 'ETHUSDT']}"""

        # Добавляем направление к системному промпту
        system_prompt = f"Рассматривай только {direction} {prompt2}"

        ai_response = await deep_seek(
            data=str(data),
            prompt=system_prompt
        )

        parsed_data = parse_ai_response(ai_response)

        if parsed_data and isinstance(parsed_data, dict) and 'pairs' in parsed_data:
            return parsed_data

        # Fallback: возвращаем случайные пары из исходных данных
        available_pairs = list(data.keys())[:5]
        return {'pairs': available_pairs}

    except Exception as e:
        logger.error(f"Ошибка при анализе с ИИ: {e}")
        return None


async def final_ai_analysis(data: Dict, direction: str) -> Optional[str]:
    """Финальный анализ с ИИ на расширенных данных (с prompt.txt)."""
    try:
        # Читаем основной промпт из prompt.txt
        try:
            with open('prompt.txt', 'r', encoding='utf-8') as file:
                main_prompt = file.read()
        except FileNotFoundError:
            main_prompt = "Ты опытный трейдер. Проанализируй данные и дай рекомендации."

        # Добавляем направление к системному промпту
        system_prompt = f"Рассматривай только {direction} {main_prompt}"

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


async def run_trading_analysis(direction: str) -> Optional[str]:
    """
    Основная функция анализа торговых сигналов.

    Args:
        direction: Направление торговли ('long' или 'short')

    Returns:
        Результат финального анализа от ИИ или None при ошибке
    """
    try:
        logger.info(f"Запуск анализа для направления: {direction}")

        # Шаг 1: Сбор ВСЕХ данных сразу (100 свечей для каждой пары)
        all_data = await collect_all_data()
        if not all_data:
            logger.error("Нет данных для анализа")
            return None

        logger.info(f"Загружено данных по {len(all_data)} парам")

        # Шаг 2: Поиск свечных паттернов (используем последние 3 свечи)
        pattern_data = extract_data_for_patterns(all_data)
        candlestick_signals = detect_candlestick_signals(pattern_data)
        selected_pairs = candlestick_signals[direction]

        if not selected_pairs:
            logger.warning(f"Нет паттернов для направления {direction}")
            return None

        logger.info(f"Найдено паттернов {direction}: {selected_pairs}")

        # Шаг 3: Извлекаем данные для первичного анализа (20 свечей)
        detailed_data = extract_data_subset(all_data, selected_pairs, "candles_20")
        if not detailed_data:
            logger.error("Не удалось извлечь детальные данные")
            return None

        # Шаг 4: Первичный анализ с ИИ
        ai_analysis = await analyze_with_ai(detailed_data, direction)
        if not ai_analysis or 'pairs' not in ai_analysis:
            logger.error("Не удалось получить анализ от ИИ")
            return None

        final_pairs = ai_analysis['pairs']
        logger.info(f"ИИ рекомендует: {final_pairs}")

        # Шаг 5: Извлекаем расширенные данные (100 свечей) для финального анализа
        if final_pairs:
            extended_data = extract_data_subset(all_data, final_pairs, "candles_full")

            # Шаг 6: Финальный анализ
            final_result = await final_ai_analysis(extended_data, direction)
            return final_result

        return None

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        return None


async def main():
    """Главная функция программы."""
    logger.info("=" * 50)
    logger.info("ЗАПУСК ТОРГОВОГО БОТА (ОПТИМИЗИРОВАННАЯ ВЕРСИЯ)")
    logger.info("=" * 50)

    try:
        # Получаем направление от пользователя
        direction = get_user_direction()

        # Запускаем анализ
        result = await run_trading_analysis(direction)

        # Логируем результат
        if result:
            logger.info("=" * 50)
            logger.info("РЕЗУЛЬТАТ АНАЛИЗА:")
            logger.info("=" * 50)
            logger.info(result)
            logger.info("=" * 50)
        else:
            logger.error("Анализ не завершен успешно")

    except KeyboardInterrupt:
        logger.info("Программа прервана пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка в main: {e}")
    finally:
        logger.info("Программа завершена")


if __name__ == "__main__":
    asyncio.run(main())