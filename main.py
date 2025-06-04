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
    """Оптимизированная обработка одной торговой пары."""
    try:
        candles_raw = await get_klines_async(symbol=pair, interval=interval, limit=limit)

        if not candles_raw or len(candles_raw) < 4:
            return None

        # Убрали лишние проверки порядка - данные уже приходят правильно из func_async

        # Быстрая фильтрация по ATR с адаптивным порогом
        atr_candles = candles_raw[-20:] if len(candles_raw) >= 20 else candles_raw
        atr = calculate_atr(atr_candles)

        # Адаптивный порог ATR на основе цены
        price = float(candles_raw[-1][4])  # Последняя цена закрытия
        min_atr = 0.01 if price < 1 else 0.02 if price < 100 else 0.05

        if atr <= min_atr:
            return None

        return pair, {
            "candles_full": candles_raw,
            "candles_3": candles_raw[-3:],
            "candles_20": candles_raw[-20:] if len(candles_raw) >= 20 else candles_raw,
            "atr": atr
        }

    except Exception as e:
        logger.error(f"Ошибка обработки {pair}: {e}")
        return None


async def collect_all_data() -> Dict[str, Dict]:
    """Оптимизированный сбор данных с улучшенным контролем параллелизма."""
    start_time = time.time()
    logger.info("Сбор данных по торговым парам...")

    try:
        usdt_pairs = await get_usdt_linear_symbols()
        # Уменьшили семафор для стабильности
        semaphore = asyncio.Semaphore(30)

        async def process_with_semaphore(pair):
            async with semaphore:
                return await process_single_pair_full(pair, limit=100)

        # Батч-обработка для лучшей производительности
        batch_size = 50
        filtered_data = {}
        error_count = 0

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

            # Короткая пауза между батчами
            if i + batch_size < len(usdt_pairs):
                await asyncio.sleep(0.1)

        elapsed_time = time.time() - start_time
        logger.info(f"Загружено {len(filtered_data)} пар за {elapsed_time:.2f}с (ошибок: {error_count})")
        return filtered_data

    except Exception as e:
        logger.error(f"Критическая ошибка при сборе данных: {e}")
        return {}


def extract_data_for_patterns(all_data: Dict[str, Dict]) -> Dict[str, Dict]:
    """Упрощенное извлечение данных для паттернов без лишних проверок."""
    return {
        pair: {"candles": data["candles_3"]}
        for pair, data in all_data.items()
        if "candles_3" in data and len(data["candles_3"]) >= 3
    }


def extract_data_subset(all_data: Dict[str, Dict], pairs: List[str], candle_key: str) -> Dict[str, List]:
    """Упрощенное извлечение подмножества данных."""
    return {
        pair: all_data[pair][candle_key]
        for pair in pairs
        if pair in all_data and candle_key in all_data[pair]
    }


def parse_ai_response(ai_response: str) -> Optional[Dict]:
    """Оптимизированный парсинг ответа ИИ."""
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


async def analyze_with_ai(data: Dict) -> Optional[Dict]:
    """Оптимизированный первичный анализ с ИИ."""
    try:
        # Читаем промпт с кэшированием
        try:
            with open("prompt2.txt", 'r', encoding='utf-8') as file:
                prompt2 = file.read()
        except FileNotFoundError:
            prompt2 = """Проанализируй торговые данные и верни результат в виде Python словаря.
                       Формат: {'pairs': ['BTCUSDT', 'ETHUSDT']}. Выбери до 10 лучших пар."""

        system_prompt = f"""{prompt2}

        ДАННЫЕ: Свечи в хронологическом порядке (от старых к новым).
        Последний индекс = текущая свеча. Выбери наиболее перспективные пары. Рассматривай Long/Short сделки"""

        logger.info(f"Первичный анализ ИИ: {len(data)} пар")

        ai_response = await deep_seek(
            data=str(data),
            prompt=system_prompt,
            max_tokens=2000,  # Уменьшили для скорости
            timeout=45  # Уменьшили таймаут
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


async def final_ai_analysis(data: Dict, direction: str) -> Optional[str]:
    """Оптимизированный финальный анализ с учетом направления торговли."""
    try:
        try:
            with open('prompt.txt', 'r', encoding='utf-8') as file:
                main_prompt = file.read()
        except FileNotFoundError:
            main_prompt = "Ты опытный трейдер. Проанализируй данные и дай рекомендации."

        # Добавляем направление пользователя в системный промпт
        direction_text = "LONG" if direction == "long" else "SHORT"

        system_prompt = f"""
        Один из возможных сценариев, основанный на свечных паттернах — движение в сторону {direction_text}. Используй это как один из факторов в анализе, не считая его окончательным выводом.

        {main_prompt}

        ДАННЫЕ: Свечи в хронологическом порядке (от старых к новым).
        Формат: [timestamp, open, high, low, close, volume, turnover]
        Последний индекс = текущая свеча.
        """

        logger.info(f"Финальный анализ ИИ: {len(data)} пар, направление: {direction_text}")

        return await deep_seek(
            data=str(data),
            prompt=system_prompt,
            timeout=60
        )

    except Exception as e:
        logger.error(f"Ошибка при финальном анализе: {e}")
        return None


def get_user_direction() -> str:
    """Получение направления торговли от пользователя."""
    while True:
        direction = input('Направление (long/short): ').strip().lower()
        if direction in ['long', 'short']:
            return direction
        print("Введите 'long' или 'short'")


def filter_pairs_by_direction(direction: str, candlestick_signals: Dict) -> List[str]:
    """Быстрая фильтрация пар по направлению."""
    selected_pairs = candlestick_signals.get(direction, [])

    if selected_pairs:
        logger.info(f"Найдено {len(selected_pairs)} паттернов {direction}")
    else:
        logger.warning(f"Нет паттернов для {direction}")

    return selected_pairs


async def run_trading_analysis(direction: str) -> Optional[str]:
    """Основная оптимизированная функция анализа."""
    try:
        logger.info(f"АНАЛИЗ: {direction.upper()}")

        # Этап 1: Сбор данных
        all_data = await collect_all_data()
        if not all_data:
            return None

        # Этап 2: Поиск паттернов
        pattern_data = extract_data_for_patterns(all_data)
        candlestick_signals = detect_candlestick_signals(pattern_data)

        # Этап 3: Фильтрация по направлению
        selected_pairs = filter_pairs_by_direction(direction, candlestick_signals)
        if not selected_pairs:
            return f"Нет паттернов для {direction}. Попробуйте позже."

        # Этап 4: Первичный анализ ИИ
        detailed_data = extract_data_subset(all_data, selected_pairs, "candles_20")
        if not detailed_data:
            return None

        ai_analysis = await analyze_with_ai(detailed_data)
        if not ai_analysis or 'pairs' not in ai_analysis:
            return None

        final_pairs = ai_analysis['pairs']
        logger.info(f"ИИ выбрал: {len(final_pairs)} пар")

        # Этап 5: Финальный анализ с передачей направления
        if final_pairs:
            extended_data = extract_data_subset(all_data, final_pairs, "candles_full")
            if extended_data:
                return await final_ai_analysis(extended_data, direction)

        return None

    except Exception as e:
        logger.error(f"Ошибка в анализе: {e}")
        return None


async def main():
    """Главная функция."""
    logger.info("СТАРТ ТОРГОВОГО БОТА")

    try:
        direction = get_user_direction()
        result = await run_trading_analysis(direction)

        if result:
            print(f"\n{result}\n")
        else:
            logger.error("Анализ не завершен")

    except KeyboardInterrupt:
        logger.info("Остановлено пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
    finally:
        logger.info("ЗАВЕРШЕНИЕ")


if __name__ == "__main__":
    asyncio.run(main())