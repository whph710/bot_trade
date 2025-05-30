import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Tuple

from func_async import get_usdt_linear_symbols, get_klines_async
from func_trade import calculate_atr, detect_candlestick_signals
from deepseek import deep_seek_streaming

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


async def process_single_pair(pair: str, limit: int, interval: str = "15") -> Optional[Tuple[str, Dict]]:
    """
    Асинхронная обработка одной торговой пары.

    Args:
        pair: Название торговой пары
        limit: Количество свечей для получения
        interval: Интервал свечей (по умолчанию 15m)

    Returns:
        Кортеж (pair, data) или None если недостаточно данных или ошибка
    """
    try:
        candles = await get_klines_async(symbol=pair, interval=interval, limit=limit)
        candles = candles[::-1]  # Переворачиваем для правильного порядка (от старых к новым)

        if not candles or len(candles) < 2:
            logger.warning(f"Недостаточно данных для пары {pair}")
            return None

        # Фильтруем по ATR (только пары с достаточной волатильностью)
        atr = calculate_atr(candles)
        if atr <= 0.02:
            logger.debug(f"Низкий ATR ({atr:.4f}) для пары {pair}, пропускаем")
            return None

        logger.debug(f"Обработана пара {pair}, ATR: {atr:.4f}")
        return pair, {"candles": candles}

    except Exception as e:
        logger.error(f"Ошибка при обработке пары {pair}: {e}")
        return None


async def collect_initial_data() -> Dict[str, Dict]:
    """
    Собирает начальные данные по всем торговым парам.

    Returns:
        Словарь с данными по парам, прошедшим фильтрацию
    """
    start_time = time.time()
    logger.info("Начинаем сбор данных по торговым парам...")

    try:
        # Получаем список всех USDT пар
        usdt_pairs = await get_usdt_linear_symbols()
        logger.info(f"Получено {len(usdt_pairs)} торговых пар")

        # Создаем задачи для параллельной обработки
        tasks = [process_single_pair(pair, limit=4) for pair in usdt_pairs]

        # Выполняем все задачи параллельно
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Фильтруем успешные результаты
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
        logger.info(f"Сбор данных завершен: {len(filtered_data)} пар обработано за {elapsed_time:.2f}с")

        if error_count > 0:
            logger.warning(f"Ошибок при обработке: {error_count}")

        return filtered_data

    except Exception as e:
        logger.error(f"Критическая ошибка при сборе данных: {e}")
        return {}


async def get_detailed_data_for_pairs(pairs: List[str], limit: int = 20) -> Dict[str, List]:
    """
    Получает детальные данные по списку торговых пар.

    Args:
        pairs: Список названий торговых пар
        limit: Количество свечей для получения

    Returns:
        Словарь с данными свечей по каждой паре
    """
    logger.info(f"Получаем детальные данные для {len(pairs)} пар...")

    detailed_data = {}
    tasks = [get_klines_async(symbol=pair, limit=limit) for pair in pairs]

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for pair, result in zip(pairs, results):
            if isinstance(result, Exception):
                logger.error(f"Ошибка получения данных для {pair}: {result}")
                continue
            detailed_data[pair] = result

        logger.info(f"Получены детальные данные для {len(detailed_data)} пар")
        return detailed_data

    except Exception as e:
        logger.error(f"Ошибка при получении детальных данных: {e}")
        return {}


import json
import re


async def analyze_with_ai(data: Dict, direction: str, prompt_file: str = 'prompt2.txt') -> Optional[Dict]:
    """
    Анализирует данные с помощью ИИ.
    ИСПРАВЛЕНА для более безопасной обработки ответов.

    Args:
        data: Данные для анализа
        direction: Направление торговли ('long' или 'short')
        prompt_file: Файл с промптом для ИИ

    Returns:
        Результат анализа ИИ в виде словаря или None при ошибке
    """
    try:
        logger.info(f"Начинаем анализ с ИИ для направления: {direction}")

        # Читаем промпт
        try:
            with open("prompt2.txt", 'r', encoding='utf-8') as file:
                prompt = file.read()
        except FileNotFoundError:
            logger.warning("Файл prompt2.txt не найден, используем базовый промпт")
            prompt = """Проанализируй торговые данные и верни результат в виде Python словаря с ключом 'pairs', 
                       содержащим список рекомендуемых торговых пар для анализа. 
                       Формат ответа: {'pairs': ['BTCUSDT', 'ETHUSDT']}"""

        # Получаем ответ от ИИ
        ai_response = await deep_seek_streaming(
            data=str(data),
            trade1=str(direction),
            prompt=prompt
        )

        logger.info("Получен ответ от ИИ")

        # ИСПРАВЛЕНИЕ: Более безопасная обработка ответа
        if not ai_response or ai_response.strip() == "":
            logger.error("Получен пустой ответ от ИИ")
            return None

        logger.debug(f"Первые 200 символов ответа: {ai_response[:200]}")

        # Пытаемся найти JSON в ответе
        parsed_data = None

        # Метод 1: Прямое преобразование в JSON
        try:
            parsed_data = json.loads(ai_response.strip())
            logger.info("Успешно распознан JSON")
        except json.JSONDecodeError:
            logger.debug("Не удалось распознать как JSON, пробуем другие методы")

        # Метод 2: Поиск JSON блока в тексте
        if parsed_data is None:
            json_pattern = r'\{[^{}]*"pairs"[^{}]*\[[^\]]*\][^{}]*\}'
            json_matches = re.findall(json_pattern, ai_response, re.DOTALL)

            for match in json_matches:
                try:
                    parsed_data = json.loads(match)
                    logger.info("Найден JSON блок в тексте")
                    break
                except json.JSONDecodeError:
                    continue

        # Метод 3: Поиск списка пар в тексте
        if parsed_data is None:
            pairs_pattern = r'["\']([A-Z]+USDT)["\']'
            found_pairs = re.findall(pairs_pattern, ai_response)

            if found_pairs:
                # Убираем дубликаты и берем первые 5-10 пар
                unique_pairs = list(dict.fromkeys(found_pairs))[:10]
                parsed_data = {'pairs': unique_pairs}
                logger.info(f"Извлечены пары из текста: {len(unique_pairs)} пар")

        # Метод 4: Безопасное использование eval (только если другие методы не сработали)
        if parsed_data is None:
            try:
                # Очищаем строку от лишних символов
                clean_response = ai_response.strip()
                if clean_response.startswith('```') and clean_response.endswith('```'):
                    clean_response = clean_response[3:-3].strip()
                if clean_response.startswith('python'):
                    clean_response = clean_response[6:].strip()

                # Пробуем eval только если строка выглядит как словарь
                if clean_response.startswith('{') and clean_response.endswith('}'):
                    parsed_data = eval(clean_response)
                    logger.info("Успешно использован eval")
            except Exception as e:
                logger.warning(f"Eval не сработал: {e}")

        # Проверяем результат
        if parsed_data and isinstance(parsed_data, dict):
            if 'pairs' in parsed_data and isinstance(parsed_data['pairs'], list):
                logger.info(f"Успешно обработан ответ ИИ: {len(parsed_data['pairs'])} пар")
                return parsed_data
            else:
                logger.error("В ответе ИИ отсутствует корректный ключ 'pairs'")
        else:
            logger.error("Не удалось распознать структуру ответа ИИ")

        # Если ничего не получилось, создаем fallback ответ
        logger.warning("Используем fallback: возвращаем случайные пары из исходных данных")
        available_pairs = list(data.keys())[:5]  # Берем первые 5 пар
        return {'pairs': available_pairs}

    except Exception as e:
        logger.error(f"Ошибка при анализе с ИИ: {e}")
        return None



async def final_ai_analysis(data: Dict, direction: str) -> Optional[str]:
    """
    Финальный анализ с ИИ на расширенных данных.

    Args:
        data: Расширенные данные для анализа
        direction: Направление торговли

    Returns:
        Финальные рекомендации от ИИ
    """
    try:
        logger.info("Начинаем финальный анализ с ИИ...")

        final_response = await deep_seek_streaming(
            data=str(data),
            trade1=str(direction)
        )

        logger.info("Получен финальный анализ от ИИ")
        return final_response

    except Exception as e:
        logger.error(f"Ошибка при финальном анализе: {e}")
        return None


def get_user_direction() -> str:
    """
    Получает от пользователя направление торговли.

    Returns:
        'long' или 'short'
    """
    while True:
        direction = input('Выберите направление торговли (long/short): ').strip().lower()
        if direction in ['long', 'short']:
            logger.info(f"Выбрано направление: {direction}")
            return direction
        logger.warning("Некорректный ввод. Введите 'long' или 'short'")


async def process_trading_signals():
    """
    Основная функция обработки торговых сигналов.

    Выполняет полный цикл:
    1. Сбор данных по парам
    2. Поиск свечных паттернов
    3. Анализ с ИИ
    4. Финальные рекомендации
    """
    try:
        logger.info("=" * 60)
        logger.info("ЗАПУСК АНАЛИЗА ТОРГОВЫХ СИГНАЛОВ")
        logger.info("=" * 60)

        # Шаг 1: Сбор начальных данных
        all_data = await collect_initial_data()
        if not all_data:
            logger.error("Нет данных для анализа. Завершение программы.")
            return

        # Шаг 2: Поиск свечных паттернов
        logger.info("Поиск свечных паттернов...")
        candlestick_signals = detect_candlestick_signals(all_data)

        total_long = len(candlestick_signals['long'])
        total_short = len(candlestick_signals['short'])
        logger.info(f"Найдено паттернов: {total_long} long, {total_short} short")

        if total_long == 0 and total_short == 0:
            logger.warning("Свечные паттерны не найдены")
            return

        # Шаг 3: Выбор направления пользователем
        direction = get_user_direction()
        selected_pairs = candlestick_signals[direction]

        if not selected_pairs:
            logger.warning(f"Нет паттернов для направления {direction}")
            return

        logger.info(f"Выбрано для анализа: {len(selected_pairs)} пар")

        # Шаг 4: Получение детальных данных
        detailed_data = await get_detailed_data_for_pairs(selected_pairs, limit=20)
        if not detailed_data:
            logger.error("Не удалось получить детальные данные")
            return

        # Шаг 5: Первичный анализ с ИИ
        ai_analysis = await analyze_with_ai(detailed_data, direction)
        if not ai_analysis or 'pairs' not in ai_analysis:
            logger.error("Не удалось получить анализ от ИИ")
            return

        final_pairs = ai_analysis['pairs']
        logger.info(f"ИИ рекомендует для детального анализа: {len(final_pairs)} пар")

        # Шаг 6: Получение расширенных данных для финального анализа
        if final_pairs:
            extended_data = await get_detailed_data_for_pairs(final_pairs, limit=100)

            # Шаг 7: Финальный анализ
            final_recommendation = await final_ai_analysis(extended_data, direction)
            if final_recommendation:
                logger.info("=" * 60)
                logger.info("ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ:")
                logger.info("=" * 60)
                logger.info(final_recommendation)
            else:
                logger.error("Не удалось получить финальные рекомендации")

        logger.info("=" * 60)
        logger.info("АНАЛИЗ ЗАВЕРШЕН")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("Программа прервана пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка в основной функции: {e}")


if __name__ == "__main__":
    asyncio.run(process_trading_signals())