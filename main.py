import asyncio
import time
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from func_async import get_usdt_linear_symbols, get_klines_async
from func_trade import calculate_atr, detect_candlestick_signals
from deepseek import deepseek_chat, deepseek_reasoner, test_deepseek_connection

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


def parse_ai_response(ai_response: str) -> Optional[Dict]:
    """
    Упрощенная функция для парсинга ответа ИИ.

    Args:
        ai_response: Ответ от ИИ

    Returns:
        Словарь с ключом 'pairs' или None
    """
    if not ai_response or ai_response.strip() == "":
        logger.error("Получен пустой ответ от ИИ")
        return None

    logger.debug(f"Первые 200 символов ответа: {ai_response[:200]}")

    # Метод 1: Прямое преобразование в JSON
    try:
        parsed_data = json.loads(ai_response.strip())
        if isinstance(parsed_data, dict) and 'pairs' in parsed_data:
            logger.info("Успешно распознан JSON")
            return parsed_data
    except json.JSONDecodeError:
        pass

    # Метод 2: Поиск JSON блока в тексте
    json_pattern = r'\{[^{}]*"pairs"[^{}]*\[[^\]]*\][^{}]*\}'
    json_matches = re.findall(json_pattern, ai_response, re.DOTALL)

    for match in json_matches:
        try:
            parsed_data = json.loads(match)
            if isinstance(parsed_data, dict) and 'pairs' in parsed_data:
                logger.info("Найден JSON блок в тексте")
                return parsed_data
        except json.JSONDecodeError:
            continue

    # Метод 3: Поиск списка пар в тексте
    pairs_pattern = r'["\']([A-Z]+USDT)["\']'
    found_pairs = re.findall(pairs_pattern, ai_response)

    if found_pairs:
        unique_pairs = list(dict.fromkeys(found_pairs))[:10]
        logger.info(f"Извлечены пары из текста: {len(unique_pairs)} пар")
        return {'pairs': unique_pairs}

    logger.error("Не удалось распознать структуру ответа ИИ")
    return None


async def analyze_with_ai(data: Dict, direction: str) -> Optional[Dict]:
    """
    Анализирует данные с помощью ИИ используя deepseek_chat.

    Args:
        data: Данные для анализа
        direction: Направление торговли ('long' или 'short')

    Returns:
        Результат анализа ИИ в виде словаря или None при ошибке
    """
    try:
        logger.info(f"Начинаем первичный анализ с ИИ (Chat) для направления: {direction}")

        # Читаем промпт
        try:
            with open("prompt2.txt", 'r', encoding='utf-8') as file:
                prompt = file.read()
        except FileNotFoundError:
            logger.warning("Файл prompt2.txt не найден, используем базовый промпт")
            prompt = f"""Ты опытный трейдер. Проанализируй торговые данные для направления {direction}.
                       Верни результат ТОЛЬКО в виде JSON словаря с ключом 'pairs', содержащим список 
                       рекомендуемых торговых пар для детального анализа.
                       Пример формата ответа: {{"pairs": ["BTCUSDT", "ETHUSDT"]}}
                       Выбери максимум 5-7 лучших пар."""

        # Формируем данные для отправки
        analysis_data = f"Направление: {direction}\nДанные: {str(data)}"

        # Получаем ответ от ИИ (используем deepseek_chat для первичного анализа)
        ai_response = await deepseek_chat(prompt, analysis_data)

        if "Ошибка" in ai_response:
            logger.error(f"Ошибка от ИИ: {ai_response}")
            return None

        # Парсим ответ
        parsed_data = parse_ai_response(ai_response)

        if parsed_data and 'pairs' in parsed_data:
            logger.info(f"Успешно обработан ответ ИИ: {len(parsed_data['pairs'])} пар")
            return parsed_data
        else:
            # Fallback: возвращаем случайные пары из исходных данных
            logger.warning("Используем fallback: возвращаем первые пары из исходных данных")
            available_pairs = list(data.keys())[:5]
            return {'pairs': available_pairs}

    except Exception as e:
        logger.error(f"Ошибка при анализе с ИИ: {e}")
        return None


async def final_ai_analysis(data: Dict, direction: str) -> Optional[str]:
    """Оптимизированная версия функции финального анализа"""
    try:
        logger.info("Начинаем оптимизированный финальный анализ...")

        # 1. Чтение промпта (оставляем без изменений)
        try:
            with open("prompt.txt", 'r', encoding='utf-8') as file:
                prompt = file.read()
        except FileNotFoundError:
            prompt = f"Ты опытный трейдер. Проведи глубокий анализ торговых данных для направления {direction}."

        # 2. Оптимизация данных
        optimized_data = {}
        for pair, candles in data.items():
            if not candles or len(candles) < 5:
                continue

            # Берем только ключевые метрики
            optimized_data[pair] = {
                "current_price": float(candles[-1][4]),
                "price_24h_change": f"{(float(candles[-1][4]) - float(candles[0][1])) / float(candles[0][1]) * 100:.2f} % ",
                "volume_24h": sum(float(c[5]) for c in candles[-24:]) / 24,
                "support_level": min(float(c[3]) for c in candles[-10:]),
                "resistance_level": max(float(c[2]) for c in candles[-10:]),
                "trend": "up" if float(candles[-1][4]) > float(candles[-5][4]) else "down"
                }

        # 3. Формируем компактные данные для анализа
        analysis_data = {
            "direction": direction,
            "pairs_count": len(optimized_data),
            "analysis_data": optimized_data
        }

        logger.info(f"Оптимизированный размер данных: {len(str(analysis_data))} символов")

        # 4. Отправка в Reasoner
        final_response = await deepseek_reasoner(prompt, json.dumps(analysis_data))

        if not final_response or len(final_response.strip()) < 50:
            logger.warning("ИИ не смог сформировать полный ответ, используем резервный метод...")
            # Резервный метод с ручной генерацией рекомендаций
            return generate_fallback_recommendations(optimized_data, direction)

        return final_response

    except Exception as e:
        logger.error(f"Ошибка при оптимизированном анализе: {e}")
        return None


def generate_fallback_recommendations(data: Dict, direction: str) -> str:
    """Генерация рекомендаций при невозможности получения ответа от ИИ"""
    recommendations = []
    for pair, metrics in data.items():
        entry_price = metrics['support_level'] * 0.99 if direction == "long" else metrics['resistance_level'] * 1.01
        recommendations.append(
            f"Пара: {pair}\n"
            f"Рекомендация: {'BUY' if direction == 'long' else 'SELL'}\n"
            f"Точка входа: {entry_price:.6f}\n"
            f"Стоп-лосс: {entry_price * 0.98 if direction == 'long' else entry_price * 1.02:.6f}\n"
            f"Тейк-профит: {entry_price * 1.03 if direction == 'long' else entry_price * 0.97:.6f}\n"
        )

    return "\n".join([
        f"Аналитический отчет ({direction})",
        f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 50,
        *recommendations
    ])


async def validate_api_before_analysis():
    """
    Проверяет работоспособность API перед началом анализа.
    """
    logger.info("Проверяем работоспособность DeepSeek API...")

    if not await test_deepseek_connection():
        logger.error("API недоступен. Проверьте подключение к интернету и API ключ.")
        return False

    logger.info("API проверка пройдена успешно")
    return True


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
    1. Проверка API
    2. Сбор данных по парам
    3. Поиск свечных паттернов
    4. Первичный анализ с ИИ (deepseek_chat)
    5. Финальные рекомендации (deepseek_reasoner)
    """
    try:
        logger.info("=" * 60)
        logger.info("ЗАПУСК АНАЛИЗА ТОРГОВЫХ СИГНАЛОВ")
        logger.info("=" * 60)

        # Шаг 0: Проверка API
        if not await validate_api_before_analysis():
            logger.error("Не удалось подключиться к API. Завершение программы.")
            return

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

        # Шаг 5: Первичный анализ с ИИ (deepseek_chat)
        logger.info("Этап 1: Первичный анализ с DeepSeek Chat...")
        ai_analysis = await analyze_with_ai(detailed_data, direction)
        if not ai_analysis or 'pairs' not in ai_analysis:
            logger.error("Не удалось получить анализ от ИИ")
            return

        final_pairs = ai_analysis['pairs']
        logger.info(f"ИИ рекомендует для детального анализа: {len(final_pairs)} пар")
        logger.info(f"Выбранные пары: {final_pairs}")

        # Шаг 6: Получение расширенных данных для финального анализа
        if final_pairs:
            extended_data = await get_detailed_data_for_pairs(final_pairs, limit=100)  # Уменьшил лимит

            # Шаг 7: Финальный анализ (deepseek_reasoner)
            logger.info("Этап 2: Глубокий анализ с DeepSeek Reasoner...")
            final_recommendation = await final_ai_analysis(extended_data, direction)

            if final_recommendation and len(final_recommendation.strip()) > 50:
                logger.info("=" * 60)
                logger.info("ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ:")
                logger.info("=" * 60)
                print(final_recommendation)

                # Сохраняем результат в файл
                try:
                    with open(f'analysis_result_{direction}.txt', 'w', encoding='utf-8') as f:
                        f.write(f"Анализ от {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Направление: {direction}\n")
                        f.write(f"Анализируемые пары: {final_pairs}\n")
                        f.write("=" * 60 + "\n")
                        f.write(final_recommendation)
                    logger.info(f"Результат сохранен в файл: analysis_result_{direction}.txt")
                except Exception as e:
                    logger.error(f"Ошибка при сохранении файла: {e}")
            else:
                logger.error("Не удалось получить качественные финальные рекомендации")
                logger.info("Попробуйте запустить анализ повторно через несколько минут")

        logger.info("=" * 60)
        logger.info("АНАЛИЗ ЗАВЕРШЕН")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("Программа прервана пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка в основной функции: {e}")
        import traceback
        logger.error(f"Детали ошибки: {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(process_trading_signals())