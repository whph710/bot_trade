import asyncio
import time
from func_async import get_usdt_linear_symbols, get_klines_async
from func_trade import calculate_atr, compute_indicators
from deepseek import deep_seek, deep_seek_check

# Параметры индикаторов
params = {
    'EMA': [9, 21],
    'RSI': 7,
    'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
    'BBANDS': {'period': 20, 'dev': 2},
    'CVD': True,  # Для расчета CVD
    'EMA_volume': [9, 21],  # Для EMA по объему
    'OBV': True,  # Для расчета OBV
}


async def process_pair(pair, interval="15"):
    """Асинхронная обработка одной торговой пары"""
    try:
        candles = await get_klines_async(symbol=pair, interval=interval)
        candles = candles[::-1]  # Переворачиваем список для правильного порядка

        if not candles or len(candles) < 2:
            print(f"Недостаточно данных для пары {pair}")
            return None

        atr = calculate_atr(candles)
        if atr > 0.005:
            # Вычисляем индикаторы на полном наборе данных
            indicators = compute_indicators(candles[:-1], params)

            # Формируем итоговый результат
            return pair, {
                "candles": candles,
                "indicators": indicators
            }
        return None
    except Exception as e:
        print(f"Ошибка при обработке пары {pair}: {e}")
        return None


async def main():
    start_time = time.time()  # начало замера времени

    try:
        # Получаем список всех пар
        usdt_pairs = await get_usdt_linear_symbols()
        print(f"Получено {len(usdt_pairs)} торговых пар")

        # Для ограничения количества одновременных запросов
        # Можно ограничить количество пар для тестирования
        # usdt_pairs = usdt_pairs[:10]  # Раскомментируйте для тестирования с меньшим количеством пар

        # Создаем список задач для асинхронного выполнения
        tasks = [process_pair(pair) for pair in usdt_pairs]

        # Выполняем все задачи параллельно
        results = await asyncio.gather(*tasks)

        # Фильтруем None результаты и собираем данные
        all_data = {pair: data for pair_data in results if pair_data is not None
                    for pair, data in [pair_data]}

        end_time = time.time()  # конец замера времени
        elapsed_time = end_time - start_time

        print(f"Собрано данных по {len(all_data)} парам.")
        print(f"Время выполнения: {elapsed_time:.2f} секунд.")

        return all_data

    except Exception as e:
        print(f"Ошибка в main(): {e}")
        return {}


async def process_data():
    try:
        print("Начинаем сбор данных...")
        all_data = await main()

        if not all_data:
            print("Нет данных для обработки.")
            return

        print(f"Начинаем анализ {len(all_data)} пар через DeepSeek...")
        pairs_deepseek = deep_seek_check(data=all_data)
        print(str(pairs_deepseek))
        # Обрабатываем каждую пару последовательно
        for pair, data in all_data.items():
            print(f"Отправляем данные по {pair} в DeepSeek...")

            # Получаем текстовый результат из функции deep_seek
            result = await deep_seek(data)

            if result:
                print(f"\n===== Результат анализа {pair} =====")
                print(result)
                print(f"===== Конец результата {pair} =====\n")
            else:
                print(f"Не получен результат для {pair}")

    except Exception as e:
        print(f"Ошибка при обработке данных: {e}")


if __name__ == "__main__":
    asyncio.run(process_data())