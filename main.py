import asyncio
import time
from func_async import get_usdt_linear_symbols, get_klines_async
from func_trade import calculate_atr, compute_indicators
from deepseek import deep_seek

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


async def process_pair(pair, interval=15):
    """Асинхронная обработка одной торговой пары"""
    candles = await get_klines_async(symbol=pair, interval=interval)
    candles = candles[::-1]  # Переворачиваем список для правильного порядка
    atr = calculate_atr(candles)
    if atr > 0.005:
        # Вычисляем индикаторы на полном наборе данных
        indicators = compute_indicators(candles[:-1], params)
        final_candles = candles

        final_indicators = {}
        for indicator_name, indicator_values in indicators.items():
            final_indicators[indicator_name] = indicator_values

        return pair, {
            "candles": final_candles,
            "indicators": final_indicators
        }
    return None


async def main():
    start_time = time.time()  # начало замера времени

    # Получаем список всех пар
    usdt_pairs = await get_usdt_linear_symbols()

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

    print(type(all_data).__name__)
    return all_data



async def process_data():
    all_data = await main()
    for pair, data in all_data.items():
        result = await deep_seek(str(data))
        print(f"{pair}:\n{result}")

if __name__ == "__main__":
    asyncio.run(process_data())