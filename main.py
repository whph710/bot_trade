import asyncio
import time
from func_async import get_usdt_linear_symbols, get_klines_async
from func_trade import calculate_atr, detect_candlestick_signals, compute_cvd_signals, compute_trend_signals
from deepseek import deep_seek, deep_seek_streaming
from chat_gpt import chat_gpt


async def process_pair(pair, limit, interval="15" ):
    """Асинхронная обработка одной торговой пары"""
    try:
        candles = await get_klines_async(symbol=pair, interval=interval, limit=limit)
        candles = candles[::-1]  # Переворачиваем список для правильного порядка

        if not candles or len(candles) < 2:
            print(f"Недостаточно данных для пары {pair}")
            return None

        atr = calculate_atr(candles)
        if atr > 0.02:
            # Вычисляем индикаторы на полном наборе данных
            #indicators = compute_indicators(candles[:-1], params)

            # Формируем итоговый результат
            return pair, {
                "candles": candles #,
                #"indicators": indicators
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

        # Создаем список задач для асинхронного выполнения
        tasks = [process_pair(pair, limit=4) for pair in usdt_pairs]

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
        else:
            candlestick_signals = detect_candlestick_signals(all_data)
        print(candlestick_signals)
        for signal, pairs in candlestick_signals.items():
            for pair in pairs:
                kline = await get_klines_async(symbol=pair)
                signal_cvd = compute_trend_signals(kline)
                if signal_cvd[-1] != signal:
                    candlestick_signals[signal].remove(pair)

        print(candlestick_signals)
        direction = input('long/short: ')
        check_candlestick_signals = {}
        if direction == 'long':
            pairs = candlestick_signals['long']
            for pair in pairs:
                kline = await get_klines_async(symbol=pair)
                check_candlestick_signals[pair] = kline

        elif direction == 'short':
            pairs = candlestick_signals['short']
            for pair in pairs:
                kline = await get_klines_async(symbol=pair)
                check_candlestick_signals[pair] = kline
        print(await deep_seek_streaming(str(check_candlestick_signals)))



    except Exception as e:
        print(f"Ошибка при обработке данных: {e}")


if __name__ == "__main__":
    asyncio.run(process_data())

