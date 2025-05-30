import asyncio
from bybit.tranding_analiz import process_trading_signals


if __name__ == "__main__":
    asyncio.run(process_trading_signals())