import aiohttp
import asyncio

async def get_klines_async(symbol, interval=15, limit=200):
    """Асинхронная версия получения свечных данных"""
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()

            if data["retCode"] != 0:
                raise Exception(f"API Error: {data['retMsg']}")

            return data["result"]["list"][:-1]  # возвращает список списков свечей

async def get_usdt_linear_symbols():
    """Асинхронная версия получения списка торговых пар"""
    url = "https://api.bybit.com/v5/market/instruments-info"
    params = {
        "category": "linear"  # linear — это USDT-маржинальные фьючерсы
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()

            if data["retCode"] != 0:
                raise Exception(f"API Error: {data['retMsg']}")

            symbols = [
                item["symbol"]
                for item in data["result"]["list"]
                if item["quoteCoin"] == "USDT" and "-" not in item["symbol"] and "1" not in item["symbol"]
            ]
            return symbols