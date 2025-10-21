"""
Runner для запуска backtesting
Файл: backtest_runner.py (в корне проекта)
"""

import asyncio
import sys
from pathlib import Path

# Добавить папку с ботом в PATH
BOT_DIR = Path(__file__).parent / "trade_bot_programm"
sys.path.insert(0, str(BOT_DIR))

from trade_bot_programm.backtester import run_backtest


async def main():
    """Запускает backtest"""
    print("=" * 70)
    print("ЗАПУСК BACKTESTING")
    print("=" * 70)

    stats = await run_backtest(max_signals=100)

    if stats:
        print("\n" + "=" * 70)
        print("ИТОГИ")
        print("=" * 70)
        print(f"Win rate: {stats['win_rate']}%")
        print(f"Average R/R: {stats['avg_realized_rr']}:1")
        print(f"Average hold time: {stats['avg_hold_time_hours']:.1f}h")
        print(f"Average drawdown: {stats['avg_max_drawdown_pct']:.2f}%")
        print("=" * 70)
    else:
        print("\n⚠️ Нет результатов для отображения")


if __name__ == "__main__":
    asyncio.run(main())