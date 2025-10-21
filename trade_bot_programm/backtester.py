"""
Backtester для оценки качества исторических сигналов
Файл: trade_bot_programm/backtester.py
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List
from pathlib import Path
import json

from func_async import fetch_klines
from logging_config import setup_module_logger

logger = setup_module_logger(__name__)


class SignalBacktester:
    """Backtester для оценки качества сигналов"""

    def __init__(self):
        self.results = []

    async def evaluate_signal(
            self,
            signal: Dict,
            symbol: str,
            entry_time: datetime,
            max_hold_hours: int = 72
    ) -> Dict:
        """
        Оценивает один сигнал исторически

        Returns:
            {
                'outcome': 'TP1/TP2/TP3/STOP/TIMEOUT',
                'hold_duration_minutes': int,
                'realized_rr': float,
                'max_drawdown_pct': float
            }
        """
        try:
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            tp_levels = signal['take_profit_levels']
            signal_type = signal['signal']

            # Загружаем свечи после entry_time
            klines = await fetch_klines(symbol, '60', limit=max_hold_hours + 10)

            if not klines or len(klines) < 10:
                return self._timeout_result("Insufficient data")

            # Фильтруем свечи после entry_time
            entry_timestamp = int(entry_time.timestamp() * 1000)
            relevant_candles = [c for c in klines if int(c[0]) >= entry_timestamp]

            if not relevant_candles:
                return self._timeout_result("No candles after entry")

            risk = abs(entry_price - stop_loss)
            max_drawdown = 0.0

            for i, candle in enumerate(relevant_candles[:max_hold_hours]):
                high = float(candle[2])
                low = float(candle[3])
                minutes_elapsed = i * 60

                # Drawdown
                if signal_type == 'LONG':
                    current_drawdown = ((entry_price - low) / entry_price) * 100
                else:
                    current_drawdown = ((high - entry_price) / entry_price) * 100

                max_drawdown = max(max_drawdown, current_drawdown)

                # Stop loss
                if signal_type == 'LONG' and low <= stop_loss:
                    return {
                        'outcome': 'STOP',
                        'hold_duration_minutes': minutes_elapsed,
                        'realized_rr': -1.0,
                        'max_drawdown_pct': max_drawdown,
                        'exit_price': stop_loss
                    }

                if signal_type == 'SHORT' and high >= stop_loss:
                    return {
                        'outcome': 'STOP',
                        'hold_duration_minutes': minutes_elapsed,
                        'realized_rr': -1.0,
                        'max_drawdown_pct': max_drawdown,
                        'exit_price': stop_loss
                    }

                # TP3
                if signal_type == 'LONG' and high >= tp_levels[2]:
                    reward = abs(tp_levels[2] - entry_price)
                    return {
                        'outcome': 'TP3',
                        'hold_duration_minutes': minutes_elapsed,
                        'realized_rr': round(reward / risk, 2),
                        'max_drawdown_pct': max_drawdown,
                        'exit_price': tp_levels[2]
                    }

                if signal_type == 'SHORT' and low <= tp_levels[2]:
                    reward = abs(entry_price - tp_levels[2])
                    return {
                        'outcome': 'TP3',
                        'hold_duration_minutes': minutes_elapsed,