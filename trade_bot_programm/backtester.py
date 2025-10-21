"""
Backtester для оценки качества исторических сигналов
Файл: trade_bot_programm/backtester.py
FIXED: Полная логика TP1/TP2/TP3 + timeout + статистика
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
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
                'max_drawdown_pct': float,
                'exit_price': float
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

                # Drawdown calculation
                if signal_type == 'LONG':
                    current_drawdown = ((entry_price - low) / entry_price) * 100
                else:
                    current_drawdown = ((high - entry_price) / entry_price) * 100

                max_drawdown = max(max_drawdown, current_drawdown)

                # Stop loss check (highest priority)
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

                # TP3 check (highest reward)
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
                        'realized_rr': round(reward / risk, 2),
                        'max_drawdown_pct': max_drawdown,
                        'exit_price': tp_levels[2]
                    }

                # TP2 check
                if signal_type == 'LONG' and high >= tp_levels[1]:
                    reward = abs(tp_levels[1] - entry_price)
                    return {
                        'outcome': 'TP2',
                        'hold_duration_minutes': minutes_elapsed,
                        'realized_rr': round(reward / risk, 2),
                        'max_drawdown_pct': max_drawdown,
                        'exit_price': tp_levels[1]
                    }

                if signal_type == 'SHORT' and low <= tp_levels[1]:
                    reward = abs(entry_price - tp_levels[1])
                    return {
                        'outcome': 'TP2',
                        'hold_duration_minutes': minutes_elapsed,
                        'realized_rr': round(reward / risk, 2),
                        'max_drawdown_pct': max_drawdown,
                        'exit_price': tp_levels[1]
                    }

                # TP1 check
                if signal_type == 'LONG' and high >= tp_levels[0]:
                    reward = abs(tp_levels[0] - entry_price)
                    return {
                        'outcome': 'TP1',
                        'hold_duration_minutes': minutes_elapsed,
                        'realized_rr': round(reward / risk, 2),
                        'max_drawdown_pct': max_drawdown,
                        'exit_price': tp_levels[0]
                    }

                if signal_type == 'SHORT' and low <= tp_levels[0]:
                    reward = abs(entry_price - tp_levels[0])
                    return {
                        'outcome': 'TP1',
                        'hold_duration_minutes': minutes_elapsed,
                        'realized_rr': round(reward / risk, 2),
                        'max_drawdown_pct': max_drawdown,
                        'exit_price': tp_levels[0]
                    }

            # Timeout - ни один TP не достигнут
            last_candle = relevant_candles[min(len(relevant_candles) - 1, max_hold_hours - 1)]
            exit_price = float(last_candle[4])  # close price

            if signal_type == 'LONG':
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price

            realized_rr = round(pnl / risk, 2) if risk > 0 else 0

            return {
                'outcome': 'TIMEOUT',
                'hold_duration_minutes': max_hold_hours * 60,
                'realized_rr': realized_rr,
                'max_drawdown_pct': max_drawdown,
                'exit_price': exit_price
            }

        except Exception as e:
            logger.error(f"Error evaluating signal for {symbol}: {e}")
            return self._timeout_result(f"Error: {str(e)[:50]}")

    def _timeout_result(self, reason: str) -> Dict:
        """Fallback result for timeouts/errors"""
        return {
            'outcome': 'TIMEOUT',
            'hold_duration_minutes': 0,
            'realized_rr': 0.0,
            'max_drawdown_pct': 0.0,
            'exit_price': 0.0,
            'error': reason
        }

    async def backtest_multiple_signals(
            self,
            signals: List[Dict],
            max_hold_hours: int = 72
    ) -> Dict:
        """
        Тестирует несколько сигналов

        Returns:
            Статистика по всем сигналам
        """
        tasks = []

        for signal in signals:
            symbol = signal['symbol']
            entry_time_str = signal.get('timestamp', datetime.now().isoformat())

            try:
                entry_time = datetime.fromisoformat(entry_time_str)
            except:
                entry_time = datetime.now() - timedelta(hours=max_hold_hours)

            task = self.evaluate_signal(signal, symbol, entry_time, max_hold_hours)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate statistics
        valid_results = [r for r in results if isinstance(r, dict) and 'outcome' in r]

        if not valid_results:
            return {
                'total_signals': len(signals),
                'tested': 0,
                'error': 'No valid results'
            }

        outcome_counts = {}
        total_rr = 0
        total_hold_time = 0
        total_drawdown = 0
        wins = 0

        for result in valid_results:
            outcome = result['outcome']
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

            total_rr += result['realized_rr']
            total_hold_time += result['hold_duration_minutes']
            total_drawdown += result['max_drawdown_pct']

            if result['outcome'] in ['TP1', 'TP2', 'TP3']:
                wins += 1

        tested = len(valid_results)
        win_rate = (wins / tested * 100) if tested > 0 else 0
        avg_rr = total_rr / tested if tested > 0 else 0
        avg_hold_time = total_hold_time / tested / 60 if tested > 0 else 0  # hours
        avg_drawdown = total_drawdown / tested if tested > 0 else 0

        return {
            'total_signals': len(signals),
            'tested': tested,
            'outcomes': outcome_counts,
            'win_rate': round(win_rate, 2),
            'avg_realized_rr': round(avg_rr, 2),
            'avg_hold_time_hours': round(avg_hold_time, 1),
            'avg_max_drawdown_pct': round(avg_drawdown, 2),
            'detailed_results': valid_results
        }


async def run_backtest(max_signals: int = 100) -> Optional[Dict]:
    """
    Запуск backtest на последних validated сигналах

    Args:
        max_signals: Максимальное количество сигналов для тестирования

    Returns:
        Статистика или None
    """
    from data_storage import storage

    # Load last N signals from storage
    signals_to_test = []

    for symbol_dir in storage.signals_dir.iterdir():
        if not symbol_dir.is_dir():
            continue

        signal_files = sorted(
            symbol_dir.glob("signal_*.json*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:max_signals]

        for filepath in signal_files:
            try:
                if filepath.suffix == '.gz':
                    import gzip
                    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                        signal = json.load(f)
                else:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        signal = json.load(f)

                signals_to_test.append(signal)

                if len(signals_to_test) >= max_signals:
                    break

            except Exception as e:
                logger.error(f"Error loading signal {filepath.name}: {e}")
                continue

        if len(signals_to_test) >= max_signals:
            break

    if not signals_to_test:
        logger.warning("No signals found for backtesting")
        return None

    logger.info(f"Starting backtest on {len(signals_to_test)} signal(s)")

    backtester = SignalBacktester()
    stats = await backtester.backtest_multiple_signals(signals_to_test, max_hold_hours=72)

    # Save results
    results_dir = Path("backtest_results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f"backtest_{timestamp}.json"

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"Backtest results saved: {results_file.name}")
    logger.info(f"Win rate: {stats['win_rate']}% | Avg R/R: {stats['avg_realized_rr']}:1")

    return stats