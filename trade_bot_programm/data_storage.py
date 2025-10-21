"""
Persistent storage для торговых данных, сигналов и статистики
Файл: trade_bot_programm/data_storage.py
"""

import json
import gzip
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
from logging_config import setup_module_logger

logger = setup_module_logger(__name__)


class DataStorage:
    """Централизованный менеджер хранения данных"""

    def __init__(self, base_dir: str = "bot_data"):
        self.base_dir = Path(base_dir)
        self.signals_dir = self.base_dir / "signals"
        self.pairs_dir = self.base_dir / "pairs"
        self.statistics_dir = self.base_dir / "statistics"

        self._create_directories()

    def _create_directories(self):
        """Создаёт структуру директорий"""
        for directory in [self.signals_dir, self.pairs_dir, self.statistics_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Data storage initialized: {self.base_dir}")

    def save_signal(self, signal: Dict, compress: bool = True) -> str:
        """
        Сохраняет validated сигнал с полными данными анализа

        Structure: bot_data/signals/SYMBOL/signal_YYYYMMDD_HHMMSS.json.gz

        Args:
            signal: Данные сигнала
            compress: Использовать gzip сжатие

        Returns:
            Путь к сохранённому файлу
        """
        symbol = signal.get('symbol', 'UNKNOWN')
        timestamp = signal.get('timestamp', datetime.now().isoformat())

        try:
            dt = datetime.fromisoformat(timestamp)
            filename = f"signal_{dt.strftime('%Y%m%d_%H%M%S')}.json"
        except:
            filename = f"signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        symbol_dir = self.signals_dir / symbol
        symbol_dir.mkdir(exist_ok=True)

        filepath = symbol_dir / filename

        # Compress analysis_data если присутствует
        signal_copy = signal.copy()
        if 'analysis_data' in signal and 'candles_1h' in signal['analysis_data']:
            signal_copy['analysis_data']['candles_1h'] = signal['analysis_data']['candles_1h'][-50:]
        if 'analysis_data' in signal and 'candles_4h' in signal['analysis_data']:
            signal_copy['analysis_data']['candles_4h'] = signal['analysis_data']['candles_4h'][-50:]

        if compress:
            filepath = filepath.with_suffix('.json.gz')
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(signal_copy, f, ensure_ascii=False)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(signal_copy, f, indent=2, ensure_ascii=False)

        logger.debug(f"Signal saved: {filepath.name}")
        return str(filepath)

    def save_daily_statistics(self, stats: Dict) -> str:
        """
        Сохраняет дневную статистику

        Structure: bot_data/statistics/daily/stats_YYYYMMDD.json

        Args:
            stats: Статистика цикла

        Returns:
            Путь к файлу
        """
        date = datetime.now().strftime('%Y%m%d')
        filepath = self.statistics_dir / "daily" / f"stats_{date}.json"
        filepath.parent.mkdir(exist_ok=True)

        # Load existing if present
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                existing = json.load(f)

            existing['runs'].append({
                'timestamp': datetime.now().isoformat(),
                'stats': stats
            })
            existing['total_runs'] += 1
            existing['total_signals_generated'] += stats.get('validated_signals', 0)

            data = existing
        else:
            data = {
                'date': date,
                'total_runs': 1,
                'total_signals_generated': stats.get('validated_signals', 0),
                'runs': [{
                    'timestamp': datetime.now().isoformat(),
                    'stats': stats
                }]
            }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Daily statistics saved: {filepath.name}")
        return str(filepath)

    def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """
        Удаляет данные старше N дней

        Args:
            days_to_keep: Сколько дней хранить

        Returns:
            Количество удалённых файлов
        """
        cutoff = datetime.now() - timedelta(days=days_to_keep)
        deleted_count = 0

        # Cleanup signals
        for symbol_dir in self.signals_dir.iterdir():
            if not symbol_dir.is_dir():
                continue

            for filepath in symbol_dir.glob("signal_*.json*"):
                file_time = datetime.fromtimestamp(filepath.stat().st_mtime)
                if file_time < cutoff:
                    filepath.unlink()
                    deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Cleanup: Removed {deleted_count} old files (>{days_to_keep} days)")

        return deleted_count


# Singleton instance
storage = DataStorage()