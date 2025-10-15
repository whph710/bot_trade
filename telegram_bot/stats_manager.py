"""
Менеджер статистики для торгового бота
Файл: telegram_bot/stats_manager.py
"""

import json
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, Any
from logging_config import setup_module_logger

logger = setup_module_logger(__name__)


class StatsManager:
    """Управление статистикой запусков бота"""

    def __init__(self, stats_file: str = "bot_statistics.json"):
        self.stats_file = Path(__file__).parent / stats_file
        self.stats = self._load_stats()

    def _load_stats(self) -> Dict[str, Any]:
        """Загрузить статистику из файла"""
        if not self.stats_file.exists():
            return {
                'total_runs': 0,
                'daily_stats': {},
                'last_run': None
            }

        try:
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading stats: {e}")
            return {
                'total_runs': 0,
                'daily_stats': {},
                'last_run': None
            }

    def _save_stats(self) -> None:
        """Сохранить статистику в файл"""
        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving stats: {e}")

    def increment_run(self) -> Dict[str, int]:
        """
        Инкрементировать счетчик запусков

        Returns:
            Dict с total_runs и today_runs
        """
        today = str(date.today())

        # Общий счетчик
        self.stats['total_runs'] = self.stats.get('total_runs', 0) + 1

        # Дневной счетчик
        if today not in self.stats['daily_stats']:
            self.stats['daily_stats'][today] = 0

        self.stats['daily_stats'][today] += 1

        # Время последнего запуска
        self.stats['last_run'] = datetime.now().isoformat()

        self._save_stats()

        return {
            'total_runs': self.stats['total_runs'],
            'today_runs': self.stats['daily_stats'][today]
        }

    def get_stats(self) -> Dict[str, Any]:
        """Получить текущую статистику"""
        today = str(date.today())

        return {
            'total_runs': self.stats.get('total_runs', 0),
            'today_runs': self.stats.get('daily_stats', {}).get(today, 0),
            'last_run': self.stats.get('last_run'),
            'daily_stats': self.stats.get('daily_stats', {})
        }

    def get_stats_text(self) -> str:
        """Получить статистику в виде форматированного текста"""
        stats = self.get_stats()

        text = f"📊 <b>СТАТИСТИКА ЗАПУСКОВ</b>\n\n"
        text += f"🔢 Всего запусков: <b>{stats['total_runs']}</b>\n"
        text += f"📅 Сегодня: <b>{stats['today_runs']}</b>\n"

        if stats['last_run']:
            try:
                last_run_dt = datetime.fromisoformat(stats['last_run'])
                text += f"⏰ Последний: <b>{last_run_dt.strftime('%Y-%m-%d %H:%M:%S')}</b>\n"
            except:
                pass

        return text

    def cleanup_old_daily_stats(self, days_to_keep: int = 30) -> None:
        """Очистить старую дневную статистику"""
        if 'daily_stats' not in self.stats:
            return

        today = date.today()
        cutoff_date = today - timedelta(days=days_to_keep)

        keys_to_remove = []
        for date_str in self.stats['daily_stats'].keys():
            try:
                stat_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                if stat_date < cutoff_date:
                    keys_to_remove.append(date_str)
            except:
                continue

        for key in keys_to_remove:
            del self.stats['daily_stats'][key]

        if keys_to_remove:
            logger.info(f"Cleaned up {len(keys_to_remove)} old daily stats")
            self._save_stats()