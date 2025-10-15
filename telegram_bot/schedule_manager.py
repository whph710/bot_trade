# schedule_manager.py
import asyncio
import logging
from datetime import datetime, timedelta, time as dtime
import pytz

logger = logging.getLogger(__name__)


class ScheduleManager:
    """
    Управление расписанием запуска бота для aiogram.
    Реализовано через фоновую задачу, которая ожидает ближайшего запуска и вызывает callback(bot).
    """

    # Расписание запусков (Пермь UTC+5) - теперь включает все часы
    SCHEDULE_TIMES = [
        ("10:05", "11:05"),  # Первый период
        ("11:05", "12:05"),  # Второй час первого периода
        ("16:05", "17:05"),  # Второй период
        ("17:05", "18:05"),  # Второй час второго периода
        ("22:05", "23:05"),  # Третий период
        ("23:05", "00:05"),  # Второй час третьего периода
    ]

    def __init__(self):
        self.perm_tz = pytz.timezone('Asia/Yekaterinburg')
        self._scheduler_task = None
        self._stopped = False

    def setup_schedule(self, bot, callback_coro):
        """
        Запустить фоновую задачу планировщика.
        callback_coro — async функция с сигнатурой async def callback(bot)
        """
        if self._scheduler_task is None:
            self._scheduler_task = asyncio.create_task(self._run_scheduler(bot, callback_coro))
            logger.info("ScheduleManager: scheduler task started")

    async def _run_scheduler(self, bot, callback_coro):
        while not self._stopped:
            try:
                next_run = self.get_next_run_time()
                now = datetime.now(self.perm_tz)
                wait_seconds = (next_run - now).total_seconds()
                if wait_seconds <= 0:
                    # Защита от отрицательного времени (вдруг точка входа совпала)
                    wait_seconds = 1

                logger.info(f"Next scheduled run at {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')} (wait {wait_seconds:.0f}s)")
                await asyncio.sleep(wait_seconds)

                # Запускаем callback в отдельной задаче, чтобы не блокировать планировщик
                asyncio.create_task(callback_coro(bot))

                # Подождать 60 секунд чтобы избежать двойного срабатывания при близких временах
                await asyncio.sleep(60)

            except Exception as e:
                logger.exception(f"ScheduleManager error: {e}")
                await asyncio.sleep(10)

    def get_next_run_time(self) -> datetime:
        now = datetime.now(self.perm_tz)
        today = now.date()

        candidate_datetimes = []
        for start_time_str, _ in self.SCHEDULE_TIMES:
            hour, minute = map(int, start_time_str.split(":"))
            candidate = self.perm_tz.localize(datetime.combine(today, dtime(hour=hour, minute=minute)))
            if candidate > now:
                candidate_datetimes.append(candidate)

        if candidate_datetimes:
            return min(candidate_datetimes)

        # Если все времена сегодня прошли — вернуть первое завтрашнее
        tomorrow = today + timedelta(days=1)
        hour, minute = map(int, self.SCHEDULE_TIMES[0][0].split(":"))
        return self.perm_tz.localize(datetime.combine(tomorrow, dtime(hour=hour, minute=minute)))

    def is_trading_hour(self) -> bool:
        now = datetime.now(self.perm_tz)
        current = now.time()

        for start_time_str, end_time_str in self.SCHEDULE_TIMES:
            sh, sm = map(int, start_time_str.split(":"))
            eh, em = map(int, end_time_str.split(":"))
            start = dtime(hour=sh, minute=sm)
            end = dtime(hour=eh, minute=em)

            if start <= current < end:
                return True

        return False

    def stop(self):
        self._stopped = True
        if self._scheduler_task:
            self._scheduler_task.cancel()