# telegram_bot_main.py - UPDATED with signal_formatter
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
import pytz
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import Command
from aiogram.enums import ChatAction

from telegram_bot.config_tg import TG_TOKEN, TG_CHAT_ID, TG_USER_ID
from telegram_bot.schedule_manager import ScheduleManager
from telegram_bot.result_formatter import (
    format_bot_result,
    send_formatted_signals_to_group,
    send_group_message_safe
)
# МОДИФИКАЦИЯ: Заменён AI formatter на template formatter
from trade_bot_programm.signal_formatter import format_multiple_signals
from telegram_bot.stats_manager import StatsManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingBotTelegram:
    def __init__(self):
        self.bot = Bot(token=TG_TOKEN)
        self.dp = Dispatcher()
        self.schedule_manager = ScheduleManager()
        # МОДИФИКАЦИЯ: Убран self.ai_formatter
        self.stats_manager = StatsManager()
        self.trading_bot_running = False
        self._typing_task = None

        self.dp.message.register(self.start_command, Command(commands=["start"]))
        self.dp.message.register(self.handle_message, F.text & ~F.command)

    async def _start_typing_indicator(self, chat_id: int):
        """Запустить индикатор печати (typing...)"""

        async def send_typing():
            try:
                while True:
                    await self.bot.send_chat_action(
                        chat_id=chat_id,
                        action=ChatAction.TYPING
                    )
                    await asyncio.sleep(4)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error in typing indicator: {e}")

        self._typing_task = asyncio.create_task(send_typing())

    async def _stop_typing_indicator(self):
        """Остановить индикатор печати"""
        if self._typing_task:
            self._typing_task.cancel()
            try:
                await self._typing_task
            except asyncio.CancelledError:
                pass
            self._typing_task = None

    async def start_command(self, message: Message):
        user_id = message.from_user.id
        if user_id != TG_USER_ID:
            await message.reply("❌ Доступ запрещен")
            return

        keyboard = ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="▶️ Запустить сейчас")],
                [KeyboardButton(text="📊 Статус"), KeyboardButton(text="📈 Статистика")],
                [KeyboardButton(text="🛑 Остановить")]
            ],
            resize_keyboard=True
        )

        await message.answer(
            "🤖 Trading Bot активирован!\n\n"
            "Бот запускается автоматически по расписанию:\n"
            "🟢 10:05-11:05 (UTC+5)\n"
            "🟢 16:05-17:05 (UTC+5)\n"
            "🟢 22:05-23:05 (UTC+5)\n\n"
            "Или нажми кнопку ниже для ручного запуска",
            reply_markup=keyboard
        )

    async def handle_message(self, message: Message):
        user_id = message.from_user.id
        if user_id != TG_USER_ID:
            return

        text = message.text

        if text == "▶️ Запустить сейчас":
            await self.run_trading_bot_manual(message)
        elif text == "📊 Статус":
            await self.show_status(message)
        elif text == "📈 Статистика":
            await self.show_statistics(message)
        elif text == "🛑 Остановить":
            await self.stop_bot(message)

    async def run_trading_bot_manual(self, message: Message):
        """Ручной запуск торгового бота"""
        try:
            run_stats = self.stats_manager.increment_run()

            await self.bot.send_message(
                chat_id=TG_USER_ID,
                text="⏳ <b>Запуск торгового бота...</b>",
                parse_mode="HTML"
            )

            await self._start_typing_indicator(TG_USER_ID)

            try:
                from main import run_trading_bot_cycle
                result = await run_trading_bot_cycle()
            finally:
                await self._stop_typing_indicator()

            formatted_result = format_bot_result(result, run_stats)

            await self.bot.send_message(
                chat_id=TG_USER_ID,
                text=f"📈 <b>Результат анализа:</b>\n\n{formatted_result}",
                parse_mode="HTML"
            )

            if result.get('validated_signals'):
                await self._post_signals_to_group(result)
            else:
                logger.info("ℹ️ No validated signals to post")

        except Exception as e:
            await self._stop_typing_indicator()
            logger.exception("Error running trading bot manually")
            await self.bot.send_message(
                chat_id=TG_USER_ID,
                text=f"❌ <b>Ошибка:</b> {str(e)}",
                parse_mode="HTML"
            )

    async def _post_signals_to_group(self, result: Dict[str, Any]) -> None:
        """
        Форматируем сигналы через template и постим в группу
        МОДИФИКАЦИЯ: Использует signal_formatter вместо AI
        """
        try:
            validated_signals = result.get('validated_signals', [])

            if not validated_signals:
                logger.info("No validated signals to post")
                return

            await self.bot.send_message(
                chat_id=TG_USER_ID,
                text=f"📝 <b>Форматирую {len(validated_signals)} сигнал(ов)...</b>",
                parse_mode="HTML"
            )

            await self._start_typing_indicator(TG_CHAT_ID)

            try:
                # МОДИФИКАЦИЯ: Используем template formatter (БЕЗ AI)
                logger.info(f"Formatting {len(validated_signals)} signals via template...")
                formatted_signals = format_multiple_signals(validated_signals)
            finally:
                await self._stop_typing_indicator()

            if not formatted_signals:
                logger.warning("Template formatting failed, no signals to post")
                await self.bot.send_message(
                    chat_id=TG_USER_ID,
                    text="⚠️ <b>Ошибка форматирования сигналов</b>",
                    parse_mode="HTML"
                )
                return

            sent_count = await send_formatted_signals_to_group(
                self.bot,
                TG_CHAT_ID,
                formatted_signals
            )

            await self.bot.send_message(
                chat_id=TG_USER_ID,
                text=f"✅ <b>Опубликовано {sent_count}/{len(formatted_signals)} сигнал(ов) в группу</b>",
                parse_mode="HTML"
            )

            logger.info(f"✅ Posted {sent_count}/{len(formatted_signals)} signal(s) to group {TG_CHAT_ID}")

        except Exception as e:
            await self._stop_typing_indicator()
            logger.exception(f"Error posting signals to group: {e}")

            await self.bot.send_message(
                chat_id=TG_USER_ID,
                text=f"❌ <b>Ошибка при публикации:</b> {str(e)[:100]}",
                parse_mode="HTML"
            )

    async def show_status(self, message: Message):
        """Показать статус бота"""
        perm_tz = pytz.timezone('Asia/Yekaterinburg')
        perm_time = datetime.now(perm_tz)

        next_run = self.schedule_manager.get_next_run_time()

        status_text = (
            "📊 <b>Статус бота:</b>\n\n"
            f"⏰ Время (Пермь UTC+5): {perm_time.strftime('%H:%M:%S')}\n"
            f"⏳ Следующий запуск: {next_run.strftime('%Y-%m-%d %H:%M')}\n"
            f"🟢 Планировщик: активен\n"
            f"📍 Группа: {TG_CHAT_ID}\n"
        )

        await self.bot.send_message(
            chat_id=TG_USER_ID,
            text=status_text,
            parse_mode="HTML"
        )

    async def show_statistics(self, message: Message):
        """Показать статистику запусков"""
        stats_text = self.stats_manager.get_stats_text()

        await self.bot.send_message(
            chat_id=TG_USER_ID,
            text=stats_text,
            parse_mode="HTML"
        )

    async def stop_bot(self, message: Message):
        """Остановка бота"""
        await self.bot.send_message(
            chat_id=TG_USER_ID,
            text="🛑 <b>Бот остановлен.</b> Перезапустите для возобновления",
            parse_mode="HTML"
        )

    async def schedule_callback(self, bot: Bot):
        """Callback для плановых запусков"""
        try:
            logger.info("🤖 Scheduled trading bot cycle started")

            run_stats = self.stats_manager.increment_run()

            await bot.send_message(
                chat_id=TG_USER_ID,
                text="⏰ <b>Плановый запуск анализа...</b>",
                parse_mode="HTML"
            )

            await self._start_typing_indicator(TG_USER_ID)

            try:
                from main import run_trading_bot_cycle
                result = await run_trading_bot_cycle()
            finally:
                await self._stop_typing_indicator()

            formatted_result = format_bot_result(result, run_stats)

            await bot.send_message(
                chat_id=TG_USER_ID,
                text=f"📈 <b>Результат анализа:</b>\n\n{formatted_result}",
                parse_mode="HTML"
            )

            if result.get('validated_signals'):
                await self._post_signals_to_group(result)
            else:
                logger.info("ℹ️ No validated signals in this cycle")

        except Exception as e:
            await self._stop_typing_indicator()
            logger.exception("Error in scheduled cycle")

            try:
                await bot.send_message(
                    chat_id=TG_USER_ID,
                    text=f"❌ Ошибка в запланированном цикле: {str(e)[:100]}"
                )
            except Exception as send_error:
                logger.exception(f"Failed to send error message: {send_error}")

    async def start(self):
        # Очистка старой статистики при запуске
        self.stats_manager.cleanup_old_daily_stats(days_to_keep=30)

        self.schedule_manager.setup_schedule(self.bot, self.schedule_callback)
        logger.info("✅ Telegram bot setup complete")

        try:
            await self.dp.start_polling(self.bot, allowed_updates=["message"])
        finally:
            await self._stop_typing_indicator()
            await self.bot.session.close()


async def run_telegram_bot():
    bot = TradingBotTelegram()
    await bot.start()