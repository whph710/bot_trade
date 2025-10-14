# telegram_bot_main.py
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

from telegram_bot.config_tg import TG_TOKEN, TG_CHAT_ID, TG_USER_ID
from telegram_bot.schedule_manager import ScheduleManager
from telegram_bot.result_formatter import (
    format_bot_result,
    format_signal_individual,
    send_individual_signals_to_group
)

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
        self.trading_bot_running = False

        self.dp.message.register(self.start_command, Command(commands=["start"]))
        self.dp.message.register(self.handle_message, F.text & ~F.command)

    async def start_command(self, message: Message):
        user_id = message.from_user.id
        if user_id != TG_USER_ID:
            await message.reply("❌ Доступ запрещен")
            return

        keyboard = ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="▶️ Запустить сейчас")],
                [KeyboardButton(text="📊 Статус"), KeyboardButton(text="🛑 Остановить")]
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
        elif text == "🛑 Остановить":
            await self.stop_bot(message)

    async def run_trading_bot_manual(self, message: Message):
        try:
            await message.reply("⏳ Запуск торгового бота...")

            from main import run_trading_bot_cycle

            result = await run_trading_bot_cycle()

            formatted_result = format_bot_result(result)

            # Отправить результат пользователю
            await message.reply(
                f"📈 <b>Результат анализа:</b>\n\n{formatted_result}",
                parse_mode="HTML"
            )

            # Отправить сигналы в группу если они есть
            if result.get('validated_signals'):
                await self._post_signals_to_group(result)
            else:
                logger.info("ℹ️ No validated signals to post")

        except Exception as e:
            logger.exception("Error running trading bot manually")
            await message.reply(f"❌ Ошибка: {str(e)}")

    async def _post_signals_to_group(self, result: Dict[str, Any]) -> None:
        """
        Постить каждый сигнал отдельным постом в группу
        """
        try:
            validated_signals = result.get('validated_signals', [])

            if not validated_signals:
                logger.info("No validated signals to post")
                return

            # Отправляем каждый сигнал отдельным постом
            sent_count = await send_individual_signals_to_group(
                self.bot,
                TG_CHAT_ID,
                validated_signals
            )

            logger.info(f"✅ Posted {sent_count}/{len(validated_signals)} signal(s) to group {TG_CHAT_ID}")

        except Exception as e:
            logger.exception(f"Error posting signals to group: {e}")

    async def show_status(self, message: Message):
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

        await message.reply(status_text, parse_mode="HTML")

    async def stop_bot(self, message: Message):
        await message.reply("🛑 Бот остановлен. Перезапустите для возобновления")

    async def schedule_callback(self, bot: Bot):
        """Callback для плановых запусков"""
        try:
            logger.info("🤖 Scheduled trading bot cycle started")

            from main import run_trading_bot_cycle

            result = await run_trading_bot_cycle()

            formatted_result = format_bot_result(result)

            # Отправить результат пользователю
            await bot.send_message(
                chat_id=TG_USER_ID,
                text=f"📈 <b>Результат анализа:</b>\n\n{formatted_result}",
                parse_mode="HTML"
            )

            # Постить сигналы в группу если они есть
            if result.get('validated_signals'):
                await self._post_signals_to_group(result)
            else:
                logger.info("ℹ️ No validated signals in this cycle")

        except Exception as e:
            logger.exception("Error in scheduled cycle")
            try:
                await bot.send_message(
                    chat_id=TG_USER_ID,
                    text=f"❌ Ошибка в запланированном цикле: {str(e)[:100]}"
                )
            except Exception as send_error:
                logger.exception(f"Failed to send error message: {send_error}")

    async def start(self):
        self.schedule_manager.setup_schedule(self.bot, self.schedule_callback)
        logger.info("✅ Telegram bot setup complete")
        try:
            await self.dp.start_polling(self.bot, allowed_updates=["message"])
        finally:
            await self.bot.session.close()


async def run_telegram_bot():
    bot = TradingBotTelegram()
    await bot.start()


