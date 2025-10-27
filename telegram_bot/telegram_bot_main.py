# telegram_bot_main.py - FIXED: Aggressive HTML escaping
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
import pytz
from pathlib import Path
import sys
import re

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
from telegram_bot.ai_formatter import AISignalFormatter
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
        self.ai_formatter = AISignalFormatter()
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

    def _escape_html(self, text: str) -> str:
        """
        Экранировать HTML теги которые не поддерживает Telegram

        Стратегия: сначала экранируем ВСЁ, потом восстанавливаем поддерживаемые теги
        """
        if not text:
            return ""

        # Шаг 1: Экранируем ВСЕ < и >
        text = text.replace('<', '&lt;').replace('>', '&gt;')

        # Шаг 2: Восстанавливаем поддерживаемые теги
        supported_tags = [
            'b', 'i', 'code', 'pre', 'a', 'u', 's', 'tg-spoiler'
        ]

        for tag in supported_tags:
            # Открывающий тег
            text = text.replace(f'&lt;{tag}&gt;', f'<{tag}>')
            # Закрывающий тег
            text = text.replace(f'&lt;/{tag}&gt;', f'</{tag}>')
            # С атрибутами (для <a> и <code>)
            text = re.sub(f'&lt;{tag} ([^&]+)&gt;', f'<{tag} \\1>', text)

        return text

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

                result = await run_trading_bot_cycle(progress_callback=self._send_progress)

            finally:
                await self._stop_typing_indicator()

            formatted_result = format_bot_result(result, run_stats)

            await self.bot.send_message(
                chat_id=TG_USER_ID,
                text=f"📈 <b>Результат анализа:</b>\n\n{formatted_result}",
                parse_mode="HTML"
            )

            # Отправка validated + rejected signals
            if result.get('validated_signals'):
                await self._post_signals_to_group(result)

            # Отправка rejected signals в личку
            if result.get('rejected_signals'):
                await self._send_rejected_signals(result.get('rejected_signals', []))

        except Exception as e:
            await self._stop_typing_indicator()
            logger.exception("Error running trading bot manually")
            await self.bot.send_message(
                chat_id=TG_USER_ID,
                text=f"❌ <b>Ошибка:</b> {str(e)}",
                parse_mode="HTML"
            )

    async def _send_progress(self, stage: str, message: str):
        """
        Callback для отправки прогресса выполнения
        """
        try:
            emoji_map = {
                'Stage 1': '1️⃣',
                'Stage 2': '2️⃣',
                'Stage 3': '3️⃣',
                'Stage 2 Complete': '✅',
                'Stage 3 Analysis': '🔍'
            }

            emoji = emoji_map.get(stage, '📊')

            # Экранируем HTML в message
            safe_message = self._escape_html(message)

            formatted_message = f"{emoji} <b>{stage}</b>\n\n{safe_message}"

            await self.bot.send_message(
                chat_id=TG_USER_ID,
                text=formatted_message,
                parse_mode="HTML"
            )

            # Небольшая задержка чтобы не спамить
            await asyncio.sleep(0.3)

        except Exception as e:
            logger.error(f"Error sending progress update: {e}")

    async def _send_rejected_signals(self, rejected_signals: list):
        """
        ИСПРАВЛЕНО: Отправка rejected signals в личку с HTML escaping
        """
        if not rejected_signals:
            return

        try:
            # Группируем по 5 сигналов
            batch_size = 5
            for i in range(0, len(rejected_signals), batch_size):
                batch = rejected_signals[i:i + batch_size]

                message_parts = [
                    f"❌ <b>ОТКЛОНЕННЫЕ СИГНАЛЫ ({i + 1}-{min(i + batch_size, len(rejected_signals))} из {len(rejected_signals)})</b>\n"]

                for sig in batch:
                    symbol = sig.get('symbol', 'UNKNOWN')
                    reason = sig.get('rejection_reason', 'Unknown reason')

                    # Обрезаем длинные причины
                    if len(reason) > 200:
                        reason = reason[:197] + "..."

                    # ИСПРАВЛЕНО: Экранируем HTML в rejection reason
                    safe_reason = self._escape_html(reason)

                    message_parts.append(f"\n<b>{symbol}</b>")
                    message_parts.append(f"<i>{safe_reason}</i>\n")

                full_message = "\n".join(message_parts)

                await self.bot.send_message(
                    chat_id=TG_USER_ID,
                    text=full_message,
                    parse_mode="HTML"
                )

                await asyncio.sleep(0.5)

            logger.info(f"✅ Sent {len(rejected_signals)} rejected signals to user")

        except Exception as e:
            logger.error(f"Error sending rejected signals: {e}")

    async def _post_signals_to_group(self, result: Dict[str, Any]) -> None:
        """Форматирование через DeepSeek AI и публикация в группу"""
        try:
            approved_signals = result.get('validated_signals', [])

            if not approved_signals:
                logger.info("No approved signals to post")
                return

            await self.bot.send_message(
                chat_id=TG_USER_ID,
                text=f"📝 <b>Форматирую {len(approved_signals)} сигнал(ов) через DeepSeek AI...</b>",
                parse_mode="HTML"
            )

            await self._start_typing_indicator(TG_CHAT_ID)

            try:
                logger.info(f"Formatting {len(approved_signals)} signals via DeepSeek AI...")
                formatted_signals = await self.ai_formatter.format_multiple_signals(approved_signals)
            finally:
                await self._stop_typing_indicator()

            if not formatted_signals:
                logger.warning("DeepSeek AI formatting failed, no signals to post")
                await self.bot.send_message(
                    chat_id=TG_USER_ID,
                    text="⚠️ <b>Ошибка форматирования через DeepSeek AI</b>",
                    parse_mode="HTML"
                )
                return

            # Отправка в группу с обработкой ошибок
            sent_count = 0
            failed_count = 0

            for index, formatted_text in enumerate(formatted_signals, 1):
                try:
                    await self.bot.send_message(
                        chat_id=TG_CHAT_ID,
                        text=formatted_text,
                        parse_mode="HTML"
                    )
                    sent_count += 1
                    logger.info(f"✅ Sent signal {index}/{len(formatted_signals)} to group")
                    await asyncio.sleep(0.5)

                except Exception as send_error:
                    failed_count += 1
                    logger.error(f"❌ Failed to send signal {index}/{len(formatted_signals)}: {send_error}")

                    # Отправляем ошибку в личку
                    try:
                        await self.bot.send_message(
                            chat_id=TG_USER_ID,
                            text=f"⚠️ <b>Ошибка отправки сигнала {index}/{len(formatted_signals)} в группу:</b>\n\n<code>{str(send_error)[:300]}</code>",
                            parse_mode="HTML"
                        )
                    except:
                        pass

            # Итоговое сообщение
            if sent_count > 0:
                status_text = f"✅ <b>Опубликовано {sent_count}/{len(formatted_signals)} сигнал(ов) в группу</b>"
                if failed_count > 0:
                    status_text += f"\n⚠️ Не удалось отправить: {failed_count}"

                await self.bot.send_message(
                    chat_id=TG_USER_ID,
                    text=status_text,
                    parse_mode="HTML"
                )

            logger.info(
                f"✅ Posted {sent_count}/{len(formatted_signals)} signal(s) to group {TG_CHAT_ID} (failed: {failed_count})")

        except Exception as e:
            await self._stop_typing_indicator()
            logger.exception(f"Error posting signals to group: {e}")

            await self.bot.send_message(
                chat_id=TG_USER_ID,
                text=f"❌ <b>Ошибка при публикации:</b>\n\n<code>{str(e)[:300]}</code>",
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
            f"🤖 Форматирование: DeepSeek AI\n"
            f"⚡ Валидация: Stage 3 (встроенная)\n"
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

                result = await run_trading_bot_cycle(progress_callback=self._send_progress)

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

            # Отправка rejected signals
            if result.get('rejected_signals'):
                await self._send_rejected_signals(result.get('rejected_signals', []))

        except Exception as e:
            await self._stop_typing_indicator()
            logger.exception("Error in scheduled cycle")

            try:
                await bot.send_message(
                    chat_id=TG_USER_ID,
                    text=f"❌ Ошибка в запланированном цикле:\n\n<code>{str(e)[:300]}</code>",
                    parse_mode="HTML"
                )
            except Exception as send_error:
                logger.exception(f"Failed to send error message: {send_error}")

    async def start(self):
        self.stats_manager.cleanup_old_daily_stats(days_to_keep=30)

        self.schedule_manager.setup_schedule(self.bot, self.schedule_callback)
        logger.info("✅ Telegram bot setup complete (3-stage pipeline, DeepSeek formatter)")

        try:
            await self.dp.start_polling(self.bot, allowed_updates=["message"])
        finally:
            await self._stop_typing_indicator()
            await self.bot.session.close()


async def run_telegram_bot():
    bot = TradingBotTelegram()
    await bot.start()