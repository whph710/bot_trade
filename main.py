"""
Main Entry Point - Orchestrator для всех модулей
"""

import asyncio
import logging
import sys
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Добавить папку с ботом в PATH
BOT_DIR = Path(__file__).parent / "trade_bot_programm"
sys.path.insert(0, str(BOT_DIR))

# Импорты
from utils import save_bot_result, print_bot_result, cleanup_old_results


async def run_trading_bot_cycle():
    """
    Запустить полный цикл торгового бота
    """
    try:
        # Импорт функции из бота
        from bot_runner import run_trading_bot

        logger.info("=" * 60)
        logger.info("🚀 STARTING TRADING BOT")
        logger.info("=" * 60)

        # Запуск бота
        result = await run_trading_bot()

        # Вывод результата в консоль
        print_bot_result(result)

        # Сохранение результата в файл
        saved_path = save_bot_result(result, output_dir="bot_results")

        if saved_path:
            logger.info(f"📁 Result saved to: {saved_path}")

        # Очистка старых результатов (оставить последние 10)
        cleanup_old_results(results_dir="bot_results", keep_last=10)

        return result

    except Exception as e:
        logger.error(f"❌ Critical error: {e}", exc_info=True)
        return {
            'result': 'ERROR',
            'error': str(e),
            'stats': {}
        }


async def main():
    """
    Главная функция
    В будущем здесь будет интеграция с Telegram ботом
    """

    # === СЕЙЧАС: Простой запуск бота ===
    result = await run_trading_bot_cycle()

    # === БУДУЩЕЕ: Интеграция с Telegram ===
    # from telegram_bot.bot import TelegramBot
    # telegram_bot = TelegramBot()
    # await telegram_bot.send_signals(result)

    return result


if __name__ == "__main__":
    try:
        result = asyncio.run(main())

        # Exit code based on result
        if result.get('result') in ['SUCCESS', 'NO_VALIDATED_SIGNALS']:
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n⏹️  Bot stopped by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)