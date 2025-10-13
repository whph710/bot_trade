"""
Main Entry Point - Orchestrator для всех модулей
"""

import asyncio
import sys
from pathlib import Path

# Добавить папку с ботом в PATH
BOT_DIR = Path(__file__).parent / "trade_bot_programm"
sys.path.insert(0, str(BOT_DIR))

from logging_config import setup_module_logger
from utils import save_bot_result, print_bot_result, cleanup_old_results

logger = setup_module_logger(__name__)


async def run_trading_bot_cycle():
    """
    Запустить полный цикл торгового бота
    """
    try:
        from bot_runner import run_trading_bot

        logger.info("Bot cycle initialization started")

        result = await run_trading_bot()

        # Вывод результата
        print_bot_result(result)

        # Сохранение результата в файл
        saved_path = save_bot_result(result, output_dir="bot_results")

        if saved_path:
            logger.info(f"Result saved: {saved_path}")

        # Очистка старых результатов
        cleanup_old_results(results_dir="bot_results", keep_last=10)

        return result

    except Exception as e:
        logger.error(f"Critical cycle error: {e}", exc_info=True)
        return {
            'result': 'ERROR',
            'error': str(e),
            'stats': {}
        }


async def main():
    """Главная функция"""
    result = await run_trading_bot_cycle()
    return result


if __name__ == "__main__":
    try:
        result = asyncio.run(main())

        if result.get('result') in ['SUCCESS', 'NO_VALIDATED_SIGNALS']:
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️  Bot stopped by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)