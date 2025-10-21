"""
Main Entry Point - Orchestrator для всех модулей
Интегрирован с Telegram ботом
МОДИФИКАЦИЯ: Добавлен pre-check trading hours
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Добавить папку с ботом в PATH
BOT_DIR = Path(__file__).parent / "trade_bot_programm"
sys.path.insert(0, str(BOT_DIR))

from logging_config import setup_module_logger
from utils import save_bot_result, print_bot_result, cleanup_old_results

logger = setup_module_logger(__name__)


async def run_trading_bot_cycle():
    """
    Выполняет один цикл работы бота с pre-checks
    МОДИФИКАЦИЯ: Добавлена проверка trading hours ПЕРЕД запуском
    """
    from simple_validator import check_trading_hours
    from datetime import datetime

    # КРИТИЧНО: Проверяем trading hours ПЕРЕД запуском любых операций
    time_allowed, time_reason = check_trading_hours()
    if not time_allowed:
        logger.warning(f"⏰ Trading hours blocked: {time_reason}")
        return {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'result': 'TRADING_HOURS_BLOCKED',
            'reason': time_reason,
            'stats': {
                'pairs_scanned': 0,
                'signal_pairs_found': 0,
                'ai_selected': 0,
                'analyzed': 0,
                'validated_signals': 0,
                'rejected_signals': 0,
                'total_time': 0
            }
        }

    # Proceed with bot cycle
    from bot_runner import run_trading_bot
    result = await run_trading_bot()
    return result


async def run_telegram_bot():
    """
    Запустить Telegram бота
    """
    try:
        from telegram_bot.telegram_bot_main import run_telegram_bot

        logger.info("Starting Telegram Bot...")
        await run_telegram_bot()

    except Exception as e:
        logger.error(f"Telegram bot error: {e}", exc_info=True)


async def main_single_cycle():
    """Запустить один цикл торгового бота"""
    result = await run_trading_bot_cycle()
    return result


async def main_telegram():
    """Запустить Telegram бота"""
    await run_telegram_bot()


def parse_arguments():
    """Парсить аргументы командной строки"""
    parser = argparse.ArgumentParser(description='Trading Bot with Telegram')
    parser.add_argument(
        'mode',
        nargs='?',
        default='telegram',
        choices=['telegram', 'once'],
        help='Режим запуска: telegram (с расписанием) или once (один цикл)'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    try:
        if args.mode == 'once':
            # Запустить один цикл торгового бота
            logger.info("🚀 Running trading bot ONCE")
            result = asyncio.run(main_single_cycle())

            if result.get('result') in ['SUCCESS', 'NO_VALIDATED_SIGNALS', 'TRADING_HOURS_BLOCKED']:
                sys.exit(0)
            else:
                sys.exit(1)

        else:  # telegram mode (default)
            # Запустить Telegram бота с расписанием
            logger.info("🤖 Starting Telegram Bot with schedule")
            asyncio.run(main_telegram())

    except KeyboardInterrupt:
        print("\n⏹️  Bot stopped by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)