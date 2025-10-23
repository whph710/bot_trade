"""
Оптимизированная система логирования для торгового бота
UPDATED: Всё выводится КРАСНЫМ цветом в консоль
Файл: trade_bot_programm/logging_config.py
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


# ANSI цветовые коды
class ColorCodes:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class ColoredFormatter(logging.Formatter):
    """Форматтер с красным цветом для всех сообщений"""

    def __init__(self, fmt=None, datefmt=None, force_color='RED'):
        super().__init__(fmt, datefmt)
        self.force_color = force_color

        # Цвета для разных уровней (но используем только RED)
        self.COLORS = {
            'RED': ColorCodes.RED,
            'GREEN': ColorCodes.GREEN,
            'YELLOW': ColorCodes.YELLOW,
            'WHITE': ColorCodes.WHITE
        }

    def format(self, record):
        # Базовое форматирование
        log_message = super().format(record)

        # Применяем красный цвет ко ВСЕМУ сообщению
        color = self.COLORS.get(self.force_color, ColorCodes.RED)
        colored_message = f"{color}{log_message}{ColorCodes.RESET}"

        return colored_message


def setup_module_logger(module_name: str, log_dir: str = "bot_logs") -> logging.Logger:
    """
    Настроить логгер для конкретного модуля
    ВСЁ ВЫВОДИТСЯ КРАСНЫМ в консоль

    Args:
        module_name: __name__ модуля
        log_dir: директория для логов

    Returns:
        Настроенный logger
    """
    logger = logging.getLogger(module_name)

    # Если логгер уже настроен - вернуть его
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Создать директорию логов
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Формат логов (без цветов для файлов)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-8s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Формат для консоли (с КРАСНЫМ цветом)
    console_formatter = ColoredFormatter(
        '%(asctime)s [%(levelname)-8s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force_color='RED'  # ВСЁ КРАСНЫМ!
    )

    # FILE: Все логи (БЕЗ цветов)
    file_handler = logging.FileHandler(
        log_path / f"bot_{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # FILE: Только ошибки (БЕЗ цветов)
    error_handler = logging.FileHandler(
        log_path / f"bot_errors_{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)

    # CONSOLE: ВСЁ КРАСНЫМ (включая INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)  # Показываем ВСЁ
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


# Также перехватываем print() и делаем его красным
def red_print(*args, **kwargs):
    """Print в красном цвете"""
    message = ' '.join(map(str, args))
    print(f"{ColorCodes.RED}{message}{ColorCodes.RESET}", **kwargs)


# Опционально: можно заменить стандартный print
# import builtins
# builtins.print = red_print