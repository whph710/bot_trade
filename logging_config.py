"""
Оптимизированная система логирования
Файл: logging_config.py
ИЗМЕНЕНИЯ:
- Удалён red_print() как мёртвый код
- Упрощён ColoredFormatter (только красный)
- Убраны избыточные комментарии
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


class ColorCodes:
    RED = '\033[91m'
    RESET = '\033[0m'


class ColoredFormatter(logging.Formatter):
    """Красный цвет для всех сообщений в консоли"""

    def format(self, record):
        log_message = super().format(record)
        return f"{ColorCodes.RED}{log_message}{ColorCodes.RESET}"


def setup_module_logger(module_name: str, log_dir: str = "bot_logs") -> logging.Logger:
    """
    Настройка логгера для модуля

    Args:
        module_name: __name__ модуля
        log_dir: директория для файлов логов

    Returns:
        Настроенный logger
    """
    logger = logging.getLogger(module_name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Создать директорию логов
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Формат логов
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-8s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_formatter = ColoredFormatter(
        '%(asctime)s [%(levelname)-8s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # FILE: Все логи
    file_handler = logging.FileHandler(
        log_path / f"bot_{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # FILE: Только ошибки
    error_handler = logging.FileHandler(
        log_path / f"bot_errors_{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)

    # CONSOLE: Красный цвет
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Только INFO и выше в консоль
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger