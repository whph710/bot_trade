"""
Оптимизированная система логирования для торгового бота
Файл: trade_bot_programm/logging_config.py
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_module_logger(module_name: str, log_dir: str = "bot_logs") -> logging.Logger:
    """
    Настроить логгер для конкретного модуля
    Вызывается в начале каждого файла: logger = setup_module_logger(__name__)

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

    # Формат логов
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-8s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # FILE: Все логи
    file_handler = logging.FileHandler(
        log_path / f"bot_{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # FILE: Только ошибки
    error_handler = logging.FileHandler(
        log_path / f"bot_errors_{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    # CONSOLE: Только важные логи (WARNING+)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger