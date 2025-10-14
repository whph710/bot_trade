"""
Конфигурация Telegram бота
Файл: bot_trade/telegram_bot/config_tg.py
"""

import os
from pathlib import Path

# Загрузить .env файл
def load_env():
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()

# Telegram настройки
TG_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TG_USER_ID = int(os.getenv('TELEGRAM_USER_ID', '0'))
TG_CHAT_ID = os.getenv('TELEGRAM_GROUP_ID', '0')

# Проверка конфигурации
if not TG_TOKEN or TG_USER_ID == 0 or TG_CHAT_ID == 0:
    raise ValueError(
        "Ошибка конфигурации! Убедись что в .env установлены:\n"
        "TELEGRAM_BOT_TOKEN=xxxxx\n"
        "TELEGRAM_USER_ID=xxxxx\n"
        "TELEGRAM_GROUP_ID=xxxxx"
    )