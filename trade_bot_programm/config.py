"""
Конфигурация для торгового бота
Объединяет все настройки из configs_continuous.py
"""

import os
import sys
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Импортируем все настройки из главного конфига
from config import *


# ============================================================================
# Класс конфигурации для обратной совместимости
# ============================================================================
class Config:
    """Класс-обертка для доступа к конфигурации"""

    # Telegram
    TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN
    TELEGRAM_USER_ID = TELEGRAM_USER_ID
    TELEGRAM_GROUP_ID = TELEGRAM_GROUP_ID

    # AI API Keys
    DEEPSEEK_API_KEY = DEEPSEEK_API_KEY
    ANTHROPIC_API_KEY = ANTHROPIC_API_KEY
    BESTCHANGE_API_KEY = BESTCHANGE_API_KEY
    BYBIT_API_KEY = BYBIT_API_KEY
    BYBIT_API_SECRET = BYBIT_API_SECRET

    # AI Models
    DEEPSEEK_MODEL = DEEPSEEK_MODEL
    ANTHROPIC_MODEL = ANTHROPIC_MODEL
    DEEPSEEK_REASONING = DEEPSEEK_REASONING
    ANTHROPIC_THINKING = ANTHROPIC_THINKING

    # Stage providers
    STAGE2_PROVIDER = STAGE2_PROVIDER
    STAGE3_PROVIDER = STAGE3_PROVIDER
    STAGE4_PROVIDER = STAGE4_PROVIDER

    # AI Parameters
    AI_TEMPERATURE_SELECT = AI_TEMPERATURE_SELECT
    AI_TEMPERATURE_ANALYZE = AI_TEMPERATURE_ANALYZE
    AI_TEMPERATURE_VALIDATE = AI_TEMPERATURE_VALIDATE
    AI_MAX_TOKENS_SELECT = AI_MAX_TOKENS_SELECT
    AI_MAX_TOKENS_ANALYZE = AI_MAX_TOKENS_ANALYZE
    AI_MAX_TOKENS_VALIDATE = AI_MAX_TOKENS_VALIDATE

    # Trading parameters
    START_AMOUNT = START_AMOUNT
    MIN_SPREAD = MIN_SPREAD
    MIN_PROFIT_USD = MIN_PROFIT_USD
    MAX_REASONABLE_SPREAD = MAX_REASONABLE_SPREAD

    # Monitoring
    MONITORING_INTERVAL = MONITORING_INTERVAL
    DATA_RELOAD_INTERVAL = DATA_RELOAD_INTERVAL
    MIN_TIME_BETWEEN_DUPLICATE = MIN_TIME_BETWEEN_DUPLICATE

    # Liquidity
    MIN_24H_VOLUME_USDT = MIN_24H_VOLUME_USDT
    MIN_LIQUIDITY_SCORE = MIN_LIQUIDITY_SCORE
    USE_ONLY_TOP_LIQUID_COINS = USE_ONLY_TOP_LIQUID_COINS

    # Performance
    MAX_CONCURRENT_REQUESTS = MAX_CONCURRENT_REQUESTS
    REQUEST_DELAY = REQUEST_DELAY
    BATCH_SIZE = BATCH_SIZE
    MAX_RETRIES = MAX_RETRIES
    RETRY_DELAY = RETRY_DELAY
    REQUEST_TIMEOUT = REQUEST_TIMEOUT

    # Caching
    ENABLE_CACHE = ENABLE_CACHE
    CACHE_HOT_PAIRS = CACHE_HOT_PAIRS
    CACHE_TTL = CACHE_TTL

    # WebSocket
    WEBSOCKET_ENABLED = WEBSOCKET_ENABLED
    WEBSOCKET_RECONNECT_DELAY = WEBSOCKET_RECONNECT_DELAY
    WEBSOCKET_PING_INTERVAL = WEBSOCKET_PING_INTERVAL

    # API URLs
    BYBIT_API_URL = BYBIT_API_URL
    BYBIT_WS_URL = BYBIT_WS_URL

    # Logging
    LOGS_DIR = LOGS_DIR
    SAVE_OPPORTUNITIES_TO_FILE = SAVE_OPPORTUNITIES_TO_FILE
    OPPORTUNITIES_LOG_FILE = OPPORTUNITIES_LOG_FILE
    LOG_LEVEL = LOG_LEVEL

    # Coin filtering
    ENABLE_COIN_FILTER = ENABLE_COIN_FILTER
    BLACKLIST_COINS = BLACKLIST_COINS
    WHITELIST_COINS = WHITELIST_COINS

    # Results
    RESULTS_DIR = RESULTS_DIR

    # Debug
    DEBUG = DEBUG

    @classmethod
    def validate(cls):
        """Валидация конфигурации"""
        return validate_config()

    @classmethod
    def print_summary(cls):
        """Вывод сводки конфигурации"""
        return print_config_summary()


# Создаем глобальный экземпляр для импорта
config = Config()

# Для обратной совместимости - экспортируем все константы напрямую
__all__ = [
    'config',
    'Config',
    'TELEGRAM_BOT_TOKEN',
    'TELEGRAM_USER_ID',
    'TELEGRAM_GROUP_ID',
    'DEEPSEEK_API_KEY',
    'ANTHROPIC_API_KEY',
    'BESTCHANGE_API_KEY',
    'BYBIT_API_KEY',
    'BYBIT_API_SECRET',
    'DEEPSEEK_MODEL',
    'ANTHROPIC_MODEL',
    'DEEPSEEK_REASONING',
    'ANTHROPIC_THINKING',
    'STAGE2_PROVIDER',
    'STAGE3_PROVIDER',
    'STAGE4_PROVIDER',
    'AI_TEMPERATURE_SELECT',
    'AI_TEMPERATURE_ANALYZE',
    'AI_TEMPERATURE_VALIDATE',
    'AI_MAX_TOKENS_SELECT',
    'AI_MAX_TOKENS_ANALYZE',
    'AI_MAX_TOKENS_VALIDATE',
    'START_AMOUNT',
    'MIN_SPREAD',
    'MIN_PROFIT_USD',
    'MAX_REASONABLE_SPREAD',
    'MONITORING_INTERVAL',
    'DATA_RELOAD_INTERVAL',
    'MIN_TIME_BETWEEN_DUPLICATE',
    'MIN_24H_VOLUME_USDT',
    'MIN_LIQUIDITY_SCORE',
    'USE_ONLY_TOP_LIQUID_COINS',
    'MAX_CONCURRENT_REQUESTS',
    'REQUEST_DELAY',
    'BATCH_SIZE',
    'MAX_RETRIES',
    'RETRY_DELAY',
    'REQUEST_TIMEOUT',
    'ENABLE_CACHE',
    'CACHE_HOT_PAIRS',
    'CACHE_TTL',
    'WEBSOCKET_ENABLED',
    'WEBSOCKET_RECONNECT_DELAY',
    'WEBSOCKET_PING_INTERVAL',
    'BYBIT_API_URL',
    'BYBIT_WS_URL',
    'LOGS_DIR',
    'SAVE_OPPORTUNITIES_TO_FILE',
    'OPPORTUNITIES_LOG_FILE',
    'LOG_LEVEL',
    'ENABLE_COIN_FILTER',
    'BLACKLIST_COINS',
    'WHITELIST_COINS',
    'RESULTS_DIR',
    'DEBUG',
    'validate_config',
    'print_config_summary'
]

if __name__ == "__main__":
    print("\n🔧 Проверка конфигурации торгового бота...")
    if config.print_summary():
        print("✅ Конфигурация валидна!\n")
    else:
        print("❌ Конфигурация содержит ошибки!\n")