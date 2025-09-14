"""
Исправленная конфигурация для бота
"""

import os
from dataclasses import dataclass

# Загружаем переменные окружения из .env
try:
    with open('.env', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
except:
    pass


@dataclass
class Config:
    """Конфигурация бота"""

    # API НАСТРОЙКИ - исправлено чтение из .env
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK') or os.getenv('DEEPSEEK_API_KEY')
    DEEPSEEK_URL = 'https://api.deepseek.com/v1'
    DEEPSEEK_MODEL = 'deepseek-chat'

    # ЭТАПЫ ОБРАБОТКИ
    QUICK_SCAN_15M = 32
    AI_BULK_15M = 32
    AI_INDICATORS_HISTORY = 32
    FINAL_5M = 200
    FINAL_15M = 100
    FINAL_INDICATORS = 50

    # ИНДИКАТОРЫ
    EMA_FAST = 5
    EMA_MEDIUM = 8
    EMA_SLOW = 20
    RSI_PERIOD = 9
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    ATR_PERIOD = 14

    # ТОРГОВЫЕ ПОРОГИ
    MIN_CONFIDENCE = 70
    MIN_VOLUME_RATIO = 1.2
    MIN_ATR_RATIO = 0.8

    # ПРОИЗВОДИТЕЛЬНОСТЬ
    BATCH_SIZE = 50
    MAX_CONCURRENT = 10
    API_TIMEOUT = 120

    # ЛИМИТЫ
    MAX_FINAL_PAIRS = 5
    MAX_BULK_PAIRS = 15

    # ПРОМПТЫ
    SELECTION_PROMPT = 'prompt_select.txt'
    ANALYSIS_PROMPT = 'prompt_analyze.txt'

    # ИИ НАСТРОЙКИ
    AI_TEMPERATURE_SELECT = 0.3
    AI_TEMPERATURE_ANALYZE = 0.7
    AI_MAX_TOKENS_SELECT = 1000
    AI_MAX_TOKENS_ANALYZE = 2000


config = Config()


def check_config():
    """Проверка конфигурации"""
    print("Проверка конфигурации...")

    if not config.DEEPSEEK_API_KEY:
        print("ОШИБКА: DeepSeek API ключ не найден!")
        print("Создайте файл .env со строкой: DEEPSEEK=your_api_key")
        return False

    print(f"DeepSeek API ключ найден (длина: {len(config.DEEPSEEK_API_KEY)})")
    print(f"API URL: {config.DEEPSEEK_URL}")
    print(f"Модель: {config.DEEPSEEK_MODEL}")

    return True


has_api_key = check_config()