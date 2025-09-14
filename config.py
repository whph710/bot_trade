"""
Исправленная конфигурация для бота + настройки валидации
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
    QUICK_SCAN_15M = 35
    AI_BULK_15M = 35
    AI_INDICATORS_HISTORY = 30
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
    VALIDATION_PROMPT = 'prompt_validate.txt'  # НОВЫЙ промпт для валидации

    # ИИ НАСТРОЙКИ
    AI_TEMPERATURE_SELECT = 0.3
    AI_TEMPERATURE_ANALYZE = 0.7
    AI_TEMPERATURE_VALIDATE = 0.3  # НОВОЕ: низкая температура для точности валидации
    AI_MAX_TOKENS_SELECT = 1000
    AI_MAX_TOKENS_ANALYZE = 2000
    AI_MAX_TOKENS_VALIDATE = 3000  # НОВОЕ: больше токенов для детального ответа валидации

    # ВАЛИДАЦИЯ НАСТРОЙКИ - НОВЫЕ
    MIN_RISK_REWARD_RATIO = 1.5  # Минимальное соотношение риск/доходность
    MAX_HOLD_DURATION_MINUTES = 120  # Максимальное время удержания позиции (2 часа)
    MIN_HOLD_DURATION_MINUTES = 15   # Минимальное время удержания позиции
    VALIDATION_CONFIDENCE_BOOST = 5  # Бонус к уверенности после валидации


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

    # Проверка файлов промптов
    prompt_files = [
        config.SELECTION_PROMPT,
        config.ANALYSIS_PROMPT,
        config.VALIDATION_PROMPT
    ]

    missing_prompts = []
    for prompt_file in prompt_files:
        if not os.path.exists(prompt_file):
            missing_prompts.append(prompt_file)

    if missing_prompts:
        print(f"ВНИМАНИЕ: Отсутствуют файлы промптов: {missing_prompts}")
        print("Этапы с отсутствующими промптами будут работать в fallback режиме")

    print(f"Настройки валидации:")
    print(f"├─ Минимальное R/R: 1:{config.MIN_RISK_REWARD_RATIO}")
    print(f"├─ Время удержания: {config.MIN_HOLD_DURATION_MINUTES}-{config.MAX_HOLD_DURATION_MINUTES} мин")
    print(f"└─ Бонус валидации: +{config.VALIDATION_CONFIDENCE_BOOST}%")

    return True


has_api_key = check_config()