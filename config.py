"""
Обновленная конфигурация для бота с поддержкой multiple AI providers
"""

import os
from dataclasses import dataclass
from typing import Literal

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

    # API НАСТРОЙКИ
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK') or os.getenv('DEEPSEEK_API_KEY')
    DEEPSEEK_URL = 'https://api.deepseek.com/v1'
    DEEPSEEK_MODEL = 'deepseek-chat'

    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC')
    # Обновленная модель Claude Sonnet 4 для 2025 года
    ANTHROPIC_MODEL = 'claude-sonnet-4-20250514'

    # AI ПРОВАЙДЕРЫ ДЛЯ ЭТАПОВ
    # Возможные значения: 'deepseek', 'anthropic', 'fallback'
    AI_STAGE_SELECTION: Literal['deepseek', 'anthropic', 'fallback'] = 'anthropic'
    AI_STAGE_ANALYSIS: Literal['deepseek', 'anthropic', 'fallback'] = 'anthropic'
    AI_STAGE_VALIDATION: Literal['deepseek', 'anthropic', 'fallback'] = 'deepseek'

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
    MIN_RISK_REWARD_RATIO = 1.5
    MAX_HOLD_DURATION_MINUTES = 120
    MIN_HOLD_DURATION_MINUTES = 15
    VALIDATION_CONFIDENCE_BOOST = 5

    # ПРОИЗВОДИТЕЛЬНОСТЬ
    BATCH_SIZE = 50
    MAX_CONCURRENT = 10
    API_TIMEOUT = 120
    MAX_FINAL_PAIRS = 5
    MAX_BULK_PAIRS = 15

    # ПРОМПТЫ
    SELECTION_PROMPT = 'prompt_select.txt'
    ANALYSIS_PROMPT = 'prompt_analyze.txt'
    VALIDATION_PROMPT = 'prompt_validate.txt'

    # ИИ НАСТРОЙКИ
    AI_TEMPERATURE_SELECT = 0.3
    AI_TEMPERATURE_ANALYZE = 0.7
    AI_TEMPERATURE_VALIDATE = 0.3
    AI_MAX_TOKENS_SELECT = 1000
    AI_MAX_TOKENS_ANALYZE = 2000
    AI_MAX_TOKENS_VALIDATE = 3000


config = Config()


def get_available_ai_providers() -> dict:
    """Проверка доступных AI провайдеров"""
    providers = {}

    if config.DEEPSEEK_API_KEY:
        providers['deepseek'] = True
    else:
        providers['deepseek'] = False

    if config.ANTHROPIC_API_KEY:
        providers['anthropic'] = True
    else:
        providers['anthropic'] = False

    providers['fallback'] = True  # Всегда доступен

    return providers


def check_config():
    """Проверка конфигурации"""
    print("Проверка конфигурации...")

    providers = get_available_ai_providers()

    # Проверяем DeepSeek
    if providers['deepseek']:
        print(f"DeepSeek API ключ найден (длина: {len(config.DEEPSEEK_API_KEY)})")
    else:
        print("ВНИМАНИЕ: DeepSeek API ключ не найден!")

    # Проверяем Anthropic
    if providers['anthropic']:
        print(f"Anthropic API ключ найден (длина: {len(config.ANTHROPIC_API_KEY)})")
        print(f"Используемая модель: {config.ANTHROPIC_MODEL}")
    else:
        print("ВНИМАНИЕ: Anthropic API ключ не найден!")

    # Проверяем настройки этапов
    print(f"\nНастройки AI провайдеров:")
    print(f"├─ Этап отбора: {config.AI_STAGE_SELECTION} ({'✓' if providers[config.AI_STAGE_SELECTION] else '✗'})")
    print(f"├─ Этап анализа: {config.AI_STAGE_ANALYSIS} ({'✓' if providers[config.AI_STAGE_ANALYSIS] else '✗'})")
    print(f"└─ Этап валидации: {config.AI_STAGE_VALIDATION} ({'✓' if providers[config.AI_STAGE_VALIDATION] else '✗'})")

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
        print(f"\nВНИМАНИЕ: Отсутствуют файлы промптов: {missing_prompts}")

    print(f"\nНастройки торговли:")
    print(f"├─ Минимальная уверенность: {config.MIN_CONFIDENCE}%")
    print(f"├─ Минимальное R/R: 1:{config.MIN_RISK_REWARD_RATIO}")
    print(f"├─ Время удержания: {config.MIN_HOLD_DURATION_MINUTES}-{config.MAX_HOLD_DURATION_MINUTES} мин")
    print(f"└─ Бонус валидации: +{config.VALIDATION_CONFIDENCE_BOOST}%")

    return any(providers[stage] for stage in [config.AI_STAGE_SELECTION, config.AI_STAGE_ANALYSIS, config.AI_STAGE_VALIDATION])


has_ai_available = check_config()