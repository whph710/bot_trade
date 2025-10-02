"""
Универсальная конфигурация торгового бота
Гибкая настройка таймфреймов и параметров
"""

import os
from dataclasses import dataclass
from typing import Literal

# Загрузка переменных окружения
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
    """Конфигурация торгового бота"""

    # API KEYS
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK') or os.getenv('DEEPSEEK_API_KEY')
    DEEPSEEK_URL = 'https://api.deepseek.com/v1'
    DEEPSEEK_MODEL = 'deepseek-chat'

    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC')
    ANTHROPIC_MODEL = 'claude-sonnet-4-20250514'

    # AI PROVIDERS PER STAGE
    AI_STAGE_SELECTION: Literal['deepseek', 'anthropic', 'fallback'] = 'deepseek'
    AI_STAGE_ANALYSIS: Literal['deepseek', 'anthropic', 'fallback'] = 'anthropic'
    AI_STAGE_VALIDATION: Literal['deepseek', 'anthropic', 'fallback'] = 'anthropic'

    # TIMEFRAMES - универсальные параметры
    # КОРОТКИЙ таймфрейм (для детального анализа)
    TIMEFRAME_SHORT = '60'  # в минутах (60 = 1H)
    TIMEFRAME_SHORT_NAME = '1h'  # для логов

    # ДЛИННЫЙ таймфрейм (для структурного анализа)
    TIMEFRAME_LONG = '240'  # в минутах (240 = 4H)
    TIMEFRAME_LONG_NAME = '4h'  # для логов

    # КОЛИЧЕСТВО СВЕЧЕЙ
    # Для быстрого сканирования (Stage 1)
    QUICK_SCAN_CANDLES = 48  # ~8 дней на 4H

    # Для AI bulk анализа (Stage 2)
    AI_BULK_CANDLES = 48  # ~8 дней на 4H
    AI_INDICATORS_HISTORY = 40

    # Для финального анализа (Stage 3)
    FINAL_SHORT_CANDLES = 168  # ~7 дней на 1H
    FINAL_LONG_CANDLES = 84    # ~14 дней на 4H
    FINAL_INDICATORS_HISTORY = 60

    # ИНДИКАТОРЫ
    EMA_FAST = 9
    EMA_MEDIUM = 21
    EMA_SLOW = 50
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    ATR_PERIOD = 14

    # ТОРГОВЫЕ ПАРАМЕТРЫ
    MIN_CONFIDENCE = 75
    MIN_VOLUME_RATIO = 1.3
    MIN_ATR_RATIO = 0.8
    MIN_RISK_REWARD_RATIO = 2.0

    # Время удержания позиции (в минутах)
    MAX_HOLD_DURATION = 2880  # 48 часов
    MIN_HOLD_DURATION = 240   # 4 часа

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

    # AI ПАРАМЕТРЫ
    AI_TEMPERATURE_SELECT = 0.3
    AI_TEMPERATURE_ANALYZE = 0.7
    AI_TEMPERATURE_VALIDATE = 0.3
    AI_MAX_TOKENS_SELECT = 2000
    AI_MAX_TOKENS_ANALYZE = 3000
    AI_MAX_TOKENS_VALIDATE = 3500


config = Config()


def get_available_ai_providers() -> dict:
    """Проверка доступных AI провайдеров"""
    providers = {
        'deepseek': bool(config.DEEPSEEK_API_KEY),
        'anthropic': bool(config.ANTHROPIC_API_KEY),
        'fallback': True
    }
    return providers


def check_config():
    """Проверка конфигурации"""
    print("=" * 80)
    print("ПРОВЕРКА КОНФИГУРАЦИИ")
    print("=" * 80)

    providers = get_available_ai_providers()

    # AI providers
    if providers['deepseek']:
        print(f"[OK] DeepSeek API key: {len(config.DEEPSEEK_API_KEY)} chars")
    else:
        print("[WARN] DeepSeek API key not found")

    if providers['anthropic']:
        print(f"[OK] Anthropic API key: {len(config.ANTHROPIC_API_KEY)} chars")
        print(f"      Model: {config.ANTHROPIC_MODEL}")
    else:
        print("[WARN] Anthropic API key not found")

    # Stage assignments
    print(f"\nAI STAGE ASSIGNMENTS:")
    print(f"  Selection: {config.AI_STAGE_SELECTION} {'[OK]' if providers[config.AI_STAGE_SELECTION] else '[FAIL]'}")
    print(f"  Analysis: {config.AI_STAGE_ANALYSIS} {'[OK]' if providers[config.AI_STAGE_ANALYSIS] else '[FAIL]'}")
    print(f"  Validation: {config.AI_STAGE_VALIDATION} {'[OK]' if providers[config.AI_STAGE_VALIDATION] else '[FAIL]'}")

    # Промпты
    prompt_files = [config.SELECTION_PROMPT, config.ANALYSIS_PROMPT, config.VALIDATION_PROMPT]
    missing = [f for f in prompt_files if not os.path.exists(f)]

    if missing:
        print(f"\n[WARN] Missing prompt files: {missing}")
    else:
        print(f"\n[OK] All prompt files found")

    # Торговые параметры
    print(f"\nTRADING PARAMETERS:")
    print(f"  Timeframes: {config.TIMEFRAME_SHORT_NAME} (detail) + {config.TIMEFRAME_LONG_NAME} (structure)")
    print(f"  Min confidence: {config.MIN_CONFIDENCE}%")
    print(f"  Min R/R: 1:{config.MIN_RISK_REWARD_RATIO}")
    print(f"  Hold time: {config.MIN_HOLD_DURATION//60}-{config.MAX_HOLD_DURATION//60} hours")
    print(f"  EMA system: {config.EMA_FAST}/{config.EMA_MEDIUM}/{config.EMA_SLOW}")

    print("=" * 80)

    return any(providers[stage] for stage in [config.AI_STAGE_SELECTION, config.AI_STAGE_ANALYSIS, config.AI_STAGE_VALIDATION])


has_ai_available = check_config()