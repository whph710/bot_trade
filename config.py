"""
Исправленная конфигурация для переписанного бота
Устранены критические ошибки API и настроек
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Конфигурация переписанного бота"""

    # === API НАСТРОЙКИ ===
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK')  # Исправлено: ищем DEEPSEEK
    DEEPSEEK_URL = 'https://api.deepseek.com/v1'
    DEEPSEEK_MODEL = 'deepseek-chat'

    # === ЭТАП 1: Фильтрация сигналов ===
    QUICK_SCAN_15M = 32     # Исправлено: 15м для фильтрации

    # === ЭТАП 2: ИИ отбор ===
    AI_BULK_15M = 32
    AI_INDICATORS_HISTORY = 32

    # === ЭТАП 3: Детальный анализ ===
    FINAL_5M = 200
    FINAL_15M = 100
    FINAL_INDICATORS = 50

    # === ИНДИКАТОРЫ ===
    EMA_FAST = 5
    EMA_MEDIUM = 8
    EMA_SLOW = 20
    RSI_PERIOD = 9
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    ATR_PERIOD = 14

    # === ТОРГОВЫЕ ПОРОГИ ===
    MIN_CONFIDENCE = 70
    MIN_VOLUME_RATIO = 1.2
    MIN_ATR_RATIO = 0.8

    # === ПРОИЗВОДИТЕЛЬНОСТЬ ===
    BATCH_SIZE = 50
    MAX_CONCURRENT = 10
    API_TIMEOUT = 120

    # === ЛИМИТЫ ===
    MAX_FINAL_PAIRS = 5
    MAX_BULK_PAIRS = 15

    # === ПРОМПТЫ ===
    SELECTION_PROMPT = 'prompt_select.txt'
    ANALYSIS_PROMPT = 'prompt_analyze.txt'

    # === ИИ НАСТРОЙКИ ===
    AI_TEMPERATURE_SELECT = 0.3
    AI_TEMPERATURE_ANALYZE = 0.7
    AI_MAX_TOKENS_SELECT = 1000
    AI_MAX_TOKENS_ANALYZE = 2000


# Глобальный экземпляр
config = Config()

# Проверка критических настроек
if not config.DEEPSEEK_API_KEY:
    print("WARNING: DEEPSEEK API ключ не найден!")
    print("Установите переменную окружения: export DEEPSEEK=your_api_key")
else:
    print("INFO: DeepSeek API ключ найден")

print(f"INFO: Конфигурация бота загружена")
print(f"API URL: {config.DEEPSEEK_URL}")
print(f"Этапы: {config.QUICK_SCAN_15M}→{config.AI_BULK_15M}→{config.FINAL_5M}/{config.FINAL_15M} свечей")
print(f"ИИ таймаут: {config.API_TIMEOUT}сек")
print(f"Финальных пар: максимум {config.MAX_FINAL_PAIRS}")