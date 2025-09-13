"""
Упрощенная конфигурация для скальпингового бота
Убраны избыточные настройки, оставлены только критические параметры
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Единственный класс конфигурации - все в одном месте"""

    # === API НАСТРОЙКИ ===
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK', '')
    DEEPSEEK_URL = 'https://api.deepseek.com'
    DEEPSEEK_MODEL = 'deepseek-chat'

    # === ТАЙМФРЕЙМЫ ===
    # Этап 1: Быстрое сканирование
    QUICK_SCAN_5M = 30     # Минимум для базовых индикаторов
    QUICK_SCAN_15M = 20    # Минимум для контекста

    # Этап 2: ИИ отбор
    AI_SELECT_15M = 50     # Достаточно для оценки тренда
    AI_SELECT_INDICATORS = 20  # Последние значения индикаторов

    # Этап 3: Финальный анализ
    FINAL_5M = 200         # Полная картина 5m
    FINAL_15M = 100        # Полная картина 15m
    FINAL_INDICATORS = 50  # Полная история индикаторов

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
    BATCH_SIZE = 50        # Размер батча для сканирования
    MAX_CONCURRENT = 10    # Максимум параллельных запросов
    API_TIMEOUT = 30       # Таймаут для ИИ

    # === ЛИМИТЫ ===
    MAX_PAIRS_TO_AI = 15   # Максимум пар для ИИ отбора
    MAX_FINAL_PAIRS = 5    # Максимум для финального анализа

    # === ПРОМПТЫ ===
    SELECTION_PROMPT = 'prompt_select.txt'
    ANALYSIS_PROMPT = 'prompt_analyze.txt'


# Глобальный экземпляр
config = Config()

# Проверка критических настроек
if not config.DEEPSEEK_API_KEY:
    print("⚠️  DEEPSEEK API ключ не найден!")

print(f"✅ Конфигурация загружена")
print(f"   Этапы: {config.QUICK_SCAN_5M}→{config.AI_SELECT_15M}→{config.FINAL_5M} свечей")
print(f"   ИИ: макс {config.MAX_PAIRS_TO_AI} отбор → {config.MAX_FINAL_PAIRS} финал")