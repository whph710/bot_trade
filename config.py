"""
Обновленная конфигурация для переписанного бота
Убраны избыточные настройки, добавлены параметры для новой логики
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Конфигурация переписанного бота"""

    # === API НАСТРОЙКИ ===
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK', '')
    DEEPSEEK_URL = 'https://api.deepseek.com'
    DEEPSEEK_MODEL = 'deepseek-chat'

    # === ЭТАП 1: Фильтрация сигналов ===
    QUICK_SCAN_5M = 30     # Свечи для базового сканирования

    # === ЭТАП 2: ИИ отбор ===
    AI_BULK_15M = 32       # Свечи 15м для каждой пары (передаем в ИИ)
    AI_INDICATORS_HISTORY = 32  # История индикаторов для ИИ

    # === ЭТАП 3: Детальный анализ ===
    FINAL_5M = 200         # Полные данные 5м
    FINAL_15M = 100        # Полные данные 15м
    FINAL_INDICATORS = 50  # История индикаторов для финального анализа

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
    API_TIMEOUT = 60       # Увеличенный таймаут для больших данных

    # === ЛИМИТЫ ===
    MAX_FINAL_PAIRS = 5    # Максимум для финального анализа

    # === ПРОМПТЫ ===
    SELECTION_PROMPT = 'prompt_select.txt'
    ANALYSIS_PROMPT = 'prompt_analyze.txt'


# Глобальный экземпляр
config = Config()

# Проверка критических настроек
if not config.DEEPSEEK_API_KEY:
    print("⚠️  DEEPSEEK API ключ не найден!")

print(f"✅ Конфигурация переписанного бота загружена")
print(f"   Этапы: {config.QUICK_SCAN_5M}→{config.AI_BULK_15M}→{config.FINAL_5M}/{config.FINAL_15M} свечей")
print(f"   ИИ таймаут: {config.API_TIMEOUT}сек для больших данных")
print(f"   Финальных пар: максимум {config.MAX_FINAL_PAIRS}")