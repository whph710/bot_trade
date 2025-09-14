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
    # ИСПРАВЛЕНО: правильная переменная окружения
    DEEPSEEK_API_KEY ="sk-dda4182a1cea4a55b9f7a537f0f500b9" #os.getenv('DEEPSEEK_API_KEY') or os.getenv('DEEPSEEK')
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

# ИСПРАВЛЕННАЯ проверка критических настроек
def check_config():
    """Проверка и отображение конфигурации"""
    print("=" * 50)
    print("ПРОВЕРКА КОНФИГУРАЦИИ БОТА")
    print("=" * 50)

    # Проверка API ключа
    if config.DEEPSEEK_API_KEY:
        print(f"✅ DeepSeek API ключ: найден (длина: {len(config.DEEPSEEK_API_KEY)} символов)")
        # Показываем первые и последние символы для проверки
        masked_key = config.DEEPSEEK_API_KEY[:8] + "..." + config.DEEPSEEK_API_KEY[-8:]
        print(f"   Ключ: {masked_key}")
    else:
        print("❌ DeepSeek API ключ: НЕ НАЙДЕН!")
        print("   Установите переменную окружения:")
        print("   export DEEPSEEK_API_KEY=your_api_key")
        print("   или")
        print("   export DEEPSEEK=your_api_key")

    print(f"📡 API URL: {config.DEEPSEEK_URL}")
    print(f"🤖 Модель: {config.DEEPSEEK_MODEL}")

    print(f"\n⚙️ ЭТАПЫ ОБРАБОТКИ:")
    print(f"   Этап 1 (фильтр): {config.QUICK_SCAN_15M} свечей 15м")
    print(f"   Этап 2 (ИИ отбор): {config.AI_BULK_15M} свечей")
    print(f"   Этап 3 (анализ): {config.FINAL_5M}/15м и {config.FINAL_15M}/15м свечей")

    print(f"\n🎯 ТОРГОВЫЕ ЛИМИТЫ:")
    print(f"   Минимальная уверенность: {config.MIN_CONFIDENCE}%")
    print(f"   Минимальный volume ratio: {config.MIN_VOLUME_RATIO}")
    print(f"   Максимум финальных пар: {config.MAX_FINAL_PAIRS}")
    print(f"   Максимум пар для ИИ: {config.MAX_BULK_PAIRS}")

    print(f"\n⚡ ПРОИЗВОДИТЕЛЬНОСТЬ:")
    print(f"   Таймаут ИИ: {config.API_TIMEOUT}сек")
    print(f"   Максимум параллельных запросов: {config.MAX_CONCURRENT}")

    print("=" * 50)

    return bool(config.DEEPSEEK_API_KEY)

# Запуск проверки при импорте
has_api_key = check_config()