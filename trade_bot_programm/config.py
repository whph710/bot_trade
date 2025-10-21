# config.py
# Единый файл конфигурации для всего проекта (В КОРНЕ ПРОЕКТА!)
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# ============================================================================
# TELEGRAM BOT
# ============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_USER_ID = os.getenv("TELEGRAM_USER_ID", "")
TELEGRAM_GROUP_ID = os.getenv("TELEGRAM_GROUP_ID", "")

# ============================================================================
# AI API KEYS
# ============================================================================
# Поддержка старых имен переменных для обратной совместимости
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", os.getenv("DEEPSEEK", ""))
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC", ""))
BESTCHANGE_API_KEY = os.getenv("BESTCHANGE_API_KEY", "")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

# ============================================================================
# AI MODELS CONFIGURATION
# ============================================================================
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# DeepSeek URL
DEEPSEEK_URL = "https://api.deepseek.com"

# ============================================================================
# AI REASONING/THINKING MODES
# ============================================================================
DEEPSEEK_REASONING = os.getenv("DEEPSEEK_REASONING", "false").lower() in ("true", "1", "yes")
ANTHROPIC_THINKING = os.getenv("ANTHROPIC_THINKING", "false").lower() in ("true", "1", "yes")

# ============================================================================
# STAGE PROVIDER SELECTION (Multi-Stage AI Pipeline)
# ============================================================================
STAGE2_PROVIDER = os.getenv("STAGE2_PROVIDER", "deepseek").lower()  # выбор пар
STAGE3_PROVIDER = os.getenv("STAGE3_PROVIDER", "claude").lower()    # глубокий анализ
STAGE4_PROVIDER = os.getenv("STAGE4_PROVIDER", "claude").lower()    # валидация

# ============================================================================
# AI PARAMETERS
# ============================================================================
AI_TEMPERATURE_SELECT = float(os.getenv("AI_TEMPERATURE_SELECT", "0.3"))
AI_TEMPERATURE_ANALYZE = float(os.getenv("AI_TEMPERATURE_ANALYZE", "0.7"))
AI_TEMPERATURE_VALIDATE = float(os.getenv("AI_TEMPERATURE_VALIDATE", "0.3"))

AI_MAX_TOKENS_SELECT = int(os.getenv("AI_MAX_TOKENS_SELECT", "2000"))
AI_MAX_TOKENS_ANALYZE = int(os.getenv("AI_MAX_TOKENS_ANALYZE", "3000"))
AI_MAX_TOKENS_VALIDATE = int(os.getenv("AI_MAX_TOKENS_VALIDATE", "3500"))

# API Timeouts
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
API_TIMEOUT_ANALYSIS = int(os.getenv("API_TIMEOUT_ANALYSIS", "60"))

# ============================================================================
# TRADING PARAMETERS
# ============================================================================
START_AMOUNT = float(os.getenv("START_AMOUNT", "100.0"))
MIN_SPREAD = float(os.getenv("MIN_SPREAD", "0.1"))
MIN_PROFIT_USD = float(os.getenv("MIN_PROFIT_USD", "0.5"))
MAX_REASONABLE_SPREAD = float(os.getenv("MAX_REASONABLE_SPREAD", "50.0"))

# Minimum confidence
MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", "75"))
MIN_RISK_REWARD_RATIO = float(os.getenv("MIN_RISK_REWARD_RATIO", "2.5"))

# ============================================================================
# MONITORING
# ============================================================================
MONITORING_INTERVAL = float(os.getenv("MONITORING_INTERVAL", "20.0"))
DATA_RELOAD_INTERVAL = int(os.getenv("DATA_RELOAD_INTERVAL", "3600"))
MIN_TIME_BETWEEN_DUPLICATE = int(os.getenv("MIN_TIME_BETWEEN_DUPLICATE", "60"))

# ============================================================================
# LIQUIDITY FILTERING
# ============================================================================
MIN_24H_VOLUME_USDT = float(os.getenv("MIN_24H_VOLUME_USDT", "100000.0"))
MIN_LIQUIDITY_SCORE = float(os.getenv("MIN_LIQUIDITY_SCORE", "40.0"))
USE_ONLY_TOP_LIQUID_COINS = int(os.getenv("USE_ONLY_TOP_LIQUID_COINS", "200"))

# ============================================================================
# PERFORMANCE OPTIMIZATION
# ============================================================================
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "100"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "50"))  # для func_async
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.05"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "0.5"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# ============================================================================
# CACHING
# ============================================================================
ENABLE_CACHE = True
CACHE_HOT_PAIRS = int(os.getenv("CACHE_HOT_PAIRS", "100"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))

# ============================================================================
# WEBSOCKET
# ============================================================================
WEBSOCKET_ENABLED = os.getenv("WEBSOCKET_ENABLED", "True").lower() in ("true", "1", "yes")
WEBSOCKET_RECONNECT_DELAY = float(os.getenv("WEBSOCKET_RECONNECT_DELAY", "5.0"))
WEBSOCKET_PING_INTERVAL = float(os.getenv("WEBSOCKET_PING_INTERVAL", "20.0"))

# ============================================================================
# API ENDPOINTS
# ============================================================================
BYBIT_API_URL = os.getenv("BYBIT_API_URL", "https://api.bybit.com")
BYBIT_WS_URL = os.getenv("BYBIT_WS_URL", "wss://stream.bybit.com/v5/public/spot")
BINANCE_API_URL = os.getenv("BINANCE_API_URL", "https://api.binance.com")

# ============================================================================
# TRADING BOT SPECIFIC PARAMETERS
# ============================================================================
# Timeframes
TIMEFRAME_SHORT = "60"  # 1H
TIMEFRAME_LONG = "240"  # 4H
TIMEFRAME_HTF = "D"     # 1D

TIMEFRAME_SHORT_NAME = "1H"
TIMEFRAME_LONG_NAME = "4H"
TIMEFRAME_HTF_NAME = "1D"

# Candle limits for stages
QUICK_SCAN_CANDLES = 48     # Stage 1: быстрое сканирование
AI_BULK_CANDLES = 100       # Stage 2: AI выбор
FINAL_SHORT_CANDLES = 168   # Stage 3: 1H (7 дней)
FINAL_LONG_CANDLES = 84     # Stage 3: 4H (14 дней)
FINAL_HTF_CANDLES = 30      # Stage 3: 1D (30 дней)

# Indicator history lengths
AI_INDICATORS_HISTORY = 60
FINAL_INDICATORS_HISTORY = 60

# Selection limits
MAX_BULK_PAIRS = 50
MAX_FINAL_PAIRS = 3

# ============================================================================
# TECHNICAL INDICATORS PARAMETERS
# ============================================================================
EMA_FAST = 5
EMA_MEDIUM = 8
EMA_SLOW = 20

RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14

MIN_VOLUME_RATIO = 1.2

# ============================================================================
# MARKET DATA THRESHOLDS
# ============================================================================
OI_CHANGE_GROWING_THRESHOLD = 5.0   # OI рост >5% = strength
OI_CHANGE_DECLINING_THRESHOLD = -5.0 # OI падение <-5% = weakness

# ============================================================================
# PROMPTS FILES
# ============================================================================
SELECTION_PROMPT = "prompt_select.txt"
ANALYSIS_PROMPT = "prompt_analyze.txt"
VALIDATION_PROMPT = "prompt_validate.txt"

# ============================================================================
# LOGGING
# ============================================================================
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

SAVE_OPPORTUNITIES_TO_FILE = os.getenv("SAVE_OPPORTUNITIES_TO_FILE", "True").lower() in ("true", "1", "yes")
OPPORTUNITIES_LOG_FILE = LOGS_DIR / "opportunities.log"
LOG_LEVEL = int(os.getenv("LOG_LEVEL", "1"))

# ============================================================================
# COIN FILTERING
# ============================================================================
ENABLE_COIN_FILTER = os.getenv("ENABLE_COIN_FILTER", "False").lower() in ("true", "1", "yes")

# Черный список монет (через запятую)
blacklist_str = os.getenv("BLACKLIST_COINS", "")
BLACKLIST_COINS = set(coin.strip().upper() for coin in blacklist_str.split(",") if coin.strip())

# Белый список монет (через запятую)
whitelist_str = os.getenv("WHITELIST_COINS", "")
WHITELIST_COINS = set(coin.strip().upper() for coin in whitelist_str.split(",") if coin.strip())

# ============================================================================
# DIRECTORIES
# ============================================================================
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# DEBUG MODE
# ============================================================================
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

# ============================================================================
# CONFIG CLASS (для обратной совместимости)
# ============================================================================
class Config:
    """Класс-обертка для доступа к конфигурации"""

    # Все переменные доступны как атрибуты класса
    def __getattr__(self, name):
        # Возвращаем значение из глобального namespace
        return globals().get(name)

    @classmethod
    def validate(cls):
        """Валидация конфигурации"""
        return validate_config()

    @classmethod
    def print_summary(cls):
        """Вывод сводки конфигурации"""
        return print_config_summary()

# Создаем глобальный экземпляр
config = Config()

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================
def validate_config():
    """Проверяет критичные параметры конфигурации"""
    errors = []
    warnings = []

    # Проверка API ключей
    if not DEEPSEEK_API_KEY and STAGE2_PROVIDER == "deepseek":
        errors.append("DEEPSEEK_API_KEY не задан, но выбран для Stage 2")

    if not ANTHROPIC_API_KEY and (STAGE3_PROVIDER == "claude" or STAGE4_PROVIDER == "claude"):
        errors.append("ANTHROPIC_API_KEY не задан, но Claude выбран для Stage 3 или 4")

    if not BESTCHANGE_API_KEY:
        warnings.append("BESTCHANGE_API_KEY не задан (для арбитража)")

    # Проверка Telegram
    if not TELEGRAM_BOT_TOKEN:
        warnings.append("TELEGRAM_BOT_TOKEN не задан (бот не будет работать)")

    # Проверка reasoning режима
    if DEEPSEEK_REASONING and "reasoner" not in DEEPSEEK_MODEL.lower():
        warnings.append(f"DEEPSEEK_REASONING=true, но модель {DEEPSEEK_MODEL} не поддерживает reasoning")
        warnings.append("Используйте DEEPSEEK_MODEL=deepseek-reasoner для включения reasoning")

    # Проверка провайдеров
    valid_providers = ["deepseek", "claude"]
    if STAGE2_PROVIDER not in valid_providers:
        errors.append(f"STAGE2_PROVIDER={STAGE2_PROVIDER} недопустим (должен быть: {', '.join(valid_providers)})")
    if STAGE3_PROVIDER not in valid_providers:
        errors.append(f"STAGE3_PROVIDER={STAGE3_PROVIDER} недопустим (должен быть: {', '.join(valid_providers)})")
    if STAGE4_PROVIDER not in valid_providers:
        errors.append(f"STAGE4_PROVIDER={STAGE4_PROVIDER} недопустим (должен быть: {', '.join(valid_providers)})")

    # Проверка параметров
    if START_AMOUNT <= 0:
        errors.append(f"START_AMOUNT должен быть > 0 (текущее: {START_AMOUNT})")

    if MIN_SPREAD < 0:
        warnings.append(f"MIN_SPREAD < 0 ({MIN_SPREAD}%), будут показаны все возможности")

    if MAX_CONCURRENT_REQUESTS > 200:
        warnings.append(f"MAX_CONCURRENT_REQUESTS очень высокий ({MAX_CONCURRENT_REQUESTS}), возможны проблемы с rate limit")

    return errors, warnings

def print_config_summary():
    """Выводит сводку конфигурации"""
    print("\n" + "="*100)
    print("⚙️  КОНФИГУРАЦИЯ СИСТЕМЫ")
    print("="*100)

    print(f"\n🤖 AI КОНФИГУРАЦИЯ:")
    print(f"   • DeepSeek модель: {DEEPSEEK_MODEL}")
    print(f"   • DeepSeek reasoning: {'✅ Включен' if DEEPSEEK_REASONING else '❌ Выключен'}")
    print(f"   • Anthropic модель: {ANTHROPIC_MODEL}")
    print(f"   • Anthropic thinking: {'✅ Включен' if ANTHROPIC_THINKING else '❌ Выключен'}")

    print(f"\n🎯 MULTI-STAGE PIPELINE:")
    print(f"   • Stage 2 (выбор пар): {STAGE2_PROVIDER.upper()}")
    print(f"   • Stage 3 (анализ): {STAGE3_PROVIDER.upper()}")
    print(f"   • Stage 4 (валидация): {STAGE4_PROVIDER.upper()}")

    print(f"\n🌡️  AI ПАРАМЕТРЫ:")
    print(f"   • Temperature выбора: {AI_TEMPERATURE_SELECT}")
    print(f"   • Temperature анализа: {AI_TEMPERATURE_ANALYZE}")
    print(f"   • Temperature валидации: {AI_TEMPERATURE_VALIDATE}")
    print(f"   • Max tokens выбора: {AI_MAX_TOKENS_SELECT}")
    print(f"   • Max tokens анализа: {AI_MAX_TOKENS_ANALYZE}")
    print(f"   • Max tokens валидации: {AI_MAX_TOKENS_VALIDATE}")

    print(f"\n💰 ТОРГОВЛЯ:")
    print(f"   • Начальная сумма: ${START_AMOUNT}")
    print(f"   • Минимальный спред: {MIN_SPREAD}%")
    print(f"   • Минимальная прибыль: ${MIN_PROFIT_USD}")
    print(f"   • Мин. confidence: {MIN_CONFIDENCE}%")
    print(f"   • Мин. R/R: {MIN_RISK_REWARD_RATIO}:1")

    print(f"\n💧 ЛИКВИДНОСТЬ:")
    print(f"   • Мин. объем 24ч: ${MIN_24H_VOLUME_USDT:,.0f}")
    print(f"   • Мин. оценка: {MIN_LIQUIDITY_SCORE}")
    print(f"   • Топ монет: {USE_ONLY_TOP_LIQUID_COINS}")

    print(f"\n⚡ ПРОИЗВОДИТЕЛЬНОСТЬ:")
    print(f"   • Параллельных запросов: {MAX_CONCURRENT_REQUESTS}")
    print(f"   • Задержка запросов: {REQUEST_DELAY}с")
    print(f"   • Размер батча: {BATCH_SIZE}")
    print(f"   • Макс. попыток: {MAX_RETRIES}")
    print(f"   • Таймаут запроса: {REQUEST_TIMEOUT}с")

    print(f"\n🔌 ДОПОЛНИТЕЛЬНО:")
    print(f"   • WebSocket: {'✅ Включен' if WEBSOCKET_ENABLED else '❌ Выключен'}")
    print(f"   • Кэширование: {'✅ Включено' if ENABLE_CACHE else '❌ Выключено'}")
    if ENABLE_CACHE:
        print(f"     - Горячих пар: {CACHE_HOT_PAIRS}")
        print(f"     - TTL кэша: {CACHE_TTL}с")
    print(f"   • Логирование: уровень {LOG_LEVEL}")
    print(f"   • Сохранение в файл: {'Да' if SAVE_OPPORTUNITIES_TO_FILE else 'Нет'}")

    print(f"\n🔑 API КЛЮЧИ:")
    print(f"   • DeepSeek: {'✅ Задан' if DEEPSEEK_API_KEY else '❌ НЕ ЗАДАН'}")
    print(f"   • Anthropic: {'✅ Задан' if ANTHROPIC_API_KEY else '❌ НЕ ЗАДАН'}")
    print(f"   • BestChange: {'✅ Задан' if BESTCHANGE_API_KEY else '⚠️  Не задан'}")
    print(f"   • Bybit: {'✅ Задан' if BYBIT_API_KEY else '⚠️  Не задан (опционально)'}")
    print(f"   • Telegram: {'✅ Задан' if TELEGRAM_BOT_TOKEN else '❌ НЕ ЗАДАН'}")

    print(f"\n🎨 ФИЛЬТРАЦИЯ МОНЕТ:")
    print(f"   • Включена: {'Да' if ENABLE_COIN_FILTER else 'Нет'}")
    if ENABLE_COIN_FILTER:
        if WHITELIST_COINS:
            print(f"   • Белый список: {len(WHITELIST_COINS)} монет")
            print(f"     {', '.join(list(WHITELIST_COINS)[:10])}" + (" ..." if len(WHITELIST_COINS) > 10 else ""))
        if BLACKLIST_COINS:
            print(f"   • Черный список: {len(BLACKLIST_COINS)} монет")
            print(f"     {', '.join(list(BLACKLIST_COINS)[:10])}" + (" ..." if len(BLACKLIST_COINS) > 10 else ""))

    # Валидация
    errors, warnings = validate_config()

    if errors:
        print(f"\n❌ КРИТИЧНЫЕ ОШИБКИ ({len(errors)}):")
        for i, error in enumerate(errors, 1):
            print(f"   {i}. {error}")

    if warnings:
        print(f"\n⚠️  ПРЕДУПРЕЖДЕНИЯ ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"   {i}. {warning}")

    print("\n" + "="*100 + "\n")

    # Если есть критичные ошибки - останавливаем
    if errors:
        print("❌ Невозможно продолжить из-за критичных ошибок в конфигурации")
        print("   Исправьте .env файл и запустите снова\n")
        return False

    if warnings:
        print("⚠️  Есть предупреждения, но работа продолжится")
        print("   Рекомендуется исправить для оптимальной работы\n")

    return True

# ============================================================================
# AUTO-RUN
# ============================================================================
if __name__ == "__main__":
    print("\n🔧 Проверка конфигурации...")
    if print_config_summary():
        print("✅ Конфигурация валидна!\n")
    else:
        print("❌ Конфигурация содержит ошибки!\n")
        exit(1)