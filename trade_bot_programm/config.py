"""
Trading Bot Configuration
FINAL FIX: Экспорт API ключей на уровне модуля + безопасное преобразование типов
Файл: trade_bot_programm/config.py
"""

import os
from pathlib import Path


def load_env():
    """Загрузить .env файл"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


def safe_int(value, default):
    """Безопасное преобразование в int"""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def safe_float(value, default):
    """Безопасное преобразование в float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_bool(value):
    """Безопасное преобразование в bool"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ['true', '1', 'yes']
    return False


# Загружаем переменные окружения при импорте
load_env()


# ============================================================================
# API KEYS - ЭКСПОРТ НА УРОВНЕ МОДУЛЯ
# ============================================================================
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')


# ============================================================================
# TRADING BOT SPECIFIC PARAMETERS
# ============================================================================
TIMEFRAME_SHORT = "60"   # 1H
TIMEFRAME_LONG = "240"   # 4H
TIMEFRAME_HTF = "D"      # 1D

TIMEFRAME_SHORT_NAME = "1H"
TIMEFRAME_LONG_NAME = "4H"
TIMEFRAME_HTF_NAME = "1D"

QUICK_SCAN_CANDLES = 48

# Stage 2 (Compact data для DeepSeek)
STAGE2_CANDLES_1H = 30
STAGE2_CANDLES_4H = 30
STAGE2_CANDLES_1D = 10

# Stage 3 (Full data для Sonnet)
STAGE3_CANDLES_1H = 100
STAGE3_CANDLES_4H = 60
STAGE3_CANDLES_1D = 20

AI_INDICATORS_HISTORY = 30
FINAL_INDICATORS_HISTORY = 30

# Финальный лимит: сколько пар DeepSeek должен ВЫБРАТЬ из всех переданных
MAX_FINAL_PAIRS = safe_int(os.getenv('MAX_FINAL_PAIRS', '3'), 3)


# ============================================================================
# CONFIG CLASS для совместимости с импортами вида "from config import config"
# ============================================================================
class Config:
    """Класс конфигурации для удобного доступа к настройкам"""

    # API Keys
    DEEPSEEK_API_KEY = DEEPSEEK_API_KEY
    ANTHROPIC_API_KEY = ANTHROPIC_API_KEY

    # DeepSeek Settings
    DEEPSEEK_URL = os.getenv('DEEPSEEK_URL', 'https://api.deepseek.com')
    DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
    DEEPSEEK_REASONING = safe_bool(os.getenv('DEEPSEEK_REASONING', 'false'))

    # Anthropic Settings
    ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-5-20250929')
    ANTHROPIC_THINKING = safe_bool(os.getenv('ANTHROPIC_THINKING', 'false'))

    # Stage Configuration
    STAGE2_PROVIDER = os.getenv('STAGE2_PROVIDER', 'deepseek')
    STAGE2_MODEL = os.getenv('STAGE2_MODEL', 'deepseek-chat')
    STAGE2_TEMPERATURE = safe_float(os.getenv('STAGE2_TEMPERATURE', '0.3'), 0.3)
    STAGE2_MAX_TOKENS = safe_int(os.getenv('STAGE2_MAX_TOKENS', '2000'), 2000)

    STAGE3_PROVIDER = os.getenv('STAGE3_PROVIDER', 'claude')
    STAGE3_MODEL = os.getenv('STAGE3_MODEL', 'claude-sonnet-4-5-20250929')
    STAGE3_TEMPERATURE = safe_float(os.getenv('STAGE3_TEMPERATURE', '0.7'), 0.7)
    STAGE3_MAX_TOKENS = safe_int(os.getenv('STAGE3_MAX_TOKENS', '4000'), 4000)

    # Formatter Configuration
    FORMATTER_MODEL = os.getenv('FORMATTER_MODEL', 'deepseek-chat')
    FORMATTER_TEMPERATURE = safe_float(os.getenv('FORMATTER_TEMPERATURE', '0.3'), 0.3)
    FORMATTER_MAX_TOKENS = safe_int(os.getenv('FORMATTER_MAX_TOKENS', '2000'), 2000)

    # API Settings
    API_TIMEOUT = safe_int(os.getenv('API_TIMEOUT', '30'), 30)
    API_TIMEOUT_ANALYSIS = safe_int(os.getenv('API_TIMEOUT_ANALYSIS', '120'), 120)
    MAX_CONCURRENT = safe_int(os.getenv('MAX_CONCURRENT', '50'), 50)

    # Trading Parameters
    MIN_CONFIDENCE = safe_int(os.getenv('MIN_CONFIDENCE', '70'), 70)
    MIN_VOLUME_RATIO = safe_float(os.getenv('MIN_VOLUME_RATIO', '1.2'), 1.2)

    # Rate Limiting
    CLAUDE_RATE_LIMIT_DELAY = safe_int(os.getenv('CLAUDE_RATE_LIMIT_DELAY', '0'), 0)

    # Market Data Thresholds
    OI_CHANGE_GROWING_THRESHOLD = safe_float(os.getenv('OI_CHANGE_GROWING_THRESHOLD', '2.0'), 2.0)
    OI_CHANGE_DECLINING_THRESHOLD = safe_float(os.getenv('OI_CHANGE_DECLINING_THRESHOLD', '-2.0'), -2.0)

    # Technical Indicators
    EMA_FAST = safe_int(os.getenv('EMA_FAST', '5'), 5)
    EMA_MEDIUM = safe_int(os.getenv('EMA_MEDIUM', '8'), 8)
    EMA_SLOW = safe_int(os.getenv('EMA_SLOW', '20'), 20)
    RSI_PERIOD = safe_int(os.getenv('RSI_PERIOD', '14'), 14)
    MACD_FAST = safe_int(os.getenv('MACD_FAST', '12'), 12)
    MACD_SLOW = safe_int(os.getenv('MACD_SLOW', '26'), 26)
    MACD_SIGNAL = safe_int(os.getenv('MACD_SIGNAL', '9'), 9)
    ATR_PERIOD = safe_int(os.getenv('ATR_PERIOD', '14'), 14)

    # Pair Selection
    MAX_FINAL_PAIRS = MAX_FINAL_PAIRS

    # Prompts
    SELECTION_PROMPT = os.getenv('SELECTION_PROMPT', 'prompts/prompt_select.txt')
    ANALYSIS_PROMPT = os.getenv('ANALYSIS_PROMPT', 'prompts/prompt_analyze.txt')

    # AI Settings
    AI_TEMPERATURE_SELECT = safe_float(os.getenv('AI_TEMPERATURE_SELECT', '0.3'), 0.3)
    AI_TEMPERATURE_ANALYZE = safe_float(os.getenv('AI_TEMPERATURE_ANALYZE', '0.7'), 0.7)
    AI_MAX_TOKENS_SELECT = safe_int(os.getenv('AI_MAX_TOKENS_SELECT', '2000'), 2000)
    AI_MAX_TOKENS_ANALYZE = safe_int(os.getenv('AI_MAX_TOKENS_ANALYZE', '4000'), 4000)

    # Timeframes (from module-level constants)
    TIMEFRAME_SHORT = TIMEFRAME_SHORT
    TIMEFRAME_LONG = TIMEFRAME_LONG
    TIMEFRAME_HTF = TIMEFRAME_HTF
    TIMEFRAME_SHORT_NAME = TIMEFRAME_SHORT_NAME
    TIMEFRAME_LONG_NAME = TIMEFRAME_LONG_NAME
    TIMEFRAME_HTF_NAME = TIMEFRAME_HTF_NAME

    # Candles
    QUICK_SCAN_CANDLES = QUICK_SCAN_CANDLES
    STAGE2_CANDLES_1H = STAGE2_CANDLES_1H
    STAGE2_CANDLES_4H = STAGE2_CANDLES_4H
    STAGE2_CANDLES_1D = STAGE2_CANDLES_1D
    STAGE3_CANDLES_1H = STAGE3_CANDLES_1H
    STAGE3_CANDLES_4H = STAGE3_CANDLES_4H
    STAGE3_CANDLES_1D = STAGE3_CANDLES_1D

    # Indicators
    AI_INDICATORS_HISTORY = AI_INDICATORS_HISTORY
    FINAL_INDICATORS_HISTORY = FINAL_INDICATORS_HISTORY


# Создаём экземпляр для импорта
config = Config()