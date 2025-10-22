# config.py - OPTIMIZED: Reduced data + rate limit protection
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
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com"

# ============================================================================
# STAGE PROVIDER SELECTION
# ============================================================================
STAGE2_PROVIDER = os.getenv("STAGE2_PROVIDER", "claude").lower()  # Haiku для Stage 2
STAGE3_PROVIDER = os.getenv("STAGE3_PROVIDER", "claude").lower()  # Sonnet для Stage 3

# ============================================================================
# STAGE 2: PAIR SELECTION (Haiku с multi-TF)
# ============================================================================
STAGE2_MODEL = os.getenv("STAGE2_MODEL", "claude-haiku-4-5-20251001")
STAGE2_TEMPERATURE = float(os.getenv("STAGE2_TEMPERATURE", "0.3"))
STAGE2_MAX_TOKENS = int(os.getenv("STAGE2_MAX_TOKENS", "2000"))

# ============================================================================
# STAGE 3: ANALYSIS (Sonnet)
# ============================================================================
STAGE3_MODEL = os.getenv("STAGE3_MODEL", "claude-sonnet-4-20250514")
STAGE3_TEMPERATURE = float(os.getenv("STAGE3_TEMPERATURE", "0.7"))
STAGE3_MAX_TOKENS = int(os.getenv("STAGE3_MAX_TOKENS", "3000"))

# ============================================================================
# STAGE 4: REMOVED
# ============================================================================

# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
AI_TEMPERATURE_SELECT = STAGE2_TEMPERATURE
AI_TEMPERATURE_ANALYZE = STAGE3_TEMPERATURE
AI_MAX_TOKENS_SELECT = STAGE2_MAX_TOKENS
AI_MAX_TOKENS_ANALYZE = STAGE3_MAX_TOKENS

# ============================================================================
# AI REASONING/THINKING MODES
# ============================================================================
DEEPSEEK_REASONING = os.getenv("DEEPSEEK_REASONING", "false").lower() in ("true", "1", "yes")
ANTHROPIC_THINKING = os.getenv("ANTHROPIC_THINKING", "false").lower() in ("true", "1", "yes")

# ============================================================================
# API TIMEOUTS & RATE LIMITS
# ============================================================================
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
API_TIMEOUT_ANALYSIS = int(os.getenv("API_TIMEOUT_ANALYSIS", "60"))

# НОВОЕ: Rate limit protection
CLAUDE_RATE_LIMIT_DELAY = float(os.getenv("CLAUDE_RATE_LIMIT_DELAY", "65.0"))  # 1 min 5 sec между запросами

# ============================================================================
# TRADING PARAMETERS
# ============================================================================
MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", "75"))
MIN_RISK_REWARD_RATIO = float(os.getenv("MIN_RISK_REWARD_RATIO", "2.5"))

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
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "50"))
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
# TRADING BOT SPECIFIC PARAMETERS - OPTIMIZED
# ============================================================================
TIMEFRAME_SHORT = "60"   # 1H
TIMEFRAME_LONG = "240"   # 4H
TIMEFRAME_HTF = "D"      # 1D

TIMEFRAME_SHORT_NAME = "1H"
TIMEFRAME_LONG_NAME = "4H"
TIMEFRAME_HTF_NAME = "1D"

# ОПТИМИЗИРОВАНО: Уменьшены свечи для Stage 2
QUICK_SCAN_CANDLES = 48  # Stage 1: Base filtering

# Stage 2: Compact multi-TF data
STAGE2_CANDLES_1H = 30   # Последние 30 свечей 1H
STAGE2_CANDLES_4H = 30   # Последние 30 свечей 4H
STAGE2_CANDLES_1D = 10   # Последние 10 свечей 1D

# Stage 3: Full analysis (REDUCED from 168/84/30)
STAGE3_CANDLES_1H = 100  # Было 168
STAGE3_CANDLES_4H = 60   # Было 84
STAGE3_CANDLES_1D = 20   # Было 30

AI_INDICATORS_HISTORY = 30  # Было 60
FINAL_INDICATORS_HISTORY = 30  # Было 60

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
OI_CHANGE_GROWING_THRESHOLD = 5.0
OI_CHANGE_DECLINING_THRESHOLD = -5.0

# ============================================================================
# PROMPTS FILES
# ============================================================================
SELECTION_PROMPT = "prompt_select.txt"
ANALYSIS_PROMPT = "prompt_analyze.txt"

# ============================================================================
# LOGGING
# ============================================================================
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

SAVE_OPPORTUNITIES_TO_FILE = os.getenv("SAVE_OPPORTUNITIES_TO_FILE", "True").lower() in ("true", "1", "yes")
OPPORTUNITIES_LOG_FILE = LOGS_DIR / "opportunities.log"
LOG_LEVEL = int(os.getenv("LOG_LEVEL", "1"))

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
# CONFIG CLASS
# ============================================================================
class Config:
    def __getattr__(self, name):
        return globals().get(name)

config = Config()