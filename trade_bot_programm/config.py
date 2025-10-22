# config.py - FIXED: Stage-specific model configuration
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

# DeepSeek URL
DEEPSEEK_URL = "https://api.deepseek.com"

# ============================================================================
# STAGE PROVIDER SELECTION
# ============================================================================
STAGE2_PROVIDER = os.getenv("STAGE2_PROVIDER", "deepseek").lower()
STAGE3_PROVIDER = os.getenv("STAGE3_PROVIDER", "deepseek").lower()
STAGE4_PROVIDER = os.getenv("STAGE4_PROVIDER", "deepseek").lower()

# ============================================================================
# STAGE 2: PAIR SELECTION
# ============================================================================
STAGE2_MODEL = os.getenv("STAGE2_MODEL", "deepseek-chat")
STAGE2_TEMPERATURE = float(os.getenv("STAGE2_TEMPERATURE", "0.3"))
STAGE2_MAX_TOKENS = int(os.getenv("STAGE2_MAX_TOKENS", "2000"))

# ============================================================================
# STAGE 3: ANALYSIS
# ============================================================================
STAGE3_MODEL = os.getenv("STAGE3_MODEL", "deepseek-chat")
STAGE3_TEMPERATURE = float(os.getenv("STAGE3_TEMPERATURE", "0.7"))
STAGE3_MAX_TOKENS = int(os.getenv("STAGE3_MAX_TOKENS", "3000"))

# ============================================================================
# STAGE 4: VALIDATION
# ============================================================================
STAGE4_MODEL = os.getenv("STAGE4_MODEL", "deepseek-chat")
STAGE4_TEMPERATURE = float(os.getenv("STAGE4_TEMPERATURE", "0.3"))
STAGE4_MAX_TOKENS = int(os.getenv("STAGE4_MAX_TOKENS", "3500"))

# ============================================================================
# BACKWARD COMPATIBILITY (legacy)
# ============================================================================
DEEPSEEK_MODEL = STAGE2_MODEL  # Default to Stage 2 model
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

AI_TEMPERATURE_SELECT = STAGE2_TEMPERATURE
AI_TEMPERATURE_ANALYZE = STAGE3_TEMPERATURE
AI_TEMPERATURE_VALIDATE = STAGE4_TEMPERATURE

AI_MAX_TOKENS_SELECT = STAGE2_MAX_TOKENS
AI_MAX_TOKENS_ANALYZE = STAGE3_MAX_TOKENS
AI_MAX_TOKENS_VALIDATE = STAGE4_MAX_TOKENS

# ============================================================================
# AI REASONING/THINKING MODES
# ============================================================================
DEEPSEEK_REASONING = os.getenv("DEEPSEEK_REASONING", "false").lower() in ("true", "1", "yes")
ANTHROPIC_THINKING = os.getenv("ANTHROPIC_THINKING", "false").lower() in ("true", "1", "yes")

# ============================================================================
# API TIMEOUTS
# ============================================================================
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
API_TIMEOUT_ANALYSIS = int(os.getenv("API_TIMEOUT_ANALYSIS", "60"))

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
# TRADING BOT SPECIFIC PARAMETERS
# ============================================================================
TIMEFRAME_SHORT = "60"   # 1H
TIMEFRAME_LONG = "240"   # 4H
TIMEFRAME_HTF = "D"      # 1D

TIMEFRAME_SHORT_NAME = "1H"
TIMEFRAME_LONG_NAME = "4H"
TIMEFRAME_HTF_NAME = "1D"

QUICK_SCAN_CANDLES = 48
AI_BULK_CANDLES = 100
FINAL_SHORT_CANDLES = 168
FINAL_LONG_CANDLES = 84
FINAL_HTF_CANDLES = 30

AI_INDICATORS_HISTORY = 60
FINAL_INDICATORS_HISTORY = 60

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