"""
Trading bot configuration - OPTIMIZED FOR SWING TRADING (1H/4H/1D)
FIXED: Bybit API использует 'D' для дневных свечей, не '1440'
"""

import os
from dataclasses import dataclass
from pathlib import Path

try:
    with open('../.env', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
except:
    pass


@dataclass
class Config:
    """Bot configuration - Swing Trading Focus"""

    # API Keys
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK') or os.getenv('DEEPSEEK_API_KEY')
    DEEPSEEK_URL = 'https://api.deepseek.com/v1'
    DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat') #'deepseek-reasoner' / 'deepseek-chat'
    DEEPSEEK_REASONING = os.getenv('DEEPSEEK_REASONING', 'false').lower() == 'true'

    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC')
    ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-20250514')
    ANTHROPIC_THINKING = os.getenv('ANTHROPIC_THINKING', 'false').lower() == 'true'

    # AI Stage Selection
    STAGE2_PROVIDER = os.getenv('STAGE2_PROVIDER', 'deepseek')  # deepseek or claude
    STAGE3_PROVIDER = os.getenv('STAGE3_PROVIDER', 'deepseek')    # claude or deepseek
    STAGE4_PROVIDER = os.getenv('STAGE4_PROVIDER', 'deepseek')    # claude or deepseek

    # Timeframes (SWING TRADING - 1H/4H/1D)
    TIMEFRAME_SHORT = '60'      # 1 hour
    TIMEFRAME_SHORT_NAME = '1h'
    TIMEFRAME_LONG = '240'      # 4 hours
    TIMEFRAME_LONG_NAME = '4h'

    # FIXED: Bybit API format для дневных свечей
    TIMEFRAME_HTF = 'D'         # Daily (Bybit использует 'D', не '1440')
    TIMEFRAME_HTF_NAME = '1d'

    # Candles
    QUICK_SCAN_CANDLES = 48     # 4H = 8 days
    AI_BULK_CANDLES = 48
    AI_INDICATORS_HISTORY = 40
    FINAL_SHORT_CANDLES = 168   # 1H = 7 days
    FINAL_LONG_CANDLES = 84     # 4H = 14 days
    FINAL_HTF_CANDLES = 30      # 1D = 30 days для контекста
    FINAL_INDICATORS_HISTORY = 60

    # Indicators
    EMA_FAST = 9
    EMA_MEDIUM = 21
    EMA_SLOW = 50
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    ATR_PERIOD = 14

    # Trading - ENHANCED FOR SWING
    MIN_CONFIDENCE = 75
    MIN_VOLUME_RATIO = 1.3
    MIN_ATR_RATIO = 0.8
    MIN_RISK_REWARD_RATIO = 2.5  # INCREASED from 2.0 для swing quality
    MAX_HOLD_DURATION = 4320     # INCREASED: 72 hours (3 days)
    MIN_HOLD_DURATION = 240      # 4 hours minimum
    VALIDATION_CONFIDENCE_BOOST = 5

    # NEW: R/R Targets для swing trading
    RR_TARGET_AGGRESSIVE = [2.5, 4.0, 6.0]  # Strong setups
    RR_TARGET_MODERATE = [2.0, 3.0, 4.5]    # Medium setups
    RR_TARGET_CONSERVATIVE = [1.8, 2.5, 3.5]  # Weak setups

    # Performance
    BATCH_SIZE = 50
    MAX_CONCURRENT = 10
    MAX_FINAL_PAIRS = 5
    MAX_BULK_PAIRS = 15  # Используется только для Claude, DeepSeek обрабатывает все

    # API Timeouts
    API_TIMEOUT = 120
    API_TIMEOUT_SELECTION = int(os.getenv('API_TIMEOUT_SELECTION', '180'))
    API_TIMEOUT_ANALYSIS = int(os.getenv('API_TIMEOUT_ANALYSIS', '180'))
    API_TIMEOUT_VALIDATION = int(os.getenv('API_TIMEOUT_VALIDATION', '120'))

    # Prompts
    @staticmethod
    def _get_prompt_path(filename: str) -> str:
        """Get absolute path to prompt file"""
        config_dir = Path(__file__).parent
        prompt_path = config_dir / filename
        if prompt_path.exists():
            return str(prompt_path)

        root_dir = config_dir.parent
        prompt_path = root_dir / 'trade_bot_programm' / filename
        if prompt_path.exists():
            return str(prompt_path)

        return f'trade_bot_programm/{filename}'

    SELECTION_PROMPT = _get_prompt_path('prompt_select.txt')
    ANALYSIS_PROMPT = _get_prompt_path('prompt_analyze.txt')
    VALIDATION_PROMPT = _get_prompt_path('prompt_validate.txt')

    # AI Parameters
    AI_TEMPERATURE_SELECT = float(os.getenv('AI_TEMPERATURE_SELECT', '0.3'))
    AI_TEMPERATURE_ANALYZE = float(os.getenv('AI_TEMPERATURE_ANALYZE', '0.7'))
    AI_TEMPERATURE_VALIDATE = float(os.getenv('AI_TEMPERATURE_VALIDATE', '0.3'))
    AI_MAX_TOKENS_SELECT = int(os.getenv('AI_MAX_TOKENS_SELECT', '2000'))
    AI_MAX_TOKENS_ANALYZE = int(os.getenv('AI_MAX_TOKENS_ANALYZE', '3500'))
    AI_MAX_TOKENS_VALIDATE = int(os.getenv('AI_MAX_TOKENS_VALIDATE', '4000'))

    # Market Data Thresholds
    OI_CHANGE_GROWING_THRESHOLD = float(os.getenv('OI_CHANGE_GROWING_THRESHOLD', '2.0'))
    OI_CHANGE_DECLINING_THRESHOLD = float(os.getenv('OI_CHANGE_DECLINING_THRESHOLD', '-2.0'))
    SPREAD_ILLIQUID_THRESHOLD = float(os.getenv('SPREAD_ILLIQUID_THRESHOLD', '0.15'))
    SPREAD_WARNING_THRESHOLD = float(os.getenv('SPREAD_WARNING_THRESHOLD', '0.08'))

    # NEW: Session Quality Metrics (для session awareness)
    SESSION_QUALITY = {
        'asian_night': 0.3,    # 00:00-08:00 UTC - низкое качество
        'london_open': 0.9,    # 08:00-12:00 UTC - высокое (breakouts)
        'london_us_overlap': 1.0,  # 12:00-16:00 UTC - максимальное
        'us_session': 0.85,    # 16:00-20:00 UTC - высокое
        'us_close': 0.5        # 20:00-00:00 UTC - среднее
    }

    # NEW: Divergence Detection Settings
    DIVERGENCE_LOOKBACK = 20  # Свечей для поиска дивергенций
    DIVERGENCE_CONFIDENCE_BOOST = 25  # Бонус к confidence за дивергенцию


config = Config()