"""
Trading bot configuration
"""

import os
from dataclasses import dataclass

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
    """Bot configuration"""

    # API Keys
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK') or os.getenv('DEEPSEEK_API_KEY')
    DEEPSEEK_URL = 'https://api.deepseek.com/v1'
    DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
    DEEPSEEK_REASONING = os.getenv('DEEPSEEK_REASONING', 'false').lower() == 'true'

    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC')
    ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-20250514')
    ANTHROPIC_THINKING = os.getenv('ANTHROPIC_THINKING', 'false').lower() == 'true'

    # AI Stage Selection
    STAGE2_PROVIDER = os.getenv('STAGE2_PROVIDER', 'deepseek')  # deepseek or claude
    STAGE3_PROVIDER = os.getenv('STAGE3_PROVIDER', 'claude')    # claude or deepseek
    STAGE4_PROVIDER = os.getenv('STAGE4_PROVIDER', 'claude')    # claude or deepseek

    # Timeframes
    TIMEFRAME_SHORT = '60'
    TIMEFRAME_SHORT_NAME = '1h'
    TIMEFRAME_LONG = '240'
    TIMEFRAME_LONG_NAME = '4h'

    # Candles
    QUICK_SCAN_CANDLES = 48
    AI_BULK_CANDLES = 48
    AI_INDICATORS_HISTORY = 40
    FINAL_SHORT_CANDLES = 168
    FINAL_LONG_CANDLES = 84
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

    # Trading
    MIN_CONFIDENCE = 75
    MIN_VOLUME_RATIO = 1.3
    MIN_ATR_RATIO = 0.8
    MIN_RISK_REWARD_RATIO = 2.0
    MAX_HOLD_DURATION = 2880
    MIN_HOLD_DURATION = 240
    VALIDATION_CONFIDENCE_BOOST = 5

    # Performance
    BATCH_SIZE = 50
    MAX_CONCURRENT = 10
    API_TIMEOUT = 120
    MAX_FINAL_PAIRS = 5
    MAX_BULK_PAIRS = 15

    # Prompts
    SELECTION_PROMPT = 'prompt_select.txt'
    ANALYSIS_PROMPT = 'prompt_analyze.txt'
    VALIDATION_PROMPT = 'prompt_validate.txt'

    # AI Parameters
    AI_TEMPERATURE_SELECT = float(os.getenv('AI_TEMPERATURE_SELECT', '0.3'))
    AI_TEMPERATURE_ANALYZE = float(os.getenv('AI_TEMPERATURE_ANALYZE', '0.7'))
    AI_TEMPERATURE_VALIDATE = float(os.getenv('AI_TEMPERATURE_VALIDATE', '0.3'))
    AI_MAX_TOKENS_SELECT = int(os.getenv('AI_MAX_TOKENS_SELECT', '2000'))
    AI_MAX_TOKENS_ANALYZE = int(os.getenv('AI_MAX_TOKENS_ANALYZE', '3000'))
    AI_MAX_TOKENS_VALIDATE = int(os.getenv('AI_MAX_TOKENS_VALIDATE', '3500'))


config = Config()