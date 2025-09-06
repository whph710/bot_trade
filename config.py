"""
Исправленная конфигурация для максимального профита
"""

import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SystemConfig:
    """Системные настройки"""
    LOG_FILE: str = 'scalping_bot.log'
    ANALYSIS_LOG_FILE: str = 'instruction_based_analysis.log'
    LOG_LEVEL: str = 'INFO'
    LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'
    USE_EMOJI: bool = False
    ENCODING: str = 'utf-8'


@dataclass
class TimeframeConfig:
    """Настройки таймфреймов согласно инструкции"""
    CONTEXT_TF: str = '15'
    ENTRY_TF: str = '5'
    CANDLES_15M: int = 120
    CANDLES_5M: int = 240
    CANDLES_FOR_AI_SELECTION: int = 50
    CANDLES_FOR_AI_ANALYSIS: int = 150
    CANDLES_FOR_CONTEXT: int = 30
    CANDLES_FOR_ENTRY: int = 60


@dataclass
class IndicatorConfig:
    """Настройки технических индикаторов для максимального профита"""
    EMA_FAST: int = 5
    EMA_MEDIUM: int = 8
    EMA_SLOW: int = 21
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: float = 35.0
    RSI_OVERBOUGHT: float = 65.0
    RSI_EXTREME_OVERSOLD: float = 25.0
    RSI_EXTREME_OVERBOUGHT: float = 75.0
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    ATR_PERIOD: int = 14
    ATR_MULTIPLIER_STOP: float = 1.2
    ATR_MIN_RATIO: float = 0.6
    ATR_OPTIMAL_RATIO: float = 0.8
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    BB_SQUEEZE_RATIO: float = 0.85
    VOLUME_SMA: int = 20
    VOLUME_SPIKE_RATIO: float = 1.3
    VOLUME_MIN_RATIO: float = 0.8


@dataclass
class TradingConfig:
    """Параметры торговли для максимального профита"""
    MIN_CONFIDENCE: int = 60
    HIGH_CONFIDENCE: int = 80
    VALIDATION_CHECKS_REQUIRED: int = 3
    VALIDATION_CHECKS_TOTAL: int = 5
    MIN_STOP_LOSS_PERCENT: float = 0.25
    MAX_TAKE_PROFIT_PERCENT: float = 1.2
    DEFAULT_RISK_REWARD: float = 2.0
    MIN_RISK_REWARD: float = 1.8
    DEFAULT_POSITION_SIZE_PERCENT: float = 2.5
    MAX_POSITION_SIZE_PERCENT: float = 5.0
    MAX_HOLD_TIME_MINUTES: int = 60
    OPTIMAL_HOLD_TIME_MINUTES: int = 45
    MIN_LIQUIDITY_VOLUME: int = 5_000_000
    OPTIMAL_LIQUIDITY_VOLUME: int = 20_000_000
    MAX_SPREAD_PERCENT: float = 0.2
    OPTIMAL_SPREAD_PERCENT: float = 0.15


@dataclass
class PatternConfig:
    """Настройки торговых паттернов для максимального профита"""
    PATTERN_PRIORITY = {
        'MOMENTUM_BREAKOUT': 1,
        'SQUEEZE_BREAKOUT': 2,
        'PULLBACK_ENTRY': 3,
        'RANGE_SCALP': 4
    }
    PATTERN_BASE_CONFIDENCE = {
        'MOMENTUM_BREAKOUT': 80,
        'SQUEEZE_BREAKOUT': 75,
        'PULLBACK_ENTRY': 70,
        'RANGE_SCALP': 65
    }
    RANGE_MIN_SIZE_PERCENT: float = 1.5
    RANGE_BOUNDARY_PROXIMITY: float = 0.15
    PULLBACK_EMA_PROXIMITY: float = 0.008
    PULLBACK_RSI_RECOVERY: float = 50.0
    PULLBACK_RSI_WEAK: float = 50.0


@dataclass
class AIConfig:
    """Настройки для работы с ИИ"""
    API_KEY_ENV: str = 'DEEPSEEK'
    API_BASE_URL: str = 'https://api.deepseek.com'
    API_MODEL: str = 'deepseek-chat'
    DEFAULT_TIMEOUT: int = 60
    SELECTION_TIMEOUT: int = 60
    ANALYSIS_TIMEOUT: int = 80
    HEALTH_CHECK_TIMEOUT: int = 15
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 3
    MAX_TOKENS_SELECTION: int = 1500
    MAX_TOKENS_ANALYSIS: int = 4000
    MAX_TOKENS_TEST: int = 5
    TEMPERATURE_SELECTION: float = 0.4
    TEMPERATURE_ANALYSIS: float = 0.6
    TOP_P_SELECTION: float = 0.85
    TOP_P_ANALYSIS: float = 0.9
    FREQUENCY_PENALTY: float = 0.05
    PRESENCE_PENALTY_SELECTION: float = 0.05
    PRESENCE_PENALTY_ANALYSIS: float = 0.03
    SELECTION_PROMPT_FILE: str = 'prompt2.txt'
    ANALYSIS_PROMPT_FILE: str = 'prompt.txt'
    MAX_PAIRS_TO_AI: int = 10
    MAX_SELECTED_PAIRS: int = 6


@dataclass
class ExchangeConfig:
    """Настройки для работы с биржей Bybit"""
    KLINE_URL: str = "https://api.bybit.com/v5/market/kline"
    INSTRUMENTS_URL: str = "https://api.bybit.com/v5/market/instruments-info"
    API_TIMEOUT: int = 20
    API_CATEGORY: str = "linear"
    QUOTE_CURRENCY: str = 'USDT'
    INSTRUMENT_STATUS: str = 'Trading'
    MAX_CONNECTIONS: int = 10
    MAX_KEEPALIVE_CONNECTIONS: int = 5
    KEEPALIVE_TIMEOUT: int = 30
    KEEPALIVE_MAX: int = 100


@dataclass
class ProcessingConfig:
    """Настройки обработки и батчинга"""
    BATCH_SIZE: int = 30
    BATCH_DELAY: float = 0.15
    MAX_CONCURRENT_REQUESTS: int = 4
    SEMAPHORE_LIMIT: int = 4
    ENABLE_PROMPT_CACHE: bool = True
    ENABLE_HTTP_CLIENT_REUSE: bool = True


@dataclass
class ScoringConfig:
    """Настройки системы оценки сигналов для профита"""
    SCORING_WEIGHTS = {
        'pattern_quality': 4,
        'multi_tf_sync': 3,
        'volume_confirmation': 3,
        'ema_alignment': 2,
        'macd_signal': 2,
        'atr_optimal': 1
    }
    MIN_SCORE_THRESHOLD: float = 10.0
    CONFIDENCE_MODIFIERS = {
        'higher_tf_aligned': 1.15,
        'volume_spike': 1.08,
        'perfect_ema_alignment': 1.08,
        'validation_perfect': 1.15,
        'strong_momentum': 1.1,
        'support_resistance_respect': 1.12
    }


@dataclass
class LevelsConfig:
    """Настройки для определения уровней ИИ"""
    LOOKBACK_PERIODS: int = 50
    MIN_TOUCHES: int = 2
    LEVEL_TOLERANCE: float = 0.003
    LEVEL_STRENGTH_WEIGHTS = {
        'touches_count': 0.4,
        'volume_at_level': 0.3,
        'time_since_test': 0.2,
        'breakout_attempts': 0.1
    }
    ADAPTIVE_STOP_LOSS = {
        'min_atr_multiplier': 0.8,
        'max_atr_multiplier': 2.0,
        'level_distance_multiplier': 0.7,
        'volatility_adjustment': True
    }
    ADAPTIVE_TAKE_PROFIT = {
        'min_risk_reward': 1.5,
        'max_risk_reward': 4.0,
        'level_target_priority': True,
        'trend_extension_multiplier': 1.2
    }


class Config:
    """Главный класс с исправленной конфигурацией"""

    def __init__(self):
        self.system = SystemConfig()
        self.timeframe = TimeframeConfig()
        self.indicators = IndicatorConfig()
        self.trading = TradingConfig()
        self.patterns = PatternConfig()
        self.ai = AIConfig()
        self.exchange = ExchangeConfig()  # ОБЯЗАТЕЛЬНО ВКЛЮЧЕНО
        self.processing = ProcessingConfig()
        self.scoring = ScoringConfig()
        self.levels = LevelsConfig()

        # Проверяем что все атрибуты созданы
        self._validate_config()

    def _validate_config(self):
        """Валидация что все необходимые атрибуты созданы"""
        required_attrs = ['system', 'timeframe', 'indicators', 'trading',
                         'patterns', 'ai', 'exchange', 'processing', 'scoring', 'levels']

        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(f"Missing required config attribute: {attr}")

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование конфигурации в словарь"""
        return {
            'system': self.system.__dict__,
            'timeframe': self.timeframe.__dict__,
            'indicators': self.indicators.__dict__,
            'trading': self.trading.__dict__,
            'patterns': self.patterns.__dict__,
            'ai': self.ai.__dict__,
            'exchange': self.exchange.__dict__,
            'processing': self.processing.__dict__,
            'scoring': self.scoring.__dict__,
            'levels': self.levels.__dict__
        }

    def validate(self) -> bool:
        """Валидация конфигурации"""
        validations = [
            os.getenv(self.ai.API_KEY_ENV) is not None,
            self.trading.MIN_CONFIDENCE <= self.trading.HIGH_CONFIDENCE,
            self.trading.MIN_RISK_REWARD <= self.trading.DEFAULT_RISK_REWARD,
            self.trading.VALIDATION_CHECKS_REQUIRED <= self.trading.VALIDATION_CHECKS_TOTAL,
            int(self.timeframe.CONTEXT_TF) > int(self.timeframe.ENTRY_TF),
            self.indicators.EMA_FAST < self.indicators.EMA_MEDIUM < self.indicators.EMA_SLOW,
            self.indicators.MACD_FAST < self.indicators.MACD_SLOW,
            self.indicators.RSI_OVERSOLD < 50 < self.indicators.RSI_OVERBOUGHT,
            self.ai.SELECTION_TIMEOUT > 0,
            self.ai.ANALYSIS_TIMEOUT > 0,
            self.levels.ADAPTIVE_STOP_LOSS['min_atr_multiplier'] > 0,
            self.levels.ADAPTIVE_TAKE_PROFIT['min_risk_reward'] > 1.0
        ]
        return all(validations)

    @classmethod
    def load_from_env(cls) -> 'Config':
        """Загрузка конфигурации с учетом переменных окружения"""
        return cls()


# Создание глобального экземпляра конфигурации
config = Config()

# Экспорт для удобства импорта
__all__ = [
    'config',
    'Config',
    'SystemConfig',
    'TimeframeConfig',
    'IndicatorConfig',
    'TradingConfig',
    'PatternConfig',
    'AIConfig',
    'ExchangeConfig',
    'ProcessingConfig',
    'ScoringConfig',
    'LevelsConfig'
]