import os
from dataclasses import dataclass
from typing import Dict, Any


# ===========================
# ОСНОВНЫЕ НАСТРОЙКИ СИСТЕМЫ
# ===========================

@dataclass
class SystemConfig:
    """Системные настройки"""
    LOG_FILE: str = 'scalping_bot.log'
    ANALYSIS_LOG_FILE: str = 'instruction_based_analysis.log'
    LOG_LEVEL: str = 'INFO'
    LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'
    USE_EMOJI: bool = False  # Отключить эмоджи в логах
    ENCODING: str = 'utf-8'


# ===========================
# НАСТРОЙКИ ТАЙМФРЕЙМОВ
# ===========================

@dataclass
class TimeframeConfig:
    """Настройки таймфреймов - УВЕЛИЧЕНО для лучшего анализа"""
    # Основные таймфреймы для анализа
    CONTEXT_TF: str = '15'  # 15m для определения контекста и тренда
    ENTRY_TF: str = '5'  # 5m для точного определения точки входа

    # УВЕЛИЧЕННЫЕ количества свечей для полноты анализа
    CANDLES_15M: int = 200  # Больше свечей для контекстного анализа (50 часов)
    CANDLES_5M: int = 500   # Больше свечей для детального анализа (41 час)

    # ДЛЯ ИИ - МАКСИМАЛЬНЫЕ ДАННЫЕ
    CANDLES_FOR_AI_SELECTION: int = 50    # Увеличено для первичного отбора
    CANDLES_FOR_AI_ANALYSIS: int = 300    # МАКСИМУМ данных для детального анализа
    CANDLES_FOR_FULL_ANALYSIS: int = 200  # Полные данные всех индикаторов

    # Контекстные данные
    CANDLES_FOR_CONTEXT: int = 50     # Больше контекста 15m
    CANDLES_FOR_ENTRY: int = 100      # Больше данных для входа 5m


# ===========================
# ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ - КОРРЕКТИРОВАННЫЕ ДЛЯ ПРИБЫЛЬНОСТИ
# ===========================

@dataclass
class IndicatorConfig:
    """Настройки индикаторов - сбалансированы для скальпинга"""

    # EMA - чувствительные и трендовые периоды
    EMA_FAST: int = 5
    EMA_MEDIUM: int = 13
    EMA_SLOW: int = 34

    # RSI - настроен на малые TF
    RSI_PERIOD: int = 9
    RSI_OVERSOLD: float = 30.0
    RSI_OVERBOUGHT: float = 70.0
    RSI_EXTREME_OVERSOLD: float = 20.0
    RSI_EXTREME_OVERBOUGHT: float = 80.0

    # MACD - стандартные настройки
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9

    # ATR - более гибкие стопы (чуть уже, чтобы повысить профитность)
    ATR_PERIOD: int = 14
    ATR_MULTIPLIER_STOP: float = 1.4  # смягчено для появления сигналов
    ATR_MIN_RATIO: float = 0.8
    ATR_OPTIMAL_RATIO: float = 1.0

    # Bollinger Bands
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    BB_SQUEEZE_RATIO: float = 0.75  # Более строгий порог

    # Volume - смягчены требования для обнаружения импульсов
    VOLUME_SMA: int = 20
    VOLUME_SPIKE_RATIO: float = 1.3
    VOLUME_MIN_RATIO: float = 1.1

    # Дополнительные индикаторы для полного анализа
    STOCH_K_PERIOD: int = 9
    STOCH_D_PERIOD: int = 3
    STOCH_OVERSOLD: float = 20.0
    STOCH_OVERBOUGHT: float = 80.0

    # Williams %R
    WILLIAMS_PERIOD: int = 14

    # CCI
    CCI_PERIOD: int = 20
    CCI_OVERSOLD: float = -100.0
    CCI_OVERBOUGHT: float = 100.0


# ===========================
# ТОРГОВЫЕ ПАРАМЕТРЫ - СМЯГЧЕННЫЕ ДЛЯ ПРАКТИЧЕСКОГО ПОИСКА СИГНАЛОВ
# ===========================

@dataclass
class TradingConfig:
    """Параметры торговли - сбалансированы (для увеличения числа качественных сигналов)"""

    # Порог уверенности
    MIN_CONFIDENCE: int = 75  # соответствует валидации, даёт больше сигналов чем 80+
    HIGH_CONFIDENCE: int = 85

    # Более гибкая валидация (сохраняем структуру 5 проверок)
    VALIDATION_CHECKS_REQUIRED: int = 3
    VALIDATION_CHECKS_TOTAL: int = 5

    # Управление рисками (умеренно агрессивное, но контролируемое)
    MIN_STOP_LOSS_PERCENT: float = 0.5   # минимальный стоп (в % от цены)
    MAX_TAKE_PROFIT_PERCENT: float = 1.6
    DEFAULT_RISK_REWARD: float = 1.6
    MIN_RISK_REWARD: float = 1.3

    # Размер позиции (risk per trade 1-3%)
    DEFAULT_POSITION_SIZE_PERCENT: float = 2.0
    MAX_POSITION_SIZE_PERCENT: float = 4.0

    # Время удержания (короче для скальпинга)
    MAX_HOLD_TIME_MINUTES: int = 30
    OPTIMAL_HOLD_TIME_MINUTES: int = 20

    # Фильтры ликвидности (смягчены для большего покрытия пар)
    MIN_LIQUIDITY_VOLUME: int = 5_000_000
    OPTIMAL_LIQUIDITY_VOLUME: int = 20_000_000
    MAX_SPREAD_PERCENT: float = 0.20
    OPTIMAL_SPREAD_PERCENT: float = 0.07


# ===========================
# ШАБЛОНЫ ТОРГОВЫХ ПАТТЕРНОВ - АКТИВИРОВАНЫ ДЛЯ СКАЛЬПИНГА
# ===========================

@dataclass
class PatternConfig:
    """Настройки торговых паттернов - адаптированы для скальпинга"""

    # Приоритет паттернов (1 - высший)
    PATTERN_PRIORITY = {
        'MOMENTUM_BREAKOUT': 1,
        'PULLBACK_ENTRY': 2,
        'SQUEEZE_BREAKOUT': 3,
        'RANGE_SCALP': 4
    }

    # Базовая уверенность
    PATTERN_BASE_CONFIDENCE = {
        'PULLBACK_ENTRY': 78,
        'MOMENTUM_BREAKOUT': 85,
        'SQUEEZE_BREAKOUT': 75,
        'RANGE_SCALP': 65
    }

    # Более мягкие параметры для диапазонов
    RANGE_MIN_SIZE_PERCENT: float = 2.0
    RANGE_BOUNDARY_PROXIMITY: float = 0.08

    # Pullback точность
    PULLBACK_EMA_PROXIMITY: float = 0.004
    PULLBACK_RSI_RECOVERY: float = 48.0
    PULLBACK_RSI_WEAK: float = 52.0


# ===========================
# НАСТРОЙКИ ИИ - БЕЗ ИЗМЕНЕНИЙ
# ===========================

@dataclass
class AIConfig:
    """Настройки для работы с ИИ"""
    API_KEY_ENV: str = 'DEEPSEEK'
    API_BASE_URL: str = 'https://api.deepseek.com'
    API_MODEL: str = 'deepseek-chat'

    # Таймауты
    DEFAULT_TIMEOUT: int = 40
    SELECTION_TIMEOUT: int = 40
    ANALYSIS_TIMEOUT: int = 40
    HEALTH_CHECK_TIMEOUT: int = 15

    # Параметры запросов
    MAX_RETRIES: int = 2
    RETRY_DELAY: float = 1.0

    # Увеличенные токены для полного анализа
    MAX_TOKENS_SELECTION: int = 1500    # Увеличено с 1000
    MAX_TOKENS_ANALYSIS: int = 4000     # Увеличено с 3000
    MAX_TOKENS_TEST: int = 5

    # Параметры генерации
    TEMPERATURE_SELECTION: float = 0.2  # Уменьшено для большей точности
    TEMPERATURE_ANALYSIS: float = 0.5   # Уменьшено с 0.7
    TOP_P_SELECTION: float = 0.7        # Уменьшено с 0.8
    TOP_P_ANALYSIS: float = 0.8         # Уменьшено с 0.9
    FREQUENCY_PENALTY: float = 0.1
    PRESENCE_PENALTY_SELECTION: float = 0.1
    PRESENCE_PENALTY_ANALYSIS: float = 0.05

    # Промпты
    SELECTION_PROMPT_FILE: str = 'prompt2.txt'
    ANALYSIS_PROMPT_FILE: str = 'prompt.txt'

    # Лимиты - более селективно
    MAX_PAIRS_TO_AI: int = 5
    MAX_SELECTED_PAIRS: int = 3


# ===========================
# НАСТРОЙКИ API BYBIT - БЕЗ ИЗМЕНЕНИЙ
# ===========================

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


# ===========================
# НАСТРОЙКИ ОБРАБОТКИ ДАННЫХ
# ===========================

@dataclass
class ProcessingConfig:
    """Настройки обработки и батчинга"""
    BATCH_SIZE: int = 30              # Уменьшено с 40 до 30
    BATCH_DELAY: float = 0.2          # Увеличено с 0.1

    # Более консервативный параллелизм
    MAX_CONCURRENT_REQUESTS: int = 2  # Уменьшено с 3 до 2
    SEMAPHORE_LIMIT: int = 2          # Уменьшено с 3 до 2

    # Кэширование
    ENABLE_PROMPT_CACHE: bool = True
    ENABLE_HTTP_CLIENT_REUSE: bool = True


# ===========================
# НАСТРОЙКИ ОЦЕНКИ СИГНАЛОВ - СМЯГЧЕНЫ
# ===========================

@dataclass
class ScoringConfig:
    """Настройки системы оценки сигналов - более мягкие для тестов"""

    # Пересмотренные веса (баланс объёма и синхронизации)
    SCORING_WEIGHTS = {
        'volume_confirmation': 4,
        'multi_tf_sync': 4,
        'ema_alignment': 3,
        'pattern_quality': 3,
        'macd_signal': 2,
        'atr_optimal': 2
    }

    # Пониженный минимальный порог оценки для получения сигналов
    MIN_SCORE_THRESHOLD: float = 9.0  # снижено с 15.0

    # Модификаторы уверенности - слегка ослаблены
    CONFIDENCE_MODIFIERS = {
        'higher_tf_aligned': 1.12,
        'volume_spike': 1.06,
        'perfect_ema_alignment': 1.06,
        'validation_perfect': 1.12,
        'strong_trend_context': 1.08,
        'low_volatility_risk': 1.04
    }


# ===========================
# ДОПОЛНИТЕЛЬНЫЕ ФИЛЬТРЫ БЕЗОПАСНОСТИ
# ===========================

@dataclass
class SafetyConfig:
    """Дополнительные фильтры для снижения рисков"""

    # Фильтры времени (избегаем волатильные периоды)
    AVOID_HOURS_UTC = [0, 1, 2, 22, 23]  # Ночные часы UTC

    # Фильтры по парам (исключаем слишком волатильные)
    EXCLUDED_PAIRS = ['LUNAUSDT', 'USTCUSDT', 'TERRACLASSIC']
    HIGH_RISK_PAIRS = ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT']  # Мемкоины

    # Максимальная корреляция между выбранными парами
    MAX_CORRELATION: float = 0.7

    # Минимальный возраст пары на бирже (дни)
    MIN_PAIR_AGE_DAYS: int = 30

    # Максимальная просадка за последние 24 часа
    MAX_DAILY_DRAWDOWN: float = 10.0  # 10%


# ===========================
# ГЛАВНЫЙ КЛАСС КОНФИГУРАЦИИ
# ===========================

class Config:
    """Главный класс с полной конфигурацией системы"""

    def __init__(self):
        self.system = SystemConfig()
        self.timeframe = TimeframeConfig()
        self.indicators = IndicatorConfig()
        self.trading = TradingConfig()
        self.patterns = PatternConfig()
        self.ai = AIConfig()
        self.exchange = ExchangeConfig()
        self.processing = ProcessingConfig()
        self.scoring = ScoringConfig()
        self.safety = SafetyConfig()  # Новый раздел безопасности

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
            'safety': self.safety.__dict__
        }

    def validate(self) -> bool:
        """Расширенная валидация конфигурации"""
        validations = [
            # Проверка наличия API ключа
            os.getenv(self.ai.API_KEY_ENV) is not None,

            # Проверка соотношений
            self.trading.MIN_CONFIDENCE <= self.trading.HIGH_CONFIDENCE,
            self.trading.MIN_RISK_REWARD <= self.trading.DEFAULT_RISK_REWARD,

            # Проверка валидации (должно быть строже)
            self.trading.VALIDATION_CHECKS_REQUIRED >= 3,  # Минимум 3 из 5
            self.trading.VALIDATION_CHECKS_REQUIRED <= self.trading.VALIDATION_CHECKS_TOTAL,

            # Проверка таймфреймов
            int(self.timeframe.CONTEXT_TF) > int(self.timeframe.ENTRY_TF),

            # Проверка индикаторов
            self.indicators.EMA_FAST < self.indicators.EMA_MEDIUM < self.indicators.EMA_SLOW,
            self.indicators.MACD_FAST < self.indicators.MACD_SLOW,

            # Проверка RSI порогов (более консервативные)
            self.indicators.RSI_OVERSOLD >= 20,
            self.indicators.RSI_OVERBOUGHT <= 80,
            self.indicators.RSI_OVERSOLD < 50 < self.indicators.RSI_OVERBOUGHT,

            # Проверка консервативности настроек
            self.trading.MIN_CONFIDENCE >= 65,  # минимум 65%
            self.trading.MIN_STOP_LOSS_PERCENT >= 0.5,  # минимум 0.5%
            self.trading.DEFAULT_POSITION_SIZE_PERCENT <= 4.0,  # максимум 4%

            # Проверка таймаутов
            self.ai.SELECTION_TIMEOUT > 0,
            self.ai.ANALYSIS_TIMEOUT > 0
        ]

        return all(validations)

    @classmethod
    def load_from_env(cls) -> 'Config':
        """Загрузка конфигурации с учетом переменных окружения"""
        config = cls()

        # Переопределение из переменных окружения для живой торговли
        if os.getenv('MIN_CONFIDENCE'):
            config.trading.MIN_CONFIDENCE = max(65, int(os.getenv('MIN_CONFIDENCE')))

        if os.getenv('POSITION_SIZE'):
            config.trading.DEFAULT_POSITION_SIZE_PERCENT = min(4.0, float(os.getenv('POSITION_SIZE')))

        if os.getenv('RISK_REWARD'):
            config.trading.DEFAULT_RISK_REWARD = max(1.3, float(os.getenv('RISK_REWARD')))

        return config


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
    'SafetyConfig'
]
