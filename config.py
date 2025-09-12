"""
Централизованная конфигурация для скальпингового торгового бота
Мультитаймфреймный анализ по инструкции: 15m контекст + 5m точный вход
"""

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
    """Настройки таймфреймов согласно инструкции"""
    # Основные таймфреймы для анализа
    CONTEXT_TF: str = '15'  # 15m для определения контекста и тренда
    ENTRY_TF: str = '5'  # 5m для точного определения точки входа

    # Количество свечей для быстрого сканирования
    CANDLES_15M_QUICK: int = 100  # Свечи 15m для быстрого сканирования
    CANDLES_5M_QUICK: int = 200  # Свечи 5m для быстрого сканирования

    # Количество свечей для ИИ отбора
    CANDLES_FOR_AI_SELECTION: int = 30  # Для первичного отбора ИИ
    CANDLES_FOR_CONTEXT: int = 20  # Последние свечи 15m для контекста
    CANDLES_FOR_ENTRY: int = 30  # Последние свечи 5m для входа

    # НОВЫЕ: Количество свечей для ДЕТАЛЬНОГО АНАЛИЗА
    DETAILED_CANDLES_15M: int = 200  # 200 свечей 15m = ~50 часов истории
    DETAILED_CANDLES_5M: int = 500   # 500 свечей 5m = ~42 часа истории

    # Количество данных индикаторов для передачи в ИИ
    INDICATORS_HISTORY_POINTS: int = 100  # Последние 100 значений каждого индикатора


# ===========================
# ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
# ===========================

@dataclass
class IndicatorConfig:
    """Настройки технических индикаторов согласно инструкции"""

    # EMA (Exponential Moving Average) - для тренда и сжатия
    EMA_FAST: int = 5  # Быстрая EMA
    EMA_MEDIUM: int = 8  # Средняя EMA
    EMA_SLOW: int = 20  # Медленная EMA

    # RSI (Relative Strength Index) - фильтр импульса
    RSI_PERIOD: int = 9
    RSI_OVERSOLD: float = 30.0
    RSI_OVERBOUGHT: float = 70.0
    RSI_EXTREME_OVERSOLD: float = 20.0
    RSI_EXTREME_OVERBOUGHT: float = 80.0

    # MACD (Moving Average Convergence Divergence) - подтверждение направления
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9

    # ATR (Average True Range) - контроль волатильности
    ATR_PERIOD: int = 14
    ATR_MULTIPLIER_STOP: float = 1.5  # Множитель для стоп-лосса
    ATR_MIN_RATIO: float = 0.7  # Минимальное соотношение к среднему
    ATR_OPTIMAL_RATIO: float = 0.9  # Оптимальное соотношение

    # Bollinger Bands - для squeeze и breakout паттернов
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    BB_SQUEEZE_RATIO: float = 0.8  # Порог сжатия полос

    # Volume анализ - подтверждение пробоев
    VOLUME_SMA: int = 20
    VOLUME_SPIKE_RATIO: float = 1.5  # Порог всплеска объема
    VOLUME_MIN_RATIO: float = 1.0  # Минимальное соотношение


# ===========================
# ТОРГОВЫЕ ПАРАМЕТРЫ
# ===========================

@dataclass
class TradingConfig:
    """Параметры торговли и управления рисками"""

    # Пороги уверенности для сигналов
    MIN_CONFIDENCE: int = 70  # Минимальная уверенность для входа
    HIGH_CONFIDENCE: int = 85  # Высокая уверенность

    # Валидация сигналов (требуется минимум из общего количества)
    VALIDATION_CHECKS_REQUIRED: int = 3  # Минимум проверок для валидации
    VALIDATION_CHECKS_TOTAL: int = 5  # Общее количество проверок

    # Управление рисками
    MIN_STOP_LOSS_PERCENT: float = 0.4  # Минимальный стоп-лосс в %
    MAX_TAKE_PROFIT_PERCENT: float = 0.8  # Максимальный тейк-профит в %
    DEFAULT_RISK_REWARD: float = 1.5  # Соотношение риск/прибыль
    MIN_RISK_REWARD: float = 1.2  # Минимальное соотношение

    # Размер позиции
    DEFAULT_POSITION_SIZE_PERCENT: float = 2.5  # Размер позиции от депозита
    MAX_POSITION_SIZE_PERCENT: float = 5.0  # Максимальный размер

    # Время удержания позиции
    MAX_HOLD_TIME_MINUTES: int = 45  # Максимальное время в минутах
    OPTIMAL_HOLD_TIME_MINUTES: int = 30  # Оптимальное время

    # Фильтры ликвидности
    MIN_LIQUIDITY_VOLUME: int = 10_000_000  # Минимальный объем $10M
    OPTIMAL_LIQUIDITY_VOLUME: int = 50_000_000  # Оптимальный объем $50M
    MAX_SPREAD_PERCENT: float = 0.15  # Максимальный спред
    OPTIMAL_SPREAD_PERCENT: float = 0.1  # Оптимальный спред


# ===========================
# ШАБЛОНЫ ТОРГОВЫХ ПАТТЕРНОВ
# ===========================

@dataclass
class PatternConfig:
    """Настройки торговых паттернов согласно инструкции"""

    # Приоритет паттернов (1 - высший)
    PATTERN_PRIORITY = {
        'MOMENTUM_BREAKOUT': 1,
        'SQUEEZE_BREAKOUT': 2,
        'PULLBACK_ENTRY': 3,
        'RANGE_SCALP': 4
    }

    # Базовая уверенность для каждого паттерна
    PATTERN_BASE_CONFIDENCE = {
        'MOMENTUM_BREAKOUT': 85,
        'SQUEEZE_BREAKOUT': 80,
        'PULLBACK_ENTRY': 75,
        'RANGE_SCALP': 70
    }

    # Параметры для Range Scalp
    RANGE_MIN_SIZE_PERCENT: float = 2.0  # Минимальный размер диапазона
    RANGE_BOUNDARY_PROXIMITY: float = 0.1  # Близость к границе диапазона

    # Параметры для Pullback
    PULLBACK_EMA_PROXIMITY: float = 0.005  # Близость к EMA (0.5%)
    PULLBACK_RSI_RECOVERY: float = 45.0  # RSI восстановление для лонга
    PULLBACK_RSI_WEAK: float = 55.0  # RSI слабость для шорта


# ===========================
# НАСТРОЙКИ ИИ (DeepSeek)
# ===========================

@dataclass
class AIConfig:
    """Настройки для работы с ИИ"""

    # API настройки
    API_KEY_ENV: str = 'DEEPSEEK'
    API_BASE_URL: str = 'https://api.deepseek.com'
    API_MODEL: str = 'deepseek-chat'

    # Таймауты (в секундах)
    DEFAULT_TIMEOUT: int = 40
    SELECTION_TIMEOUT: int = 40  # Для быстрого отбора
    ANALYSIS_TIMEOUT: int = 60   # УВЕЛИЧЕН для детального анализа
    HEALTH_CHECK_TIMEOUT: int = 15

    # Параметры запросов
    MAX_RETRIES: int = 2
    RETRY_DELAY: float = 1.0  # Базовая задержка между попытками

    # Токены
    MAX_TOKENS_SELECTION: int = 1000  # Для отбора пар
    MAX_TOKENS_ANALYSIS: int = 4000   # УВЕЛИЧЕН для детального анализа
    MAX_TOKENS_TEST: int = 5  # Для проверки подключения

    # Параметры генерации для скальпинга
    TEMPERATURE_SELECTION: float = 0.3  # Низкая для точности отбора
    TEMPERATURE_ANALYSIS: float = 0.7  # Средняя для анализа
    TOP_P_SELECTION: float = 0.8
    TOP_P_ANALYSIS: float = 0.9
    FREQUENCY_PENALTY: float = 0.1
    PRESENCE_PENALTY_SELECTION: float = 0.1
    PRESENCE_PENALTY_ANALYSIS: float = 0.05

    # Промпты
    SELECTION_PROMPT_FILE: str = 'prompt2.txt'
    ANALYSIS_PROMPT_FILE: str = 'prompt.txt'

    # Лимиты
    MAX_PAIRS_TO_AI: int = 8  # Максимум пар для анализа ИИ
    MAX_SELECTED_PAIRS: int = 5  # Максимум выбранных пар


# ===========================
# НАСТРОЙКИ API BYBIT
# ===========================

@dataclass
class ExchangeConfig:
    """Настройки для работы с биржей Bybit"""

    # API endpoints
    KLINE_URL: str = "https://api.bybit.com/v5/market/kline"
    INSTRUMENTS_URL: str = "https://api.bybit.com/v5/market/instruments-info"

    # Параметры запросов
    API_TIMEOUT: int = 20  # Таймаут для API запросов
    API_CATEGORY: str = "linear"  # Категория инструментов

    # Фильтры инструментов
    QUOTE_CURRENCY: str = 'USDT'
    INSTRUMENT_STATUS: str = 'Trading'

    # HTTP настройки
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

    # Батчинг
    BATCH_SIZE: int = 40  # Размер батча для параллельной обработки
    BATCH_DELAY: float = 0.1  # Задержка между батчами

    # Параллелизм
    MAX_CONCURRENT_REQUESTS: int = 3  # Максимум параллельных запросов к ИИ
    SEMAPHORE_LIMIT: int = 3  # Лимит семафора

    # Кэширование
    ENABLE_PROMPT_CACHE: bool = True  # Кэширование промптов
    ENABLE_HTTP_CLIENT_REUSE: bool = True  # Переиспользование HTTP клиента


# ===========================
# НАСТРОЙКИ ОЦЕНКИ СИГНАЛОВ
# ===========================

@dataclass
class ScoringConfig:
    """Настройки системы оценки сигналов"""

    # Веса для оценки (согласно prompt2.txt)
    SCORING_WEIGHTS = {
        'volume_confirmation': 4,
        'ema_alignment': 3,
        'pattern_quality': 3,
        'macd_signal': 2,
        'multi_tf_sync': 2,
        'atr_optimal': 1
    }

    # Минимальный порог оценки
    MIN_SCORE_THRESHOLD: float = 12.0

    # Модификаторы уверенности
    CONFIDENCE_MODIFIERS = {
        'higher_tf_aligned': 1.1,  # +10% если старший TF совпадает
        'volume_spike': 1.05,  # +5% при всплеске объема
        'perfect_ema_alignment': 1.05,  # +5% при идеальном выравнивании EMA
        'validation_perfect': 1.1  # +10% при 5/5 валидации
    }


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
            'scoring': self.scoring.__dict__
        }

    def validate(self) -> bool:
        """Валидация конфигурации"""
        validations = [
            # Проверка наличия API ключа
            os.getenv(self.ai.API_KEY_ENV) is not None,

            # Проверка соотношений
            self.trading.MIN_CONFIDENCE <= self.trading.HIGH_CONFIDENCE,
            self.trading.MIN_RISK_REWARD <= self.trading.DEFAULT_RISK_REWARD,

            # Проверка валидации
            self.trading.VALIDATION_CHECKS_REQUIRED <= self.trading.VALIDATION_CHECKS_TOTAL,

            # Проверка таймфреймов
            int(self.timeframe.CONTEXT_TF) > int(self.timeframe.ENTRY_TF),

            # Проверка индикаторов
            self.indicators.EMA_FAST < self.indicators.EMA_MEDIUM < self.indicators.EMA_SLOW,
            self.indicators.MACD_FAST < self.indicators.MACD_SLOW,

            # Проверка порогов
            self.indicators.RSI_OVERSOLD < 50 < self.indicators.RSI_OVERBOUGHT,

            # Проверка таймаутов
            self.ai.SELECTION_TIMEOUT > 0,
            self.ai.ANALYSIS_TIMEOUT > 0
        ]

        return all(validations)

    @classmethod
    def load_from_env(cls) -> 'Config':
        """Загрузка конфигурации с учетом переменных окружения"""
        config = cls()

        # Здесь можно добавить загрузку из переменных окружения
        # Например:
        # if os.getenv('MIN_CONFIDENCE'):
        #     config.trading.MIN_CONFIDENCE = int(os.getenv('MIN_CONFIDENCE'))

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
    'ScoringConfig'
]