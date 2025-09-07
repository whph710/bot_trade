"""
Упрощенная конфигурация для высокоприбыльного скальпинг-бота
Основана на проверенных торговых стратегиях
"""

import os
from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()


@dataclass
class TradingConfig:
    """Основные торговые параметры - проверенные прибыльные настройки"""
    # Таймфреймы (классическая комбинация для скальпинга)
    HIGHER_TF: str = '15'  # Контекст тренда
    ENTRY_TF: str = '5'    # Точка входа

    # Количество свечей (оптимизировано для скорости)
    CANDLES_HIGHER: int = 50   # Достаточно для тренда
    CANDLES_ENTRY: int = 100   # Достаточно для паттернов

    # Индикаторы (классический profitable стек)
    EMA_FAST: int = 8      # Быстрая EMA
    EMA_SLOW: int = 21     # Медленная EMA
    RSI_PERIOD: int = 14   # Стандартный RSI
    ATR_PERIOD: int = 14   # Для волатильности
    VOLUME_PERIOD: int = 20 # Для объемного анализа

    # Пороги (проверенные значения)
    MIN_CONFIDENCE: int = 70    # Минимальная уверенность
    MIN_VOLUME_RATIO: float = 1.2  # Минимальный объем
    MIN_ATR_RATIO: float = 0.8     # Минимальная волатильность

    # Risk Management (классические значения)
    MAX_RISK_PERCENT: float = 1.0   # Максимальный риск на сделку
    MIN_RR_RATIO: float = 1.5       # Минимальный Risk/Reward
    MAX_RR_RATIO: float = 3.0       # Максимальный Risk/Reward


@dataclass
class AIConfig:
    """ИИ настройки"""
    API_KEY_ENV: str = 'DEEPSEEK'
    BASE_URL: str = 'https://api.deepseek.com'
    MODEL: str = 'deepseek-chat'

    # Таймауты (оптимизированы для скальпинга)
    SELECTION_TIMEOUT: int = 30
    ANALYSIS_TIMEOUT: int = 60

    # Токены (баланс качества и скорости)
    SELECTION_TOKENS: int = 2000
    ANALYSIS_TOKENS: int = 4000

    # Температура (для стабильности)
    TEMPERATURE: float = 0.3

    # Количество пар
    MAX_PAIRS_TO_AI: int = 15
    MAX_SELECTED_PAIRS: int = 5


@dataclass
class ExchangeConfig:
    """Настройки биржи"""
    BASE_URL: str = "https://api.bybit.com"
    KLINE_ENDPOINT: str = "/v5/market/kline"
    INSTRUMENTS_ENDPOINT: str = "/v5/market/instruments-info"
    CATEGORY: str = "linear"
    QUOTE_CURRENCY: str = "USDT"
    TIMEOUT: int = 20


@dataclass
class SystemConfig:
    """Системные настройки"""
    LOG_LEVEL: str = 'INFO'
    LOG_FILE: str = 'trading_bot.log'
    ANALYSIS_FILE: str = 'analysis_results.log'
    ENCODING: str = 'utf-8'

    # Промпты
    SELECTION_PROMPT: str = 'selection_prompt.txt'
    ANALYSIS_PROMPT: str = 'analysis_prompt.txt'


class Config:
    """Главный класс конфигурации"""

    def __init__(self):
        self.trading = TradingConfig()
        self.ai = AIConfig()
        self.exchange = ExchangeConfig()
        self.system = SystemConfig()

    def validate(self) -> bool:
        """Валидация конфигурации с детальной диагностикой"""
        import logging
        logger = logging.getLogger(__name__)

        api_key = os.getenv(self.ai.API_KEY_ENV)

        validations = {
            'API key exists': api_key is not None,
            'API key not empty': api_key is not None and len(api_key.strip()) > 0,
            'Min confidence > 0': self.trading.MIN_CONFIDENCE > 0,
            'Min RR ratio > 1.0': self.trading.MIN_RR_RATIO > 1.0,
            'EMA fast < slow': self.trading.EMA_FAST < self.trading.EMA_SLOW,
            'Selection timeout > 0': self.ai.SELECTION_TIMEOUT > 0,
            'Analysis timeout > 0': self.ai.ANALYSIS_TIMEOUT > 0
        }

        # Диагностика проблем
        failed_validations = [name for name, result in validations.items() if not result]

        if failed_validations:
            logger.error("❌ Configuration validation failed:")
            for failed in failed_validations:
                logger.error(f"   - {failed}")

            if not api_key:
                logger.error(f"   💡 Create .env file with: {self.ai.API_KEY_ENV}=your_api_key")
            elif len(api_key.strip()) == 0:
                logger.error(f"   💡 API key is empty in .env file")

            return False

        logger.info("✅ Configuration validation passed")
        return True


# Глобальный экземпляр
config = Config()