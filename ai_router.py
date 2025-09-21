"""
Маршрутизатор для выбора AI провайдера на каждом этапе
"""

import logging
from typing import List, Dict
from config import config, get_available_ai_providers

logger = logging.getLogger(__name__)

# Импорты AI клиентов
try:
    from deepseek import (
        ai_select_pairs_deepseek,
        ai_analyze_pair_deepseek,
        ai_validate_signals_deepseek,
        create_fallback_selection,
        create_fallback_validation,
        create_fallback_analysis
    )
except ImportError as e:
    logger.error(f"Ошибка импорта DeepSeek: {e}")

try:
    from anthropic_ai import anthropic_client
except ImportError as e:
    logger.error(f"Ошибка импорта Anthropic: {e}")


class AIRouter:
    """Маршрутизатор AI провайдеров для разных этапов"""

    def __init__(self):
        self.providers = get_available_ai_providers()
        logger.info(f"Доступные AI провайдеры: {[k for k, v in self.providers.items() if v]}")

    def _get_provider_for_stage(self, stage: str) -> str:
        """Получить провайдера для конкретного этапа с fallback"""
        stage_mapping = {
            'selection': config.AI_STAGE_SELECTION,
            'analysis': config.AI_STAGE_ANALYSIS,
            'validation': config.AI_STAGE_VALIDATION
        }

        preferred_provider = stage_mapping.get(stage, 'fallback')

        # Проверяем доступность провайдера
        if preferred_provider != 'fallback' and self.providers.get(preferred_provider, False):
            return preferred_provider

        # Fallback логика - ищем первый доступный
        for provider in ['deepseek', 'anthropic']:
            if self.providers.get(provider, False):
                logger.warning(f"Этап {stage}: переключение с {preferred_provider} на {provider}")
                return provider

        logger.warning(f"Этап {stage}: используется fallback режим")
        return 'fallback'

    async def select_pairs(self, pairs_data: List[Dict]) -> List[str]:
        """Отбор пар через выбранный AI провайдер"""
        provider = self._get_provider_for_stage('selection')
        logger.info(f"Этап отбора: используется {provider}")

        try:
            if provider == 'deepseek':
                return await ai_select_pairs_deepseek(pairs_data)
            elif provider == 'anthropic':
                return await anthropic_client.select_pairs(pairs_data)
            else:  # fallback
                return create_fallback_selection(pairs_data, config.MAX_FINAL_PAIRS)

        except Exception as e:
            logger.error(f"Ошибка отбора пар через {provider}: {e}")
            logger.info("Переключение на fallback режим")
            return create_fallback_selection(pairs_data, config.MAX_FINAL_PAIRS)

    async def analyze_pair(self, symbol: str, data_5m: List, data_15m: List,
                           indicators_5m: Dict, indicators_15m: Dict) -> Dict:
        """Анализ пары через выбранный AI провайдер"""
        provider = self._get_provider_for_stage('analysis')
        logger.debug(f"Анализ {symbol}: используется {provider}")

        try:
            if provider == 'deepseek':
                return await ai_analyze_pair_deepseek(symbol, data_5m, data_15m, indicators_5m, indicators_15m)
            elif provider == 'anthropic':
                return await anthropic_client.analyze_pair(symbol, data_5m, data_15m, indicators_5m, indicators_15m)
            else:  # fallback
                current_price = indicators_5m.get('current', {}).get('price', 0)
                return create_fallback_analysis(symbol, indicators_5m)

        except Exception as e:
            logger.error(f"Ошибка анализа {symbol} через {provider}: {e}")
            logger.info("Переключение на fallback режим")
            current_price = indicators_5m.get('current', {}).get('price', 0)
            return create_fallback_analysis(symbol, indicators_5m)

    async def validate_signals(self, preliminary_signals: List[Dict], market_data: Dict) -> List[Dict]:
        """Валидация сигналов через выбранный AI провайдер"""
        provider = self._get_provider_for_stage('validation')
        logger.info(f"Этап валидации: используется {provider}")

        try:
            if provider == 'deepseek':
                return await ai_validate_signals_deepseek(preliminary_signals, market_data)
            elif provider == 'anthropic':
                return await anthropic_client.validate_signals(preliminary_signals, market_data)
            else:  # fallback
                return create_fallback_validation(preliminary_signals)

        except Exception as e:
            logger.error(f"Ошибка валидации через {provider}: {e}")
            logger.info("Переключение на fallback режим")
            return create_fallback_validation(preliminary_signals)

    def get_status(self) -> Dict:
        """Получить статус всех провайдеров"""
        return {
            'providers_available': self.providers,
            'stage_assignments': {
                'selection': config.AI_STAGE_SELECTION,
                'analysis': config.AI_STAGE_ANALYSIS,
                'validation': config.AI_STAGE_VALIDATION
            },
            'effective_providers': {
                'selection': self._get_provider_for_stage('selection'),
                'analysis': self._get_provider_for_stage('analysis'),
                'validation': self._get_provider_for_stage('validation')
            }
        }


# Создаем глобальный роутер
ai_router = AIRouter()