"""
AI Router - FIXED: Добавлены методы analyze_pair_comprehensive и validate_signal_with_stage3_data
Поддерживает DeepSeek и Anthropic Claude с reasoning/thinking режимами
"""

import os
from typing import List, Dict, Optional
from deepseek import DeepSeekClient
from config import (
    DEEPSEEK_API_KEY, ANTHROPIC_API_KEY,
    DEEPSEEK_MODEL, ANTHROPIC_MODEL,
    DEEPSEEK_REASONING, ANTHROPIC_THINKING,
    STAGE2_PROVIDER, STAGE3_PROVIDER, STAGE4_PROVIDER,
    AI_TEMPERATURE_SELECT, AI_TEMPERATURE_ANALYZE, AI_TEMPERATURE_VALIDATE,
    AI_MAX_TOKENS_SELECT, AI_MAX_TOKENS_ANALYZE, AI_MAX_TOKENS_VALIDATE
)


class AIRouter:
    """
    Маршрутизатор для работы с несколькими AI провайдерами
    Поддерживает multi-stage pipeline с разными моделями на каждом этапе
    """

    def __init__(self):
        """Инициализация роутера"""
        self.deepseek_client: Optional[DeepSeekClient] = None
        self.claude_client = None  # Будет инициализирован при необходимости

        # Конфигурация из env
        self.stage_providers = {
            'stage2': STAGE2_PROVIDER,
            'stage3': STAGE3_PROVIDER,
            'stage4': STAGE4_PROVIDER
        }

        print(f"\n[AI Router] {'='*70}")
        print(f"[AI Router] {'AI ROUTER ИНИЦИАЛИЗАЦИЯ':^70}")
        print(f"[AI Router] {'='*70}")
        print(f"[AI Router] 🎯 Multi-Stage Pipeline:")
        print(f"[AI Router]    • Stage 2 (выбор пар): {STAGE2_PROVIDER.upper()}")
        print(f"[AI Router]    • Stage 3 (анализ): {STAGE3_PROVIDER.upper()}")
        print(f"[AI Router]    • Stage 4 (валидация): {STAGE4_PROVIDER.upper()}")
        print(f"[AI Router] {'='*70}\n")

    async def initialize_deepseek(self):
        """Инициализирует DeepSeek клиент"""
        if not DEEPSEEK_API_KEY:
            print(f"[AI Router] ⚠️  DEEPSEEK_API_KEY не задан")
            return False

        try:
            self.deepseek_client = DeepSeekClient(
                api_key=DEEPSEEK_API_KEY,
                model=DEEPSEEK_MODEL,
                use_reasoning=DEEPSEEK_REASONING
            )
            print(f"[AI Router] ✅ DeepSeek клиент инициализирован")
            return True
        except Exception as e:
            print(f"[AI Router] ❌ Ошибка инициализации DeepSeek: {e}")
            return False

    async def initialize_claude(self):
        """Инициализирует Anthropic Claude клиент"""
        if not ANTHROPIC_API_KEY:
            print(f"[AI Router] ⚠️  ANTHROPIC_API_KEY не задан")
            return False

        try:
            # Импортируем Anthropic SDK
            from anthropic import AsyncAnthropic

            self.claude_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

            print(f"[AI Router] {'='*70}")
            print(f"[AI Router] {'ИНИЦИАЛИЗАЦИЯ CLAUDE':^70}")
            print(f"[AI Router] {'='*70}")
            print(f"[AI Router] ║ Модель: {ANTHROPIC_MODEL:<59} ║")
            print(f"[AI Router] ║ Extended Thinking: {'✅ Включен' if ANTHROPIC_THINKING else '❌ Выключен':<52} ║")
            print(f"[AI Router] {'='*70}")

            return True
        except ImportError:
            print(f"[AI Router] ❌ Anthropic SDK не установлен: pip install anthropic")
            return False
        except Exception as e:
            print(f"[AI Router] ❌ Ошибка инициализации Claude: {e}")
            return False

    async def _get_provider_client(self, stage: str):
        """
        Получает клиент для конкретного stage

        Args:
            stage: 'stage2', 'stage3' или 'stage4'

        Returns:
            Кортеж (provider_name, client)
        """
        provider = self.stage_providers.get(stage, 'deepseek')

        if provider == 'deepseek':
            if not self.deepseek_client:
                await self.initialize_deepseek()
            return 'deepseek', self.deepseek_client

        elif provider == 'claude':
            if not self.claude_client:
                await self.initialize_claude()
            return 'claude', self.claude_client

        else:
            print(f"[AI Router] ❌ Неизвестный провайдер: {provider}")
            return None, None

    async def select_pairs(
        self,
        pairs_data: List[Dict],
        max_pairs: Optional[int] = None
    ) -> List[str]:
        """
        Stage 2: Выбирает лучшие пары через AI

        Args:
            pairs_data: Данные о парах
            max_pairs: Максимальное количество пар

        Returns:
            Список выбранных тикеров
        """
        print(f"\n[AI Router] {'='*70}")
        print(f"[AI Router] 🎯 STAGE 2: ВЫБОР ПАР")
        print(f"[AI Router] {'='*70}")
        print(f"[AI Router] 📊 Пар на входе: {len(pairs_data)}")
        print(f"[AI Router] 🎚️  Лимит: {max_pairs if max_pairs else 'без ограничений'}")

        provider_name, client = await self._get_provider_client('stage2')

        if not client:
            print(f"[AI Router] ❌ Клиент недоступен для Stage 2")
            return []

        print(f"[AI Router] 🤖 Провайдер: {provider_name.upper()}")

        try:
            if provider_name == 'deepseek':
                selected = await self.deepseek_client.select_pairs(
                    pairs_data=pairs_data,
                    max_pairs=max_pairs,
                    temperature=AI_TEMPERATURE_SELECT,
                    max_tokens=AI_MAX_TOKENS_SELECT
                )

            elif provider_name == 'claude':
                selected = await self._claude_select_pairs(
                    pairs_data=pairs_data,
                    max_pairs=max_pairs
                )

            else:
                return []

            print(f"[AI Router] ✅ Stage 2 завершен: выбрано {len(selected)} пар")
            if selected:
                print(f"[AI Router] 📋 {', '.join(selected)}")
            print(f"[AI Router] {'='*70}\n")

            return selected

        except Exception as e:
            print(f"[AI Router] ❌ Ошибка в Stage 2: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def analyze_pair_comprehensive(
        self,
        symbol: str,
        comprehensive_data: Dict
    ) -> Dict:
        """
        Stage 3: Comprehensive analysis with all data

        Args:
            symbol: Trading pair symbol
            comprehensive_data: Full data from Stage 3 (all timeframes, market data, etc)

        Returns:
            Analysis result with signal/confidence/levels
        """
        print(f"\n[AI Router] {'─'*70}")
        print(f"[AI Router] 🔬 STAGE 3: COMPREHENSIVE ANALYSIS {symbol}")
        print(f"[AI Router] {'─'*70}")

        provider_name, client = await self._get_provider_client('stage3')

        if not client:
            print(f"[AI Router] ❌ Client unavailable for Stage 3")
            return {
                'symbol': symbol,
                'signal': 'NO_SIGNAL',
                'confidence': 0,
                'rejection_reason': 'AI client unavailable'
            }

        print(f"[AI Router] 🤖 Provider: {provider_name.upper()}")

        try:
            if provider_name == 'deepseek':
                # DeepSeek не поддерживает полноценный Stage 3 анализ
                # Возвращаем базовый анализ
                return {
                    'symbol': symbol,
                    'signal': 'NO_SIGNAL',
                    'confidence': 0,
                    'rejection_reason': 'DeepSeek не поддерживает comprehensive analysis. Используйте Claude для Stage 3'
                }

            elif provider_name == 'claude':
                # Claude полноценный анализ
                from anthropic_client import AnthropicClient

                claude = AnthropicClient()

                result = await claude.analyze_comprehensive(symbol, comprehensive_data)

                if result:
                    print(f"[AI Router] ✅ Stage 3 complete for {symbol}")
                    return result
                else:
                    return {
                        'symbol': symbol,
                        'signal': 'NO_SIGNAL',
                        'confidence': 0,
                        'rejection_reason': 'Claude analysis returned no result'
                    }

            else:
                return {
                    'symbol': symbol,
                    'signal': 'NO_SIGNAL',
                    'confidence': 0,
                    'rejection_reason': f'Unknown provider: {provider_name}'
                }

        except Exception as e:
            print(f"[AI Router] ❌ Error in Stage 3 for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'symbol': symbol,
                'signal': 'NO_SIGNAL',
                'confidence': 0,
                'rejection_reason': f'Exception: {str(e)[:100]}'
            }

    async def validate_signal_with_stage3_data(
        self,
        signal: Dict,
        comprehensive_data: Dict
    ) -> Dict:
        """
        Stage 4: Validation with full Stage 3 data

        Args:
            signal: Signal from Stage 3
            comprehensive_data: Full data from Stage 3

        Returns:
            Validation result
        """
        provider_name, client = await self._get_provider_client('stage4')

        if not client:
            print(f"[AI Router] ❌ Client unavailable for Stage 4")
            # Fallback validation
            from shared_utils import fallback_validation
            return fallback_validation(signal, comprehensive_data)

        try:
            if provider_name == 'claude':
                from anthropic_client import AnthropicClient
                claude = AnthropicClient()
                return await claude.validate_signal(signal, comprehensive_data)

            elif provider_name == 'deepseek':
                # DeepSeek fallback
                from shared_utils import fallback_validation
                return fallback_validation(signal, comprehensive_data)

            else:
                from shared_utils import fallback_validation
                return fallback_validation(signal, comprehensive_data)

        except Exception as e:
            print(f"[AI Router] ❌ Error in Stage 4: {e}")
            from shared_utils import fallback_validation
            return fallback_validation(signal, comprehensive_data)

    # ========================================================================
    # CLAUDE METHODS
    # ========================================================================

    async def _claude_select_pairs(
        self,
        pairs_data: List[Dict],
        max_pairs: Optional[int] = None
    ) -> List[str]:
        """Выбор пар через Claude"""
        # Формируем промпт
        pairs_info = []
        for pair in pairs_data:
            info = (
                f"Пара: {pair['ticker']}\n"
                f"Сигнал: {pair['signal_type']} ({pair['signal_strength']}%)\n"
                f"Цена: ${pair['price']:.8f}\n"
                f"Объем 24ч: ${pair.get('volume_24h', 0):,.0f}\n"
            )
            pairs_info.append(info)

        pairs_text = "\n---\n".join(pairs_info)
        limit_text = f"максимум {max_pairs} пар" if max_pairs else "без ограничения"

        prompt = (
            f"Проанализируй {len(pairs_data)} торговых пар и выбери {limit_text} "
            f"с лучшим потенциалом:\n\n{pairs_text}\n\n"
            f"Верни ТОЛЬКО список тикеров через запятую."
        )

        # Вызов Claude API
        kwargs = {
            'model': ANTHROPIC_MODEL,
            'max_tokens': AI_MAX_TOKENS_SELECT,
            'temperature': AI_TEMPERATURE_SELECT,
            'messages': [{'role': 'user', 'content': prompt}]
        }

        # Extended thinking для поддерживающих моделей
        if ANTHROPIC_THINKING:
            kwargs['thinking'] = {'type': 'enabled', 'budget_tokens': 2000}

        response = await self.claude_client.messages.create(**kwargs)

        # Показываем thinking если есть
        if ANTHROPIC_THINKING and hasattr(response, 'thinking'):
            print(f"[Claude] 💭 Extended Thinking (первые 500 символов):")
            print(f"     {str(response.thinking)[:500]}...")

        content = response.content[0].text.strip()

        # Парсим тикеры
        selected = []
        for line in content.split('\n'):
            tokens = line.replace(',', ' ').split()
            for token in tokens:
                token = token.strip().upper()
                if 2 <= len(token) <= 10:
                    clean = token.replace('USDT', '').replace('USD', '')
                    if clean and clean.isalnum() and clean not in selected:
                        selected.append(clean)

        if max_pairs and len(selected) > max_pairs:
            selected = selected[:max_pairs]

        return selected

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_config(self) -> Dict:
        """Возвращает текущую конфигурацию"""
        return {
            'stage_providers': self.stage_providers,
            'deepseek_model': DEEPSEEK_MODEL,
            'deepseek_reasoning': DEEPSEEK_REASONING,
            'claude_model': ANTHROPIC_MODEL,
            'claude_thinking': ANTHROPIC_THINKING,
            'deepseek_initialized': self.deepseek_client is not None,
            'claude_initialized': self.claude_client is not None
        }