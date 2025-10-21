"""
AI Router - умный маршрутизатор для multi-stage AI pipeline
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

    async def analyze_pair(
        self,
        pair_data: Dict
    ) -> Optional[Dict]:
        """
        Stage 3: Анализирует конкретную пару через AI

        Args:
            pair_data: Данные о паре

        Returns:
            Результаты анализа
        """
        ticker = pair_data.get('ticker', 'unknown')

        print(f"\n[AI Router] {'─'*70}")
        print(f"[AI Router] 🔬 STAGE 3: АНАЛИЗ ПАРЫ {ticker}")
        print(f"[AI Router] {'─'*70}")

        provider_name, client = await self._get_provider_client('stage3')

        if not client:
            print(f"[AI Router] ❌ Клиент недоступен для Stage 3")
            return None

        print(f"[AI Router] 🤖 Провайдер: {provider_name.upper()}")

        try:
            if provider_name == 'deepseek':
                result = await self.deepseek_client.analyze_pair(
                    pair_data=pair_data,
                    temperature=AI_TEMPERATURE_ANALYZE,
                    max_tokens=AI_MAX_TOKENS_ANALYZE
                )

            elif provider_name == 'claude':
                result = await self._claude_analyze_pair(pair_data)

            else:
                return None

            if result:
                print(f"[AI Router] ✅ Stage 3 завершен для {ticker}")

            return result

        except Exception as e:
            print(f"[AI Router] ❌ Ошибка в Stage 3: {e}")
            return None

    async def validate_signal(
        self,
        signal_data: Dict
    ) -> Optional[Dict]:
        """
        Stage 4: Финальная валидация сигнала перед отправкой

        Args:
            signal_data: Данные сигнала для валидации

        Returns:
            Результат валидации с рекомендацией
        """
        ticker = signal_data.get('ticker', 'unknown')

        print(f"\n[AI Router] {'─'*70}")
        print(f"[AI Router] ✓ STAGE 4: ВАЛИДАЦИЯ СИГНАЛА {ticker}")
        print(f"[AI Router] {'─'*70}")

        provider_name, client = await self._get_provider_client('stage4')

        if not client:
            print(f"[AI Router] ❌ Клиент недоступен для Stage 4")
            return None

        print(f"[AI Router] 🤖 Провайдер: {provider_name.upper()}")

        try:
            if provider_name == 'deepseek':
                result = await self._deepseek_validate_signal(signal_data)

            elif provider_name == 'claude':
                result = await self._claude_validate_signal(signal_data)

            else:
                return None

            if result:
                approved = result.get('approved', False)
                status = "✅ ОДОБРЕН" if approved else "❌ ОТКЛОНЕН"
                print(f"[AI Router] {status} Stage 4 для {ticker}")

            return result

        except Exception as e:
            print(f"[AI Router] ❌ Ошибка в Stage 4: {e}")
            return None

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

    async def _claude_analyze_pair(self, pair_data: Dict) -> Optional[Dict]:
        """Анализ пары через Claude"""
        pair_info = (
            f"Тикер: {pair_data['ticker']}\n"
            f"Цена: ${pair_data['price']:.8f}\n"
            f"Объем 24ч: ${pair_data.get('volume_24h', 0):,.0f}\n"
        )

        if pair_data.get('technical_data'):
            pair_info += f"\n{pair_data['technical_data']}"

        prompt = f"Проанализируй торговую пару:\n\n{pair_info}"

        kwargs = {
            'model': ANTHROPIC_MODEL,
            'max_tokens': AI_MAX_TOKENS_ANALYZE,
            'temperature': AI_TEMPERATURE_ANALYZE,
            'messages': [{'role': 'user', 'content': prompt}]
        }

        if ANTHROPIC_THINKING:
            kwargs['thinking'] = {'type': 'enabled', 'budget_tokens': 3000}

        response = await self.claude_client.messages.create(**kwargs)

        if ANTHROPIC_THINKING and hasattr(response, 'thinking'):
            print(f"[Claude] 💭 Thinking: {str(response.thinking)[:300]}...")

        return {
            'ticker': pair_data['ticker'],
            'analysis': response.content[0].text.strip(),
            'model': ANTHROPIC_MODEL,
            'thinking_used': ANTHROPIC_THINKING
        }

    async def _claude_validate_signal(self, signal_data: Dict) -> Optional[Dict]:
        """Валидация через Claude"""
        prompt = f"Валидируй торговый сигнал:\n\n{signal_data}\n\nОдобрить? (да/нет)"

        kwargs = {
            'model': ANTHROPIC_MODEL,
            'max_tokens': AI_MAX_TOKENS_VALIDATE,
            'temperature': AI_TEMPERATURE_VALIDATE,
            'messages': [{'role': 'user', 'content': prompt}]
        }

        if ANTHROPIC_THINKING:
            kwargs['thinking'] = {'type': 'enabled', 'budget_tokens': 2000}

        response = await self.claude_client.messages.create(**kwargs)

        content = response.content[0].text.strip().lower()
        approved = 'да' in content or 'yes' in content or 'одобр' in content

        return {
            'approved': approved,
            'reasoning': content,
            'model': ANTHROPIC_MODEL
        }

    async def _deepseek_validate_signal(self, signal_data: Dict) -> Optional[Dict]:
        """Валидация через DeepSeek"""
        prompt = f"Валидируй торговый сигнал:\n\n{signal_data}\n\nОдобрить?"

        response = await self.deepseek_client.chat(
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=AI_MAX_TOKENS_VALIDATE,
            temperature=AI_TEMPERATURE_VALIDATE
        )

        approved = 'да' in response.lower() or 'yes' in response.lower()

        return {
            'approved': approved,
            'reasoning': response,
            'model': DEEPSEEK_MODEL
        }

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