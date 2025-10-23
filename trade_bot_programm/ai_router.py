"""
AI Router - FIXED: Stage 2 compact data + 65s delay + Stage 4 removed
Файл: trade_bot_programm/ai_router.py
"""

from typing import List, Dict, Optional
from deepseek import DeepSeekClient
from config import (
    DEEPSEEK_API_KEY, ANTHROPIC_API_KEY,
    STAGE2_PROVIDER, STAGE3_PROVIDER,
    STAGE2_MODEL, STAGE2_TEMPERATURE, STAGE2_MAX_TOKENS,
    STAGE3_MODEL, STAGE3_TEMPERATURE, STAGE3_MAX_TOKENS,
    DEEPSEEK_REASONING, ANTHROPIC_THINKING
)


class AIRouter:
    """AI Router с поддержкой stage-specific моделей"""

    def __init__(self):
        self.deepseek_clients: Dict[str, DeepSeekClient] = {}
        self.claude_client = None

        self.stage_providers = {
            'stage2': STAGE2_PROVIDER,
            'stage3': STAGE3_PROVIDER
        }

        print(f"\n[AI Router] {'='*70}")
        print(f"[AI Router] {'AI ROUTER ИНИЦИАЛИЗАЦИЯ':^70}")
        print(f"[AI Router] {'='*70}")
        print(f"[AI Router] 🎯 Multi-Stage Pipeline:")
        print(f"[AI Router]    • Stage 2: {STAGE2_PROVIDER.upper()} ({STAGE2_MODEL})")
        print(f"[AI Router]    • Stage 3: {STAGE3_PROVIDER.upper()} ({STAGE3_MODEL})")
        print(f"[AI Router]    • Stage 4: REMOVED (fallback only)")
        print(f"[AI Router] {'='*70}\n")

    async def _get_deepseek_client(self, stage: str) -> Optional[DeepSeekClient]:
        """Получить DeepSeek клиент для конкретного stage"""
        if stage in self.deepseek_clients:
            return self.deepseek_clients[stage]

        if not DEEPSEEK_API_KEY:
            print(f"[AI Router] ⚠️  DEEPSEEK_API_KEY не задан")
            return None

        if stage == 'stage2':
            model = STAGE2_MODEL
        elif stage == 'stage3':
            model = STAGE3_MODEL
        else:
            model = "deepseek-chat"

        try:
            client = DeepSeekClient(
                api_key=DEEPSEEK_API_KEY,
                model=model,
                use_reasoning=DEEPSEEK_REASONING
            )
            self.deepseek_clients[stage] = client
            return client
        except Exception as e:
            print(f"[AI Router] ❌ Ошибка инициализации DeepSeek для {stage}: {e}")
            return None

    async def initialize_claude(self):
        """Инициализирует Claude клиент"""
        if not ANTHROPIC_API_KEY:
            print(f"[AI Router] ⚠️  ANTHROPIC_API_KEY не задан")
            return False

        try:
            from anthropic import AsyncAnthropic

            self.claude_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

            print(f"[AI Router] {'='*70}")
            print(f"[AI Router] {'ИНИЦИАЛИЗАЦИЯ CLAUDE':^70}")
            print(f"[AI Router] {'='*70}")
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
        """Получить клиент для конкретного stage"""
        provider = self.stage_providers.get(stage, 'deepseek')

        if provider == 'deepseek':
            client = await self._get_deepseek_client(stage)
            return 'deepseek', client

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
        """Stage 2: Выбор пар (Haiku с compact multi-TF data)"""
        print(f"\n[AI Router] {'='*70}")
        print(f"[AI Router] 🎯 STAGE 2: ВЫБОР ПАР (COMPACT MULTI-TF)")
        print(f"[AI Router] {'='*70}")
        print(f"[AI Router] 📊 Пар на входе: {len(pairs_data)}")
        print(f"[AI Router] 🎚️  Лимит: {max_pairs if max_pairs else 'без ограничений'}")

        provider_name, client = await self._get_provider_client('stage2')

        if not client:
            print(f"[AI Router] ❌ Клиент недоступен для Stage 2")
            return []

        print(f"[AI Router] 🤖 Провайдер: {provider_name.upper()}")
        print(f"[AI Router] 📦 Модель: {STAGE2_MODEL}")
        print(f"[AI Router] 🌡️  Temperature: {STAGE2_TEMPERATURE}")
        print(f"[AI Router] 🎫 Max tokens: {STAGE2_MAX_TOKENS}")

        try:
            if provider_name == 'deepseek':
                selected = await client.select_pairs(
                    pairs_data=pairs_data,
                    max_pairs=max_pairs,
                    temperature=STAGE2_TEMPERATURE,
                    max_tokens=STAGE2_MAX_TOKENS
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
                print(f"[AI Router] 📋 {selected}")
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
        """Stage 3: Comprehensive analysis (Sonnet)"""
        print(f"\n[AI Router] {'─'*70}")
        print(f"[AI Router] 🔬 STAGE 3: ANALYSIS {symbol}")
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
        print(f"[AI Router] 📦 Model: {STAGE3_MODEL}")
        print(f"[AI Router] 🌡️  Temperature: {STAGE3_TEMPERATURE}")
        print(f"[AI Router] 🎫 Max tokens: {STAGE3_MAX_TOKENS}")

        try:
            if provider_name == 'claude':
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

            elif provider_name == 'deepseek':
                return {
                    'symbol': symbol,
                    'signal': 'NO_SIGNAL',
                    'confidence': 0,
                    'rejection_reason': 'DeepSeek не поддерживает comprehensive analysis'
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

    async def _claude_select_pairs(
        self,
        pairs_data: List[Dict],
        max_pairs: Optional[int] = None
    ) -> List[str]:
        """Выбор пар через Claude (Stage 2)"""
        import json

        pairs_info = []
        for pair in pairs_data:
            symbol = pair.get('symbol', 'UNKNOWN')

            # Компактная информация из multi-TF данных
            info_parts = [f"Symbol: {symbol}"]
            info_parts.append(f"Direction: {pair.get('direction', 'NONE')} ({pair.get('confidence', 0)}%)")

            # Текущие значения индикаторов
            if pair.get('indicators_1h'):
                current_1h = pair['indicators_1h'].get('current', {})
                if current_1h:
                    info_parts.append(f"1H: RSI={current_1h.get('rsi', 0):.1f}, Price=${current_1h.get('price', 0):.2f}")

            if pair.get('indicators_4h'):
                current_4h = pair['indicators_4h'].get('current', {})
                if current_4h:
                    info_parts.append(f"4H: RSI={current_4h.get('rsi', 0):.1f}, Vol ratio={current_4h.get('volume_ratio', 0):.2f}")

            pairs_info.append('\n'.join(info_parts))

        pairs_text = "\n---\n".join(pairs_info)
        limit_text = f"максимум {max_pairs} пар" if max_pairs else "без ограничения"

        prompt = (
            f"Проанализируй {len(pairs_data)} торговых пар с компактными multi-timeframe данными "
            f"и выбери {limit_text} с лучшим потенциалом:\n\n{pairs_text}\n\n"
            f"Верни ТОЛЬКО JSON: {{\"selected_pairs\": [\"BTCUSDT\", \"ETHUSDT\"]}}"
        )

        kwargs = {
            'model': STAGE2_MODEL if 'claude' in STAGE2_MODEL else 'claude-haiku-4-5-20251001',
            'max_tokens': STAGE2_MAX_TOKENS,
            'temperature': STAGE2_TEMPERATURE,
            'messages': [{'role': 'user', 'content': prompt}]
        }

        if ANTHROPIC_THINKING:
            kwargs['thinking'] = {'type': 'enabled', 'budget_tokens': 2000}

        response = await self.claude_client.messages.create(**kwargs)

        if ANTHROPIC_THINKING and hasattr(response, 'thinking'):
            print(f"[Claude] 💭 Extended Thinking (первые 500 символов):")
            print(f"     {str(response.thinking)[:500]}...")

        content = response.content[0].text.strip()

        # Парсинг ответа
        selected = []

        try:
            # Удаляем markdown блоки
            if '```json' in content:
                start = content.find('```json') + 7
                end = content.find('```', start)
                if end != -1:
                    content = content[start:end].strip()
            elif '```' in content:
                start = content.find('```') + 3
                end = content.find('```', start)
                if end != -1:
                    content = content[start:end].strip()

            data = json.loads(content)
            selected_pairs = data.get('selected_pairs', [])

            for symbol in selected_pairs:
                clean = symbol.strip().upper()
                if clean and clean not in selected:
                    selected.append(clean)

        except json.JSONDecodeError:
            # Fallback парсинг
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

    def get_config(self) -> Dict:
        """Возвращает текущую конфигурацию"""
        return {
            'stage_providers': self.stage_providers,
            'stage2_model': STAGE2_MODEL,
            'stage2_temperature': STAGE2_TEMPERATURE,
            'stage2_max_tokens': STAGE2_MAX_TOKENS,
            'stage3_model': STAGE3_MODEL,
            'stage3_temperature': STAGE3_TEMPERATURE,
            'stage3_max_tokens': STAGE3_MAX_TOKENS,
            'stage4_removed': True,
            'deepseek_reasoning': DEEPSEEK_REASONING,
            'anthropic_thinking': ANTHROPIC_THINKING
        }