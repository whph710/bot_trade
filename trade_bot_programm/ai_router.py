"""
AI Router - OPTIMIZED LOGGING
Файл: trade_bot_programm/ai_router.py
ИЗМЕНЕНИЯ:
- Удалены все print() в пользу logger
- Убраны ASCII-рамки инициализации
- Сокращены избыточные debug-сообщения
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
from logging_config import setup_module_logger

logger = setup_module_logger(__name__)


class AIRouter:
    """AI Router с поддержкой stage-specific моделей"""

    def __init__(self):
        self.deepseek_clients: Dict[str, DeepSeekClient] = {}
        self.claude_client = None

        self.stage_providers = {
            'stage2': STAGE2_PROVIDER,
            'stage3': STAGE3_PROVIDER
        }

        logger.info(f"AI Router initialized: Stage2={STAGE2_PROVIDER.upper()} ({STAGE2_MODEL}), Stage3={STAGE3_PROVIDER.upper()} ({STAGE3_MODEL})")

    async def _get_deepseek_client(self, stage: str) -> Optional[DeepSeekClient]:
        """Получить DeepSeek клиент для конкретного stage"""
        if stage in self.deepseek_clients:
            return self.deepseek_clients[stage]

        if not DEEPSEEK_API_KEY:
            logger.warning("DEEPSEEK_API_KEY not found")
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
            logger.error(f"Failed to initialize DeepSeek for {stage}: {e}")
            return None

    async def initialize_claude(self):
        """Инициализирует Claude клиент"""
        if not ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not found")
            return False

        try:
            from anthropic import AsyncAnthropic

            self.claude_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
            logger.info(f"Claude initialized: extended_thinking={'ON' if ANTHROPIC_THINKING else 'OFF'}")
            return True
        except ImportError:
            logger.error("Anthropic SDK not installed: pip install anthropic")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Claude: {e}")
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
            logger.error(f"Unknown provider: {provider}")
            return None, None

    async def select_pairs(
        self,
        pairs_data: List[Dict],
        max_pairs: Optional[int] = None
    ) -> List[str]:
        """Stage 2: Выбор пар"""
        logger.info(f"Stage 2: selecting from {len(pairs_data)} pairs (limit: {max_pairs})")

        provider_name, client = await self._get_provider_client('stage2')

        if not client:
            logger.error("Stage 2: Client unavailable")
            return []

        logger.debug(f"Stage 2: using {provider_name.upper()} (model={STAGE2_MODEL}, temp={STAGE2_TEMPERATURE})")

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

            logger.info(f"Stage 2 complete: selected {len(selected)} pairs")
            return selected

        except Exception as e:
            logger.error(f"Stage 2 error: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def analyze_pair_comprehensive(
        self,
        symbol: str,
        comprehensive_data: Dict
    ) -> Dict:
        """Stage 3: Comprehensive analysis"""
        logger.debug(f"Stage 3: analyzing {symbol}")

        provider_name, client = await self._get_provider_client('stage3')

        if not client:
            logger.error(f"Stage 3: Client unavailable for {symbol}")
            return {
                'symbol': symbol,
                'signal': 'NO_SIGNAL',
                'confidence': 0,
                'rejection_reason': 'AI client unavailable'
            }

        logger.debug(f"Stage 3: using {provider_name.upper()} (model={STAGE3_MODEL})")

        try:
            if provider_name == 'claude':
                from anthropic_client import AnthropicClient
                claude = AnthropicClient()
                result = await claude.analyze_comprehensive(symbol, comprehensive_data)

                if result:
                    logger.debug(f"Stage 3: Claude analysis complete for {symbol}")
                    return result
                else:
                    return {
                        'symbol': symbol,
                        'signal': 'NO_SIGNAL',
                        'confidence': 0,
                        'rejection_reason': 'Claude returned no result'
                    }

            elif provider_name == 'deepseek':
                result = await self._deepseek_comprehensive_analysis(symbol, comprehensive_data)

                if result and result.get('signal') != 'NO_SIGNAL':
                    logger.debug(f"Stage 3: DeepSeek analysis complete for {symbol}")
                    return result
                else:
                    return {
                        'symbol': symbol,
                        'signal': 'NO_SIGNAL',
                        'confidence': 0,
                        'rejection_reason': result.get('rejection_reason', 'DeepSeek rejected signal')
                    }

            else:
                return {
                    'symbol': symbol,
                    'signal': 'NO_SIGNAL',
                    'confidence': 0,
                    'rejection_reason': f'Unknown provider: {provider_name}'
                }

        except Exception as e:
            logger.error(f"Stage 3 error for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'symbol': symbol,
                'signal': 'NO_SIGNAL',
                'confidence': 0,
                'rejection_reason': f'Exception: {str(e)[:100]}'
            }

    async def _deepseek_comprehensive_analysis(
        self,
        symbol: str,
        comprehensive_data: Dict
    ) -> Dict:
        """DeepSeek comprehensive analysis implementation"""
        import json
        from pathlib import Path
        from shared_utils import extract_json_from_response

        try:
            prompt_path = Path(__file__).parent / "prompts" / "prompt_analyze.txt"

            if not prompt_path.exists():
                logger.error(f"Analysis prompt not found: {prompt_path}")
                return {
                    'symbol': symbol,
                    'signal': 'NO_SIGNAL',
                    'confidence': 0,
                    'rejection_reason': 'Analysis prompt file missing'
                }

            with open(prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read()

            analysis_data = {
                'symbol': symbol,
                'has_1d_data': comprehensive_data.get('has_1d_data', False),
                'candles_1h': comprehensive_data.get('candles_1h', [])[-100:],
                'candles_4h': comprehensive_data.get('candles_4h', [])[-60:],
                'candles_1d': comprehensive_data.get('candles_1d', [])[-20:],
                'indicators_1h': comprehensive_data.get('indicators_1h', {}),
                'indicators_4h': comprehensive_data.get('indicators_4h', {}),
                'indicators_1d': comprehensive_data.get('indicators_1d', {}),
                'current_price': comprehensive_data.get('current_price', 0),
                'market_data': comprehensive_data.get('market_data', {}),
                'correlation_data': comprehensive_data.get('correlation_data', {}),
                'volume_profile': comprehensive_data.get('volume_profile', {}),
                'vp_analysis': comprehensive_data.get('vp_analysis', {}),
                'btc_candles_1h': comprehensive_data.get('btc_candles_1h', [])[-100:],
                'btc_candles_4h': comprehensive_data.get('btc_candles_4h', [])[-60:],
                'btc_candles_1d': comprehensive_data.get('btc_candles_1d', [])[-20:]
            }

            data_json = json.dumps(analysis_data, separators=(',', ':'))
            logger.debug(f"Stage 3 {symbol}: analysis data size = {len(data_json)} chars")

            deepseek = await self._get_deepseek_client('stage3')

            if not deepseek:
                return {
                    'symbol': symbol,
                    'signal': 'NO_SIGNAL',
                    'confidence': 0,
                    'rejection_reason': 'DeepSeek client unavailable'
                }

            user_prompt = f"{system_prompt}\n\nData:\n{data_json}"

            response = await deepseek.chat(
                messages=[
                    {"role": "system", "content": "You are an expert institutional swing trader with 20 years experience."},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=STAGE3_MAX_TOKENS,
                temperature=STAGE3_TEMPERATURE
            )

            result = extract_json_from_response(response)

            if not result:
                logger.warning(f"Stage 3 {symbol}: invalid JSON response")
                return {
                    'symbol': symbol,
                    'signal': 'NO_SIGNAL',
                    'confidence': 0,
                    'rejection_reason': 'Invalid JSON response from DeepSeek'
                }

            result['symbol'] = symbol

            # Normalize take_profit_levels
            if 'take_profit_levels' in result:
                tp_levels = result['take_profit_levels']
                if not isinstance(tp_levels, list):
                    tp_levels = [float(tp_levels), float(tp_levels) * 1.1, float(tp_levels) * 1.2]
                elif len(tp_levels) < 3:
                    while len(tp_levels) < 3:
                        last_tp = tp_levels[-1] if tp_levels else 0
                        tp_levels.append(last_tp * 1.1)
                result['take_profit_levels'] = tp_levels

            return result

        except Exception as e:
            logger.error(f"Stage 3 DeepSeek analysis error for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'symbol': symbol,
                'signal': 'NO_SIGNAL',
                'confidence': 0,
                'rejection_reason': f'DeepSeek exception: {str(e)[:100]}'
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

            info_parts = [f"Symbol: {symbol}"]
            info_parts.append(f"Direction: {pair.get('direction', 'NONE')} ({pair.get('confidence', 0)}%)")

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
            logger.debug(f"Claude extended thinking (first 300 chars): {str(response.thinking)[:300]}")

        content = response.content[0].text.strip()

        selected = []

        try:
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
            logger.warning("Claude JSON parsing failed, using fallback")
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