"""
AI Router with provider selection
"""

import logging
import json
from typing import List, Dict
from config import config

logger = logging.getLogger(__name__)


class AIRouter:
    """Routes AI requests to selected providers"""

    def __init__(self):
        self._deepseek_client = None
        self._anthropic_client = None

    @property
    def deepseek_client(self):
        if self._deepseek_client is None:
            from deepseek import DeepSeekClient
            self._deepseek_client = DeepSeekClient()
        return self._deepseek_client

    @property
    def anthropic_client(self):
        if self._anthropic_client is None:
            from anthropic_ai import AnthropicClient
            self._anthropic_client = AnthropicClient()
        return self._anthropic_client

    async def call_ai(
            self,
            prompt: str,
            stage: str = 'analysis',
            max_tokens: int = 2000,
            temperature: float = 0.7,
            use_reasoning: bool = False
    ) -> str:
        """Universal AI call method"""
        try:
            provider = self._get_provider_for_stage(stage)

            if provider == 'claude':
                return await self.anthropic_client.call(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    use_thinking=use_reasoning
                )
            else:
                return await self.deepseek_client.call(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    use_reasoning=use_reasoning
                )

        except Exception as e:
            logger.error(f"AI call error: {e}")
            return "{}"

    def _get_provider_for_stage(self, stage: str) -> str:
        """Get AI provider for specific stage"""
        if stage == 'selection':
            return config.STAGE2_PROVIDER
        elif stage == 'analysis':
            return config.STAGE3_PROVIDER
        elif stage == 'validation':
            return config.STAGE4_PROVIDER
        return 'claude'

    async def select_pairs(self, pairs_data: List[Dict]) -> List[str]:
        """Stage 2: Pair selection"""
        logger.info(f"Stage 2: {config.STAGE2_PROVIDER.upper()} selection")

        provider = config.STAGE2_PROVIDER

        if provider == 'claude':
            return await self.anthropic_client.select_pairs(pairs_data)
        else:
            return await self.deepseek_client.select_pairs(pairs_data)

    async def analyze_pair_comprehensive(
            self,
            symbol: str,
            comprehensive_data: Dict
    ) -> Dict:
        """Stage 3: Comprehensive analysis"""
        logger.debug(f"Stage 3: {config.STAGE3_PROVIDER.upper()} unified analysis for {symbol}")

        try:
            from ai_advanced_analysis import get_unified_analysis

            result = await get_unified_analysis(
                self,
                symbol,
                comprehensive_data
            )

            if result and isinstance(result, dict) and 'signal' in result:
                if not isinstance(result.get('take_profit_levels'), list):
                    tp = result.get('take_profit_levels', 0)
                    if tp:
                        result['take_profit_levels'] = [float(tp), float(tp) * 1.1, float(tp) * 1.2]
                    else:
                        result['take_profit_levels'] = [0, 0, 0]
                return result
            else:
                logger.warning(f"Analysis failed for {symbol}")
                return self._fallback_analysis(symbol, comprehensive_data.get('current_price', 0))

        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {e}")
            return self._fallback_analysis(symbol, comprehensive_data.get('current_price', 0))

    async def validate_signal_with_stage3_data(
            self,
            signal: Dict,
            comprehensive_data: Dict
    ) -> Dict:
        """Stage 4: Signal validation"""
        symbol = signal['symbol']
        logger.debug(f"Stage 4: {config.STAGE4_PROVIDER.upper()} validation for {symbol}")

        try:
            provider = config.STAGE4_PROVIDER

            if provider == 'claude':
                return await self.anthropic_client.validate_signal(signal, comprehensive_data)
            else:
                return await self.deepseek_client.validate_signal(signal, comprehensive_data)

        except Exception as e:
            logger.error(f"Validation error for {symbol}: {e}")
            return self._fallback_validation(signal)

    def _fallback_analysis(self, symbol: str, current_price: float) -> Dict:
        """Fallback analysis result"""
        return {
            'symbol': symbol,
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'entry_price': current_price,
            'stop_loss': 0,
            'take_profit_levels': [0, 0, 0],
            'analysis': 'Analysis failed',
            'orderflow_analysis': {},
            'smc_analysis': {},
            'ai_generated': False,
            'stage': 3
        }

    def _fallback_validation(self, signal: Dict) -> Dict:
        """Fallback validation result"""
        entry = signal.get('entry_price', 0)
        stop = signal.get('stop_loss', 0)
        tp_levels = signal.get('take_profit_levels', [0, 0, 0])

        if not isinstance(tp_levels, list):
            tp_levels = [float(tp_levels), float(tp_levels) * 1.1, float(tp_levels) * 1.2]

        if entry > 0 and stop > 0 and tp_levels and tp_levels[0] > 0:
            risk = abs(entry - stop)
            reward = abs(tp_levels[1] - entry) if len(tp_levels) > 1 else abs(tp_levels[0] - entry)

            if risk > 0:
                rr_ratio = round(reward / risk, 2)
                if rr_ratio >= config.MIN_RISK_REWARD_RATIO:
                    return {
                        'approved': True,
                        'final_confidence': signal['confidence'],
                        'entry_price': entry,
                        'stop_loss': stop,
                        'take_profit_levels': tp_levels,
                        'risk_reward_ratio': rr_ratio,
                        'hold_duration_minutes': 720,
                        'validation_method': 'fallback',
                        'validation_notes': f'Fallback validation: R/R {rr_ratio}'
                    }

        return {
            'approved': False,
            'rejection_reason': 'Fallback validation failed',
            'entry_price': entry,
            'stop_loss': stop,
            'take_profit_levels': tp_levels,
            'final_confidence': signal.get('confidence', 0),
            'validation_method': 'fallback'
        }


ai_router = AIRouter()