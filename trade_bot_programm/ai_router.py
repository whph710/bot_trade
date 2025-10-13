"""
AI Router with provider selection - с оптимизированным логированием
"""

from typing import List, Dict
from config import config
from utils import fallback_validation
from logging_config import setup_module_logger

logger = setup_module_logger(__name__)


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
            logger.debug("DeepSeek client initialized")
        return self._deepseek_client

    @property
    def anthropic_client(self):
        if self._anthropic_client is None:
            from anthropic_ai import AnthropicClient
            self._anthropic_client = AnthropicClient()
            logger.debug("Anthropic client initialized")
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
            logger.debug(f"AI call: stage={stage}, provider={provider}, tokens={max_tokens}")

            if provider == 'claude':
                response = await self.anthropic_client.call(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    use_thinking=use_reasoning,
                    stage=stage
                )
            else:
                response = await self.deepseek_client.call(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    use_reasoning=use_reasoning
                )

            logger.debug(f"AI response received: {len(response)} chars")
            return response

        except Exception as e:
            logger.error(f"AI call error (stage: {stage}): {e}")
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
        logger.info(f"Stage 2 AI: Selecting from {len(pairs_data)} pair(s) using {config.STAGE2_PROVIDER}")
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
        logger.debug(f"Stage 3 AI: Comprehensive analysis for {symbol}")

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
                logger.warning(f"Analysis failed for {symbol}: invalid result structure")
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
        logger.debug(f"Stage 4 AI: Validating {symbol} using {config.STAGE4_PROVIDER}")

        try:
            provider = config.STAGE4_PROVIDER

            if provider == 'claude':
                return await self.anthropic_client.validate_signal(signal, comprehensive_data)
            else:
                return await self.deepseek_client.validate_signal(signal, comprehensive_data)

        except Exception as e:
            logger.error(f"Validation error for {symbol}: {e}")
            return fallback_validation(signal, config.MIN_RISK_REWARD_RATIO)

    def _fallback_analysis(self, symbol: str, current_price: float) -> Dict:
        """Fallback analysis result"""
        logger.debug(f"Using fallback analysis for {symbol}")
        return {
            'symbol': symbol,
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'entry_price': current_price,
            'stop_loss': 0,
            'take_profit_levels': [0, 0, 0],
            'analysis': 'Analysis failed - fallback',
            'orderflow_analysis': {},
            'smc_analysis': {},
            'ai_generated': False,
            'stage': 3
        }


ai_router = AIRouter()