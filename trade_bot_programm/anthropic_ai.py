"""
Anthropic Claude API client - FIXED VERSION
"""

import asyncio
import json
import logging
from typing import List, Dict, Optional
import httpx
from config import config
from utils import fallback_validation, extract_json_from_response

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Async Anthropic Claude API client"""

    def __init__(self):
        self.api_key = config.ANTHROPIC_API_KEY
        self.model = config.ANTHROPIC_MODEL
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.use_thinking = config.ANTHROPIC_THINKING

    def _get_timeout_for_stage(self, stage: str = None) -> float:
        """Get dynamic timeout based on stage"""
        if stage == 'analysis':
            return getattr(config, 'API_TIMEOUT_ANALYSIS', 180)
        elif stage == 'selection':
            return getattr(config, 'API_TIMEOUT_SELECTION', 90)
        elif stage == 'validation':
            return getattr(config, 'API_TIMEOUT_VALIDATION', 120)
        else:
            return config.API_TIMEOUT

    async def call(
            self,
            prompt: str,
            max_tokens: int = 1000,
            temperature: float = 0.7,
            use_thinking: bool = None,
            stage: str = None
    ) -> str:
        """Make API request with dynamic timeout"""
        if not self.api_key:
            raise ValueError("Anthropic API key not found")

        if use_thinking is None:
            use_thinking = self.use_thinking

        timeout = self._get_timeout_for_stage(stage)

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }

        if use_thinking:
            payload["thinking"] = {"type": "enabled", "budget_tokens": 1000}

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )

                response.raise_for_status()
                data = response.json()
                return data["content"][0]["text"]

        except httpx.TimeoutException:
            logger.error(f"Anthropic API timeout (stage: {stage}, timeout: {timeout}s)")
            raise asyncio.TimeoutError(f"Anthropic API timeout after {timeout}s")
        except httpx.HTTPStatusError as e:
            logger.error(f"Anthropic HTTP error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Anthropic error: {e}")
            raise

    async def select_pairs(self, pairs_data: List[Dict]) -> List[str]:
        """Select pairs for analysis"""
        if not pairs_data:
            return []

        try:
            from deepseek import load_prompt_cached

            compact_data = {}
            for item in pairs_data[:config.MAX_BULK_PAIRS]:
                symbol = item['symbol']
                if 'indicators_15m' not in item or 'candles_15m' not in item:
                    continue

                candles_15m = item.get('candles_15m', [])
                indicators_15m = item.get('indicators_15m', {})

                if not candles_15m or not indicators_15m:
                    continue

                compact_data[symbol] = {
                    'base_signal': {
                        'direction': item.get('direction', 'NONE'),
                        'confidence': item.get('confidence', 0)
                    },
                    'candles_60m': candles_15m[-30:],
                    'indicators': {
                        'ema5': indicators_15m.get('ema5_history', [])[-30:],
                        'ema8': indicators_15m.get('ema8_history', [])[-30:],
                        'ema20': indicators_15m.get('ema20_history', [])[-30:],
                        'rsi': indicators_15m.get('rsi_history', [])[-30:],
                        'macd_histogram': indicators_15m.get('macd_histogram_history', [])[-30:],
                        'volume_ratio': indicators_15m.get('volume_ratio_history', [])[-30:]
                    },
                    'current_state': indicators_15m.get('current', {})
                }

            if not compact_data:
                return []

            prompt = load_prompt_cached(config.SELECTION_PROMPT)
            data_json = json.dumps(compact_data, separators=(',', ':'))

            response_text = await self.call(
                prompt=f"{prompt}\n\nData:\n{data_json}",
                max_tokens=config.AI_MAX_TOKENS_SELECT,
                temperature=config.AI_TEMPERATURE_SELECT,
                stage='selection'
            )

            result = extract_json_from_response(response_text)
            if result and 'selected_pairs' in result:
                selected_pairs = result['selected_pairs'][:config.MAX_FINAL_PAIRS]
                logger.info(f"Claude selected {len(selected_pairs)} pairs")
                return selected_pairs

            logger.warning("Claude returned no pairs")
            return []

        except Exception as e:
            logger.error(f"Claude pair selection error: {e}")
            return []

    async def validate_signal(self, signal: Dict, comprehensive_data: Dict) -> Dict:
        """Validate trading signal"""
        try:
            from deepseek import load_prompt_cached

            validation_input = {
                'signal': {
                    'symbol': signal['symbol'],
                    'signal': signal['signal'],
                    'confidence': signal['confidence'],
                    'entry_price': signal['entry_price'],
                    'stop_loss': signal['stop_loss'],
                    'take_profit_levels': signal.get('take_profit_levels', [0, 0, 0]),
                    'analysis': signal.get('analysis', '')
                },
                'comprehensive_data': {
                    'market_data': comprehensive_data.get('market_data', {}),
                    'correlation_data': comprehensive_data.get('correlation_data', {}),
                    'volume_profile': comprehensive_data.get('volume_profile', {}),
                    'orderflow_analysis': signal.get('orderflow_analysis', {}),
                    'smc_analysis': signal.get('smc_analysis', {}),
                    'current_price': comprehensive_data.get('current_price', 0)
                }
            }

            prompt = load_prompt_cached(config.VALIDATION_PROMPT)
            data_json = json.dumps(validation_input, separators=(',', ':'))

            response_text = await self.call(
                prompt=f"{prompt}\n\nValidation data:\n{data_json}",
                max_tokens=config.AI_MAX_TOKENS_VALIDATE,
                temperature=config.AI_TEMPERATURE_VALIDATE,
                stage='validation'
            )

            result = extract_json_from_response(response_text)

            if result and 'final_signals' in result:
                final_signals = result.get('final_signals', [])
                if final_signals:
                    validated = final_signals[0]
                    tp_levels = validated.get('take_profit_levels', signal.get('take_profit_levels', [0, 0, 0]))

                    if not isinstance(tp_levels, list):
                        tp_levels = [float(tp_levels), float(tp_levels) * 1.1, float(tp_levels) * 1.2]

                    return {
                        'approved': True,
                        'final_confidence': validated.get('confidence', signal['confidence']),
                        'entry_price': validated.get('entry_price', signal['entry_price']),
                        'stop_loss': validated.get('stop_loss', signal['stop_loss']),
                        'take_profit_levels': tp_levels,
                        'risk_reward_ratio': validated.get('risk_reward_ratio', 0),
                        'hold_duration_minutes': validated.get('hold_duration_minutes', 720),
                        'validation_notes': validated.get('validation_notes', ''),
                        'market_conditions': validated.get('market_conditions', ''),
                        'key_levels': validated.get('key_levels', ''),
                        'validation_method': 'claude'
                    }
                else:
                    rejected_info = result.get('rejected_signals', [{}])[0]
                    return {
                        'approved': False,
                        'rejection_reason': rejected_info.get('reason', 'Claude rejected'),
                        'entry_price': signal.get('entry_price', 0),
                        'stop_loss': signal.get('stop_loss', 0),
                        'take_profit_levels': signal.get('take_profit_levels', [0, 0, 0]),
                        'final_confidence': signal.get('confidence', 0),
                        'validation_method': 'claude'
                    }

            return fallback_validation(signal, config.MIN_RISK_REWARD_RATIO)

        except Exception as e:
            logger.error(f"Claude validation error: {e}")
            return fallback_validation(signal, config.MIN_RISK_REWARD_RATIO)


anthropic_client = AnthropicClient()