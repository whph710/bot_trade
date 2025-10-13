"""
DeepSeek API client - FIXED VERSION
"""

import asyncio
import json
import logging
import os
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from config import config
from utils import fallback_validation, extract_json_from_response

logger = logging.getLogger(__name__)

_prompts_cache = {}


def load_prompt_cached(filename: str) -> str:
    """Load prompt with caching"""
    if filename in _prompts_cache:
        return _prompts_cache[filename]

    if not os.path.exists(filename):
        logger.error(f"Prompt file {filename} not found")
        raise FileNotFoundError(f"Prompt file {filename} not found")

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                raise ValueError(f"Prompt file {filename} is empty")
            _prompts_cache[filename] = content
            return content
    except Exception as e:
        logger.error(f"Error loading prompt {filename}: {e}")
        raise


def safe_float_conversion(value) -> float:
    """Safe float conversion"""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


class DeepSeekClient:
    """DeepSeek API client"""

    def __init__(self):
        self.api_key = config.DEEPSEEK_API_KEY
        self.model = config.DEEPSEEK_MODEL
        self.base_url = config.DEEPSEEK_URL
        self.use_reasoning = config.DEEPSEEK_REASONING

    async def call(
            self,
            prompt: str,
            max_tokens: int = 2000,
            temperature: float = 0.7,
            use_reasoning: bool = None
    ) -> str:
        """Make API request"""
        if not self.api_key:
            raise ValueError("DeepSeek API key not found")

        if use_reasoning is None:
            use_reasoning = self.use_reasoning

        try:
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            messages = [{"role": "user", "content": prompt}]

            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            if use_reasoning:
                kwargs["reasoning"] = {"enabled": True}

            response = await asyncio.wait_for(
                client.chat.completions.create(**kwargs),
                timeout=config.API_TIMEOUT
            )

            return response.choices[0].message.content

        except asyncio.TimeoutError:
            logger.error("DeepSeek API timeout")
            raise
        except Exception as e:
            logger.error(f"DeepSeek error: {e}")
            raise

    async def select_pairs(self, pairs_data: List[Dict]) -> List[str]:
        """Select pairs for analysis"""
        if not pairs_data:
            return []

        try:
            if len(pairs_data) > config.MAX_BULK_PAIRS:
                pairs_data = sorted(pairs_data, key=lambda x: x.get('confidence', 0), reverse=True)[:config.MAX_BULK_PAIRS]

            compact_data = {}
            for item in pairs_data:
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
                    'candles_15m': candles_15m[-30:],
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
            json_payload = json.dumps(compact_data, separators=(',', ':'))

            response = await self.call(
                prompt=f"{prompt}\n\nData:\n{json_payload}",
                max_tokens=config.AI_MAX_TOKENS_SELECT,
                temperature=config.AI_TEMPERATURE_SELECT
            )

            result = extract_json_from_response(response)

            if result and 'selected_pairs' in result:
                selected_pairs = result['selected_pairs'][:config.MAX_FINAL_PAIRS]
                logger.info(f"DeepSeek selected {len(selected_pairs)} pairs")
                return selected_pairs

            logger.info("DeepSeek returned no pairs")
            return []

        except asyncio.TimeoutError:
            logger.error("DeepSeek selection timeout")
            return []
        except Exception as e:
            logger.error(f"DeepSeek selection error: {e}")
            return []

    async def validate_signal(self, signal: Dict, comprehensive_data: Dict) -> Dict:
        """Validate trading signal"""
        try:
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
                    'current_price': comprehensive_data.get('current_price', 0)
                }
            }

            prompt = load_prompt_cached(config.VALIDATION_PROMPT)
            data_json = json.dumps(validation_input, separators=(',', ':'))

            response = await self.call(
                prompt=f"{prompt}\n\nValidation data:\n{data_json}",
                max_tokens=config.AI_MAX_TOKENS_VALIDATE,
                temperature=config.AI_TEMPERATURE_VALIDATE
            )

            result = extract_json_from_response(response)

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
                        'validation_method': 'deepseek'
                    }
                else:
                    rejected_info = result.get('rejected_signals', [{}])[0]
                    return {
                        'approved': False,
                        'rejection_reason': rejected_info.get('reason', 'DeepSeek rejected'),
                        'entry_price': signal.get('entry_price', 0),
                        'stop_loss': signal.get('stop_loss', 0),
                        'take_profit_levels': signal.get('take_profit_levels', [0, 0, 0]),
                        'final_confidence': signal.get('confidence', 0),
                        'validation_method': 'deepseek'
                    }

            return fallback_validation(signal, config.MIN_RISK_REWARD_RATIO)

        except Exception as e:
            logger.error(f"DeepSeek validation error: {e}")
            return fallback_validation(signal, config.MIN_RISK_REWARD_RATIO)