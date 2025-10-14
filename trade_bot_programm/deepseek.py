"""
DeepSeek API client - FIXED: гарантируем market_conditions и key_levels
"""

import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from config import config
from shared_utils import fallback_validation, extract_json_from_response
from logging_config import setup_module_logger

logger = setup_module_logger(__name__)

_prompts_cache = {}


def load_prompt_cached(filename: str) -> str:
    """Load prompt with caching - FIXED: Better path resolution"""
    if filename in _prompts_cache:
        logger.debug(f"Prompt loaded from cache: {filename}")
        return _prompts_cache[filename]

    # Try direct path
    filepath = Path(filename)
    if not filepath.exists():
        # Try relative to this file
        filepath = Path(__file__).parent / Path(filename).name

    if not filepath.exists():
        # Try in parent/trade_bot_programm
        filepath = Path(__file__).parent.parent / 'trade_bot_programm' / Path(filename).name

    if not filepath.exists():
        logger.error(f"Prompt file not found: {filename}")
        logger.error(f"Tried paths: {filename}, {Path(__file__).parent / Path(filename).name}")
        raise FileNotFoundError(f"Prompt file {filename} not found")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                raise ValueError(f"Prompt file is empty: {filepath}")
            _prompts_cache[filename] = content
            logger.debug(f"Prompt cached: {filepath.name} ({len(content)} chars)")
            return content
    except Exception as e:
        logger.error(f"Error loading prompt {filepath}: {e}")
        raise


def safe_float_conversion(value) -> float:
    """Safe float conversion"""
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
        logger.debug(f"DeepSeek client initialized: model={self.model}, reasoning={self.use_reasoning}")

    async def call(
            self,
            prompt: str,
            max_tokens: int = 2000,
            temperature: float = 0.7,
            use_reasoning: bool = None
    ) -> str:
        """Make API request"""
        if not self.api_key:
            logger.error("DeepSeek API key not configured")
            raise ValueError("DeepSeek API key not found")

        if use_reasoning is None:
            use_reasoning = self.use_reasoning

        try:
            logger.debug(f"DeepSeek API call: max_tokens={max_tokens}, reasoning={use_reasoning}")

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

            result = response.choices[0].message.content
            logger.debug(f"DeepSeek response: {len(result)} chars")
            return result

        except asyncio.TimeoutError:
            logger.error(f"DeepSeek timeout: {config.API_TIMEOUT}s")
            raise
        except Exception as e:
            logger.error(f"DeepSeek error: {e}")
            raise

    async def select_pairs(self, pairs_data: List[Dict]) -> List[str]:
        """Select pairs for analysis"""
        if not pairs_data:
            logger.warning("No pairs data for DeepSeek selection")
            return []

        try:
            if len(pairs_data) > config.MAX_BULK_PAIRS:
                pairs_data = sorted(pairs_data, key=lambda x: x.get('confidence', 0), reverse=True)[:config.MAX_BULK_PAIRS]
                logger.debug(f"Selection limited to top {config.MAX_BULK_PAIRS} pairs")

            logger.info(f"DeepSeek: Selecting pairs from {len(pairs_data)} candidates")

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
                logger.warning("No valid compact data for DeepSeek selection")
                return []

            prompt = load_prompt_cached(config.SELECTION_PROMPT)
            json_payload = json.dumps(compact_data, separators=(',', ':'))

            logger.debug(f"Selection data size: {len(json_payload)} chars")

            response = await self.call(
                prompt=f"{prompt}\n\nData:\n{json_payload}",
                max_tokens=config.AI_MAX_TOKENS_SELECT,
                temperature=config.AI_TEMPERATURE_SELECT
            )

            result = extract_json_from_response(response)

            if result and 'selected_pairs' in result:
                selected_pairs = result['selected_pairs'][:config.MAX_FINAL_PAIRS]
                logger.info(f"DeepSeek selected {len(selected_pairs)} pairs: {selected_pairs}")
                return selected_pairs

            logger.warning("DeepSeek returned no pairs in response")
            return []

        except asyncio.TimeoutError:
            logger.error("DeepSeek selection timeout")
            return []
        except Exception as e:
            logger.error(f"DeepSeek selection error: {e}")
            return []

    async def validate_signal(self, signal: Dict, comprehensive_data: Dict) -> Dict:
        """Validate trading signal - FIXED: гарантируем все поля"""
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            logger.debug(f"DeepSeek: Validating signal for {symbol}")

            validation_input = {
                'signal': {
                    'symbol': symbol,
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

            logger.debug(f"Validation data size: {len(data_json)} chars")

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

                    # FIXED: Гарантируем наличие всех полей
                    market_conditions = validated.get('market_conditions', '').strip()
                    key_levels = validated.get('key_levels', '').strip()

                    # Если AI не заполнил - заполняем сами
                    if not market_conditions:
                        market_data = comprehensive_data.get('market_data', {})
                        funding = market_data.get('funding_rate', {})
                        oi = market_data.get('open_interest', {})
                        orderbook = market_data.get('orderbook', {})

                        funding_rate = funding.get('funding_rate', 0) if funding else 0
                        oi_trend = oi.get('oi_trend', 'UNKNOWN') if oi else 'UNKNOWN'
                        spread_pct = orderbook.get('spread_pct', 0) if orderbook else 0

                        market_conditions = f"Funding: {funding_rate:.4f}%, OI: {oi_trend}, Spread: {spread_pct:.4f}%"

                    if not key_levels:
                        entry = validated.get('entry_price', signal['entry_price'])
                        stop = validated.get('stop_loss', signal['stop_loss'])
                        key_levels = f"Entry: ${entry:.4f}, Stop: ${stop:.4f}, TP1: ${tp_levels[0]:.4f}, TP2: ${tp_levels[1]:.4f}, TP3: ${tp_levels[2]:.4f}"

                    logger.debug(f"DeepSeek: Approved {symbol} with R/R {validated.get('risk_reward_ratio', 0)}")

                    return {
                        'approved': True,
                        'final_confidence': validated.get('confidence', signal['confidence']),
                        'entry_price': validated.get('entry_price', signal['entry_price']),
                        'stop_loss': validated.get('stop_loss', signal['stop_loss']),
                        'take_profit_levels': tp_levels,
                        'risk_reward_ratio': validated.get('risk_reward_ratio', 0),
                        'hold_duration_minutes': validated.get('hold_duration_minutes', 720),
                        'validation_notes': validated.get('validation_notes', ''),
                        'market_conditions': market_conditions,  # FIXED: гарантировано заполнено
                        'key_levels': key_levels,  # FIXED: гарантировано заполнено
                        'validation_method': 'deepseek'
                    }
                else:
                    rejected_info = result.get('rejected_signals', [{}])[0]
                    reason = rejected_info.get('reason', 'DeepSeek rejected')
                    logger.debug(f"DeepSeek: Rejected {symbol} - {reason}")

                    return {
                        'approved': False,
                        'rejection_reason': reason,
                        'entry_price': signal.get('entry_price', 0),
                        'stop_loss': signal.get('stop_loss', 0),
                        'take_profit_levels': signal.get('take_profit_levels', [0, 0, 0]),
                        'final_confidence': signal.get('confidence', 0),
                        'validation_method': 'deepseek'
                    }

            logger.warning(f"DeepSeek: Invalid validation response for {symbol}")
            return fallback_validation(signal, config.MIN_RISK_REWARD_RATIO)

        except asyncio.TimeoutError:
            logger.error(f"DeepSeek validation timeout for {symbol}")
            return fallback_validation(signal, config.MIN_RISK_REWARD_RATIO)
        except Exception as e:
            logger.error(f"DeepSeek validation error for {symbol}: {e}")
            return fallback_validation(signal, config.MIN_RISK_REWARD_RATIO)