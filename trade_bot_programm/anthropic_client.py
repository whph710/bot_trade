"""
Anthropic Claude AI Client - FIXED: Unified prompt loading
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Optional
from anthropic import AsyncAnthropic
from config import config
from shared_utils import fallback_validation, extract_json_from_response
from logging_config import setup_module_logger

logger = setup_module_logger(__name__)

_prompts_cache = {}


def load_prompt_cached(filename: str) -> str:
    """
    Load prompt with caching - FIXED: Unified search strategy

    Ищет промпт в следующем порядке:
    1. Прямой путь (если передан полный путь)
    2. trade_bot_programm/prompts/
    3. prompts/ (корень проекта)
    4. ../prompts/ (родительская директория)
    """
    if filename in _prompts_cache:
        logger.debug(f"Prompt loaded from cache: {filename}")
        return _prompts_cache[filename]

    # Список путей для поиска
    search_paths = [
        Path(filename),  # Прямой путь
        Path(__file__).parent / "prompts" / Path(filename).name,  # trade_bot_programm/prompts/
        Path(__file__).parent.parent / "prompts" / Path(filename).name,  # Корень проекта
        Path(__file__).parent.parent.parent / "prompts" / Path(filename).name,  # Выше корня
    ]

    filepath = None
    for path in search_paths:
        if path.exists() and path.is_file():
            filepath = path
            logger.debug(f"Prompt found at: {filepath}")
            break

    if not filepath:
        error_msg = f"Prompt file '{filename}' not found. Searched in:\n"
        for path in search_paths:
            error_msg += f"  - {path.absolute()}\n"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                raise ValueError(f"Prompt file is empty: {filename}")
            _prompts_cache[filename] = content
            logger.info(f"Prompt cached: {filepath.name} ({len(content)} chars)")
            return content
    except Exception as e:
        logger.error(f"Error loading prompt {filename}: {e}")
        raise


class AnthropicClient:
    """Anthropic Claude API client"""

    def __init__(self):
        self.api_key = config.ANTHROPIC_API_KEY
        self.model = config.ANTHROPIC_MODEL
        self.use_thinking = config.ANTHROPIC_THINKING
        logger.debug(f"Anthropic client initialized: model={self.model}, thinking={self.use_thinking}")

    async def call(
            self,
            prompt: str,
            max_tokens: int = 2000,
            temperature: float = 0.7,
            use_thinking: bool = None,
            stage: str = 'analysis'
    ) -> str:
        """Make API request to Claude"""
        if not self.api_key:
            logger.error("Anthropic API key not configured")
            raise ValueError("Anthropic API key not found")

        if use_thinking is None:
            use_thinking = self.use_thinking

        try:
            logger.debug(f"Claude API call: stage={stage}, max_tokens={max_tokens}, thinking={use_thinking}")

            client = AsyncAnthropic(api_key=self.api_key)

            if use_thinking:
                budget_tokens = min(10000, max_tokens * 3)
                response = await asyncio.wait_for(
                    client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        thinking={
                            "type": "enabled",
                            "budget_tokens": budget_tokens
                        },
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature
                    ),
                    timeout=config.API_TIMEOUT_ANALYSIS if stage == 'analysis' else config.API_TIMEOUT
                )
            else:
                response = await asyncio.wait_for(
                    client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature
                    ),
                    timeout=config.API_TIMEOUT_ANALYSIS if stage == 'analysis' else config.API_TIMEOUT
                )

            result = response.content[0].text
            logger.debug(f"Claude response: {len(result)} chars")
            return result

        except asyncio.TimeoutError:
            logger.error(f"Claude timeout: {config.API_TIMEOUT}s")
            raise
        except Exception as e:
            logger.error(f"Claude error: {e}")
            raise

    async def select_pairs(self, pairs_data: List[Dict]) -> List[str]:
        """Select pairs for analysis"""
        if not pairs_data:
            logger.warning("No pairs data for Claude selection")
            return []

        try:
            if len(pairs_data) > config.MAX_BULK_PAIRS:
                pairs_data = sorted(pairs_data, key=lambda x: x.get('confidence', 0), reverse=True)[:config.MAX_BULK_PAIRS]
                logger.debug(f"Selection limited to top {config.MAX_BULK_PAIRS} pairs")

            logger.info(f"Claude: Selecting pairs from {len(pairs_data)} candidates")

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
                logger.warning("No valid compact data for Claude selection")
                return []

            # FIXED: Используем load_prompt_cached
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
                logger.info(f"Claude selected {len(selected_pairs)} pairs: {selected_pairs}")
                return selected_pairs

            logger.warning("Claude returned no pairs in response")
            return []

        except asyncio.TimeoutError:
            logger.error("Claude selection timeout")
            return []
        except Exception as e:
            logger.error(f"Claude selection error: {e}")
            return []

    async def analyze_comprehensive(self, symbol: str, comprehensive_data: Dict) -> Dict:
        """
        Comprehensive analysis using full Stage 3 data
        """
        try:
            logger.debug(f"Claude: Comprehensive analysis for {symbol}")

            # FIXED: Используем load_prompt_cached
            prompt = load_prompt_cached(config.ANALYSIS_PROMPT)

            # Формируем JSON с данными
            data_json = json.dumps(comprehensive_data, separators=(',', ':'))

            logger.debug(f"Analysis data size: {len(data_json)} chars")

            response = await self.call(
                prompt=f"{prompt}\n\nData:\n{data_json}",
                max_tokens=config.AI_MAX_TOKENS_ANALYZE,
                temperature=config.AI_TEMPERATURE_ANALYZE,
                stage='analysis'
            )

            result = extract_json_from_response(response)

            if result:
                result['symbol'] = symbol
                logger.debug(f"Claude: Analysis complete for {symbol}")
                return result
            else:
                logger.warning(f"Claude: Invalid response for {symbol}")
                return {
                    'symbol': symbol,
                    'signal': 'NO_SIGNAL',
                    'confidence': 0,
                    'rejection_reason': 'Invalid Claude response'
                }

        except Exception as e:
            logger.error(f"Claude analysis error for {symbol}: {e}")
            return {
                'symbol': symbol,
                'signal': 'NO_SIGNAL',
                'confidence': 0,
                'rejection_reason': f'Exception: {str(e)[:100]}'
            }

    async def validate_signal(self, signal: Dict, comprehensive_data: Dict) -> Dict:
        """Validate trading signal - FIXED: гарантируем все поля"""
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            logger.debug(f"Claude: Validating signal for {symbol}")

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

            # FIXED: Используем load_prompt_cached
            prompt = load_prompt_cached(config.VALIDATION_PROMPT)
            data_json = json.dumps(validation_input, separators=(',', ':'))

            logger.debug(f"Validation data size: {len(data_json)} chars")

            response = await self.call(
                prompt=f"{prompt}\n\nValidation data:\n{data_json}",
                max_tokens=config.AI_MAX_TOKENS_VALIDATE,
                temperature=config.AI_TEMPERATURE_VALIDATE,
                stage='validation'
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

                    logger.debug(f"Claude: Approved {symbol} with R/R {validated.get('risk_reward_ratio', 0)}")

                    return {
                        'approved': True,
                        'final_confidence': validated.get('confidence', signal['confidence']),
                        'entry_price': validated.get('entry_price', signal['entry_price']),
                        'stop_loss': validated.get('stop_loss', signal['stop_loss']),
                        'take_profit_levels': tp_levels,
                        'risk_reward_ratio': validated.get('risk_reward_ratio', 0),
                        'hold_duration_minutes': validated.get('hold_duration_minutes', 720),
                        'validation_notes': validated.get('validation_notes', ''),
                        'market_conditions': market_conditions,
                        'key_levels': key_levels,
                        'validation_method': 'claude'
                    }
                else:
                    rejected_info = result.get('rejected_signals', [{}])[0]
                    reason = rejected_info.get('reason', 'Claude rejected')
                    logger.debug(f"Claude: Rejected {symbol} - {reason}")

                    return {
                        'approved': False,
                        'rejection_reason': reason,
                        'entry_price': signal.get('entry_price', 0),
                        'stop_loss': signal.get('stop_loss', 0),
                        'take_profit_levels': signal.get('take_profit_levels', [0, 0, 0]),
                        'final_confidence': signal.get('confidence', 0),
                        'validation_method': 'claude'
                    }

            logger.warning(f"Claude: Invalid validation response for {symbol}")
            return fallback_validation(signal, comprehensive_data)

        except asyncio.TimeoutError:
            logger.error(f"Claude validation timeout for {symbol}")
            return fallback_validation(signal, comprehensive_data)
        except Exception as e:
            logger.error(f"Claude validation error for {symbol}: {e}")
            return fallback_validation(signal, comprehensive_data)