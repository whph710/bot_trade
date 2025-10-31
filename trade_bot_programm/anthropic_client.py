"""
Anthropic Claude AI Client - БЕЗ 1D ДАННЫХ
Файл: trade_bot_programm/anthropic_client.py
ИЗМЕНЕНИЯ:
- Убраны все упоминания 1D свечей
- Убраны indicators_1d
- Убран флаг has_1d_data
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Optional
from anthropic import AsyncAnthropic
from config import config
from shared_utils import extract_json_from_response
from logging_config import setup_module_logger

logger = setup_module_logger(__name__)

_prompts_cache = {}


def load_prompt_cached(filename: str) -> str:
    """Load prompt with caching"""
    if filename in _prompts_cache:
        logger.debug(f"Prompt loaded from cache: {filename}")
        return _prompts_cache[filename]

    search_paths = [
        Path(filename),
        Path(__file__).parent / "prompts" / Path(filename).name,
        Path(__file__).parent.parent / "prompts" / Path(filename).name,
        Path(__file__).parent.parent.parent / "prompts" / Path(filename).name,
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
    """Anthropic Claude API client - Stage 3 only (БЕЗ 1D)"""

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
        """Select pairs for analysis (Stage 2 - БЕЗ 1D)"""
        if not pairs_data:
            logger.warning("No pairs data for Claude selection")
            return []

        try:
            logger.info(f"Claude: Selecting pairs from ALL {len(pairs_data)} candidates")

            compact_data = {}
            for item in pairs_data:
                symbol = item['symbol']

                indicators_1h = item.get('indicators_1h', {})
                indicators_4h = item.get('indicators_4h', {})

                if not indicators_1h or not indicators_4h:
                    continue

                compact_data[symbol] = {
                    'base_signal': {
                        'direction': item.get('direction', 'NONE'),
                        'confidence': item.get('confidence', 0)
                    },
                    'candles_1h': item.get('candles_1h', [])[-30:],
                    'candles_4h': item.get('candles_4h', [])[-30:],
                    'indicators_1h': indicators_1h,
                    'indicators_4h': indicators_4h
                }

            if not compact_data:
                logger.warning("No valid compact data for Claude selection")
                return []

            prompt = load_prompt_cached(config.SELECTION_PROMPT)
            json_payload = json.dumps(compact_data, separators=(',', ':'))

            logger.debug(f"Selection data size: {len(json_payload)} chars for {len(compact_data)} pairs")

            response = await self.call(
                prompt=f"{prompt}\n\nData:\n{json_payload}",
                max_tokens=config.AI_MAX_TOKENS_SELECT,
                temperature=config.AI_TEMPERATURE_SELECT
            )

            result = extract_json_from_response(response)

            if result and 'selected_pairs' in result:
                selected_pairs = result['selected_pairs'][:config.MAX_FINAL_PAIRS]
                logger.info(f"Claude selected {len(selected_pairs)} out of {len(compact_data)} pairs: {selected_pairs}")
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
        """Comprehensive analysis using full Stage 3 data (БЕЗ 1D)"""
        try:
            logger.debug(f"Claude: Comprehensive analysis for {symbol}")

            prompt = load_prompt_cached(config.ANALYSIS_PROMPT)
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