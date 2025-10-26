"""
DeepSeek AI клиент - OPTIMIZED LOGGING
Файл: trade_bot_programm/deepseek.py
ИЗМЕНЕНИЯ:
- Удалены все print() в пользу logger
- Упрощены ASCII-рамки инициализации
- Убраны избыточные debug-сообщения
"""

import os
import json
from openai import AsyncOpenAI
from typing import Optional, Dict, List
from pathlib import Path
from logging_config import setup_module_logger

logger = setup_module_logger(__name__)


def load_prompt_unified(prompt_file: str) -> str:
    """Unified prompt loader"""
    search_paths = [
        Path(prompt_file),
        Path(__file__).parent / "prompts" / Path(prompt_file).name,
        Path(__file__).parent.parent / "prompts" / Path(prompt_file).name,
        Path(__file__).parent.parent.parent / "prompts" / Path(prompt_file).name,
    ]

    for path in search_paths:
        if path.exists() and path.is_file():
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.debug(f"Prompt loaded: {path.name} ({len(content)} chars)")
                return content

    error_msg = f"Prompt file '{prompt_file}' not found in search paths"
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)


class DeepSeekClient:
    """Клиент для работы с DeepSeek API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        use_reasoning: Optional[bool] = None,
        base_url: str = "https://api.deepseek.com"
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment")

        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        reasoning_env = os.getenv("DEEPSEEK_REASONING", "false").lower()
        self.use_reasoning = use_reasoning if use_reasoning is not None else (reasoning_env in ["true", "1", "yes"])

        self.is_reasoning_model = "reasoner" in self.model.lower() or self.model == "deepseek-reasoner"

        if self.use_reasoning and not self.is_reasoning_model:
            logger.warning(f"DEEPSEEK_REASONING=true but model {self.model} doesn't support reasoning - disabling")
            self.use_reasoning = False

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url
        )

        self.prompts_cache: Dict[str, str] = {}

        logger.info(f"DeepSeek initialized: model={self.model}, reasoning={'ON' if self.use_reasoning else 'OFF'}")

    def _load_prompt(self, prompt_file: str) -> str:
        """Загрузить промпт из файла с кэшированием"""
        if prompt_file in self.prompts_cache:
            return self.prompts_cache[prompt_file]

        content = load_prompt_unified(prompt_file)
        self.prompts_cache[prompt_file] = content
        return content

    async def select_pairs(
        self,
        pairs_data: List[Dict],
        max_pairs: Optional[int] = None,
        system_prompt_file: str = "prompt_select.txt",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> List[str]:
        """Выбор пар через DeepSeek (Stage 2 - compact multi-TF)"""
        try:
            if temperature is None:
                temperature = float(os.getenv("AI_TEMPERATURE_SELECT", "0.3"))
            if max_tokens is None:
                max_tokens = int(os.getenv("AI_MAX_TOKENS_SELECT", "2000"))

            system_prompt = self._load_prompt(system_prompt_file)

            pairs_info = []
            for pair in pairs_data:
                symbol = pair.get('symbol', 'UNKNOWN')

                info = [f"Symbol: {symbol}"]
                info.append(f"Direction: {pair.get('direction', 'NONE')} ({pair.get('confidence', 0)}%)")

                if pair.get('indicators_1h'):
                    current_1h = pair['indicators_1h'].get('current', {})
                    if current_1h:
                        info.append(f"1H: RSI={current_1h.get('rsi', 0):.1f}, Price=${current_1h.get('price', 0):.2f}")

                if pair.get('indicators_4h'):
                    current_4h = pair['indicators_4h'].get('current', {})
                    if current_4h:
                        info.append(f"4H: RSI={current_4h.get('rsi', 0):.1f}, Vol={current_4h.get('volume_ratio', 0):.2f}")

                if pair.get('indicators_1d'):
                    current_1d = pair['indicators_1d'].get('current', {})
                    if current_1d:
                        info.append(f"1D: RSI={current_1d.get('rsi', 0):.1f}")

                pairs_info.append('\n'.join(info))

            pairs_text = "\n---\n".join(pairs_info)

            limit_text = f"максимум {max_pairs} пар" if max_pairs else "без ограничения количества"
            user_prompt = (
                f"Проанализируй следующие {len(pairs_data)} торговых пар с компактными "
                f"multi-timeframe данными (1H/4H/1D) и выбери {limit_text} "
                f"с наилучшим потенциалом для swing trading:\n\n{pairs_text}\n\n"
                f"Верни ТОЛЬКО JSON в формате: {{\"selected_pairs\": [\"BTCUSDT\", \"ETHUSDT\"]}}"
            )

            logger.info(f"DeepSeek Stage 2: analyzing {len(pairs_data)} pairs (limit: {max_pairs}, data size: {len(pairs_text)} chars)")

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            if self.use_reasoning and self.is_reasoning_model:
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    reasoning = response.choices[0].message.reasoning_content
                    if reasoning:
                        logger.debug(f"DeepSeek reasoning (first 500 chars): {reasoning[:500]}")

            content = response.choices[0].message.content.strip()
            logger.debug(f"DeepSeek response (first 150 chars): {content[:150]}")

            # Parse JSON response
            selected = []

            try:
                # Remove markdown code blocks
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
                    if isinstance(symbol, str):
                        clean_symbol = symbol.strip().strip('"').strip("'").strip('[').strip(']').upper()
                        if clean_symbol and clean_symbol not in selected:
                            selected.append(clean_symbol)

            except json.JSONDecodeError:
                logger.warning("JSON parsing failed, using fallback")
                for line in content.split('\n'):
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('//'):
                        continue

                    tokens = line.replace(',', ' ').replace('"', ' ').replace("'", ' ').replace('[', ' ').replace(']', ' ').split()
                    for token in tokens:
                        token = token.strip().upper()
                        if 2 <= len(token) <= 15 and token not in selected:
                            if 'USDT' in token or token.replace('USDT', '').isalpha():
                                selected.append(token)

            # Apply limit
            if max_pairs:
                if len(selected) > max_pairs:
                    logger.debug(f"Trimming from {len(selected)} to {max_pairs} pairs")
                    selected = selected[:max_pairs]
                elif len(selected) == 0:
                    logger.warning("Model returned no pairs")

            logger.info(f"DeepSeek Stage 2 result: selected {len(selected)} pairs (limit: {max_pairs})")
            if selected:
                logger.debug(f"Selected pairs: {selected}")

            return selected

        except Exception as e:
            logger.error(f"DeepSeek pair selection error: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """Общий метод для чата с DeepSeek (используется для Stage 3)"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            if self.use_reasoning and self.is_reasoning_model:
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    reasoning = response.choices[0].message.reasoning_content
                    if reasoning:
                        logger.debug(f"DeepSeek reasoning (first 300 chars): {reasoning[:300]}")

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"DeepSeek chat error: {e}")
            raise