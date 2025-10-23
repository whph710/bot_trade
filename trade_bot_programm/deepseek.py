"""
DeepSeek AI клиент - FIXED: Stage 2 compact multi-TF support + max_pairs enforcement
Файл: trade_bot_programm/deepseek.py
"""

import os
import json
from openai import AsyncOpenAI
from typing import Optional, Dict, List
from pathlib import Path


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
                print(f"[DeepSeek] 📄 Промпт загружен: {path.name} ({len(content)} символов)")
                return content

    error_msg = f"Промпт файл '{prompt_file}' не найден. Искал в:\n"
    for path in search_paths:
        error_msg += f"  - {path.absolute()}\n"
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
            raise ValueError("DEEPSEEK_API_KEY не задан в .env")

        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        reasoning_env = os.getenv("DEEPSEEK_REASONING", "false").lower()
        self.use_reasoning = use_reasoning if use_reasoning is not None else (reasoning_env in ["true", "1", "yes"])

        self.is_reasoning_model = "reasoner" in self.model.lower() or self.model == "deepseek-reasoner"

        if self.use_reasoning and not self.is_reasoning_model:
            print(f"[DeepSeek] ⚠️  ВНИМАНИЕ: DEEPSEEK_REASONING=true, но модель {self.model} не поддерживает reasoning")
            print(f"[DeepSeek] 🔄 Автоматически отключаем reasoning режим")
            self.use_reasoning = False

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url
        )

        self.prompts_cache: Dict[str, str] = {}

        print(f"[DeepSeek] ╔{'═'*60}╗")
        print(f"[DeepSeek] ║ {'ИНИЦИАЛИЗАЦИЯ DEEPSEEK':^60} ║")
        print(f"[DeepSeek] ╠{'═'*60}╣")
        print(f"[DeepSeek] ║ Модель: {self.model:<49} ║")
        print(f"[DeepSeek] ║ Reasoning модель: {'Да' if self.is_reasoning_model else 'Нет':<43} ║")
        print(f"[DeepSeek] ║ Reasoning режим: {'✅ Включен' if self.use_reasoning else '❌ Выключен':<44} ║")
        print(f"[DeepSeek] ║ Base URL: {base_url:<47} ║")
        print(f"[DeepSeek] ╚{'═'*60}╝")

    def _load_prompt(self, prompt_file: str) -> str:
        """Загружает промпт из файла с кэшированием"""
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

                # Компактная информация из multi-TF данных
                info = [f"Symbol: {symbol}"]
                info.append(f"Direction: {pair.get('direction', 'NONE')} ({pair.get('confidence', 0)}%)")

                # 1H текущие значения
                if pair.get('indicators_1h'):
                    current_1h = pair['indicators_1h'].get('current', {})
                    if current_1h:
                        info.append(f"1H: RSI={current_1h.get('rsi', 0):.1f}, Price=${current_1h.get('price', 0):.2f}")

                # 4H текущие значения
                if pair.get('indicators_4h'):
                    current_4h = pair['indicators_4h'].get('current', {})
                    if current_4h:
                        info.append(f"4H: RSI={current_4h.get('rsi', 0):.1f}, Vol={current_4h.get('volume_ratio', 0):.2f}")

                # 1D текущие значения (если есть)
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

            print(f"\n[DeepSeek] {'─'*60}")
            print(f"[DeepSeek] 🎯 STAGE 2: ВЫБОР ПАР (COMPACT MULTI-TF)")
            print(f"[DeepSeek] {'─'*60}")
            print(f"[DeepSeek] 📊 Пар на анализ: {len(pairs_data)}")
            print(f"[DeepSeek] 🎚️  Лимит выбора: {limit_text}")
            print(f"[DeepSeek] 📏 Размер данных: {len(pairs_text)} символов")
            print(f"[DeepSeek] 🤖 Модель: {self.model}")
            print(f"[DeepSeek] 🌡️  Temperature: {temperature}")
            print(f"[DeepSeek] 🎫 Max tokens: {max_tokens}")
            print(f"[DeepSeek] 💭 Reasoning: {'✅' if self.use_reasoning else '❌'}")

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
                        print(f"\n[DeepSeek] {'='*60}")
                        print(f"[DeepSeek] 💭 РАССУЖДЕНИЯ МОДЕЛИ (первые 800 символов):")
                        print(f"[DeepSeek] {'='*60}")
                        print(f"{reasoning[:800]}...")
                        print(f"[DeepSeek] {'='*60}\n")

            content = response.choices[0].message.content.strip()

            print(f"[DeepSeek] 📝 Ответ модели (первые 200 символов):")
            print(f"[DeepSeek]    {content[:200]}...")

            # Парсим JSON response
            selected = []

            try:
                # Удаляем markdown блоки если есть
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

                # Парсим JSON
                data = json.loads(content)
                selected_pairs = data.get('selected_pairs', [])

                # Очищаем от лишних символов
                for symbol in selected_pairs:
                    if isinstance(symbol, str):
                        clean_symbol = symbol.strip().strip('"').strip("'").strip('[').strip(']').upper()
                        if clean_symbol and clean_symbol not in selected:
                            selected.append(clean_symbol)

            except json.JSONDecodeError:
                # Fallback: ищем символы вручную
                print(f"[DeepSeek] ⚠️  JSON parsing failed, using fallback")
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

            # КРИТИЧНО: Применяем лимит СТРОГО
            if max_pairs:
                if len(selected) > max_pairs:
                    print(f"[DeepSeek] ⚠️  Обрезаем с {len(selected)} до {max_pairs} пар")
                    selected = selected[:max_pairs]
                elif len(selected) == 0:
                    print(f"[DeepSeek] ⚠️  Модель не вернула пары - попробуйте снова")

            print(f"\n[DeepSeek] {'='*60}")
            print(f"[DeepSeek] ✅ РЕЗУЛЬТАТ: Выбрано {len(selected)} пар (лимит: {max_pairs if max_pairs else 'нет'})")
            if selected:
                print(f"[DeepSeek] 📋 Список: {selected}")
            print(f"[DeepSeek] {'='*60}\n")

            return selected

        except Exception as e:
            print(f"\n[DeepSeek] ❌ ОШИБКА при выборе пар: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """Общий метод для чата с DeepSeek"""
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
                        print(f"[DeepSeek] 💭 Рассуждения модели:")
                        print(f"      {reasoning[:500]}...")

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[DeepSeek] ❌ Ошибка чата: {e}")
            raise