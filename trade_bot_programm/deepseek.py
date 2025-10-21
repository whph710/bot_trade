"""
DeepSeek AI клиент с полной поддержкой reasoning режима через .env
FIXED: Изменен 'ticker' на 'symbol' для совместимости с ai_router.py
"""

import os
from openai import AsyncOpenAI
from typing import Optional, Dict, List
from pathlib import Path


class DeepSeekClient:
    """Клиент для работы с DeepSeek API (совместим с OpenAI SDK)"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        use_reasoning: Optional[bool] = None,
        base_url: str = "https://api.deepseek.com"
    ):
        """
        Инициализация клиента DeepSeek

        Args:
            api_key: API ключ (если None, берется из DEEPSEEK_API_KEY)
            model: Название модели (если None, берется из DEEPSEEK_MODEL)
            use_reasoning: Использовать reasoning (если None, берется из DEEPSEEK_REASONING)
            base_url: Базовый URL API
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY не задан в .env")

        # Определяем модель из env или используем по умолчанию
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        # Определяем режим reasoning из env
        reasoning_env = os.getenv("DEEPSEEK_REASONING", "false").lower()
        self.use_reasoning = use_reasoning if use_reasoning is not None else (reasoning_env in ["true", "1", "yes"])

        # Проверяем, является ли модель reasoning моделью
        self.is_reasoning_model = "reasoner" in self.model.lower() or self.model == "deepseek-reasoner"

        # Предупреждение если режим reasoning включен для non-reasoning модели
        if self.use_reasoning and not self.is_reasoning_model:
            print(f"[DeepSeek] ⚠️  ВНИМАНИЕ: DEEPSEEK_REASONING=true, но модель {self.model} не поддерживает reasoning")
            print(f"[DeepSeek] ⚠️  Используйте DEEPSEEK_MODEL=deepseek-reasoner для reasoning режима")
            print(f"[DeepSeek] 🔄 Автоматически отключаем reasoning режим")
            self.use_reasoning = False

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url
        )

        self.prompts_cache: Dict[str, str] = {}

        # Логируем конфигурацию
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

        prompts_dir = Path(__file__).parent.parent / "prompts"
        prompt_path = prompts_dir / prompt_file

        if not prompt_path.exists():
            raise FileNotFoundError(f"Промпт файл не найден: {prompt_path}")

        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.prompts_cache[prompt_file] = content
        print(f"[DeepSeek] 📄 Промпт закэширован: {prompt_file} ({len(content)} символов)")

        return content

    async def select_pairs(
        self,
        pairs_data: List[Dict],
        max_pairs: Optional[int] = None,
        system_prompt_file: str = "prompt_select.txt",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> List[str]:
        """
        Выбирает лучшие торговые пары из предложенных
        FIXED: Использует 'symbol' вместо 'ticker'

        Args:
            pairs_data: Список данных о парах
            max_pairs: Максимальное количество пар для выбора
            system_prompt_file: Файл с системным промптом
            temperature: Температура (из .env AI_TEMPERATURE_SELECT если None)
            max_tokens: Максимум токенов (из .env AI_MAX_TOKENS_SELECT если None)

        Returns:
            Список выбранных тикеров
        """
        try:
            # Параметры из env
            if temperature is None:
                temperature = float(os.getenv("AI_TEMPERATURE_SELECT", "0.3"))
            if max_tokens is None:
                max_tokens = int(os.getenv("AI_MAX_TOKENS_SELECT", "2000"))

            # Загружаем системный промпт
            system_prompt = self._load_prompt(system_prompt_file)

            # FIXED: Формируем данные о парах (используем 'symbol')
            pairs_info = []
            for pair in pairs_data:
                symbol = pair.get('symbol', 'UNKNOWN')  # ← ИСПРАВЛЕНО

                info = (
                    f"Symbol: {symbol}\n"
                    f"Direction: {pair.get('direction', 'NONE')} ({pair.get('confidence', 0)}%)\n"
                )

                # Добавляем дополнительные данные если есть
                if pair.get('candles_15m'):
                    info += f"Candles: {len(pair['candles_15m'])} bars\n"

                if pair.get('indicators_15m'):
                    current = pair['indicators_15m'].get('current', {})
                    if current:
                        info += f"Price: ${current.get('price', 0):.4f}\n"
                        info += f"RSI: {current.get('rsi', 0):.1f}\n"
                        info += f"Volume ratio: {current.get('volume_ratio', 0):.2f}\n"

                pairs_info.append(info)

            pairs_text = "\n---\n".join(pairs_info)

            # Формируем user промпт
            limit_text = f"максимум {max_pairs} пар" if max_pairs else "без ограничения количества"
            user_prompt = (
                f"Проанализируй следующие {len(pairs_data)} торговых пар и выбери {limit_text} "
                f"с наилучшим потенциалом для торговли:\n\n{pairs_text}\n\n"
                f"Верни ТОЛЬКО список символов через запятую (например: BTCUSDT, ETHUSDT, SOLUSDT)"
            )

            print(f"\n[DeepSeek] {'─'*60}")
            print(f"[DeepSeek] 🎯 STAGE 2: ВЫБОР ПАР")
            print(f"[DeepSeek] {'─'*60}")
            print(f"[DeepSeek] 📊 Пар на анализ: {len(pairs_data)}")
            print(f"[DeepSeek] 🎚️  Лимит выбора: {limit_text}")
            print(f"[DeepSeek] 📏 Размер данных: {len(pairs_text)} символов")
            print(f"[DeepSeek] 🤖 Модель: {self.model}")
            print(f"[DeepSeek] 🌡️  Temperature: {temperature}")
            print(f"[DeepSeek] 🎫 Max tokens: {max_tokens}")
            print(f"[DeepSeek] 💭 Reasoning: {'✅' if self.use_reasoning else '❌'}")

            # Вызываем API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Показываем reasoning для reasoning моделей
            if self.use_reasoning and self.is_reasoning_model:
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    reasoning = response.choices[0].message.reasoning_content
                    if reasoning:
                        print(f"\n[DeepSeek] {'='*60}")
                        print(f"[DeepSeek] 💭 РАССУЖДЕНИЯ МОДЕЛИ (первые 800 символов):")
                        print(f"[DeepSeek] {'='*60}")
                        print(f"{reasoning[:800]}...")
                        print(f"[DeepSeek] {'='*60}\n")

            # Извлекаем ответ
            content = response.choices[0].message.content.strip()

            print(f"[DeepSeek] 📝 Ответ модели (первые 200 символов):")
            print(f"[DeepSeek]    {content[:200]}...")

            # Парсим список тикеров
            selected = []
            for line in content.split('\n'):
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('//'):
                    continue

                # Ищем тикеры
                tokens = line.replace(',', ' ').split()
                for token in tokens:
                    token = token.strip().upper()
                    if 2 <= len(token) <= 15 and token not in selected:
                        # Проверяем что это похоже на криптопару
                        if 'USDT' in token or token.replace('USDT', '').isalpha():
                            selected.append(token)

            # Применяем лимит
            if max_pairs and len(selected) > max_pairs:
                selected = selected[:max_pairs]

            print(f"\n[DeepSeek] {'='*60}")
            print(f"[DeepSeek] ✅ РЕЗУЛЬТАТ: Выбрано {len(selected)} пар")
            if selected:
                print(f"[DeepSeek] 📋 Список: {', '.join(selected)}")
            print(f"[DeepSeek] {'='*60}\n")

            return selected

        except Exception as e:
            print(f"\n[DeepSeek] ❌ ОШИБКА при выборе пар: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def analyze_pair(
        self,
        pair_data: Dict,
        analysis_prompt_file: str = "prompt_analyze.txt",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Анализирует конкретную торговую пару
        FIXED: Использует 'symbol' вместо 'ticker'

        Args:
            pair_data: Данные о паре для анализа
            analysis_prompt_file: Файл с промптом для анализа
            temperature: Температура (из .env AI_TEMPERATURE_ANALYZE если None)
            max_tokens: Максимум токенов (из .env AI_MAX_TOKENS_ANALYZE если None)

        Returns:
            Словарь с результатами анализа или None при ошибке
        """
        try:
            # Параметры из env
            if temperature is None:
                temperature = float(os.getenv("AI_TEMPERATURE_ANALYZE", "0.7"))
            if max_tokens is None:
                max_tokens = int(os.getenv("AI_MAX_TOKENS_ANALYZE", "3000"))

            system_prompt = self._load_prompt(analysis_prompt_file)

            # FIXED: Используем 'symbol'
            symbol = pair_data.get('symbol', 'UNKNOWN')

            # Формируем данные о паре
            pair_info = f"Symbol: {symbol}\n"

            if pair_data.get('candles_15m'):
                pair_info += f"Candles: {len(pair_data['candles_15m'])} bars\n"

            if pair_data.get('indicators_15m'):
                current = pair_data['indicators_15m'].get('current', {})
                if current:
                    pair_info += f"Price: ${current.get('price', 0):.4f}\n"
                    pair_info += f"RSI: {current.get('rsi', 0):.1f}\n"

            user_prompt = f"Проанализируй следующую торговую пару:\n\n{pair_info}"

            print(f"\n[DeepSeek] 🔬 Анализ пары: {symbol}")
            print(f"[DeepSeek] 🌡️  Temperature: {temperature}, Max tokens: {max_tokens}")

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Показываем reasoning
            if self.use_reasoning and self.is_reasoning_model:
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    reasoning = response.choices[0].message.reasoning_content
                    if reasoning:
                        print(f"[DeepSeek] 💭 Рассуждения (первые 300 символов):")
                        print(f"      {reasoning[:300]}...")

            content = response.choices[0].message.content.strip()

            print(f"[DeepSeek] ✅ Анализ завершен для {symbol}")

            return {
                'symbol': symbol,  # ← ИСПРАВЛЕНО
                'analysis': content,
                'model': self.model,
                'reasoning_used': self.use_reasoning and self.is_reasoning_model
            }

        except Exception as e:
            print(f"[DeepSeek] ❌ Ошибка при анализе {pair_data.get('symbol', 'unknown')}: {e}")
            return None

    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """
        Общий метод для чата с DeepSeek

        Args:
            messages: Список сообщений в формате OpenAI
            max_tokens: Максимум токенов в ответе
            temperature: Температура генерации

        Returns:
            Ответ модели
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Показываем reasoning
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