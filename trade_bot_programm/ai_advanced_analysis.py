"""
AI Router - маршрутизатор запросов к AI моделям
Поддерживает DeepSeek (chat и reasoner)
"""

import os
from typing import List, Dict, Optional
from deepseek import DeepSeekClient


class AIRouter:
    """Маршрутизатор для работы с разными AI провайдерами"""

    def __init__(self):
        """Инициализация роутера"""
        self.deepseek_client: Optional[DeepSeekClient] = None
        self.default_provider = "deepseek"

        # Загружаем конфигурацию из env
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        print(f"[AI Router] Инициализация:")
        print(f"  • Провайдер по умолчанию: {self.default_provider}")
        print(f"  • DeepSeek модель: {self.deepseek_model}")

    async def initialize_deepseek(self):
        """Инициализирует DeepSeek клиент"""
        if not self.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY не задан в .env")

        try:
            self.deepseek_client = DeepSeekClient(
                api_key=self.deepseek_api_key,
                model=self.deepseek_model
            )
            print(f"[AI Router] ✅ DeepSeek клиент инициализирован")
            return True
        except Exception as e:
            print(f"[AI Router] ❌ Ошибка инициализации DeepSeek: {e}")
            return False

    async def select_pairs(
        self,
        pairs_data: List[Dict],
        max_pairs: Optional[int] = None,
        provider: str = None
    ) -> List[str]:
        """
        Выбирает лучшие пары через AI

        Args:
            pairs_data: Данные о парах
            max_pairs: Максимальное количество пар
            provider: Провайдер (по умолчанию - deepseek)

        Returns:
            Список выбранных тикеров
        """
        provider = provider or self.default_provider

        print(f"[AI Router] Выбор из {len(pairs_data)} пар через {provider}")
        print(f"[AI Router] Лимит: {max_pairs if max_pairs else 'без ограничений'}")

        if provider == "deepseek":
            if not self.deepseek_client:
                await self.initialize_deepseek()

            if not self.deepseek_client:
                print(f"[AI Router] ❌ DeepSeek клиент недоступен")
                return []

            try:
                selected = await self.deepseek_client.select_pairs(
                    pairs_data=pairs_data,
                    max_pairs=max_pairs
                )

                print(f"[AI Router] ✅ {provider} выбрал {len(selected)} пар")
                return selected

            except Exception as e:
                print(f"[AI Router] ❌ Ошибка при выборе через {provider}: {e}")
                import traceback
                traceback.print_exc()
                return []
        else:
            print(f"[AI Router] ❌ Неизвестный провайдер: {provider}")
            return []

    async def analyze_pair(
        self,
        pair_data: Dict,
        provider: str = None
    ) -> Optional[Dict]:
        """
        Анализирует конкретную пару через AI

        Args:
            pair_data: Данные о паре
            provider: Провайдер (по умолчанию - deepseek)

        Returns:
            Результаты анализа
        """
        provider = provider or self.default_provider

        print(f"[AI Router] Анализ пары {pair_data.get('ticker', 'unknown')} через {provider}")

        if provider == "deepseek":
            if not self.deepseek_client:
                await self.initialize_deepseek()

            if not self.deepseek_client:
                print(f"[AI Router] ❌ DeepSeek клиент недоступен")
                return None

            try:
                result = await self.deepseek_client.analyze_pair(pair_data)

                if result:
                    print(f"[AI Router] ✅ Анализ завершен для {pair_data.get('ticker')}")

                return result

            except Exception as e:
                print(f"[AI Router] ❌ Ошибка при анализе через {provider}: {e}")
                return None
        else:
            print(f"[AI Router] ❌ Неизвестный провайдер: {provider}")
            return None

    async def chat(
        self,
        messages: List[Dict[str, str]],
        provider: str = None,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> Optional[str]:
        """
        Общий метод для чата с AI

        Args:
            messages: Сообщения в формате OpenAI
            provider: Провайдер
            max_tokens: Максимум токенов
            temperature: Температура

        Returns:
            Ответ модели
        """
        provider = provider or self.default_provider

        if provider == "deepseek":
            if not self.deepseek_client:
                await self.initialize_deepseek()

            if not self.deepseek_client:
                return None

            try:
                response = await self.deepseek_client.chat(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response
            except Exception as e:
                print(f"[AI Router] ❌ Ошибка чата: {e}")
                return None
        else:
            print(f"[AI Router] ❌ Неизвестный провайдер: {provider}")
            return None

    def get_available_providers(self) -> List[str]:
        """Возвращает список доступных провайдеров"""
        providers = []

        if self.deepseek_api_key:
            providers.append("deepseek")

        return providers

    def get_current_config(self) -> Dict:
        """Возвращает текущую конфигурацию"""
        return {
            'default_provider': self.default_provider,
            'available_providers': self.get_available_providers(),
            'deepseek_model': self.deepseek_model,
            'deepseek_initialized': self.deepseek_client is not None
        }