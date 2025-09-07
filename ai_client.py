"""
Упрощенный ИИ клиент для анализа торговых сигналов
Фокус на скорости и точности результатов
"""

import asyncio
import os
import json
import logging
from typing import Optional, List, Dict, Any
import httpx
from openai import AsyncOpenAI
from config import config

logger = logging.getLogger(__name__)


class AIClient:
    """Упрощенный ИИ клиент"""

    def __init__(self):
        self.client = None
        self.http_client = None
        self.selection_prompt = self._load_prompt(config.system.SELECTION_PROMPT)
        self.analysis_prompt = self._load_prompt(config.system.ANALYSIS_PROMPT)

    def _load_prompt(self, filename: str) -> str:
        """Загрузка промпта из файла"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
                logger.info(f"Loaded prompt from {filename}")
                return prompt
        except FileNotFoundError:
            logger.warning(f"Prompt file {filename} not found, using default")
            return "You are an expert crypto trader. Analyze the data and provide trading recommendations."

    async def _get_client(self):
        """Получение ИИ клиента"""
        if self.client is None:
            api_key = os.getenv(config.ai.API_KEY_ENV)
            if not api_key:
                raise ValueError(f"API key {config.ai.API_KEY_ENV} not found")

            if self.http_client is None:
                self.http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(config.ai.ANALYSIS_TIMEOUT),
                    limits=httpx.Limits(max_connections=5)
                )

            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=config.ai.BASE_URL,
                http_client=self.http_client
            )

        return self.client

    async def close(self):
        """Закрытие клиента"""
        if self.http_client and not self.http_client.is_closed:
            await self.http_client.aclose()

    async def select_pairs(self, signals: List[Dict]) -> List[str]:
        """
        Быстрый отбор лучших пар для анализа
        """
        if not signals:
            return []

        logger.info(f"AI selecting from {len(signals)} signals")

        # Подготавливаем данные для ИИ (только ключевая информация)
        selection_data = []
        for signal in signals[:config.ai.MAX_PAIRS_TO_AI]:
            selection_data.append({
                'pair': signal.get('pair'),
                'signal': signal.get('signal'),
                'confidence': signal.get('confidence'),
                'pattern': signal.get('pattern'),
                'volume_ratio': signal.get('volume_ratio', 1.0),
                'trend_strength': signal.get('trend_strength', 0)
            })

        message = f"""
{self.selection_prompt}

SIGNALS TO ANALYZE:
{json.dumps(selection_data, indent=2)}

Select maximum {config.ai.MAX_SELECTED_PAIRS} best pairs.
Return JSON: {{"pairs": ["BTCUSDT", "ETHUSDT"]}}
"""

        try:
            client = await self._get_client()
            response = await client.chat.completions.create(
                model=config.ai.MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert trader selecting the best trading opportunities."},
                    {"role": "user", "content": message}
                ],
                max_tokens=config.ai.SELECTION_TOKENS,
                temperature=config.ai.TEMPERATURE,
                timeout=config.ai.SELECTION_TIMEOUT
            )

            result = response.choices[0].message.content
            selected_pairs = self._parse_pairs(result)

            logger.info(f"AI selected {len(selected_pairs)} pairs: {selected_pairs}")
            return selected_pairs

        except Exception as e:
            logger.error(f"AI selection error: {e}")
            # Fallback - выбираем по confidence
            fallback = sorted(signals, key=lambda x: x.get('confidence', 0), reverse=True)
            return [s['pair'] for s in fallback[:config.ai.MAX_SELECTED_PAIRS]]

    async def analyze_pair(self, pair: str, candles_5m: List, candles_15m: List, indicators: Dict) -> Optional[str]:
        """
        Детальный анализ конкретной пары
        """
        logger.info(f"AI analyzing {pair}")

        # Подготавливаем полные данные для анализа
        analysis_data = {
            'pair': pair,
            'current_price': indicators.get('current_price'),
            'timeframes': {
                '5m_candles_count': len(candles_5m),
                '15m_candles_count': len(candles_15m)
            },
            'indicators': {
                'rsi': indicators.get('current_rsi'),
                'volume_ratio': indicators.get('volume_ratio'),
                'atr': indicators.get('current_atr'),
                'trend_htf': indicators.get('htf_trend', {}),
                'signal_data': indicators.get('signal_data', {})
            },
            'recent_candles_5m': [
                {
                    'timestamp': int(c[0]),
                    'open': float(c[1]),
                    'high': float(c[2]),
                    'low': float(c[3]),
                    'close': float(c[4]),
                    'volume': float(c[5])
                } for c in candles_5m[-20:]  # Последние 20 свечей для контекста
            ],
            'recent_candles_15m': [
                {
                    'timestamp': int(c[0]),
                    'open': float(c[1]),
                    'high': float(c[2]),
                    'low': float(c[3]),
                    'close': float(c[4]),
                    'volume': float(c[5])
                } for c in candles_15m[-10:]  # Последние 10 свечей для контекста
            ]
        }

        message = f"""
{self.analysis_prompt}

PAIR ANALYSIS DATA:
{json.dumps(analysis_data, indent=2)}

Provide detailed trading analysis and specific entry/exit recommendations.
"""

        try:
            client = await self._get_client()
            response = await client.chat.completions.create(
                model=config.ai.MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert crypto trader providing detailed analysis."},
                    {"role": "user", "content": message}
                ],
                max_tokens=config.ai.ANALYSIS_TOKENS,
                temperature=config.ai.TEMPERATURE,
                timeout=config.ai.ANALYSIS_TIMEOUT
            )

            result = response.choices[0].message.content
            logger.info(f"AI analysis completed for {pair}")
            return result

        except Exception as e:
            logger.error(f"AI analysis error for {pair}: {e}")
            return None

    def _parse_pairs(self, ai_response: str) -> List[str]:
        """Парсинг ответа ИИ для извлечения пар"""
        try:
            # Ищем JSON в ответе
            import re
            json_match = re.search(r'\{[^}]*"pairs"[^}]*\}', ai_response)
            if json_match:
                data = json.loads(json_match.group())
                pairs = data.get('pairs', [])
                return pairs[:config.ai.MAX_SELECTED_PAIRS]

            # Если JSON не найден, пробуем найти символы
            pairs = re.findall(r'[A-Z]{3,10}USDT', ai_response)
            return list(set(pairs))[:config.ai.MAX_SELECTED_PAIRS]

        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return []

    async def health_check(self) -> bool:
        """Проверка работоспособности ИИ"""
        try:
            client = await self._get_client()
            response = await client.chat.completions.create(
                model=config.ai.MODEL,
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=10,
                timeout=15
            )

            logger.info("AI health check passed")
            return True

        except Exception as e:
            logger.error(f"AI health check failed: {e}")
            return False


# Глобальный ИИ клиент
ai_client = AIClient()


async def select_best_pairs(signals: List[Dict]) -> List[str]:
    """Удобная функция для отбора пар"""
    return await ai_client.select_pairs(signals)


async def analyze_pair_detailed(pair: str, candles_5m: List, candles_15m: List, indicators: Dict) -> Optional[str]:
    """Удобная функция для анализа пары"""
    return await ai_client.analyze_pair(pair, candles_5m, candles_15m, indicators)


async def check_ai_health() -> bool:
    """Удобная функция для проверки ИИ"""
    return await ai_client.health_check()


async def cleanup_ai():
    """Очистка ресурсов ИИ"""
    await ai_client.close()