"""
Клиент для работы с Anthropic Claude API
"""

import asyncio
import json
import logging
from typing import List, Dict, Optional
import httpx
from config import config

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Асинхронный клиент для Anthropic Claude API"""

    def __init__(self):
        self.api_key = config.ANTHROPIC_API_KEY
        self.model = config.ANTHROPIC_MODEL
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.timeout = config.API_TIMEOUT

    async def _make_request(self, messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.7) -> Optional[
        str]:
        """Базовый запрос к Anthropic API"""
        if not self.api_key:
            raise ValueError("Anthropic API key не найден")

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )

                response.raise_for_status()
                data = response.json()

                return data["content"][0]["text"]

        except httpx.TimeoutException:
            logger.error("Таймаут запроса к Anthropic API")
            raise asyncio.TimeoutError("Anthropic API timeout")
        except httpx.HTTPError as e:
            logger.error(f"HTTP ошибка Anthropic API: {e}")
            raise
        except Exception as e:
            logger.error(f"Общая ошибка Anthropic API: {e}")
            raise

    def extract_json(self, text: str) -> Optional[Dict]:
        """Извлечение JSON из ответа Claude"""
        if not text:
            return None

        try:
            # Убираем markdown блоки
            if '```json' in text:
                start = text.find('```json') + 7
                end = text.find('```', start)
                if end != -1:
                    text = text[start:end].strip()
            elif '```' in text:
                # Ищем JSON между любыми ``` блоками
                parts = text.split('```')
                for part in parts:
                    try:
                        if part.strip().startswith('{'):
                            return json.loads(part.strip())
                    except:
                        continue

            # Поиск JSON объекта в тексте
            start_idx = text.find('{')
            if start_idx == -1:
                return None

            # Подсчет скобок для нахождения полного JSON
            brace_count = 0
            for i, char in enumerate(text[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = text[start_idx:i + 1]
                        return json.loads(json_str)

            return None

        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON от Claude: {e}")
            return None
        except Exception as e:
            logger.error(f"Общая ошибка извлечения JSON: {e}")
            return None

    async def select_pairs(self, pairs_data: List[Dict]) -> List[str]:
        """Отбор пар через Claude"""
        if not pairs_data:
            return []

        try:
            from deepseek import load_prompt_cached

            # Подготавливаем компактные данные
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
            data_json = json.dumps(compact_data, separators=(',', ':'))

            messages = [
                {
                    "role": "user",
                    "content": f"{prompt}\n\nДанные для анализа:\n{data_json}\n\nВерни JSON с выбранными парами."
                }
            ]

            response_text = await self._make_request(
                messages=messages,
                max_tokens=config.AI_MAX_TOKENS_SELECT,
                temperature=config.AI_TEMPERATURE_SELECT
            )

            result = self.extract_json(response_text)
            if result and 'selected_pairs' in result:
                selected_pairs = result['selected_pairs'][:config.MAX_FINAL_PAIRS]
                logger.info(f"Claude выбрал {len(selected_pairs)} пар")
                return selected_pairs

            logger.warning("Claude не вернул корректный результат отбора")
            return []

        except Exception as e:
            logger.error(f"Ошибка отбора пар через Claude: {e}")
            return []

    async def analyze_pair(self, symbol: str, data_5m: List, data_15m: List,
                           indicators_5m: Dict, indicators_15m: Dict) -> Dict:
        """Анализ пары через Claude"""
        try:
            from deepseek import load_prompt_cached

            current_price = indicators_5m.get('current', {}).get('price', 0)
            if current_price <= 0:
                return self._create_fallback_analysis(symbol, current_price)

            analysis_data = {
                'symbol': symbol,
                'current_price': current_price,
                'timeframes': {
                    '5m': {
                        'candles': data_5m[-80:],
                        'indicators': {
                            'ema5': indicators_5m.get('ema5_history', [])[-80:],
                            'ema8': indicators_5m.get('ema8_history', [])[-80:],
                            'ema20': indicators_5m.get('ema20_history', [])[-80:],
                            'rsi': indicators_5m.get('rsi_history', [])[-80:],
                            'macd_histogram': indicators_5m.get('macd_histogram_history', [])[-80:],
                            'volume_ratio': indicators_5m.get('volume_ratio_history', [])[-80:]
                        }
                    },
                    '15m': {
                        'candles': data_15m[-40:],
                        'indicators': {
                            'ema5': indicators_15m.get('ema5_history', [])[-40:],
                            'ema8': indicators_15m.get('ema8_history', [])[-40:],
                            'ema20': indicators_15m.get('ema20_history', [])[-40:],
                            'rsi': indicators_15m.get('rsi_history', [])[-40:],
                            'macd_histogram': indicators_15m.get('macd_histogram_history', [])[-40:]
                        }
                    }
                },
                'current_state': {
                    'price': current_price,
                    'atr': indicators_5m.get('current', {}).get('atr', 0),
                    'trend_5m': 'UP' if indicators_5m.get('current', {}).get('ema5', 0) > indicators_5m.get('current',
                                                                                                            {}).get(
                        'ema20', 0) else 'DOWN',
                    'trend_15m': 'UP' if indicators_15m.get('current', {}).get('ema5', 0) > indicators_15m.get(
                        'current', {}).get('ema20', 0) else 'DOWN',
                    'rsi_5m': indicators_5m.get('current', {}).get('rsi', 50),
                    'rsi_15m': indicators_15m.get('current', {}).get('rsi', 50),
                    'volume_ratio': indicators_5m.get('current', {}).get('volume_ratio', 1.0),
                    'macd_momentum': indicators_5m.get('current', {}).get('macd_histogram', 0)
                }
            }

            prompt = load_prompt_cached(config.ANALYSIS_PROMPT)
            data_json = json.dumps(analysis_data, separators=(',', ':'))

            messages = [
                {
                    "role": "user",
                    "content": f"{prompt}\n\nДанные для анализа:\n{data_json}\n\nВерни JSON с анализом."
                }
            ]

            response_text = await self._make_request(
                messages=messages,
                max_tokens=config.AI_MAX_TOKENS_ANALYZE,
                temperature=config.AI_TEMPERATURE_ANALYZE
            )

            result = self.extract_json(response_text)

            if result:
                signal = str(result.get('signal', 'NO_SIGNAL')).upper()
                confidence = max(0, min(100, int(result.get('confidence', 0))))
                entry_price = float(result.get('entry_price', current_price))
                stop_loss = float(result.get('stop_loss', 0))
                take_profit = float(result.get('take_profit', 0))
                analysis = str(result.get('analysis', 'Анализ от Claude'))

                # Валидация уровней
                if signal in ['LONG', 'SHORT'] and entry_price > 0:
                    if stop_loss <= 0:
                        stop_loss = entry_price * (0.98 if signal == 'LONG' else 1.02)
                    if take_profit <= 0:
                        risk = abs(entry_price - stop_loss)
                        take_profit = entry_price + (risk * 2 if signal == 'LONG' else -risk * 2)

                return {
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': confidence,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'analysis': analysis,
                    'ai_generated': True
                }

            return self._create_fallback_analysis(symbol, current_price)

        except Exception as e:
            logger.error(f"Ошибка анализа {symbol} через Claude: {e}")
            return self._create_fallback_analysis(symbol, indicators_5m.get('current', {}).get('price', 0))

    async def validate_signals(self, preliminary_signals: List[Dict], market_data: Dict) -> List[Dict]:
        """Валидация сигналов через Claude"""
        if not preliminary_signals:
            return []

        try:
            from deepseek import load_prompt_cached

            validation_data = {
                'preliminary_signals': preliminary_signals,
                'market_data': market_data
            }

            prompt = load_prompt_cached(config.VALIDATION_PROMPT)
            data_json = json.dumps(validation_data, separators=(',', ':'))

            messages = [
                {
                    "role": "user",
                    "content": f"{prompt}\n\nДанные для валидации:\n{data_json}\n\nВерни JSON с результатами валидации."
                }
            ]

            response_text = await self._make_request(
                messages=messages,
                max_tokens=config.AI_MAX_TOKENS_VALIDATE,
                temperature=config.AI_TEMPERATURE_VALIDATE
            )

            result = self.extract_json(response_text)

            if result and 'final_signals' in result:
                final_signals = result.get('final_signals', [])
                rejected_count = len(result.get('rejected_signals', []))
                logger.info(f"Claude валидация: подтверждено {len(final_signals)}, отклонено {rejected_count}")
                return final_signals

            logger.warning("Claude не вернул корректный результат валидации")
            return self._create_fallback_validation(preliminary_signals)

        except Exception as e:
            logger.error(f"Ошибка валидации через Claude: {e}")
            return self._create_fallback_validation(preliminary_signals)

    def _create_fallback_analysis(self, symbol: str, current_price: float) -> Dict:
        """Fallback анализ"""
        return {
            'symbol': symbol,
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'entry_price': current_price,
            'stop_loss': 0,
            'take_profit': 0,
            'analysis': 'Claude анализ недоступен - fallback режим',
            'ai_generated': False
        }

    def _create_fallback_validation(self, preliminary_signals: List[Dict]) -> List[Dict]:
        """Fallback валидация"""
        validated_signals = []

        for signal in preliminary_signals:
            entry = signal.get('entry_price', 0)
            stop = signal.get('stop_loss', 0)
            profit = signal.get('take_profit', 0)

            if entry > 0 and stop > 0 and profit > 0:
                risk = abs(entry - stop)
                reward = abs(profit - entry)

                if risk > 0:
                    rr_ratio = round(reward / risk, 2)
                    if rr_ratio >= config.MIN_RISK_REWARD_RATIO:
                        validated_signal = signal.copy()
                        validated_signal.update({
                            'risk_reward_ratio': rr_ratio,
                            'hold_duration_minutes': 30,
                            'validation_notes': f'Fallback валидация: R/R {rr_ratio}',
                            'action': 'APPROVED',
                            'market_conditions': 'Автоматическая оценка',
                            'key_levels': f'Вход: {entry}, Стоп: {stop}, Профит: {profit}'
                        })
                        validated_signals.append(validated_signal)

        logger.info(f"Fallback валидация Claude: подтверждено {len(validated_signals)} сигналов")
        return validated_signals


# Создаем глобальный экземпляр
anthropic_client = AnthropicClient()