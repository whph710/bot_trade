"""
Маршрутизатор для выбора AI провайдера на каждом этапе
ОБНОВЛЕНО: Добавлен метод analyze_pair_comprehensive для Stage 3
"""

import logging
from typing import List, Dict, Optional
from config import config, get_available_ai_providers

logger = logging.getLogger(__name__)

# Импорты AI клиентов
try:
    from deepseek import (
        ai_select_pairs_deepseek,
        ai_analyze_pair_deepseek,
        ai_validate_signals_deepseek,
        create_fallback_selection,
        create_fallback_validation,
        create_fallback_analysis
    )
except ImportError as e:
    logger.error(f"Ошибка импорта DeepSeek: {e}")

try:
    from anthropic_ai import anthropic_client
except ImportError as e:
    logger.error(f"Ошибка импорта Anthropic: {e}")


class AIRouter:
    """Маршрутизатор AI провайдеров для разных этапов"""

    def __init__(self):
        self.providers = get_available_ai_providers()
        logger.info(f"Доступные AI провайдеры: {[k for k, v in self.providers.items() if v]}")

    def _get_provider_for_stage(self, stage: str) -> str:
        """Получить провайдера для конкретного этапа с fallback"""
        stage_mapping = {
            'selection': config.AI_STAGE_SELECTION,
            'analysis': config.AI_STAGE_ANALYSIS,
            'validation': config.AI_STAGE_VALIDATION
        }

        preferred_provider = stage_mapping.get(stage, 'fallback')

        if preferred_provider != 'fallback' and self.providers.get(preferred_provider, False):
            return preferred_provider

        for provider in ['deepseek', 'anthropic']:
            if self.providers.get(provider, False):
                logger.warning(f"Этап {stage}: переключение с {preferred_provider} на {provider}")
                return provider

        logger.warning(f"Этап {stage}: используется fallback режим")
        return 'fallback'

    async def call_ai(
            self,
            prompt: str,
            stage: str = 'analysis',
            max_tokens: int = 2000,
            temperature: float = 0.7
    ) -> str:
        """
        Универсальный метод для вызова AI с произвольным промптом
        """
        provider = self._get_provider_for_stage(stage)
        logger.debug(f"call_ai через {provider}")

        try:
            if provider == 'deepseek':
                import asyncio
                from openai import AsyncOpenAI

                client = AsyncOpenAI(
                    api_key=config.DEEPSEEK_API_KEY,
                    base_url=config.DEEPSEEK_URL
                )

                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=config.DEEPSEEK_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature
                    ),
                    timeout=config.API_TIMEOUT
                )

                return response.choices[0].message.content

            elif provider == 'anthropic':
                messages = [{"role": "user", "content": prompt}]
                return await anthropic_client._make_request(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

            else:
                logger.warning("call_ai: fallback mode, returning empty response")
                return "{}"

        except Exception as e:
            logger.error(f"Ошибка call_ai через {provider}: {e}")
            return "{}"

    async def select_pairs(self, pairs_data: List[Dict]) -> List[str]:
        """Отбор пар через выбранный AI провайдер"""
        provider = self._get_provider_for_stage('selection')
        logger.info(f"Этап отбора: используется {provider}")

        try:
            if provider == 'deepseek':
                return await ai_select_pairs_deepseek(pairs_data)
            elif provider == 'anthropic':
                return await anthropic_client.select_pairs(pairs_data)
            else:
                return create_fallback_selection(pairs_data, config.MAX_FINAL_PAIRS)

        except Exception as e:
            logger.error(f"Ошибка отбора пар через {provider}: {e}")
            logger.info("Переключение на fallback режим")
            return create_fallback_selection(pairs_data, config.MAX_FINAL_PAIRS)

    async def analyze_pair(self, symbol: str, data_5m: List, data_15m: List,
                           indicators_5m: Dict, indicators_15m: Dict) -> Dict:
        """Анализ пары через выбранный AI провайдер (старый метод)"""
        provider = self._get_provider_for_stage('analysis')
        logger.debug(f"Анализ {symbol}: используется {provider}")

        try:
            if provider == 'deepseek':
                return await ai_analyze_pair_deepseek(symbol, data_5m, data_15m, indicators_5m, indicators_15m)
            elif provider == 'anthropic':
                return await anthropic_client.analyze_pair(symbol, data_5m, data_15m, indicators_5m, indicators_15m)
            else:
                current_price = indicators_5m.get('current', {}).get('price', 0)
                return create_fallback_analysis(symbol, indicators_5m)

        except Exception as e:
            logger.error(f"Ошибка анализа {symbol} через {provider}: {e}")
            logger.info("Переключение на fallback режим")
            current_price = indicators_5m.get('current', {}).get('price', 0)
            return create_fallback_analysis(symbol, indicators_5m)

    async def analyze_pair_comprehensive(
            self,
            symbol: str,
            comprehensive_data: Dict
    ) -> Dict:
        """
        НОВЫЙ метод для Stage 3 - анализ с ПОЛНЫМИ данными

        Args:
            symbol: торговая пара
            comprehensive_data: все собранные данные включая market_data, correlations, VP, etc.

        Returns:
            Полный результат анализа с сигналом
        """
        provider = self._get_provider_for_stage('analysis')
        logger.debug(f"Comprehensive анализ {symbol}: используется {provider}")

        try:
            # Извлекаем базовые данные
            candles_1h = comprehensive_data.get('candles_1h', [])
            candles_4h = comprehensive_data.get('candles_4h', [])
            indicators_1h = comprehensive_data.get('indicators_1h', {})
            indicators_4h = comprehensive_data.get('indicators_4h', {})
            current_price = comprehensive_data.get('current_price', 0)

            # Извлекаем расширенные данные
            market_data = comprehensive_data.get('market_data', {})
            corr_data = comprehensive_data.get('correlation_data', {})
            vp_data = comprehensive_data.get('volume_profile', {})
            orderflow_ai = comprehensive_data.get('orderflow_ai', {})
            smc_ai = comprehensive_data.get('smc_ai', {})

            # Формируем КОМПАКТНЫЙ пакет для AI (без избыточных данных)
            ai_input = {
                'symbol': symbol,
                'current_price': current_price,
                'timeframes': {
                    '1h': {
                        'candles': candles_1h[-80:],  # Последние 80 свечей
                        'indicators': {
                            'ema5': indicators_1h.get('ema5_history', [])[-80:],
                            'ema8': indicators_1h.get('ema8_history', [])[-80:],
                            'ema20': indicators_1h.get('ema20_history', [])[-80:],
                            'rsi': indicators_1h.get('rsi_history', [])[-80:],
                            'macd_histogram': indicators_1h.get('macd_histogram_history', [])[-80:],
                            'volume_ratio': indicators_1h.get('volume_ratio_history', [])[-80:]
                        }
                    },
                    '4h': {
                        'candles': candles_4h[-40:],  # Последние 40 свечей
                        'indicators': {
                            'ema5': indicators_4h.get('ema5_history', [])[-40:],
                            'ema8': indicators_4h.get('ema8_history', [])[-40:],
                            'ema20': indicators_4h.get('ema20_history', [])[-40:],
                            'rsi': indicators_4h.get('rsi_history', [])[-40:],
                            'macd_histogram': indicators_4h.get('macd_histogram_history', [])[-40:]
                        }
                    }
                },
                'current_state': {
                    'price': current_price,
                    'atr': indicators_1h.get('current', {}).get('atr', 0),
                    'trend_1h': self._determine_trend(indicators_1h),
                    'trend_4h': self._determine_trend(indicators_4h),
                    'rsi_1h': indicators_1h.get('current', {}).get('rsi', 50),
                    'rsi_4h': indicators_4h.get('current', {}).get('rsi', 50),
                    'volume_ratio': indicators_1h.get('current', {}).get('volume_ratio', 1.0),
                    'macd_momentum': indicators_1h.get('current', {}).get('macd_histogram', 0)
                },
                # Добавляем КРАТКУЮ информацию из расширенных данных
                'market_context': {
                    'funding_rate': market_data.get('funding_rate', {}).get('funding_rate', 0) if market_data.get('funding_rate') else 0,
                    'oi_trend': market_data.get('open_interest', {}).get('oi_trend', 'UNKNOWN') if market_data.get('open_interest') else 'UNKNOWN',
                    'spread_pct': market_data.get('orderbook', {}).get('spread_pct', 0) if market_data.get('orderbook') else 0,
                    'buy_pressure': market_data.get('taker_volume', {}).get('buy_pressure', 0.5) if market_data.get('taker_volume') else 0.5
                },
                'correlation_context': {
                    'btc_correlation': corr_data.get('btc_correlation', {}).get('correlation', 0) if corr_data else 0,
                    'btc_trend': corr_data.get('btc_trend', 'UNKNOWN') if corr_data else 'UNKNOWN'
                },
                'volume_profile_context': {
                    'poc': vp_data.get('poc', 0) if vp_data else 0,
                    'value_area': [vp_data.get('value_area_low', 0), vp_data.get('value_area_high', 0)] if vp_data else [0, 0]
                },
                'orderflow_context': {
                    'direction': orderflow_ai.get('orderflow_direction', 'UNKNOWN') if orderflow_ai else 'UNKNOWN',
                    'spoofing_risk': orderflow_ai.get('spoofing_risk', 'UNKNOWN') if orderflow_ai else 'UNKNOWN'
                },
                'smc_context': {
                    'order_blocks': len(smc_ai.get('order_blocks', [])) if smc_ai else 0,
                    'patterns_alignment': smc_ai.get('patterns_alignment', 'MIXED') if smc_ai else 'MIXED'
                }
            }

            # Вызываем AI с компактными данными
            if provider == 'deepseek':
                return await self._analyze_comprehensive_deepseek(symbol, ai_input)
            elif provider == 'anthropic':
                return await self._analyze_comprehensive_anthropic(symbol, ai_input)
            else:
                return create_fallback_analysis(symbol, indicators_1h)

        except Exception as e:
            logger.error(f"Ошибка comprehensive анализа {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return create_fallback_analysis(symbol, comprehensive_data.get('indicators_1h', {}))

    async def _analyze_comprehensive_deepseek(self, symbol: str, ai_input: Dict) -> Dict:
        """DeepSeek реализация comprehensive анализа"""
        import asyncio
        import json
        from openai import AsyncOpenAI
        from deepseek import load_prompt_cached, extract_json_optimized, safe_float_conversion

        try:
            client = AsyncOpenAI(
                api_key=config.DEEPSEEK_API_KEY,
                base_url=config.DEEPSEEK_URL
            )

            prompt = load_prompt_cached(config.ANALYSIS_PROMPT)
            data_json = json.dumps(ai_input, separators=(',', ':'))

            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=config.DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": data_json}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=config.AI_MAX_TOKENS_ANALYZE,
                    temperature=config.AI_TEMPERATURE_ANALYZE
                ),
                timeout=config.API_TIMEOUT
            )

            result_text = response.choices[0].message.content
            json_result = extract_json_optimized(result_text)

            if json_result:
                return self._format_analysis_result(symbol, json_result, ai_input['current_price'])
            else:
                return create_fallback_analysis(symbol, {'current': {'price': ai_input['current_price']}})

        except Exception as e:
            logger.error(f"DeepSeek comprehensive error for {symbol}: {e}")
            return create_fallback_analysis(symbol, {'current': {'price': ai_input['current_price']}})

    async def _analyze_comprehensive_anthropic(self, symbol: str, ai_input: Dict) -> Dict:
        """Anthropic реализация comprehensive анализа"""
        import json
        from deepseek import load_prompt_cached

        try:
            prompt = load_prompt_cached(config.ANALYSIS_PROMPT)
            data_json = json.dumps(ai_input, separators=(',', ':'))

            messages = [
                {
                    "role": "user",
                    "content": f"{prompt}\n\nДанные для анализа:\n{data_json}\n\nВерни JSON с анализом."
                }
            ]

            response_text = await anthropic_client._make_request(
                messages=messages,
                max_tokens=config.AI_MAX_TOKENS_ANALYZE,
                temperature=config.AI_TEMPERATURE_ANALYZE
            )

            result = anthropic_client.extract_json(response_text)

            if result:
                return self._format_analysis_result(symbol, result, ai_input['current_price'])
            else:
                return create_fallback_analysis(symbol, {'current': {'price': ai_input['current_price']}})

        except Exception as e:
            logger.error(f"Anthropic comprehensive error for {symbol}: {e}")
            return create_fallback_analysis(symbol, {'current': {'price': ai_input['current_price']}})

    def _format_analysis_result(self, symbol: str, ai_result: Dict, current_price: float) -> Dict:
        """Форматирование результата AI анализа"""
        from deepseek import safe_float_conversion

        signal = str(ai_result.get('signal', 'NO_SIGNAL')).upper()
        confidence = max(0, min(100, int(safe_float_conversion(ai_result.get('confidence', 0)))))
        entry_price = safe_float_conversion(ai_result.get('entry_price', current_price))
        stop_loss = safe_float_conversion(ai_result.get('stop_loss', 0))
        take_profit = safe_float_conversion(ai_result.get('take_profit', 0))
        analysis = str(ai_result.get('analysis', 'AI analysis'))

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

    def _determine_trend(self, indicators: Dict) -> str:
        """Определить тренд по индикаторам"""
        if not indicators or 'current' not in indicators:
            return 'UNKNOWN'

        current = indicators['current']
        ema5 = current.get('ema5', 0)
        ema20 = current.get('ema20', 0)

        if ema5 > 0 and ema20 > 0:
            return 'UP' if ema5 > ema20 else 'DOWN'
        return 'FLAT'

    async def validate_signals(self, preliminary_signals: List[Dict], market_data: Dict) -> List[Dict]:
        """Валидация сигналов через выбранный AI провайдер"""
        provider = self._get_provider_for_stage('validation')
        logger.info(f"Этап валидации: используется {provider}")

        try:
            if provider == 'deepseek':
                return await ai_validate_signals_deepseek(preliminary_signals, market_data)
            elif provider == 'anthropic':
                return await anthropic_client.validate_signals(preliminary_signals, market_data)
            else:
                return create_fallback_validation(preliminary_signals)

        except Exception as e:
            logger.error(f"Ошибка валидации через {provider}: {e}")
            logger.info("Переключение на fallback режим")
            return create_fallback_validation(preliminary_signals)

    def get_status(self) -> Dict:
        """Получить статус всех провайдеров"""
        return {
            'providers_available': self.providers,
            'stage_assignments': {
                'selection': config.AI_STAGE_SELECTION,
                'analysis': config.AI_STAGE_ANALYSIS,
                'validation': config.AI_STAGE_VALIDATION
            },
            'effective_providers': {
                'selection': self._get_provider_for_stage('selection'),
                'analysis': self._get_provider_for_stage('analysis'),
                'validation': self._get_provider_for_stage('validation')
            }
        }


# Создаем глобальный роутер
ai_router = AIRouter()