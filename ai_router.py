"""
AI Router - упрощенная версия
Stage 2: DeepSeek
Stage 3 & 4: Claude (Anthropic)
"""

import logging
import json
import asyncio
from typing import List, Dict
from config import config

logger = logging.getLogger(__name__)

# Импорты
from deepseek import ai_select_pairs_deepseek, load_prompt_cached, extract_json_optimized
from anthropic_ai import anthropic_client


class AIRouter:
    """Упрощенный роутер: DeepSeek для отбора, Claude для анализа"""

    def __init__(self):
        pass

    async def call_ai(
            self,
            prompt: str,
            stage: str = 'analysis',
            max_tokens: int = 2000,
            temperature: float = 0.7
    ) -> str:
        """
        Универсальный метод для вызова AI
        Используется в ai_advanced_analysis.py
        """
        try:
            messages = [{"role": "user", "content": prompt}]

            response_text = await anthropic_client._make_request(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            return response_text if response_text else "{}"

        except Exception as e:
            logger.error(f"call_ai error: {e}")
            return "{}"

    async def select_pairs(self, pairs_data: List[Dict]) -> List[str]:
        """Stage 2: DeepSeek отбор"""
        logger.info("Stage 2: DeepSeek selection")
        return await ai_select_pairs_deepseek(pairs_data)

    async def analyze_pair_comprehensive(
            self,
            symbol: str,
            comprehensive_data: Dict
    ) -> Dict:
        """
        Stage 3: Claude ПОЛНЫЙ анализ

        Args:
            symbol: пара
            comprehensive_data: ВСЕ данные (свечи, индикаторы, market_data, корреляции, VP, AI анализы)
        """
        logger.debug(f"Stage 3: Claude comprehensive analysis for {symbol}")

        try:
            # Формируем компактный JSON для Claude
            ai_input = self._prepare_analysis_input(symbol, comprehensive_data)

            # Загружаем промпт
            prompt = load_prompt_cached(config.ANALYSIS_PROMPT)
            data_json = json.dumps(ai_input, separators=(',', ':'))

            messages = [
                {
                    "role": "user",
                    "content": f"{prompt}\n\nДанные для анализа:\n{data_json}\n\nВерни JSON с полным анализом."
                }
            ]

            response_text = await anthropic_client._make_request(
                messages=messages,
                max_tokens=config.AI_MAX_TOKENS_ANALYZE,
                temperature=config.AI_TEMPERATURE_ANALYZE
            )

            result = anthropic_client.extract_json(response_text)

            if result and isinstance(result, dict):
                # Проверяем что есть обязательные поля
                if 'signal' in result:
                    return self._format_analysis_result(symbol, result, comprehensive_data['current_price'])
                else:
                    logger.warning(f"Claude result missing 'signal' field for {symbol}")
                    return self._fallback_analysis(symbol, comprehensive_data['current_price'])
            else:
                logger.warning(f"Claude returned no valid JSON for {symbol}, response: {response_text[:200]}")
                return self._fallback_analysis(symbol, comprehensive_data['current_price'])

        except Exception as e:
            logger.error(f"Claude analysis error for {symbol}: {e}")
            return self._fallback_analysis(symbol, comprehensive_data['current_price'])

    async def validate_signal_with_stage3_data(
            self,
            signal: Dict,
            comprehensive_data: Dict
    ) -> Dict:
        """
        Stage 4: Claude валидация

        Args:
            signal: результат Stage 3
            comprehensive_data: те же данные что были в Stage 3
        """
        symbol = signal['symbol']
        logger.debug(f"Stage 4: Claude validation for {symbol}")

        try:
            # Формируем validation input
            validation_input = {
                'signal': signal,
                'comprehensive_data': {
                    'market_data': comprehensive_data.get('market_data', {}),
                    'correlation_data': comprehensive_data.get('correlation_data', {}),
                    'volume_profile': comprehensive_data.get('volume_profile', {}),
                    'vp_analysis': comprehensive_data.get('vp_analysis', {}),
                    'orderflow_ai': comprehensive_data.get('orderflow_ai', {}),
                    'smc_ai': comprehensive_data.get('smc_ai', {}),
                    'current_price': comprehensive_data.get('current_price', 0)
                }
            }

            prompt = load_prompt_cached(config.VALIDATION_PROMPT)
            data_json = json.dumps(validation_input, separators=(',', ':'))

            messages = [
                {
                    "role": "user",
                    "content": f"{prompt}\n\nДанные для валидации:\n{data_json}\n\nВерни JSON с результатом валидации."
                }
            ]

            response_text = await anthropic_client._make_request(
                messages=messages,
                max_tokens=config.AI_MAX_TOKENS_VALIDATE,
                temperature=config.AI_TEMPERATURE_VALIDATE
            )

            result = anthropic_client.extract_json(response_text)

            if result and 'final_signals' in result:
                # Claude вернул валидацию
                final_signals = result.get('final_signals', [])
                if final_signals:
                    validated = final_signals[0]  # Берем первый (наш символ)
                    validated['validation_method'] = 'claude'
                    return validated
                else:
                    # Rejected
                    rejected_info = result.get('rejected_signals', [{}])[0]
                    return {
                        'symbol': symbol,
                        'approved': False,
                        'rejection_reason': rejected_info.get('reason', 'Claude rejected'),
                        'validation_method': 'claude'
                    }
            else:
                # Fallback если Claude не вернул корректный формат
                return self._fallback_validation(signal)

        except Exception as e:
            logger.error(f"Claude validation error for {symbol}: {e}")
            return self._fallback_validation(signal)

    def _prepare_analysis_input(self, symbol: str, comprehensive_data: Dict) -> Dict:
        """Подготовка компактного JSON для Stage 3"""
        candles_1h = comprehensive_data.get('candles_1h', [])
        candles_4h = comprehensive_data.get('candles_4h', [])
        indicators_1h = comprehensive_data.get('indicators_1h', {})
        indicators_4h = comprehensive_data.get('indicators_4h', {})
        current_price = comprehensive_data.get('current_price', 0)

        return {
            'symbol': symbol,
            'current_price': current_price,
            'timeframes': {
                '1h': {
                    'candles': candles_1h[-80:],
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
                    'candles': candles_4h[-40:],
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
            # Краткая инфа из расширенных данных
            'market_context': self._extract_market_context(comprehensive_data),
            'correlation_context': self._extract_correlation_context(comprehensive_data),
            'volume_profile_context': self._extract_vp_context(comprehensive_data),
            'orderflow_context': self._extract_orderflow_context(comprehensive_data),
            'smc_context': self._extract_smc_context(comprehensive_data)
        }

    def _extract_market_context(self, data: Dict) -> Dict:
        """Извлечь краткий market context"""
        market_data = data.get('market_data', {})
        return {
            'funding_rate': market_data.get('funding_rate', {}).get('funding_rate', 0) if market_data.get('funding_rate') else 0,
            'oi_trend': market_data.get('open_interest', {}).get('oi_trend', 'UNKNOWN') if market_data.get('open_interest') else 'UNKNOWN',
            'spread_pct': market_data.get('orderbook', {}).get('spread_pct', 0) if market_data.get('orderbook') else 0,
            'buy_pressure': market_data.get('taker_volume', {}).get('buy_pressure', 0.5) if market_data.get('taker_volume') else 0.5
        }

    def _extract_correlation_context(self, data: Dict) -> Dict:
        """Извлечь correlation context"""
        corr_data = data.get('correlation_data', {})
        return {
            'btc_correlation': corr_data.get('btc_correlation', {}).get('correlation', 0) if corr_data else 0,
            'btc_trend': corr_data.get('btc_trend', 'UNKNOWN') if corr_data else 'UNKNOWN'
        }

    def _extract_vp_context(self, data: Dict) -> Dict:
        """Извлечь Volume Profile context"""
        vp_data = data.get('volume_profile', {})
        return {
            'poc': vp_data.get('poc', 0) if vp_data else 0,
            'value_area': [vp_data.get('value_area_low', 0), vp_data.get('value_area_high', 0)] if vp_data else [0, 0]
        }

    def _extract_orderflow_context(self, data: Dict) -> Dict:
        """Извлечь OrderFlow context"""
        orderflow = data.get('orderflow_ai', {})
        return {
            'direction': orderflow.get('orderflow_direction', 'UNKNOWN') if orderflow else 'UNKNOWN',
            'spoofing_risk': orderflow.get('spoofing_risk', 'UNKNOWN') if orderflow else 'UNKNOWN'
        }

    def _extract_smc_context(self, data: Dict) -> Dict:
        """Извлечь SMC context"""
        smc = data.get('smc_ai', {})
        return {
            'order_blocks': len(smc.get('order_blocks', [])) if smc else 0,
            'patterns_alignment': smc.get('patterns_alignment', 'MIXED') if smc else 'MIXED'
        }

    def _determine_trend(self, indicators: Dict) -> str:
        """Определить тренд"""
        if not indicators or 'current' not in indicators:
            return 'UNKNOWN'
        current = indicators['current']
        ema5 = current.get('ema5', 0)
        ema20 = current.get('ema20', 0)
        if ema5 > 0 and ema20 > 0:
            return 'UP' if ema5 > ema20 else 'DOWN'
        return 'FLAT'

    def _format_analysis_result(self, symbol: str, ai_result: Dict, current_price: float) -> Dict:
        """Форматирование результата Claude анализа"""
        try:
            signal = str(ai_result.get('signal', 'NO_SIGNAL')).upper()
            confidence = max(0, min(100, int(float(ai_result.get('confidence', 0)))))
            entry_price = float(ai_result.get('entry_price', current_price) or current_price)
            stop_loss = float(ai_result.get('stop_loss', 0) or 0)
            take_profit = float(ai_result.get('take_profit', 0) or 0)
            analysis = str(ai_result.get('analysis', 'Claude analysis'))

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
                'ai_generated': True,
                'stage': 3
            }
        except Exception as e:
            logger.error(f"Error formatting Claude result for {symbol}: {e}")
            return self._fallback_analysis(symbol, current_price)

    def _fallback_analysis(self, symbol: str, current_price: float) -> Dict:
        """Fallback если Claude не сработал"""
        return {
            'symbol': symbol,
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'entry_price': current_price,
            'stop_loss': 0,
            'take_profit': 0,
            'analysis': 'Claude analysis failed',
            'ai_generated': False,
            'stage': 3
        }

    def _fallback_validation(self, signal: Dict) -> Dict:
        """Fallback валидация"""
        entry = signal.get('entry_price', 0)
        stop = signal.get('stop_loss', 0)
        profit = signal.get('take_profit', 0)

        if entry > 0 and stop > 0 and profit > 0:
            risk = abs(entry - stop)
            reward = abs(profit - entry)
            if risk > 0:
                rr_ratio = round(reward / risk, 2)
                if rr_ratio >= config.MIN_RISK_REWARD_RATIO:
                    return {
                        'symbol': signal['symbol'],
                        'approved': True,
                        'final_confidence': signal['confidence'],
                        'risk_reward_ratio': rr_ratio,
                        'validation_method': 'fallback',
                        'validation_notes': f'Fallback validation: R/R {rr_ratio}'
                    }

        return {
            'symbol': signal['symbol'],
            'approved': False,
            'rejection_reason': 'Fallback validation failed',
            'validation_method': 'fallback'
        }


# Глобальный экземпляр
ai_router = AIRouter()