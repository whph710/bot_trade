"""
AI Router - UPDATED VERSION
Использует unified analysis вместо трех отдельных запросов
"""

import logging
import json
import asyncio
from typing import List, Dict
from config import config

logger = logging.getLogger(__name__)

from deepseek import ai_select_pairs_deepseek, load_prompt_cached, extract_json_optimized
from anthropic_ai import anthropic_client


class AIRouter:
    """Роутер: DeepSeek для отбора, Claude для unified анализа"""

    def __init__(self):
        pass

    async def call_ai(
            self,
            prompt: str,
            stage: str = 'analysis',
            max_tokens: int = 2000,
            temperature: float = 0.7
    ) -> str:
        """Универсальный метод для вызова AI"""
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
        Stage 3: Claude UNIFIED анализ
        Один запрос вместо трех (OrderFlow + SMC + Main)

        Args:
            symbol: пара
            comprehensive_data: ВСЕ данные

        Returns:
            {
                'signal': 'LONG'/'SHORT'/'NO_SIGNAL',
                'confidence': 85,
                'entry_price': 43251.25,
                'stop_loss': 42980.50,
                'take_profit_levels': [43892.75, 44500.00, 45200.00],
                'analysis': '...',
                'orderflow_analysis': {...},
                'smc_analysis': {...}
            }
        """
        logger.debug(f"Stage 3: Claude unified analysis for {symbol}")

        try:
            # Используем unified analyzer
            from ai_advanced_analysis import get_unified_analysis

            result = await get_unified_analysis(
                self,
                symbol,
                comprehensive_data
            )

            # Проверяем что есть обязательные поля
            if result and isinstance(result, dict) and 'signal' in result:
                # Убеждаемся что TP levels это список
                if not isinstance(result.get('take_profit_levels'), list):
                    logger.warning(f"{symbol}: take_profit_levels not a list, fixing...")
                    tp = result.get('take_profit_levels', 0)
                    if tp:
                        result['take_profit_levels'] = [float(tp), float(tp) * 1.1, float(tp) * 1.2]
                    else:
                        result['take_profit_levels'] = [0, 0, 0]

                return result
            else:
                logger.warning(f"Claude unified analysis failed for {symbol}")
                return self._fallback_analysis(symbol, comprehensive_data.get('current_price', 0))

        except Exception as e:
            logger.error(f"Claude unified analysis error for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._fallback_analysis(symbol, comprehensive_data.get('current_price', 0))

    async def validate_signal_with_stage3_data(
            self,
            signal: Dict,
            comprehensive_data: Dict
    ) -> Dict:
        """
        Stage 4: Claude валидация с ПОЛНЫМИ данными

        Returns:
            {
                'approved': True/False,
                'final_confidence': 85,
                'rejection_reason': '...' if rejected,
                'entry_price': ...,
                'stop_loss': ...,
                'take_profit_levels': [tp1, tp2, tp3],
                'risk_reward_ratio': 2.4,
                'hold_duration_minutes': 720,
                'validation_notes': '...'
            }
        """
        symbol = signal['symbol']
        logger.debug(f"Stage 4: Claude validation for {symbol}")

        try:
            # Формируем validation input
            validation_input = {
                'signal': {
                    'symbol': signal['symbol'],
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
                    'orderflow_analysis': signal.get('orderflow_analysis', {}),
                    'smc_analysis': signal.get('smc_analysis', {}),
                    'current_price': comprehensive_data.get('current_price', 0)
                }
            }

            prompt = load_prompt_cached(config.VALIDATION_PROMPT)
            data_json = json.dumps(validation_input, separators=(',', ':'))

            messages = [
                {
                    "role": "user",
                    "content": f"{prompt}\n\nДанные для валидации:\n{data_json}\n\nВерни JSON с результатом."
                }
            ]

            response_text = await anthropic_client._make_request(
                messages=messages,
                max_tokens=config.AI_MAX_TOKENS_VALIDATE,
                temperature=config.AI_TEMPERATURE_VALIDATE
            )

            result = anthropic_client.extract_json(response_text)

            if result and 'final_signals' in result:
                final_signals = result.get('final_signals', [])
                if final_signals:
                    validated = final_signals[0]

                    # Убеждаемся что TP levels это список
                    tp_levels = validated.get('take_profit_levels', signal.get('take_profit_levels', [0, 0, 0]))
                    if not isinstance(tp_levels, list):
                        tp_levels = [float(tp_levels), float(tp_levels) * 1.1, float(tp_levels) * 1.2]

                    return {
                        'approved': True,
                        'final_confidence': validated.get('confidence', signal['confidence']),
                        'entry_price': validated.get('entry_price', signal['entry_price']),
                        'stop_loss': validated.get('stop_loss', signal['stop_loss']),
                        'take_profit_levels': tp_levels,
                        'risk_reward_ratio': validated.get('risk_reward_ratio', 0),
                        'hold_duration_minutes': validated.get('hold_duration_minutes', 720),
                        'validation_notes': validated.get('validation_notes', ''),
                        'market_conditions': validated.get('market_conditions', ''),
                        'key_levels': validated.get('key_levels', ''),
                        'validation_method': 'claude'
                    }
                else:
                    # Rejected
                    rejected_info = result.get('rejected_signals', [{}])[0]

                    # ВАЖНО: Возвращаем уровни даже при rejection
                    return {
                        'approved': False,
                        'rejection_reason': rejected_info.get('reason', 'Claude rejected'),
                        'entry_price': signal.get('entry_price', 0),
                        'stop_loss': signal.get('stop_loss', 0),
                        'take_profit_levels': signal.get('take_profit_levels', [0, 0, 0]),
                        'final_confidence': signal.get('confidence', 0),
                        'validation_method': 'claude'
                    }
            else:
                # Fallback если Claude не вернул корректный формат
                return self._fallback_validation(signal)

        except Exception as e:
            logger.error(f"Claude validation error for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._fallback_validation(signal)

    def _fallback_analysis(self, symbol: str, current_price: float) -> Dict:
        """Fallback если Claude не сработал"""
        return {
            'symbol': symbol,
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'entry_price': current_price,
            'stop_loss': 0,
            'take_profit_levels': [0, 0, 0],
            'analysis': 'Claude unified analysis failed',
            'orderflow_analysis': {},
            'smc_analysis': {},
            'ai_generated': False,
            'stage': 3
        }

    def _fallback_validation(self, signal: Dict) -> Dict:
        """Fallback валидация с сохранением уровней"""
        entry = signal.get('entry_price', 0)
        stop = signal.get('stop_loss', 0)
        tp_levels = signal.get('take_profit_levels', [0, 0, 0])

        # Убеждаемся что TP levels это список
        if not isinstance(tp_levels, list):
            tp_levels = [float(tp_levels), float(tp_levels) * 1.1, float(tp_levels) * 1.2]

        if entry > 0 and stop > 0 and tp_levels and tp_levels[0] > 0:
            risk = abs(entry - stop)
            reward = abs(tp_levels[1] - entry) if len(tp_levels) > 1 else abs(tp_levels[0] - entry)

            if risk > 0:
                rr_ratio = round(reward / risk, 2)
                if rr_ratio >= config.MIN_RISK_REWARD_RATIO:
                    return {
                        'approved': True,
                        'final_confidence': signal['confidence'],
                        'entry_price': entry,
                        'stop_loss': stop,
                        'take_profit_levels': tp_levels,
                        'risk_reward_ratio': rr_ratio,
                        'hold_duration_minutes': 720,
                        'validation_method': 'fallback',
                        'validation_notes': f'Fallback validation: R/R {rr_ratio}'
                    }

        return {
            'approved': False,
            'rejection_reason': 'Fallback validation failed - poor R/R or invalid levels',
            'entry_price': entry,
            'stop_loss': stop,
            'take_profit_levels': tp_levels,
            'final_confidence': signal.get('confidence', 0),
            'validation_method': 'fallback'
        }


# Глобальный экземпляр
ai_router = AIRouter()