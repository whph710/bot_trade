"""
Упрощенный валидатор - без избыточных проверок
Просто передает данные в Claude для финальной валидации
"""

import logging
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


async def validate_signals_simple(
        ai_router,
        preliminary_signals: List[Dict]
) -> Dict:
    """
    Простая валидация через Claude

    Args:
        ai_router: роутер для вызова Claude
        preliminary_signals: сигналы из Stage 3 (с comprehensive_data)

    Returns:
        {'validated': [...], 'rejected': [...]}
    """
    if not preliminary_signals:
        return {'validated': [], 'rejected': []}

    validated = []
    rejected = []

    for signal in preliminary_signals:
        try:
            symbol = signal['symbol']
            comprehensive_data = signal.get('comprehensive_data', {})

            logger.debug(f"Validating {symbol}...")

            # Claude валидация
            validation_result = await ai_router.validate_signal_with_stage3_data(
                signal,
                comprehensive_data
            )

            if validation_result.get('approved', False):
                # Approved
                validated_signal = {
                    'symbol': symbol,
                    'signal': signal['signal'],
                    'confidence': validation_result.get('final_confidence', signal['confidence']),
                    'entry_price': signal['entry_price'],
                    'stop_loss': signal['stop_loss'],
                    'take_profit': signal['take_profit'],
                    'analysis': signal.get('analysis', ''),
                    'risk_reward_ratio': validation_result.get('risk_reward_ratio', 0),
                    'hold_duration_minutes': validation_result.get('hold_duration_minutes', 720),
                    'validation_notes': validation_result.get('validation_notes', ''),
                    'market_conditions': validation_result.get('market_conditions', ''),
                    'key_levels': validation_result.get('key_levels', ''),
                    'validation_method': 'claude',
                    'timestamp': signal.get('timestamp', datetime.now().isoformat())
                }
                validated.append(validated_signal)
                logger.info(f"✓ {symbol} VALIDATED")
            else:
                # Rejected
                rejected_signal = {
                    'symbol': symbol,
                    'signal': signal['signal'],
                    'original_confidence': signal['confidence'],
                    'rejection_reason': validation_result.get('rejection_reason', 'Claude validation failed')
                }
                rejected.append(rejected_signal)
                logger.info(f"✗ {symbol} REJECTED: {rejected_signal['rejection_reason']}")

        except Exception as e:
            logger.error(f"Validation error for {signal['symbol']}: {e}")
            rejected.append({
                'symbol': signal['symbol'],
                'rejection_reason': f'Validation exception: {str(e)}'
            })
            continue

    return {
        'validated': validated,
        'rejected': rejected
    }


def calculate_validation_stats(validated: List[Dict], rejected: List[Dict]) -> Dict:
    """Простая статистика валидации"""
    total = len(validated) + len(rejected)

    if total == 0:
        return {
            'total': 0,
            'approved': 0,
            'rejected': 0,
            'approval_rate': 0.0
        }

    approval_rate = (len(validated) / total) * 100 if total > 0 else 0

    # Средний R/R
    rr_ratios = [sig.get('risk_reward_ratio', 0) for sig in validated if sig.get('risk_reward_ratio', 0) > 0]
    avg_rr = sum(rr_ratios) / len(rr_ratios) if rr_ratios else 0

    # Топ причины отклонения
    rejection_reasons = {}
    for rej in rejected:
        reason = rej.get('rejection_reason', 'Unknown')
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    top_rejections = sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        'total': total,
        'approved': len(validated),
        'rejected': len(rejected),
        'approval_rate': round(approval_rate, 1),
        'avg_risk_reward': round(avg_rr, 2),
        'top_rejection_reasons': [reason for reason, count in top_rejections]
    }