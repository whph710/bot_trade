"""
Simple validator with full level support
"""

import logging
from typing import Dict, List
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)


def check_trading_hours(perm_time=None) -> tuple[bool, str]:
    """
    Проверка торговых часов (по пермскому времени UTC+5)

    Returns:
        (is_allowed: bool, reason: str)
    """
    if perm_time is None:
        # Получить текущее время в Перми
        perm_tz = pytz.timezone('Asia/Yekaterinburg')  # UTC+5
        perm_time = datetime.now(perm_tz)

    hour = perm_time.hour

    # 🔴 КРИТИЧЕСКИЕ ПЕРИОДЫ - НЕ ТОРГОВАТЬ
    # 00:00–03:00 (19:00–22:00 UTC) — конец US сессии, низкие объёмы
    if 0 <= hour < 3:
        return False, f"❌ BLOCKED: {hour}:00–03:00 (конец US сессии, низкие объёмы)"

    # 04:00–08:00 (23:00–03:00 UTC) — азиатская ночь, высокие спреды
    if 4 <= hour < 8:
        return False, f"❌ BLOCKED: {hour}:00–08:00 (азиатская ночь, спреды >0.15%)"

    # 12:00–13:00 (07:00–08:00 UTC) — gap после Asian close
    if 12 <= hour < 13:
        return False, f"❌ BLOCKED: 12:00–13:00 (gap после Asian close, фиксация позиций)"

    return True, "✅ Торговые часы OK"


async def validate_signals_simple(ai_router, preliminary_signals: List[Dict]) -> Dict:
    """Simple validation through AI"""

    # Проверка времени перед началом валидации
    time_allowed, time_reason = check_trading_hours()
    logger.warning(time_reason)

    if not time_allowed:
        logger.warning(f"Валидация пропущена: {time_reason}")
        return {
            'validated': [],
            'rejected': [],
            'validation_skipped_reason': time_reason
        }

    if not preliminary_signals:
        return {'validated': [], 'rejected': []}

    validated = []
    rejected = []

    for signal in preliminary_signals:
        try:
            symbol = signal['symbol']
            comprehensive_data = signal.get('comprehensive_data', {})

            logger.debug(f"Validating {symbol}...")

            validation_result = await ai_router.validate_signal_with_stage3_data(
                signal,
                comprehensive_data
            )

            tp_levels = validation_result.get('take_profit_levels', signal.get('take_profit_levels', [0, 0, 0]))

            if not isinstance(tp_levels, list):
                tp_levels = [float(tp_levels), float(tp_levels) * 1.1, float(tp_levels) * 1.2]
            elif len(tp_levels) < 3:
                while len(tp_levels) < 3:
                    last_tp = tp_levels[-1] if tp_levels else 0
                    tp_levels.append(last_tp * 1.1)

            if validation_result.get('approved', False):
                validated_signal = {
                    'symbol': symbol,
                    'signal': signal['signal'],
                    'confidence': validation_result.get('final_confidence', signal['confidence']),
                    'entry_price': validation_result.get('entry_price', signal['entry_price']),
                    'stop_loss': validation_result.get('stop_loss', signal['stop_loss']),
                    'take_profit_levels': tp_levels,
                    'analysis': signal.get('analysis', ''),
                    'risk_reward_ratio': validation_result.get('risk_reward_ratio', 0),
                    'hold_duration_minutes': validation_result.get('hold_duration_minutes', 720),
                    'validation_notes': validation_result.get('validation_notes', ''),
                    'market_conditions': validation_result.get('market_conditions', ''),
                    'key_levels': validation_result.get('key_levels', ''),
                    'validation_method': validation_result.get('validation_method', 'ai'),
                    'timestamp': signal.get('timestamp', datetime.now().isoformat())
                }
                validated.append(validated_signal)
                logger.info(f"✅ Validated {symbol} | R/R: {validation_result.get('risk_reward_ratio', 0)}")
            else:
                rejected_signal = {
                    'symbol': symbol,
                    'signal': signal.get('signal', 'UNKNOWN'),
                    'original_confidence': signal.get('confidence', 0),
                    'entry_price': validation_result.get('entry_price', signal.get('entry_price', 0)),
                    'stop_loss': validation_result.get('stop_loss', signal.get('stop_loss', 0)),
                    'take_profit_levels': tp_levels,
                    'rejection_reason': validation_result.get('rejection_reason', 'Validation failed'),
                    'timestamp': signal.get('timestamp', datetime.now().isoformat())
                }
                rejected.append(rejected_signal)
                logger.info(f"❌ Rejected {symbol}: {rejected_signal['rejection_reason']}")

        except Exception as e:
            logger.error(f"Validation error for {signal['symbol']}: {e}")

            tp_levels = signal.get('take_profit_levels', [0, 0, 0])
            if not isinstance(tp_levels, list):
                tp_levels = [0, 0, 0]

            rejected.append({
                'symbol': signal['symbol'],
                'signal': signal.get('signal', 'UNKNOWN'),
                'original_confidence': signal.get('confidence', 0),
                'entry_price': signal.get('entry_price', 0),
                'stop_loss': signal.get('stop_loss', 0),
                'take_profit_levels': tp_levels,
                'rejection_reason': f'Validation exception: {str(e)}',
                'timestamp': signal.get('timestamp', datetime.now().isoformat())
            })
            continue

    return {
        'validated': validated,
        'rejected': rejected
    }


def calculate_validation_stats(validated: List[Dict], rejected: List[Dict]) -> Dict:
    """Calculate validation statistics"""
    total = len(validated) + len(rejected)

    if total == 0:
        return {
            'total': 0,
            'approved': 0,
            'rejected': 0,
            'approval_rate': 0.0,
            'avg_risk_reward': 0.0,
            'top_rejection_reasons': [],
            'rr_stats': {
                'min_rr': 0.0,
                'max_rr': 0.0,
                'samples_counted': 0
            }
        }

    approval_rate = (len(validated) / total) * 100 if total > 0 else 0

    # ✅ ИСПРАВЛЕНИЕ: Фильтруем нулевые R/R перед расчётом среднего
    rr_ratios = [
        sig.get('risk_reward_ratio', 0)
        for sig in validated
        if sig.get('risk_reward_ratio', 0) > 0  # Только положительные значения
    ]

    avg_rr = sum(rr_ratios) / len(rr_ratios) if rr_ratios else 0
    min_rr = min(rr_ratios) if rr_ratios else 0.0
    max_rr = max(rr_ratios) if rr_ratios else 0.0

    rejection_reasons = {}
    for rej in rejected:
        reason = rej.get('rejection_reason', 'Unknown')
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    top_rejections = sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:3]

    stats = {
        'total': total,
        'approved': len(validated),
        'rejected': len(rejected),
        'approval_rate': round(approval_rate, 1),
        'avg_risk_reward': round(avg_rr, 2),
        'top_rejection_reasons': [reason for reason, count in top_rejections],
        'rr_stats': {
            'min_rr': round(min_rr, 2),
            'max_rr': round(max_rr, 2),
            'samples_counted': len(rr_ratios),
            'signals_with_zero_rr': len(validated) - len(rr_ratios)  # Отладка
        }
    }

    # Логируем статистику
    logger.info(f"📊 VALIDATION STATS:")
    logger.info(f"  Total: {total} | Approved: {len(validated)} | Rejected: {len(rejected)}")
    logger.info(f"  Approval rate: {approval_rate:.1f}%")
    logger.info(f"  R/R: min={min_rr:.2f}, avg={avg_rr:.2f}, max={max_rr:.2f} ({len(rr_ratios)} samples)")
    if stats['top_rejection_reasons']:
        logger.info(f"  Top rejections: {', '.join(stats['top_rejection_reasons'][:2])}")

    return stats