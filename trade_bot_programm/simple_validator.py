"""
Simple validator - DEPRECATED
Файл: trade_bot_programm/simple_validator.py

МОДИФИКАЦИЯ: Stage 4 validation полностью удалён.
Этот модуль сохранён только для совместимости с импортами.
Вся валидация теперь происходит в Stage 3 (AI analysis).
"""

from typing import Dict, List
from datetime import datetime
import pytz
from logging_config import setup_module_logger

logger = setup_module_logger(__name__)


def check_trading_hours(perm_time=None) -> tuple[bool, str]:
    """
    Проверка торговых часов (по пермскому времени UTC+5)

    МОДИФИКАЦИЯ: Все часы разрешены (ограничения отключены)

    Returns:
        (is_allowed: bool, reason: str)
    """
    # Все часы разрешены
    return True, "Trading hours: OK"


async def validate_signals_simple(ai_router, preliminary_signals: List[Dict]) -> Dict:
    """
    DEPRECATED: Stage 4 validation removed

    Этот метод теперь просто возвращает сигналы без валидации,
    т.к. вся валидация происходит в Stage 3.

    Returns:
        {
            'validated': preliminary_signals (as is),
            'rejected': []
        }
    """

    if not preliminary_signals:
        logger.warning("No signals received for 'validation' (deprecated)")
        return {
            'validated': [],
            'rejected': []
        }

    logger.info(f"Stage 4 validation SKIPPED (deprecated). Passing {len(preliminary_signals)} signal(s) as is.")

    # Просто возвращаем сигналы без изменений
    return {
        'validated': preliminary_signals,
        'rejected': []
    }


def calculate_validation_stats(validated: List[Dict], rejected: List[Dict]) -> Dict:
    """
    Calculate validation statistics
    DEPRECATED: Stage 4 removed, но метод сохранён для совместимости
    """
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

    # Filter only positive R/R
    rr_ratios = [
        sig.get('risk_reward_ratio', 0)
        for sig in validated
        if sig.get('risk_reward_ratio', 0) > 0
    ]

    avg_rr = sum(rr_ratios) / len(rr_ratios) if rr_ratios else 0
    min_rr = min(rr_ratios) if rr_ratios else 0.0
    max_rr = max(rr_ratios) if rr_ratios else 0.0

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
        'top_rejection_reasons': [reason for reason, count in top_rejections],
        'rr_stats': {
            'min_rr': round(min_rr, 2),
            'max_rr': round(max_rr, 2),
            'samples_counted': len(rr_ratios)
        }
    }