"""
Shared utilities for trading bot - FINAL
Файл: trade_bot_programm/shared_utils.py
"""

import json
import logging
from typing import Dict, Optional
from validation_engine import ValidationEngine

logger = logging.getLogger(__name__)


def fallback_validation(signal: Dict, comp_data: Dict) -> Dict:
    """
    Fallback validation с использованием ValidationEngine

    МОДИФИКАЦИЯ: Единственный метод валидации после удаления Stage 4
    """
    symbol = signal.get('symbol', 'UNKNOWN')

    # Используем ValidationEngine для критических проверок
    passed, reasons = ValidationEngine.run_all_checks(signal, comp_data)

    if not passed:
        return {
            'approved': False,
            'rejection_reason': '; '.join(reasons),
            'symbol': symbol,
            'confidence': 0,
            'fallback_used': True,
            'entry_price': signal.get('entry_price', 0),
            'stop_loss': signal.get('stop_loss', 0),
            'take_profit_levels': signal.get('take_profit_levels', [0, 0, 0]),
            'validation_method': 'fallback_blocked',
            'market_conditions': 'Blocked by ValidationEngine',
            'key_levels': ''
        }

    # Если прошло - базовая валидация R/R
    entry = signal.get('entry_price', 0)
    stop = signal.get('stop_loss', 0)
    tp_levels = signal.get('take_profit_levels', [0, 0, 0])

    if not isinstance(tp_levels, list):
        tp_levels = [float(tp_levels), float(tp_levels) * 1.1, float(tp_levels) * 1.2]

    # R/R check
    if entry > 0 and stop > 0 and tp_levels and tp_levels[0] > 0:
        risk = abs(entry - stop)
        reward = abs(tp_levels[1] - entry) if len(tp_levels) > 1 else abs(tp_levels[0] - entry)

        if risk > 0:
            rr_ratio = round(reward / risk, 2)

            if rr_ratio >= 2.5:  # Минимум для swing
                # Извлекаем данные для market_conditions
                market_data = comp_data.get('market_data', {})
                funding = market_data.get('funding_rate', {})
                oi = market_data.get('open_interest', {})
                orderbook = market_data.get('orderbook', {})

                funding_rate = funding.get('funding_rate', 0) if funding else 0
                oi_trend = oi.get('oi_trend', 'UNKNOWN') if oi else 'UNKNOWN'
                spread_pct = orderbook.get('spread_pct', 0) if orderbook else 0

                market_conditions = f"Funding: {funding_rate:.4f}%, OI: {oi_trend}, Spread: {spread_pct:.4f}%"
                key_levels = f"Entry: ${entry:.4f}, Stop: ${stop:.4f}, TP1: ${tp_levels[0]:.4f}, TP2: ${tp_levels[1]:.4f}, TP3: ${tp_levels[2]:.4f}"

                return {
                    'approved': True,
                    'symbol': symbol,
                    'confidence': signal.get('confidence', 75),
                    'entry_price': entry,
                    'stop_loss': stop,
                    'take_profit_levels': tp_levels,
                    'risk_reward_ratio': rr_ratio,
                    'hold_duration_minutes': 720,
                    'fallback_used': True,
                    'validation_notes': f'Fallback validation passed (R/R {rr_ratio}:1)',
                    'validation_method': 'fallback_enhanced',
                    'market_conditions': market_conditions,
                    'key_levels': key_levels
                }

    return {
        'approved': False,
        'rejection_reason': 'Fallback validation failed: insufficient R/R (<2.5:1)',
        'symbol': symbol,
        'confidence': 0,
        'fallback_used': True,
        'entry_price': entry,
        'stop_loss': stop,
        'take_profit_levels': tp_levels,
        'validation_method': 'fallback_blocked',
        'market_conditions': 'R/R too low',
        'key_levels': ''
    }


def extract_json_from_response(text: str) -> Optional[Dict]:
    """Extract JSON from AI response (unified parser)"""
    if not text or len(text) < 10:
        return None

    try:
        text = text.strip()

        # Remove markdown code blocks
        if '```json' in text:
            start = text.find('```json') + 7
            end = text.find('```', start)
            if end != -1:
                text = text[start:end].strip()
        elif '```' in text:
            lines = text.split('\n')
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith('```'):
                    if in_json:
                        break
                    in_json = True
                    continue
                if in_json:
                    json_lines.append(line)
            text = '\n'.join(json_lines)

        # Find JSON object
        start_idx = text.find('{')
        if start_idx == -1:
            return None

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
        logger.warning(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        logger.error(f"Parsing error: {e}")
        return None


def normalize_take_profit_levels(tp_levels, entry_price: float = 0, signal_type: str = 'LONG') -> list:
    """Normalize take profit levels to always be a list of 3 values"""
    if isinstance(tp_levels, list) and len(tp_levels) >= 3:
        return [float(tp) for tp in tp_levels[:3]]
    elif isinstance(tp_levels, list) and len(tp_levels) > 0:
        base_tp = float(tp_levels[0])
        return [base_tp, base_tp * 1.1, base_tp * 1.2]
    elif tp_levels and entry_price > 0:
        # Single value provided
        base_tp = float(tp_levels)
        return [base_tp, base_tp * 1.1, base_tp * 1.2]
    else:
        # Fallback to zeros
        return [0, 0, 0]