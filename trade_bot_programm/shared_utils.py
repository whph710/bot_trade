"""
Shared utilities for trading bot - ENHANCED WITH CRITICAL BLOCKERS
"""

import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def fallback_validation(signal: Dict, min_risk_reward_ratio: float = 2.0) -> Dict:
    """
    Enhanced fallback validation with CRITICAL SAFETY CHECKS

    КРИТИЧНО: Добавлены проверки которые были пропущены в ALICEUSDT
    - BTC correlation blocking
    - Volume Profile overextension
    - RSI exhaustion
    """
    entry = signal.get('entry_price', 0)
    stop = signal.get('stop_loss', 0)
    tp_levels = signal.get('take_profit_levels', [0, 0, 0])

    if not isinstance(tp_levels, list):
        tp_levels = [float(tp_levels), float(tp_levels) * 1.1, float(tp_levels) * 1.2]

    # ========== НОВЫЕ КРИТИЧЕСКИЕ ПРОВЕРКИ ==========

    comp_data = signal.get('comprehensive_data', {})

    # 1. БЛОКИРОВКА ПО КОРРЕЛЯЦИИ (HIGHEST PRIORITY)
    corr_data = comp_data.get('correlation_data', {})

    if corr_data.get('should_block_signal', False):
        logger.warning(f"❌ BLOCKED: BTC correlation conflict for {signal.get('symbol', 'UNKNOWN')}")
        return {
            'approved': False,
            'rejection_reason': 'BTC correlation conflict - blocked by correlation analysis',
            'entry_price': entry,
            'stop_loss': stop,
            'take_profit_levels': tp_levels,
            'final_confidence': signal.get('confidence', 0),
            'validation_method': 'fallback_blocked',
            'market_conditions': 'Blocked by correlation',
            'key_levels': ''
        }

    # 2. ПРОВЕРКА OVEREXTENSION (Volume Profile)
    vp_analysis = comp_data.get('vp_analysis', {})
    poc_distance = vp_analysis.get('poc_analysis', {}).get('distance_to_poc_pct', 0)
    market_condition = vp_analysis.get('value_area_analysis', {}).get('market_condition', '')

    if market_condition == 'OVEREXTENDED' and poc_distance > 15:
        logger.warning(f"❌ BLOCKED: Price overextended {poc_distance:.1f}% from POC for {signal.get('symbol', 'UNKNOWN')}")
        return {
            'approved': False,
            'rejection_reason': f'Price overextended {poc_distance:.1f}% from POC - high reversion risk',
            'entry_price': entry,
            'stop_loss': stop,
            'take_profit_levels': tp_levels,
            'final_confidence': signal.get('confidence', 0),
            'validation_method': 'fallback_blocked',
            'market_conditions': f'Overextended {poc_distance:.1f}% from fair value',
            'key_levels': ''
        }

    # 3. ПРОВЕРКА RSI EXHAUSTION
    indicators_1h = comp_data.get('indicators_1h', {})
    indicators_4h = comp_data.get('indicators_4h', {})

    rsi_1h = indicators_1h.get('current', {}).get('rsi', 50)
    rsi_4h = indicators_4h.get('current', {}).get('rsi', 50)
    signal_type = signal.get('signal', 'NONE')

    # LONG при экстремальной перекупленности
    if signal_type == 'LONG' and rsi_1h > 75:
        logger.warning(f"❌ BLOCKED: RSI 1H extremely overbought ({rsi_1h:.1f}) for {signal.get('symbol', 'UNKNOWN')}")
        return {
            'approved': False,
            'rejection_reason': f'RSI 1H extremely overbought ({rsi_1h:.1f}) - momentum exhaustion',
            'entry_price': entry,
            'stop_loss': stop,
            'take_profit_levels': tp_levels,
            'final_confidence': signal.get('confidence', 0),
            'validation_method': 'fallback_blocked',
            'market_conditions': f'RSI exhaustion: 1H={rsi_1h:.1f}',
            'key_levels': ''
        }

    # SHORT при экстремальной перепроданности
    if signal_type == 'SHORT' and rsi_1h < 25:
        logger.warning(f"❌ BLOCKED: RSI 1H extremely oversold ({rsi_1h:.1f}) for {signal.get('symbol', 'UNKNOWN')}")
        return {
            'approved': False,
            'rejection_reason': f'RSI 1H extremely oversold ({rsi_1h:.1f}) - momentum exhaustion',
            'entry_price': entry,
            'stop_loss': stop,
            'take_profit_levels': tp_levels,
            'final_confidence': signal.get('confidence', 0),
            'validation_method': 'fallback_blocked',
            'market_conditions': f'RSI exhaustion: 1H={rsi_1h:.1f}',
            'key_levels': ''
        }

    # Оба таймфрейма в экстремальных зонах
    if signal_type == 'LONG' and rsi_1h > 70 and rsi_4h > 65:
        logger.warning(f"❌ BLOCKED: Both timeframes overbought (1H={rsi_1h:.1f}, 4H={rsi_4h:.1f}) for {signal.get('symbol', 'UNKNOWN')}")
        return {
            'approved': False,
            'rejection_reason': f'Both timeframes overbought: 1H={rsi_1h:.1f}, 4H={rsi_4h:.1f}',
            'entry_price': entry,
            'stop_loss': stop,
            'take_profit_levels': tp_levels,
            'final_confidence': signal.get('confidence', 0),
            'validation_method': 'fallback_blocked',
            'market_conditions': f'Multi-timeframe exhaustion: 1H={rsi_1h:.1f}, 4H={rsi_4h:.1f}',
            'key_levels': ''
        }

    if signal_type == 'SHORT' and rsi_1h < 30 and rsi_4h < 35:
        logger.warning(f"❌ BLOCKED: Both timeframes oversold (1H={rsi_1h:.1f}, 4H={rsi_4h:.1f}) for {signal.get('symbol', 'UNKNOWN')}")
        return {
            'approved': False,
            'rejection_reason': f'Both timeframes oversold: 1H={rsi_1h:.1f}, 4H={rsi_4h:.1f}',
            'entry_price': entry,
            'stop_loss': stop,
            'take_profit_levels': tp_levels,
            'final_confidence': signal.get('confidence', 0),
            'validation_method': 'fallback_blocked',
            'market_conditions': f'Multi-timeframe exhaustion: 1H={rsi_1h:.1f}, 4H={rsi_4h:.1f}',
            'key_levels': ''
        }

    # ========== ПРОДОЛЖЕНИЕ СУЩЕСТВУЮЩЕЙ ЛОГИКИ R/R ==========

    if entry > 0 and stop > 0 and tp_levels and tp_levels[0] > 0:
        risk = abs(entry - stop)
        reward = abs(tp_levels[1] - entry) if len(tp_levels) > 1 else abs(tp_levels[0] - entry)

        if risk > 0:
            rr_ratio = round(reward / risk, 2)
            if rr_ratio >= min_risk_reward_ratio:
                # Извлекаем данные для market_conditions
                market_data = comp_data.get('market_data', {})
                funding = market_data.get('funding_rate', {})
                oi = market_data.get('open_interest', {})
                orderbook = market_data.get('orderbook', {})

                funding_rate = funding.get('funding_rate', 0) if funding else 0
                oi_trend = oi.get('oi_trend', 'UNKNOWN') if oi else 'UNKNOWN'
                spread_pct = orderbook.get('spread_pct', 0) if orderbook else 0

                market_conditions = f"Funding: {funding_rate:.4f}%, OI: {oi_trend}, Spread: {spread_pct:.4f}%"

                # Формируем key_levels
                analysis_text = signal.get('analysis', '')
                if 'OB' in analysis_text or 'Order Block' in analysis_text:
                    key_levels = f"Entry: ${entry:.4f}, Stop: ${stop:.4f}, TP1: ${tp_levels[0]:.4f}, TP2: ${tp_levels[1]:.4f}, TP3: ${tp_levels[2]:.4f}"
                else:
                    key_levels = f"Entry: ${entry:.4f}, Stop: ${stop:.4f} (swing low/high), TP levels: ${tp_levels[0]:.4f}/${tp_levels[1]:.4f}/${tp_levels[2]:.4f}"

                logger.info(f"✓ Fallback validation PASSED: R/R {rr_ratio}, all critical checks OK")

                return {
                    'approved': True,
                    'final_confidence': signal['confidence'],
                    'entry_price': entry,
                    'stop_loss': stop,
                    'take_profit_levels': tp_levels,
                    'risk_reward_ratio': rr_ratio,
                    'hold_duration_minutes': 720,
                    'validation_method': 'fallback_enhanced',
                    'validation_notes': f'Enhanced fallback: R/R {rr_ratio}, passed critical checks (correlation, VP, RSI)',
                    'market_conditions': market_conditions,
                    'key_levels': key_levels
                }

    return {
        'approved': False,
        'rejection_reason': 'Fallback validation failed: insufficient R/R or invalid data',
        'entry_price': entry,
        'stop_loss': stop,
        'take_profit_levels': tp_levels,
        'final_confidence': signal.get('confidence', 0),
        'validation_method': 'fallback_blocked',
        'market_conditions': 'Validation failed',
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