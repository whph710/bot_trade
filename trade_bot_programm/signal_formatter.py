"""
Template-based сигнальный форматтер
Заменяет AI форматирование на простой Python template
Файл: trade_bot_programm/signal_formatter.py
"""

from typing import Dict, List
from logging_config import setup_module_logger

logger = setup_module_logger(__name__)

SIGNAL_TEMPLATE = """
{emoji} <b>{symbol}</b> | {signal}
━━━━━━━━━━━━━━━━━━━━━━

<b>📊 ПАРАМЕТРЫ:</b>

• Confidence: <b>{confidence}%</b>
• Risk/Reward: <b>1:{rr_ratio:.1f}</b>
• Длительность: <b>{hold_duration} мин</b>

<b>💰 УРОВНИ ВХОДА/ВЫХОДА:</b>

• Entry:  <code>${entry_price:.4f}</code>
• Stop:   <code>${stop_loss:.4f}</code>
• TP1:    <code>${tp1:.4f}</code>
• TP2:    <code>${tp2:.4f}</code>
• TP3:    <code>${tp3:.4f}</code>

<b>📈 РЫНОК:</b>

{market_conditions}

<b>📝 АНАЛИЗ:</b>

<i>{analysis}</i>

<b>🏁 ВАЛИДАЦИЯ:</b>

{validation_notes}
"""


def format_signal_simple(signal_data: Dict) -> str:
    """
    Форматирует сигнал используя template (БЕЗ AI)

    Args:
        signal_data: Данные сигнала

    Returns:
        Отформатированная строка для Telegram
    """
    try:
        emoji = '🟢' if signal_data.get('signal') == 'LONG' else '🔴'

        tp_levels = signal_data.get('take_profit_levels', [0, 0, 0])
        if len(tp_levels) < 3:
            tp_levels = tp_levels + [0] * (3 - len(tp_levels))

        # Рассчитываем R/R
        entry = signal_data.get('entry_price', 0)
        stop = signal_data.get('stop_loss', 0)
        risk = abs(entry - stop)
        reward = abs(tp_levels[1] - entry) if len(tp_levels) > 1 else 0
        rr_ratio = (reward / risk) if risk > 0 else 0

        # Market conditions
        market_cond = signal_data.get('market_conditions', 'N/A')
        if isinstance(market_cond, dict):
            market_cond = f"Trend: {market_cond.get('trend', 'N/A')}, Vol: {market_cond.get('volume', 'N/A')}"

        # Analysis
        analysis = signal_data.get('analysis', 'No detailed analysis available')
        if len(analysis) > 500:
            analysis = analysis[:497] + "..."

        # Validation notes
        validation = signal_data.get('validation_notes', 'Signal validated')

        return SIGNAL_TEMPLATE.format(
            emoji=emoji,
            symbol=signal_data.get('symbol', 'UNKNOWN'),
            signal=signal_data.get('signal', 'NONE'),
            confidence=signal_data.get('confidence', 0),
            rr_ratio=rr_ratio,
            hold_duration=signal_data.get('hold_duration_minutes', 720),
            entry_price=entry,
            stop_loss=stop,
            tp1=tp_levels[0],
            tp2=tp_levels[1],
            tp3=tp_levels[2],
            market_conditions=market_cond,
            analysis=analysis,
            validation_notes=validation
        ).strip()

    except Exception as e:
        logger.error(f"Error formatting signal: {e}")
        return f"⚠️ Error formatting signal for {signal_data.get('symbol', 'UNKNOWN')}"


def format_multiple_signals(signals: List[Dict]) -> List[str]:
    """
    Форматирует несколько сигналов

    Args:
        signals: Список JSON данных сигналов

    Returns:
        Список отформатированных текстов
    """
    if not signals:
        return []

    formatted = []
    for signal in signals:
        try:
            formatted_text = format_signal_simple(signal)
            formatted.append(formatted_text)
        except Exception as e:
            logger.error(f"Error formatting signal {signal.get('symbol', 'UNKNOWN')}: {e}")
            continue

    logger.info(f"Formatted {len(formatted)}/{len(signals)} signals successfully")
    return formatted