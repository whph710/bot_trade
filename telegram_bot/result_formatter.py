"""
Форматирование результатов торгового бота для Telegram
Файл: bot_trade/telegram_bot/result_formatter.py
"""

from typing import Dict, Any


def format_bot_result(result: Dict[str, Any]) -> str:
    """Форматировать результат работы торгового бота для вывода в Telegram"""

    bot_result = result.get('result', 'UNKNOWN')
    total_time = result.get('total_time', 0)
    stats = result.get('stats', {})

    # Определить эмодзи по результату
    emoji_map = {
        'SUCCESS': '✅',
        'NO_VALIDATED_SIGNALS': '⚠️',
        'NO_SIGNAL_PAIRS': '❌',
        'NO_AI_SELECTION': '❌',
        'NO_ANALYSIS_SIGNALS': '❌',
        'VALIDATION_SKIPPED': '⏱️',
        'ERROR': '💥'
    }

    emoji = emoji_map.get(bot_result, '❓')

    result_text = (
        f"<b>{emoji} РЕЗУЛЬТАТ: {bot_result}</b>\n\n"
        f"⏱️ <b>Время выполнения:</b> {total_time:.1f}s\n\n"
    )

    # Статистика
    result_text += "<b>📊 СТАТИСТИКА:</b>\n"
    result_text += f"  • Пар отсканировано: {stats.get('pairs_scanned', 0)}\n"
    result_text += f"  • Сигналов найдено: {stats.get('signal_pairs_found', 0)}\n"
    result_text += f"  • AI отобрал: {stats.get('ai_selected', 0)}\n"
    result_text += f"  • Проанализировано: {stats.get('analyzed', 0)}\n"
    result_text += f"  • ✅ Одобрено: {stats.get('validated_signals', 0)}\n"
    result_text += f"  • ❌ Отклонено: {stats.get('rejected_signals', 0)}\n"

    if stats.get('processing_speed'):
        result_text += f"  • Скорость: {stats.get('processing_speed', 0):.1f} пар/сек\n"

    # Дополнительная информация
    if result.get('validation_skipped_reason'):
        result_text += f"\n⏰ <b>Причина пропуска:</b>\n{result['validation_skipped_reason']}"

    if result.get('error'):
        result_text += f"\n❌ <b>Ошибка:</b>\n{result['error']}"

    # Статистика валидации
    vstats = result.get('validation_stats', {})
    if vstats.get('total', 0) > 0:
        result_text += f"\n\n<b>📈 СТАТИСТИКА ВАЛИДАЦИИ:</b>\n"
        result_text += f"  • Успешность: {vstats.get('approval_rate', 0):.1f}%\n"
        result_text += f"  • Средний R/R: 1:{vstats.get('avg_risk_reward', 0):.2f}\n"

        rr_stats = vstats.get('rr_stats', {})
        if rr_stats.get('samples_counted', 0) > 0:
            result_text += f"  • Диапазон R/R: 1:{rr_stats.get('min_rr', 0):.2f} - 1:{rr_stats.get('max_rr', 0):.2f}\n"

    return result_text


def format_signal_for_group(signal: Dict[str, Any], index: int = 1) -> str:
    """Форматировать один сигнал для группы"""

    symbol = signal.get('symbol', 'UNKNOWN')
    signal_type = signal.get('signal', 'UNKNOWN')
    confidence = signal.get('confidence', 0)
    entry = signal.get('entry_price', 0)
    stop = signal.get('stop_loss', 0)
    tp_levels = signal.get('take_profit_levels', [0, 0, 0])
    rr_ratio = signal.get('risk_reward_ratio', 0)
    hold_time = signal.get('hold_duration_minutes', 0)
    notes = signal.get('validation_notes', '')

    emoji = '🔴' if signal_type == 'SHORT' else '🟢'

    text = (
        f"{emoji} <b>#{index}. {symbol} {signal_type}</b>\n"
        f"📊 Confidence: <b>{confidence}%</b>\n"
        f"💰 Entry: <code>${entry:.2f}</code>\n"
        f"🛑 Stop: <code>${stop:.2f}</code>\n"
        f"🎯 TP1: <code>${tp_levels[0]:.2f}</code>\n"
        f"🎯 TP2: <code>${tp_levels[1]:.2f}</code>\n"
        f"🎯 TP3: <code>${tp_levels[2]:.2f}</code>\n"
        f"📈 R/R: <b>1:{rr_ratio:.2f}</b>\n"
        f"⏱️ Hold: <b>{hold_time}min</b>\n"
    )

    if notes:
        text += f"📝 <i>{notes[:100]}</i>\n"

    return text