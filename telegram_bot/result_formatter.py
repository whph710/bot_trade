"""
Обновленный result_formatter.py - отдельный пост для каждого сигнала с анализом
"""

from typing import Dict, Any, List


def format_bot_result(result: Dict[str, Any]) -> str:
    """Форматировать результат работы торгового бота для вывода в Telegram"""

    bot_result = result.get('result', 'UNKNOWN')
    total_time = result.get('total_time', 0)
    stats = result.get('stats', {})

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
        f"⏱️ <b>Время:</b> {total_time:.1f}s\n\n"
    )

    result_text += "<b>📊 СТАТИСТИКА:</b>\n"
    result_text += f"  • Пар отсканировано: {stats.get('pairs_scanned', 0)}\n"
    result_text += f"  • Сигналов найдено: {stats.get('signal_pairs_found', 0)}\n"
    result_text += f"  • AI отобрал: {stats.get('ai_selected', 0)}\n"
    result_text += f"  • Проанализировано: {stats.get('analyzed', 0)}\n"
    result_text += f"  • ✅ Одобрено: {stats.get('validated_signals', 0)}\n"
    result_text += f"  • ❌ Отклонено: {stats.get('rejected_signals', 0)}\n"

    if stats.get('processing_speed'):
        result_text += f"  • Скорость: {stats.get('processing_speed', 0):.1f} пар/сек\n"

    if result.get('validation_skipped_reason'):
        result_text += f"\n⏰ <b>Причина пропуска:</b>\n{result['validation_skipped_reason']}"

    if result.get('error'):
        result_text += f"\n❌ <b>Ошибка:</b>\n{result['error']}"

    return result_text


def format_signal_individual(signal: Dict[str, Any], index: int = 1, total: int = 1) -> str:
    """
    Форматировать один сигнал для отдельного поста

    Args:
        signal: Словарь с данными сигнала
        index: Номер сигнала
        total: Всего сигналов

    Returns:
        Отформатированный текст для поста
    """
    symbol = signal.get('symbol', 'N/A')
    signal_type = signal.get('signal', 'N/A')
    confidence = signal.get('confidence', 0)
    entry = signal.get('entry_price', 0)
    stop = signal.get('stop_loss', 0)
    tp_levels = signal.get('take_profit_levels', [0, 0, 0])
    rr = signal.get('risk_reward_ratio', 0)
    hold = signal.get('hold_duration_minutes', 0)
    analysis = signal.get('analysis', '')
    market_cond = signal.get('market_conditions', '')
    validation_notes = signal.get('validation_notes', '')

    # Направление с цветным эмодзи
    dir_emoji = '🟢' if signal_type.upper() == 'LONG' else '🔴'

    # Основная информация - начинаем с тикера
    message = f"{dir_emoji} <b>{symbol}</b> | {signal_type}\n"
    message += f"{'━' * 40}\n\n"

    # Основные параметры
    message += f"<b>📊 ПАРАМЕТРЫ:</b>\n"
    message += f"  • Confidence: <b>{confidence}%</b>\n"
    message += f"  • Risk/Reward: <b>1:{rr:.2f}</b>\n"
    message += f"  • Длительность: <b>{hold} мин</b>\n\n"

    # Уровни входа/выхода
    message += f"<b>💰 УРОВНИ ВХОДА/ВЫХОДА:</b>\n"
    message += f"  • Entry:  <code>${entry:.4f}</code>\n"
    message += f"  • Stop:   <code>${stop:.4f}</code>\n"
    message += f"  • TP1:    <code>${tp_levels[0]:.4f}</code>\n"
    message += f"  • TP2:    <code>${tp_levels[1]:.4f}</code>\n"
    message += f"  • TP3:    <code>${tp_levels[2]:.4f}</code>\n\n"

    # Рыночные условия если есть
    if market_cond:
        message += f"<b>📈 РЫНОК:</b>\n"
        message += f"  {market_cond}\n\n"

    # Обоснование анализа (главная часть)
    if analysis:
        message += f"<b>📝 АНАЛИЗ:</b>\n"
        message += f"<i>{analysis}</i>\n\n"

    # Заметки валидации
    if validation_notes:
        message += f"<b>✓ ВАЛИДАЦИЯ:</b>\n"
        message += f"  {validation_notes}\n"

    return message


async def send_individual_signals_to_group(bot, chat_id: int, signals: List[Dict[str, Any]]) -> int:
    """
    Отправить каждый сигнал отдельным постом в группу

    Args:
        bot: Экземпляр Bot
        chat_id: ID чата
        signals: Список сигналов

    Returns:
        Количество успешно отправленных сигналов
    """
    if not signals:
        return 0

    sent_count = 0
    total_signals = len(signals)

    for index, signal in enumerate(signals, 1):
        try:
            formatted_signal = format_signal_individual(signal, index, total_signals)

            await bot.send_message(
                chat_id=chat_id,
                text=formatted_signal,
                parse_mode="HTML"
            )

            sent_count += 1
            print(f"✅ Sent signal {index}/{total_signals} to group")

            # Небольшая задержка между постами (чтобы Telegram не заблокировал)
            import asyncio
            await asyncio.sleep(0.5)

        except Exception as e:
            print(f"❌ Error sending signal {index}/{total_signals}: {e}")
            continue

    return sent_count


async def send_group_message_safe(bot, chat_id: int, text: str, max_length: int = 4096) -> bool:
    """
    Безопасно отправить одно сообщение, разбивая на части если нужно

    Args:
        bot: Экземпляр Bot
        chat_id: ID чата
        text: Текст сообщения
        max_length: Максимальная длина сообщения Telegram

    Returns:
        True если успешно, False если ошибка
    """
    try:
        if len(text) <= max_length:
            await bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode="HTML"
            )
            return True
        else:
            # Разбить на части по max_length
            parts = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            for part in parts:
                await bot.send_message(
                    chat_id=chat_id,
                    text=part,
                    parse_mode="HTML"
                )
            return True
    except Exception as e:
        print(f"❌ Error sending message: {e}")
        return False