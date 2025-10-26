"""
Упрощенный result_formatter.py - базовое форматирование результатов
МОДИФИКАЦИЯ: Stage 4 убран, термины изменены
"""

from typing import Dict, Any


def format_bot_result(result: Dict[str, Any], run_stats: Dict[str, int] = None) -> str:
    """
    Форматировать результат работы торгового бота для вывода в Telegram

    Args:
        result: Результат работы бота
        run_stats: Статистика запусков (total_runs, today_runs)
    """

    bot_result = result.get('result', 'UNKNOWN')
    total_time = result.get('total_time', 0)
    stats = result.get('stats', {})

    emoji_map = {
        'SUCCESS': '✅',
        'NO_APPROVED_SIGNALS': '⚠️',
        'NO_SIGNAL_PAIRS': '❌',
        'NO_AI_SELECTION': '❌',
        'NO_ANALYSIS_SIGNALS': '❌',  # Новое название
        'TRADING_HOURS_BLOCKED': '⏱️',
        'ERROR': '💥'
    }

    emoji = emoji_map.get(bot_result, '❓')

    result_text = (
        f"<b>{emoji} РЕЗУЛЬТАТ: {bot_result}</b>\n\n"
        f"⏱️ <b>Время:</b> {total_time:.1f}s\n\n"
    )

    result_text += "<b>📊 СТАТИСТИКА АНАЛИЗА:</b>\n"
    result_text += f"  • Пар отсканировано: {stats.get('pairs_scanned', 0)}\n"
    result_text += f"  • Сигналов найдено: {stats.get('signal_pairs_found', 0)}\n"
    result_text += f"  • AI отобрал: {stats.get('ai_selected', 0)}\n"
    result_text += f"  • Проанализировано: {stats.get('analyzed', 0)}\n"
    result_text += f"  • ✅ Одобрено (Stage 3): {stats.get('approved_signals', 0)}\n"

    if stats.get('processing_speed'):
        result_text += f"  • Скорость: {stats.get('processing_speed', 0):.1f} пар/сек\n"

    # Добавляем статистику запусков если передана
    if run_stats:
        result_text += f"\n<b>🔢 СТАТИСТИКА ЗАПУСКОВ:</b>\n"
        result_text += f"  • Всего запусков: {run_stats.get('total_runs', 0)}\n"
        result_text += f"  • Сегодня: {run_stats.get('today_runs', 0)}\n"

    if result.get('validation_skipped_reason'):
        result_text += f"\n⏰ <b>Причина пропуска:</b>\n{result['validation_skipped_reason']}"

    if result.get('error'):
        result_text += f"\n❌ <b>Ошибка:</b>\n{result['error']}"

    return result_text


async def send_formatted_signals_to_group(bot, chat_id: int, formatted_signals: list[str]) -> int:
    """
    Отправить уже отформатированные сигналы в группу

    Args:
        bot: Экземпляр Bot
        chat_id: ID чата
        formatted_signals: Список уже отформатированных HTML-текстов

    Returns:
        Количество успешно отправленных сигналов
    """
    if not formatted_signals:
        return 0

    sent_count = 0
    total_signals = len(formatted_signals)

    for index, formatted_text in enumerate(formatted_signals, 1):
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=formatted_text,
                parse_mode="HTML"
            )

            sent_count += 1
            print(f"✅ Sent signal {index}/{total_signals} to group")

            # Небольшая задержка между постами
            import asyncio
            await asyncio.sleep(0.5)

        except Exception as e:
            print(f"❌ Error sending signal {index}/{total_signals}: {e}")
            continue

    return sent_count


async def send_group_message_safe(bot, chat_id: int, text: str, max_length: int = 4096) -> bool:
    """
    Безопасно отправить одно сообщение, разбивая на части если нужно
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