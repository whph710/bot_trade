import asyncio
import json
import logging
from func_trade import get_signal_details

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def main():
    """
    Главная функция - запуск EMA+TSI анализа торговых пар
    """
    try:
        logger.info("🚀 ЗАПУСК EMA+TSI ТОРГОВОГО БОТА")
        logger.info("=" * 70)

        # Создаем экземпляр анализатора
        # Можно настроить параметры индикаторов:
        analyzer = get_signal_details(
            ema1_period=7,  # Быстрая EMA
            ema2_period=14,  # Средняя EMA
            ema3_period=28,  # Медленная EMA
            tsi_long=25,  # Длинный период TSI
            tsi_short=13,  # Короткий период TSI
            tsi_signal=13  # Период сигнальной линии TSI
        )

        # Запускаем полный анализ
        result = await analyzer.get_signal_details()

        # Обрабатываем результат
        if result['success']:
            logger.info("=" * 70)
            logger.info("✅ АНАЛИЗ УСПЕШНО ЗАВЕРШЕН")
            logger.info("=" * 70)

            # Выводим основную статистику
            logger.info(f"⏱️  Время выполнения: {result['execution_time']:.2f} секунд")
            logger.info(f"📊 Проверено пар: {result['total_pairs_checked']}")

            # Детальная статистика сигналов
            signal_counts = result['signal_counts']
            logger.info(f"📈 LONG сигналы: {signal_counts['LONG']}")
            logger.info(f"📉 SHORT сигналы: {signal_counts['SHORT']}")
            logger.info(f"⚪ Без сигналов: {signal_counts['NO_SIGNAL']}")

            total_signals = signal_counts['LONG'] + signal_counts['SHORT']
            logger.info(f"🎯 Всего сигналов найдено: {total_signals}")

            # Анализ ИИ
            ai_result = result['result']
            if ai_result and 'ai_analysis' in ai_result:
                logger.info("=" * 70)
                logger.info("🤖 РЕЗУЛЬТАТ АНАЛИЗА НЕЙРОСЕТИ:")
                logger.info("=" * 70)
                print(ai_result['ai_analysis'])

                # Сохраняем результат в файл для дальнейшего использования
                save_results_to_file(result)

            else:
                logger.warning("⚠️  Анализ ИИ не выполнен или вернул пустой результат")

        else:
            logger.error("❌ АНАЛИЗ ЗАВЕРШИЛСЯ С ОШИБКОЙ")
            logger.error(f"Причина: {result.get('message', 'Unknown error')}")

            if 'signal_counts' in result:
                logger.info(f"📊 Проверено пар: {result['total_pairs_checked']}")
                logger.info(f"Статистика: {result['signal_counts']}")

    except KeyboardInterrupt:
        logger.info("🛑 Анализ прерван пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка в main(): {e}")
        import traceback
        logger.error(traceback.format_exc())


def save_results_to_file(result):
    """
    Сохранение результатов анализа в файл

    Args:
        result: Результат анализа
    """
    try:
        # Подготавливаем данные для сохранения
        save_data = {
            'execution_time': result['execution_time'],
            'total_pairs_checked': result['total_pairs_checked'],
            'signal_counts': result['signal_counts'],
            'ai_analysis': result['result'].get('ai_analysis', ''),
            'signals_found': []
        }

        # Добавляем информацию о найденных сигналах
        if 'pairs_data' in result['result']:
            for pair_info in result['result']['pairs_data']:
                save_data['signals_found'].append({
                    'pair': pair_info['pair'],
                    'signal': pair_info['signal'],
                    'last_price': pair_info['details']['last_price'],
                    'ema_alignment': pair_info['details']['ema_alignment'],
                    'tsi_crossover_direction': pair_info['details']['tsi_crossover_direction'],
                    'reason': pair_info['details']['reason']
                })

        # Сохраняем в JSON файл
        with open('trading_analysis_result.json', 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info("💾 Результаты сохранены в файл 'trading_analysis_result.json'")

    except Exception as e:
        logger.error(f"❌ Ошибка сохранения результатов: {e}")


if __name__ == "__main__":
    """
    Точка входа в программу
    """
    try:
        # Запускаем асинхронную главную функцию
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Программа остановлена пользователем")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()