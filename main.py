"""
Исправленный скальпинговый бот - 3 четких этапа без избыточности
ЭТАП 1: Быстрое сканирование всех пар → базовые индикаторы
ЭТАП 2: ИИ отбор лучших (15m данные + индикаторы) → 3-5 пар
ЭТАП 3: Детальный анализ финалистов (полные 5m+15m данные) → сигналы
"""

import asyncio
import logging
import time
import json
from typing import List, Dict, Any

from config import config
# ИСПРАВЛЕННЫЕ ИМПОРТЫ:
from func_async import get_trading_pairs, fetch_klines, batch_fetch_klines, cleanup as cleanup_api
from func_trade import calculate_basic_indicators, calculate_ai_indicators, check_basic_signal
from deepseek import ai_select_pairs, ai_analyze_pair

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SimplifiedScalpingBot:
    """Упрощенный скальпинговый бот - только необходимые функции"""

    def __init__(self):
        self.processed_pairs = 0
        self.session_start = time.time()

    async def stage1_quick_scan(self) -> List[Dict]:
        """
        ЭТАП 1: Быстрое сканирование всех пар
        Возвращает пары с базовыми сигналами
        """
        start_time = time.time()
        logger.info("🔍 ЭТАП 1: Быстрое сканирование...")

        # Получаем список торговых пар
        pairs = await get_trading_pairs()
        if not pairs:
            logger.error("Пары не получены")
            return []

        logger.info(f"   Сканируем {len(pairs)} пар")

        # Подготавливаем запросы для быстрого сканирования
        requests = [
            {'symbol': pair, 'interval': '5', 'limit': config.QUICK_SCAN_5M}
            for pair in pairs
        ]

        # Массовое получение данных
        results = await batch_fetch_klines(requests)

        promising_pairs = []

        # Обрабатываем результаты
        for result in results:
            if not result.get('success') or len(result['klines']) < 20:
                continue

            symbol = result['symbol']
            klines = result['klines']

            # Рассчитываем базовые индикаторы
            indicators = calculate_basic_indicators(klines)
            if not indicators:
                continue

            # Проверяем базовый сигнал
            signal_check = check_basic_signal(indicators)

            if signal_check['signal'] and signal_check['confidence'] >= config.MIN_CONFIDENCE:
                promising_pairs.append({
                    'symbol': symbol,
                    'confidence': signal_check['confidence'],
                    'direction': signal_check['direction'],
                    'indicators': indicators
                })

        # Сортируем по уверенности
        promising_pairs.sort(key=lambda x: x['confidence'], reverse=True)

        elapsed = time.time() - start_time
        self.processed_pairs = len(results)

        logger.info(f"   ✅ Найдено {len(promising_pairs)} перспективных пар за {elapsed:.1f}сек")
        logger.info(f"   Скорость: {len(pairs) / elapsed:.0f} пар/сек")

        return promising_pairs[:config.MAX_PAIRS_TO_AI]

    async def stage2_ai_selection(self, promising_pairs: List[Dict]) -> List[str]:
        """
        ЭТАП 2: ИИ отбор лучших пар
        Загружает 15m данные + индикаторы для каждой пары
        """
        start_time = time.time()
        logger.info(f"🤖 ЭТАП 2: ИИ отбор из {len(promising_pairs)} пар...")

        if not promising_pairs:
            return []

        # Получаем 15m данные для ИИ анализа
        requests_15m = [
            {'symbol': pair['symbol'], 'interval': '15', 'limit': config.AI_SELECT_15M}
            for pair in promising_pairs
        ]

        results_15m = await batch_fetch_klines(requests_15m)

        # Подготавливаем данные для ИИ
        ai_data = []
        for pair_data in promising_pairs:
            symbol = pair_data['symbol']

            # Находим соответствующие 15m данные
            candles_15m = None
            for result in results_15m:
                if result['symbol'] == symbol and result.get('success'):
                    candles_15m = result['klines']
                    break

            if not candles_15m or len(candles_15m) < 20:
                continue

            # Рассчитываем индикаторы для ИИ
            indicators_15m = calculate_ai_indicators(candles_15m, config.AI_SELECT_INDICATORS)

            ai_data.append({
                'symbol': symbol,
                'confidence': pair_data['confidence'],
                'direction': pair_data['direction'],
                'candles_15m': candles_15m,
                'indicators': indicators_15m
            })

        if not ai_data:
            logger.warning("   Нет данных для ИИ отбора")
            return []

        # ИИ отбор
        selected_pairs = await ai_select_pairs(ai_data)

        elapsed = time.time() - start_time
        logger.info(f"   ✅ ИИ выбрал {len(selected_pairs)} пар за {elapsed:.1f}сек")
        if selected_pairs:
            logger.info(f"   Финалисты: {', '.join(selected_pairs)}")

        return selected_pairs

    async def stage3_detailed_analysis(self, selected_pairs: List[str]) -> List[Dict]:
        """
        ЭТАП 3: Детальный анализ финалистов
        Загружает полные данные 5m+15m для каждой пары отдельно
        """
        start_time = time.time()
        logger.info(f"📊 ЭТАП 3: Детальный анализ {len(selected_pairs)} пар...")

        if not selected_pairs:
            return []

        final_signals = []

        # Анализируем каждую пару отдельно для максимальной детализации
        for symbol in selected_pairs:
            logger.info(f"   Анализ {symbol}...")

            # Получаем полные данные для обоих таймфреймов
            klines_5m_task = fetch_klines(symbol, '5', config.FINAL_5M)
            klines_15m_task = fetch_klines(symbol, '15', config.FINAL_15M)

            klines_5m, klines_15m = await asyncio.gather(klines_5m_task, klines_15m_task)

            if (not klines_5m or len(klines_5m) < 100 or
                    not klines_15m or len(klines_15m) < 50):
                logger.warning(f"   Недостаточно данных для {symbol}")
                continue

            # Рассчитываем полные индикаторы
            indicators_5m = calculate_ai_indicators(klines_5m, config.FINAL_INDICATORS)
            indicators_15m = calculate_ai_indicators(klines_15m, config.FINAL_INDICATORS)

            if not indicators_5m or not indicators_15m:
                logger.warning(f"   Ошибка расчета индикаторов для {symbol}")
                continue

            # Детальный ИИ анализ
            analysis = await ai_analyze_pair(
                symbol, klines_5m, klines_15m, indicators_5m, indicators_15m
            )

            if analysis['signal'] != 'NO_SIGNAL' and analysis['confidence'] >= config.MIN_CONFIDENCE:
                final_signals.append(analysis)
                logger.info(f"   ✅ {symbol}: {analysis['signal']} ({analysis['confidence']}%)")
            else:
                logger.info(f"   ❌ {symbol}: {analysis['signal']} ({analysis['confidence']}%)")

        elapsed = time.time() - start_time
        logger.info(f"   ✅ Анализ завершен за {elapsed:.1f}сек")

        return final_signals

    async def run_full_cycle(self) -> Dict[str, Any]:
        """Полный цикл работы бота"""
        cycle_start = time.time()

        logger.info("🚀 ЗАПУСК ПОЛНОГО ЦИКЛА")
        logger.info("=" * 50)

        try:
            # ЭТАП 1: Быстрое сканирование
            stage1_start = time.time()
            promising_pairs = await self.stage1_quick_scan()
            stage1_time = time.time() - stage1_start

            if not promising_pairs:
                return {
                    'result': 'NO_PROMISING_PAIRS',
                    'stage1_time': stage1_time,
                    'total_time': time.time() - cycle_start
                }

            # ЭТАП 2: ИИ отбор
            stage2_start = time.time()
            selected_pairs = await self.stage2_ai_selection(promising_pairs)
            stage2_time = time.time() - stage2_start

            if not selected_pairs:
                return {
                    'result': 'NO_AI_SELECTION',
                    'stage1_time': stage1_time,
                    'stage2_time': stage2_time,
                    'promising_pairs': len(promising_pairs),
                    'total_time': time.time() - cycle_start
                }

            # ЭТАП 3: Детальный анализ
            stage3_start = time.time()
            final_signals = await self.stage3_detailed_analysis(selected_pairs)
            stage3_time = time.time() - stage3_start

            total_time = time.time() - cycle_start

            # Формируем результат
            result = {
                'result': 'SUCCESS' if final_signals else 'NO_FINAL_SIGNALS',
                'timing': {
                    'stage1_scan': round(stage1_time, 2),
                    'stage2_ai_select': round(stage2_time, 2),
                    'stage3_analysis': round(stage3_time, 2),
                    'total': round(total_time, 2)
                },
                'stats': {
                    'pairs_processed': self.processed_pairs,
                    'promising_found': len(promising_pairs),
                    'ai_selected': len(selected_pairs),
                    'final_signals': len(final_signals),
                    'processing_speed': round(self.processed_pairs / stage1_time, 1)
                },
                'selected_pairs': selected_pairs,
                'signals': final_signals
            }

            # Логирование результата
            logger.info("📈 РЕЗУЛЬТАТЫ ЦИКЛА:")
            logger.info(
                f"   Время: сканирование {stage1_time:.1f}с | ИИ отбор {stage2_time:.1f}с | анализ {stage3_time:.1f}с")
            logger.info(
                f"   Воронка: {self.processed_pairs} → {len(promising_pairs)} → {len(selected_pairs)} → {len(final_signals)}")
            logger.info(f"   Скорость: {self.processed_pairs / stage1_time:.0f} пар/сек")

            if final_signals:
                logger.info("🎯 ТОРГОВЫЕ СИГНАЛЫ:")
                for signal in final_signals:
                    logger.info(f"   {signal['symbol']}: {signal['signal']} ({signal['confidence']}%)")

            return result

        except Exception as e:
            logger.error(f"❌ Критическая ошибка цикла: {e}")
            return {
                'result': 'ERROR',
                'error': str(e),
                'total_time': time.time() - cycle_start
            }

    async def cleanup(self):
        """Очистка ресурсов"""
        await cleanup_api()


async def main():
    """Главная функция"""
    print("🤖 УПРОЩЕННЫЙ СКАЛЬПИНГОВЫЙ БОТ")
    print(f"⚙️  Конфигурация:")
    print(f"   Этапы данных: {config.QUICK_SCAN_5M}→{config.AI_SELECT_15M}→{config.FINAL_5M}/{config.FINAL_15M}")
    print(f"   Лимиты: {config.MAX_PAIRS_TO_AI} ИИ отбор → {config.MAX_FINAL_PAIRS} финал")
    print(f"   ИИ: {'✅' if config.DEEPSEEK_API_KEY else '❌'} DeepSeek")
    print("=" * 60)

    bot = SimplifiedScalpingBot()

    try:
        # Запуск одного полного цикла
        result = await bot.run_full_cycle()

        # Красивый вывод результата
        print(f"\n📊 ИТОГОВЫЙ РЕЗУЛЬТАТ:")
        print(f"   Статус: {result['result']}")

        if 'timing' in result:
            t = result['timing']
            print(
                f"   Время: {t['total']}сек (сканирование: {t['stage1_scan']}с, ИИ: {t['stage2_ai_select']}с, анализ: {t['stage3_analysis']}с)")

        if 'stats' in result:
            s = result['stats']
            print(
                f"   Статистика: {s['pairs_processed']} → {s['promising_found']} → {s['ai_selected']} → {s['final_signals']}")
            print(f"   Производительность: {s['processing_speed']} пар/сек")

        if result.get('signals'):
            print(f"\n🎯 ТОРГОВЫЕ СИГНАЛЫ ({len(result['signals'])}):")
            for signal in result['signals']:
                trend_mark = "🔄" if signal.get('trend_alignment') else "⚠️"
                volume_mark = "📈" if signal.get('volume_confirmation') else "📉"
                print(f"   {signal['symbol']}: {signal['signal']} {signal['confidence']}% {trend_mark}{volume_mark}")

                # Показываем краткий анализ
                if signal.get('analysis'):
                    analysis_short = signal['analysis'][:150] + "..." if len(signal['analysis']) > 150 else signal[
                        'analysis']
                    print(f"      → {analysis_short}")

        # Сохраняем результат в файл
        with open('bot_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Результат сохранен в bot_result.json")

    except KeyboardInterrupt:
        print("\n⚠️  Остановлено пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
    finally:
        await bot.cleanup()
        print("🧹 Ресурсы очищены")


if __name__ == "__main__":
    asyncio.run(main())