"""
Переписанный скальпинговый бот согласно новым требованиям:
ЭТАП 1: Отсеивает пары БЕЗ сигналов
ЭТАП 2: Передает ВСЕ пары с сигналами + 32 свечи + индикаторы одним запросом
ЭТАП 3: Детальный анализ каждой отобранной пары с уровнями
"""

import asyncio
import logging
import time
import json
from typing import List, Dict, Any

from config import config
from func_async import get_trading_pairs, fetch_klines, batch_fetch_klines, cleanup as cleanup_api
from func_trade import calculate_basic_indicators, calculate_ai_indicators, check_basic_signal
from deepseek import ai_select_pairs, ai_analyze_pair

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class RewrittenScalpingBot:
    """Переписанный скальпинговый бот по новой логике"""

    def __init__(self):
        self.processed_pairs = 0
        self.session_start = time.time()

    async def stage1_filter_signals(self) -> List[Dict]:
        """
        ЭТАП 1: Отсеиваем пары БЕЗ торговых сигналов
        Возвращает ТОЛЬКО пары с четкими сигналами
        """
        start_time = time.time()
        logger.info("🔍 ЭТАП 1: Фильтрация пар с сигналами...")

        # Получаем список торговых пар
        pairs = await get_trading_pairs()
        if not pairs:
            logger.error("Пары не получены")
            return []

        logger.info(f"   Проверяем {len(pairs)} пар на наличие сигналов")

        # Подготавливаем запросы для сканирования
        requests = [
            {'symbol': pair, 'interval': '5', 'limit': config.QUICK_SCAN_5M}
            for pair in pairs
        ]

        # Массовое получение данных
        results = await batch_fetch_klines(requests)

        pairs_with_signals = []

        # Обрабатываем результаты - ОСТАВЛЯЕМ ТОЛЬКО С СИГНАЛАМИ
        for result in results:
            if not result.get('success') or len(result['klines']) < 20:
                continue

            symbol = result['symbol']
            klines = result['klines']

            # Рассчитываем базовые индикаторы
            indicators = calculate_basic_indicators(klines)
            if not indicators:
                continue

            # Проверяем базовый сигнал - СТРОГАЯ ФИЛЬТРАЦИЯ
            signal_check = check_basic_signal(indicators)

            # ОСТАВЛЯЕМ ТОЛЬКО пары с четкими сигналами
            if signal_check['signal'] and signal_check['confidence'] >= config.MIN_CONFIDENCE:
                pairs_with_signals.append({
                    'symbol': symbol,
                    'confidence': signal_check['confidence'],
                    'direction': signal_check['direction'],
                    'base_indicators': indicators
                })

        # Сортируем по уверенности
        pairs_with_signals.sort(key=lambda x: x['confidence'], reverse=True)

        elapsed = time.time() - start_time
        self.processed_pairs = len(results)

        logger.info(f"   ✅ Отфильтровано {len(pairs_with_signals)} пар с сигналами за {elapsed:.1f}сек")
        logger.info(f"   Скорость: {len(pairs) / elapsed:.0f} пар/сек")

        return pairs_with_signals

    async def stage2_ai_bulk_select(self, signal_pairs: List[Dict]) -> List[str]:
        """
        ЭТАП 2: ИИ отбор - передаем ВСЕ пары с сигналами одним запросом
        + 32 свечи 15м + 32 значения индикаторов для каждой пары
        """
        start_time = time.time()
        logger.info(f"🤖 ЭТАП 2: ИИ анализ {len(signal_pairs)} пар с сигналами...")

        if not signal_pairs:
            return []

        # Получаем 15м данные для ВСЕХ пар с сигналами
        requests_15m = [
            {'symbol': pair['symbol'], 'interval': '15', 'limit': 32}  # Ровно 32 свечи
            for pair in signal_pairs
        ]

        results_15m = await batch_fetch_klines(requests_15m)

        # Подготавливаем данные для ИИ - ВСЕ пары одним запросом
        ai_input_data = []

        for pair_data in signal_pairs:
            symbol = pair_data['symbol']

            # Находим соответствующие 15м данные
            candles_15m = None
            for result in results_15m:
                if result['symbol'] == symbol and result.get('success'):
                    candles_15m = result['klines']
                    break

            if not candles_15m or len(candles_15m) < 20:
                logger.warning(f"   Недостаточно 15м данных для {symbol}")
                continue

            # Рассчитываем индикаторы с историей на 32 значения
            indicators_15m = calculate_ai_indicators(candles_15m, 32)
            if not indicators_15m:
                continue

            # Добавляем в общий список для ИИ
            ai_input_data.append({
                'symbol': symbol,
                'confidence': pair_data['confidence'],
                'direction': pair_data['direction'],
                'candles_15m': candles_15m,  # 32 свечи
                'indicators_15m': indicators_15m  # 32 значения каждого индикатора
            })

        if not ai_input_data:
            logger.warning("   Нет данных для ИИ анализа")
            return []

        logger.info(f"   Подготовлено {len(ai_input_data)} пар для ИИ")
        logger.info(f"   Размер данных: ~{len(json.dumps(ai_input_data, separators=(',', ':'))) // 1000}KB")

        # ИИ анализ ВСЕХ пар одним запросом
        selected_pairs = await ai_select_pairs(ai_input_data)

        elapsed = time.time() - start_time
        logger.info(f"   ✅ ИИ выбрал {len(selected_pairs)} пар за {elapsed:.1f}сек")
        if selected_pairs:
            logger.info(f"   Финалисты: {', '.join(selected_pairs)}")

        return selected_pairs

    async def stage3_detailed_analysis(self, selected_pairs: List[str]) -> List[Dict]:
        """
        ЭТАП 3: Детальный анализ каждой пары отдельно
        Полные данные 5м + 15м + все индикаторы → структурированный JSON с уровнями
        """
        start_time = time.time()
        logger.info(f"📊 ЭТАП 3: Детальный анализ {len(selected_pairs)} пар...")

        if not selected_pairs:
            return []

        final_signals = []

        # Анализируем каждую пару отдельно
        for symbol in selected_pairs:
            logger.info(f"   Анализ {symbol}...")

            # Получаем ПОЛНЫЕ данные для обоих таймфреймов
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

            # Детальный ИИ анализ с получением структурированного JSON
            analysis = await ai_analyze_pair(
                symbol, klines_5m, klines_15m, indicators_5m, indicators_15m
            )

            # Проверяем результат
            if (analysis['signal'] != 'NO_SIGNAL' and
                analysis['confidence'] >= config.MIN_CONFIDENCE):

                final_signals.append(analysis)
                logger.info(f"   ✅ {symbol}: {analysis['signal']} ({analysis['confidence']}%) "
                          f"Вход: {analysis.get('entry_price', 0)}")
            else:
                logger.info(f"   ❌ {symbol}: {analysis['signal']} ({analysis['confidence']}%)")

        elapsed = time.time() - start_time
        logger.info(f"   ✅ Детальный анализ завершен за {elapsed:.1f}сек")

        return final_signals

    async def run_full_cycle(self) -> Dict[str, Any]:
        """Полный цикл работы переписанного бота"""
        cycle_start = time.time()

        logger.info("🚀 ЗАПУСК ПЕРЕПИСАННОГО ЦИКЛА")
        logger.info("=" * 60)

        try:
            # ЭТАП 1: Фильтрация пар с сигналами
            stage1_start = time.time()
            signal_pairs = await self.stage1_filter_signals()
            stage1_time = time.time() - stage1_start

            if not signal_pairs:
                return {
                    'result': 'NO_SIGNAL_PAIRS',
                    'stage1_time': stage1_time,
                    'total_time': time.time() - cycle_start,
                    'message': 'Нет пар с торговыми сигналами'
                }

            # ЭТАП 2: ИИ отбор всех пар одним запросом
            stage2_start = time.time()
            selected_pairs = await self.stage2_ai_bulk_select(signal_pairs)
            stage2_time = time.time() - stage2_start

            if not selected_pairs:
                return {
                    'result': 'NO_AI_SELECTION',
                    'stage1_time': stage1_time,
                    'stage2_time': stage2_time,
                    'signal_pairs': len(signal_pairs),
                    'total_time': time.time() - cycle_start,
                    'message': 'ИИ не выбрал подходящих пар'
                }

            # ЭТАП 3: Детальный анализ с уровнями
            stage3_start = time.time()
            final_signals = await self.stage3_detailed_analysis(selected_pairs)
            stage3_time = time.time() - stage3_start

            total_time = time.time() - cycle_start

            # Формируем результат
            result = {
                'result': 'SUCCESS' if final_signals else 'NO_FINAL_SIGNALS',
                'timing': {
                    'stage1_filter': round(stage1_time, 2),
                    'stage2_ai_bulk': round(stage2_time, 2),
                    'stage3_detailed': round(stage3_time, 2),
                    'total': round(total_time, 2)
                },
                'stats': {
                    'pairs_scanned': self.processed_pairs,
                    'signal_pairs_found': len(signal_pairs),
                    'ai_selected': len(selected_pairs),
                    'final_signals': len(final_signals),
                    'processing_speed': round(self.processed_pairs / stage1_time, 1)
                },
                'pipeline': {
                    'stage1_pairs': [p['symbol'] for p in signal_pairs[:10]],  # Показываем топ-10
                    'stage2_selected': selected_pairs,
                    'stage3_signals': [s['symbol'] for s in final_signals]
                },
                'signals': final_signals
            }

            # Логирование результата
            logger.info("📈 РЕЗУЛЬТАТЫ ПЕРЕПИСАННОГО ЦИКЛА:")
            logger.info(f"   Время: фильтр {stage1_time:.1f}с | ИИ отбор {stage2_time:.1f}с | анализ {stage3_time:.1f}с")
            logger.info(f"   Пайплайн: {self.processed_pairs} → {len(signal_pairs)} → {len(selected_pairs)} → {len(final_signals)}")
            logger.info(f"   Скорость: {self.processed_pairs / stage1_time:.0f} пар/сек")

            if final_signals:
                logger.info("🎯 ТОРГОВЫЕ СИГНАЛЫ С УРОВНЯМИ:")
                for signal in final_signals:
                    entry = signal.get('entry_price', 0)
                    stop = signal.get('stop_loss', 0)
                    profit = signal.get('take_profit', 0)
                    logger.info(f"   {signal['symbol']}: {signal['signal']} ({signal['confidence']}%)")
                    if entry and stop and profit:
                        risk_reward = round((profit - entry) / (entry - stop), 2) if entry != stop else 0
                        logger.info(f"      Вход: {entry} | Стоп: {stop} | Профит: {profit} | R/R: 1:{risk_reward}")

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
    """Главная функция переписанного бота"""
    print("🤖 ПЕРЕПИСАННЫЙ СКАЛЬПИНГОВЫЙ БОТ")
    print(f"⚙️  Новая логика:")
    print(f"   ЭТАП 1: Фильтр пар с сигналами (5м данные)")
    print(f"   ЭТАП 2: ИИ отбор (все пары + 32 свечи 15м одним запросом)")
    print(f"   ЭТАП 3: Детальный анализ (полные данные 5м+15м, уровни входа)")
    print(f"   ИИ: {'✅' if config.DEEPSEEK_API_KEY else '❌'} DeepSeek")
    print("=" * 70)

    bot = RewrittenScalpingBot()

    try:
        # Запуск полного цикла
        result = await bot.run_full_cycle()

        # Красивый вывод результата
        print(f"\n📊 ИТОГОВЫЙ РЕЗУЛЬТАТ:")
        print(f"   Статус: {result['result']}")

        if 'timing' in result:
            t = result['timing']
            print(f"   Время: {t['total']}сек (фильтр: {t['stage1_filter']}с, ИИ: {t['stage2_ai_bulk']}с, анализ: {t['stage3_detailed']}с)")

        if 'stats' in result:
            s = result['stats']
            print(f"   Статистика: {s['pairs_scanned']} → {s['signal_pairs_found']} → {s['ai_selected']} → {s['final_signals']}")
            print(f"   Производительность: {s['processing_speed']} пар/сек")

        if 'pipeline' in result:
            p = result['pipeline']
            if p['stage1_pairs']:
                print(f"\n🔍 ЭТАП 1 - Пары с сигналами (топ-10): {', '.join(p['stage1_pairs'])}")
            if p['stage2_selected']:
                print(f"🤖 ЭТАП 2 - ИИ отбор: {', '.join(p['stage2_selected'])}")
            if p['stage3_signals']:
                print(f"📊 ЭТАП 3 - Финальные сигналы: {', '.join(p['stage3_signals'])}")

        if result.get('signals'):
            print(f"\n🎯 ТОРГОВЫЕ СИГНАЛЫ С УРОВНЯМИ ({len(result['signals'])}):")
            for signal in result['signals']:
                symbol = signal['symbol']
                direction = signal['signal']
                confidence = signal['confidence']
                entry = signal.get('entry_price', 0)
                stop = signal.get('stop_loss', 0)
                profit = signal.get('take_profit', 0)

                trend_mark = "🔄" if signal.get('trend_alignment') else "⚠️"
                volume_mark = "📈" if signal.get('volume_confirmation') else "📉"

                print(f"   {symbol}: {direction} {confidence}% {trend_mark}{volume_mark}")

                if entry and stop and profit:
                    risk_reward = round(abs(profit - entry) / abs(entry - stop), 2) if entry != stop else 0
                    print(f"      💰 Вход: {entry:.4f}")
                    print(f"      🛑 Стоп: {stop:.4f}")
                    print(f"      🎯 Профит: {profit:.4f}")
                    print(f"      📊 Риск/Доход: 1:{risk_reward}")

                # Краткий анализ
                if signal.get('analysis'):
                    analysis_short = signal['analysis'][:200] + "..." if len(signal['analysis']) > 200 else signal['analysis']
                    print(f"      💡 {analysis_short}")
                print()

        # Сохраняем результат в файл
        with open('bot_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"💾 Результат сохранен в bot_result.json")

    except KeyboardInterrupt:
        print("\n⚠️  Остановлено пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
    finally:
        await bot.cleanup()
        print("🧹 Ресурсы очищены")


if __name__ == "__main__":
    asyncio.run(main())