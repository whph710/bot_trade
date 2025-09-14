"""
Переписанный скальпинговый бот - ИСПРАВЛЕННАЯ ВЕРСИЯ
Устранены критические ошибки:
- Правильная передача данных между этапами
- Исправлена обработка индикаторов
- Добавлено детальное логирование каждого шага
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from typing import List, Dict, Any

from config import config, has_api_key
from func_async import get_trading_pairs, fetch_klines, batch_fetch_klines, cleanup as cleanup_api
from func_trade import calculate_basic_indicators, calculate_ai_indicators, check_basic_signal
from deepseek import ai_select_pairs, ai_analyze_pair

# Настройка логирования с временными метками
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class RewrittenScalpingBot:
    """Переписанный скальпинговый бот по новой логике - ИСПРАВЛЕН"""

    def __init__(self):
        self.processed_pairs = 0
        self.session_start = time.time()

    async def stage1_filter_signals(self) -> List[Dict]:
        """
        ЭТАП 1: Отсеиваем пары БЕЗ торговых сигналов (15М данные)
        """
        start_time = time.time()
        logger.info("=" * 50)
        logger.info("ЭТАП 1: Фильтрация пар с сигналами...")
        logger.info("=" * 50)

        # Получаем список торговых пар
        logger.info("Получение списка торговых пар...")
        pairs = await get_trading_pairs()
        if not pairs:
            logger.error("❌ Пары не получены")
            return []

        logger.info(f"✅ Получено {len(pairs)} торговых пар")
        logger.info(f"📊 Проверяем каждую пару на наличие сигналов...")

        # Подготавливаем запросы для сканирования 15М данных
        requests = [
            {'symbol': pair, 'interval': '15', 'limit': config.QUICK_SCAN_15M}
            for pair in pairs
        ]

        # Массовое получение данных
        logger.info(f"🔄 Загружаем 15м данные для {len(requests)} пар...")
        results = await batch_fetch_klines(requests)
        logger.info(f"📥 Получено данных по {len(results)} парам")

        pairs_with_signals = []

        # Обрабатываем результаты - ОСТАВЛЯЕМ ТОЛЬКО С СИГНАЛАМИ
        for i, result in enumerate(results):
            if not result.get('success') or len(result['klines']) < 20:
                continue

            symbol = result['symbol']
            klines = result['klines']

            # Логируем прогресс каждые 50 пар
            if i % 50 == 0:
                logger.info(f"📈 Обработано {i}/{len(results)} пар...")

            # Рассчитываем базовые индикаторы
            try:
                indicators = calculate_basic_indicators(klines)
                if not indicators:
                    logger.debug(f"⚠️ {symbol}: не удалось рассчитать индикаторы")
                    continue

                # Проверяем базовый сигнал - СТРОГАЯ ФИЛЬТРАЦИЯ
                signal_check = check_basic_signal(indicators)

                # ОСТАВЛЯЕМ ТОЛЬКО пары с четкими сигналами
                if signal_check['signal'] and signal_check['confidence'] >= config.MIN_CONFIDENCE:
                    pair_data = {
                        'symbol': symbol,
                        'confidence': signal_check['confidence'],
                        'direction': signal_check['direction'],
                        'base_indicators': indicators,
                        # ИСПРАВЛЕНО: сохраняем свечи для следующего этапа
                        'stage1_klines': klines
                    }
                    pairs_with_signals.append(pair_data)

                    logger.info(f"✅ {symbol}: {signal_check['direction']} ({signal_check['confidence']}%)")

            except Exception as e:
                logger.error(f"❌ Ошибка обработки {symbol}: {e}")
                continue

        # Сортируем по уверенности
        pairs_with_signals.sort(key=lambda x: x['confidence'], reverse=True)

        elapsed = time.time() - start_time
        self.processed_pairs = len(results)

        logger.info("=" * 50)
        logger.info(f"📊 РЕЗУЛЬТАТЫ ЭТАПА 1:")
        logger.info(f"   Обработано: {len(results)} пар")
        logger.info(f"   С сигналами: {len(pairs_with_signals)} пар")
        logger.info(f"   Время: {elapsed:.1f}сек")
        logger.info(f"   Скорость: {len(pairs) / elapsed:.0f} пар/сек")

        if pairs_with_signals:
            top_pairs = [(p['symbol'], p['confidence'], p['direction']) for p in pairs_with_signals[:10]]
            logger.info(f"🏆 Топ-10 пар с сигналами:")
            for symbol, conf, direction in top_pairs:
                logger.info(f"   {symbol}: {direction} ({conf}%)")

        logger.info("=" * 50)

        return pairs_with_signals

    async def stage2_ai_bulk_select(self, signal_pairs: List[Dict]) -> List[str]:
        """
        ЭТАП 2: ИИ отбор - передаем ВСЕ пары с сигналами одним запросом
        ИСПРАВЛЕНО: правильная передача данных
        """
        start_time = time.time()
        logger.info("=" * 50)
        logger.info(f"ЭТАП 2: ИИ анализ {len(signal_pairs)} пар с сигналами...")
        logger.info("=" * 50)

        if not signal_pairs:
            logger.warning("❌ Нет пар для ИИ анализа")
            return []

        # ИСПРАВЛЕНО: подготавливаем данные для ИИ правильно
        logger.info("🔄 Подготовка данных для ИИ анализа...")

        ai_input_data = []

        for i, pair_data in enumerate(signal_pairs):
            symbol = pair_data['symbol']

            logger.info(f"📊 Подготовка {symbol} ({i+1}/{len(signal_pairs)})...")

            # ИСПРАВЛЕНО: используем уже полученные свечи из этапа 1
            if 'stage1_klines' in pair_data:
                candles_15m = pair_data['stage1_klines']
                logger.debug(f"   Используются свечи из этапа 1: {len(candles_15m)} штук")
            else:
                # Fallback - получаем новые данные
                logger.debug(f"   Получение новых 15м данных для {symbol}...")
                candles_15m = await fetch_klines(symbol, '15', config.AI_BULK_15M)

            if not candles_15m or len(candles_15m) < 20:
                logger.warning(f"⚠️ {symbol}: недостаточно 15м данных ({len(candles_15m) if candles_15m else 0} свечей)")
                continue

            # Рассчитываем индикаторы с историей для ИИ
            try:
                indicators_15m = calculate_ai_indicators(candles_15m, config.AI_INDICATORS_HISTORY)
                if not indicators_15m:
                    logger.warning(f"⚠️ {symbol}: ошибка расчета индикаторов для ИИ")
                    continue

                # ИСПРАВЛЕНО: правильная структура данных для ИИ
                pair_ai_data = {
                    'symbol': symbol,
                    'confidence': pair_data['confidence'],
                    'direction': pair_data['direction'],
                    'candles_15m': candles_15m[-config.AI_BULK_15M:],  # Ограничиваем размер
                    'indicators_15m': indicators_15m
                }

                ai_input_data.append(pair_ai_data)
                logger.debug(f"✅ {symbol}: данные для ИИ подготовлены")

            except Exception as e:
                logger.error(f"❌ Ошибка подготовки данных для ИИ {symbol}: {e}")
                continue

        if not ai_input_data:
            logger.error("❌ Нет подготовленных данных для ИИ анализа!")
            logger.error("Проверьте:")
            logger.error("1. Получены ли свечи на этапе 1")
            logger.error("2. Рассчитываются ли индикаторы")
            logger.error("3. Нет ли ошибок в calculate_ai_indicators")
            return []

        logger.info(f"✅ Подготовлено {len(ai_input_data)} пар для ИИ из {len(signal_pairs)}")

        # Размер данных для ИИ
        try:
            json_data = json.dumps(ai_input_data, separators=(',', ':'))
            data_size = len(json_data)
            logger.info(f"📊 Размер данных для ИИ: {data_size:,} байт ({data_size/1024:.1f} KB)")

            if data_size > 1024 * 1024:  # > 1MB
                logger.warning(f"⚠️ Большой размер данных для ИИ: {data_size/1024/1024:.1f} MB")

        except Exception as e:
            logger.error(f"❌ Ошибка сериализации данных для ИИ: {e}")
            return []

        # Проверяем доступность ИИ
        if not has_api_key:
            logger.error("❌ DeepSeek API ключ недоступен!")
            logger.error("Установите переменную окружения DEEPSEEK_API_KEY")
            return []

        # ИИ анализ ВСЕХ пар одним запросом
        logger.info("🤖 Отправляем данные в ИИ для анализа...")
        selected_pairs = await ai_select_pairs(ai_input_data)

        elapsed = time.time() - start_time

        logger.info("=" * 50)
        logger.info(f"📊 РЕЗУЛЬТАТЫ ЭТАПА 2:")
        logger.info(f"   Отправлено в ИИ: {len(ai_input_data)} пар")
        logger.info(f"   Выбрано ИИ: {len(selected_pairs)} пар")
        logger.info(f"   Время: {elapsed:.1f}сек")

        if selected_pairs:
            logger.info(f"🎯 Пары для детального анализа: {', '.join(selected_pairs)}")
        else:
            logger.warning("⚠️ ИИ не выбрал ни одной пары для детального анализа")

        logger.info("=" * 50)

        return selected_pairs

    async def stage3_detailed_analysis(self, selected_pairs: List[str]) -> List[Dict]:
        """
        ЭТАП 3: Детальный анализ каждой пары отдельно
        """
        start_time = time.time()
        logger.info("=" * 50)
        logger.info(f"ЭТАП 3: Детальный анализ {len(selected_pairs)} пар...")
        logger.info("=" * 50)

        if not selected_pairs:
            logger.warning("❌ Нет пар для детального анализа")
            return []

        final_signals = []

        # Анализируем каждую пару отдельно
        for i, symbol in enumerate(selected_pairs):
            logger.info(f"🔍 Анализ {symbol} ({i+1}/{len(selected_pairs)})...")

            try:
                # Получаем ПОЛНЫЕ данные для обоих таймфреймов
                logger.debug(f"   Загрузка 5м данных ({config.FINAL_5M} свечей)...")
                logger.debug(f"   Загрузка 15м данных ({config.FINAL_15M} свечей)...")

                klines_5m_task = fetch_klines(symbol, '5', config.FINAL_5M)
                klines_15m_task = fetch_klines(symbol, '15', config.FINAL_15M)

                klines_5m, klines_15m = await asyncio.gather(klines_5m_task, klines_15m_task)

                if (not klines_5m or len(klines_5m) < 100 or
                        not klines_15m or len(klines_15m) < 50):
                    logger.warning(f"❌ {symbol}: недостаточно данных (5м: {len(klines_5m) if klines_5m else 0}, 15м: {len(klines_15m) if klines_15m else 0})")
                    continue

                logger.debug(f"✅ {symbol}: данные получены (5м: {len(klines_5m)}, 15м: {len(klines_15m)})")

                # Рассчитываем полные индикаторы
                logger.debug(f"   Расчет индикаторов...")
                indicators_5m = calculate_ai_indicators(klines_5m, config.FINAL_INDICATORS)
                indicators_15m = calculate_ai_indicators(klines_15m, config.FINAL_INDICATORS)

                if not indicators_5m or not indicators_15m:
                    logger.warning(f"❌ {symbol}: ошибка расчета индикаторов")
                    continue

                # Детальный ИИ анализ
                logger.debug(f"   ИИ анализ с полными данными...")
                analysis = await ai_analyze_pair(
                    symbol, klines_5m, klines_15m, indicators_5m, indicators_15m
                )

                # Проверяем результат
                if (analysis['signal'] != 'NO_SIGNAL' and
                    analysis['confidence'] >= config.MIN_CONFIDENCE):

                    final_signals.append(analysis)
                    entry = analysis.get('entry_price', 0)
                    stop = analysis.get('stop_loss', 0)
                    profit = analysis.get('take_profit', 0)

                    logger.info(f"✅ {symbol}: {analysis['signal']} ({analysis['confidence']}%)")
                    if entry and stop and profit:
                        risk_reward = round(abs(profit - entry) / abs(entry - stop), 2) if entry != stop else 0
                        logger.info(f"   📊 Вход: {entry:.4f} | Стоп: {stop:.4f} | Профит: {profit:.4f} | R/R: 1:{risk_reward}")
                else:
                    logger.info(f"⚠️ {symbol}: {analysis['signal']} ({analysis['confidence']}%) - не прошел фильтр")

            except Exception as e:
                logger.error(f"❌ Ошибка анализа {symbol}: {e}")
                continue

        elapsed = time.time() - start_time

        logger.info("=" * 50)
        logger.info(f"📊 РЕЗУЛЬТАТЫ ЭТАПА 3:")
        logger.info(f"   Анализировано: {len(selected_pairs)} пар")
        logger.info(f"   Финальных сигналов: {len(final_signals)}")
        logger.info(f"   Время: {elapsed:.1f}сек")

        if final_signals:
            logger.info(f"🎯 Торговые сигналы:")
            for signal in final_signals:
                logger.info(f"   {signal['symbol']}: {signal['signal']} ({signal['confidence']}%)")

        logger.info("=" * 50)

        return final_signals

    async def run_full_cycle(self) -> Dict[str, Any]:
        """Полный цикл работы переписанного бота - ИСПРАВЛЕН"""
        cycle_start = time.time()

        logger.info("🚀 ЗАПУСК ПЕРЕПИСАННОГО ЦИКЛА АНАЛИЗА")
        logger.info(f"⏰ Время запуска: {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"🔑 DeepSeek API: {'✅ Доступен' if has_api_key else '❌ Недоступен'}")

        try:
            # ЭТАП 1: Фильтрация пар с сигналами (15М)
            stage1_start = time.time()
            signal_pairs = await self.stage1_filter_signals()
            stage1_time = time.time() - stage1_start

            if not signal_pairs:
                return {
                    'result': 'NO_SIGNAL_PAIRS',
                    'stage1_time': stage1_time,
                    'total_time': time.time() - cycle_start,
                    'pairs_scanned': self.processed_pairs,
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
                    'pairs_scanned': self.processed_pairs,
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
                    'processing_speed': round(self.processed_pairs / stage1_time, 1) if stage1_time > 0 else 0
                },
                'pipeline': {
                    'stage1_pairs': [p['symbol'] for p in signal_pairs[:10]],
                    'stage2_selected': selected_pairs,
                    'stage3_signals': [s['symbol'] for s in final_signals]
                },
                'signals': final_signals,
                'api_available': has_api_key
            }

            # Финальное логирование
            logger.info("🎉 ЗАВЕРШЕНИЕ ЦИКЛА АНАЛИЗА")
            logger.info(f"📊 Пайплайн: {self.processed_pairs} → {len(signal_pairs)} → {len(selected_pairs)} → {len(final_signals)}")
            logger.info(f"⏱️ Время: общее {total_time:.1f}с (фильтр: {stage1_time:.1f}с | ИИ: {stage2_time:.1f}с | анализ: {stage3_time:.1f}с)")
            logger.info(f"⚡ Скорость: {self.processed_pairs / stage1_time:.0f} пар/сек")

            return result

        except Exception as e:
            logger.error(f"💥 Критическая ошибка цикла: {e}")
            import traceback
            logger.error(f"Стек ошибки: {traceback.format_exc()}")
            return {
                'result': 'ERROR',
                'error': str(e),
                'total_time': time.time() - cycle_start
            }

    async def cleanup(self):
        """Очистка ресурсов"""
        logger.info("🧹 Очистка ресурсов...")
        await cleanup_api()


async def main():
    """Главная функция переписанного бота - ИСПРАВЛЕНА"""
    print("🤖 ПЕРЕПИСАННЫЙ СКАЛЬПИНГОВЫЙ БОТ - ИСПРАВЛЕННАЯ ВЕРСИЯ")
    print(f"⏰ Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("📋 ЛОГИКА РАБОТЫ:")
    print("   ЭТАП 1: Фильтр пар с сигналами (15м данные)")
    print("   ЭТАП 2: ИИ отбор (все пары + индикаторы одним запросом)")
    print("   ЭТАП 3: Детальный анализ (полные данные 5м+15м, уровни)")
    print()
    print(f"🔑 DeepSeek ИИ: {'✅ Доступен' if has_api_key else '❌ Недоступен (fallback режим)'}")
    print("=" * 70)

    bot = RewrittenScalpingBot()

    try:
        # Запуск полного цикла
        result = await bot.run_full_cycle()

        # Красивый вывод результата
        print(f"\n📊 ИТОГОВЫЙ РЕЗУЛЬТАТ:")
        print(f"   Статус: {result['result']}")
        print(f"   ИИ доступность: {'✅' if result.get('api_available') else '❌'}")

        if 'timing' in result:
            t = result['timing']
            print(f"   ⏱️ Время: {t['total']}сек")
            print(f"      ├─ Фильтрация: {t['stage1_filter']}сек")
            print(f"      ├─ ИИ отбор: {t['stage2_ai_bulk']}сек")
            print(f"      └─ Детальный анализ: {t['stage3_detailed']}сек")

        if 'stats' in result:
            s = result['stats']
            print(f"   📈 Статистика: {s['pairs_scanned']} → {s['signal_pairs_found']} → {s['ai_selected']} → {s['final_signals']}")
            print(f"   ⚡ Производительность: {s['processing_speed']} пар/сек")

        if 'pipeline' in result and result['pipeline']:
            p = result['pipeline']
            if p.get('stage1_pairs'):
                print(f"\n🔍 ЭТАП 1 - Пары с сигналами (топ-10):")
                print(f"   {', '.join(p['stage1_pairs'])}")

            if p.get('stage2_selected'):
                print(f"\n🤖 ЭТАП 2 - ИИ отбор:")