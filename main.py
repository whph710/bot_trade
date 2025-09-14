"""
Исправленный скальпинговый бот с диагностикой
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

# Настройка логирования с диагностикой
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class ScalpingBot:
    """Исправленный скальпинговый бот с диагностикой"""

    def __init__(self):
        self.processed_pairs = 0
        self.session_start = time.time()

    async def stage1_filter_signals(self) -> List[Dict]:
        """ЭТАП 1: Фильтрация пар с сигналами"""
        start_time = time.time()
        logger.info("ЭТАП 1: Фильтрация пар с сигналами")

        # Получаем пары
        pairs = await get_trading_pairs()
        if not pairs:
            logger.error("Не удалось получить торговые пары")
            return []

        logger.info(f"Получено {len(pairs)} торговых пар")

        # Подготавливаем запросы
        requests = [
            {'symbol': pair, 'interval': '15', 'limit': config.QUICK_SCAN_15M}
            for pair in pairs
        ]

        # Массовое получение данных
        logger.info(f"Загружаем данные для {len(requests)} пар")
        results = await batch_fetch_klines(requests)
        logger.info(f"Получено данных по {len(results)} парам")

        pairs_with_signals = []
        error_count = 0

        # Обрабатываем результаты
        for i, result in enumerate(results):
            if not result.get('success') or len(result['klines']) < 20:
                continue

            symbol = result['symbol']
            klines = result['klines']

            if i % 50 == 0:
                logger.info(f"Обработано {i}/{len(results)} пар")

            try:
                # ДИАГНОСТИКА: проверяем данные свечей
                if len(klines) < 20:
                    logger.debug(f"{symbol}: мало свечей ({len(klines)})")
                    continue

                # Проверим формат свечей
                first_candle = klines[0]
                if len(first_candle) < 6:
                    logger.debug(f"{symbol}: неправильный формат свечи")
                    continue

                indicators = calculate_basic_indicators(klines)
                if not indicators:
                    error_count += 1
                    logger.debug(f"{symbol}: не удалось рассчитать индикаторы")
                    continue

                # ДИАГНОСТИКА: проверяем индикаторы
                price = indicators.get('price', 0)
                ema5 = indicators.get('ema5', 0)
                if price <= 0 or ema5 <= 0:
                    logger.debug(f"{symbol}: некорректные индикаторы (price: {price}, ema5: {ema5})")
                    continue

                signal_check = check_basic_signal(indicators)

                if signal_check['signal'] and signal_check['confidence'] >= config.MIN_CONFIDENCE:
                    pair_data = {
                        'symbol': symbol,
                        'confidence': signal_check['confidence'],
                        'direction': signal_check['direction'],
                        'base_indicators': indicators,
                        'stage1_klines': klines
                    }
                    pairs_with_signals.append(pair_data)
                    logger.info(f"{symbol}: {signal_check['direction']} ({signal_check['confidence']}%)")

            except Exception as e:
                error_count += 1
                logger.error(f"Ошибка обработки {symbol}: {e}")
                continue

        # Сортируем по уверенности
        pairs_with_signals.sort(key=lambda x: x['confidence'], reverse=True)

        elapsed = time.time() - start_time
        self.processed_pairs = len(results)

        logger.info(f"РЕЗУЛЬТАТЫ ЭТАПА 1:")
        logger.info(f"Обработано: {len(results)} пар")
        logger.info(f"Ошибок обработки: {error_count}")
        logger.info(f"С сигналами: {len(pairs_with_signals)} пар")
        logger.info(f"Время: {elapsed:.1f}сек")

        return pairs_with_signals

    async def stage2_ai_bulk_select(self, signal_pairs: List[Dict]) -> List[str]:
        """ЭТАП 2: ИИ отбор пар"""
        start_time = time.time()
        logger.info(f"ЭТАП 2: ИИ анализ {len(signal_pairs)} пар с сигналами")

        if not signal_pairs:
            logger.warning("Нет пар для ИИ анализа")
            return []

        # Подготавливаем данные для ИИ
        logger.info("Подготовка данных для ИИ анализа")

        ai_input_data = []
        preparation_errors = 0

        for i, pair_data in enumerate(signal_pairs):
            symbol = pair_data['symbol']

            logger.info(f"Подготовка {symbol} ({i+1}/{len(signal_pairs)})")

            try:
                # Используем свечи из этапа 1 или получаем новые
                if 'stage1_klines' in pair_data and len(pair_data['stage1_klines']) >= 20:
                    candles_15m = pair_data['stage1_klines']
                else:
                    candles_15m = await fetch_klines(symbol, '15', config.AI_BULK_15M)

                if not candles_15m or len(candles_15m) < 20:
                    logger.warning(f"{symbol}: недостаточно данных ({len(candles_15m) if candles_15m else 0} свечей)")
                    preparation_errors += 1
                    continue

                # ДИАГНОСТИКА: проверим данные перед расчетом
                logger.debug(f"{symbol}: данные OK, свечей: {len(candles_15m)}")

                # Рассчитываем индикаторы с историей
                indicators_15m = calculate_ai_indicators(candles_15m, config.AI_INDICATORS_HISTORY)
                if not indicators_15m:
                    logger.warning(f"{symbol}: ошибка расчета индикаторов")
                    preparation_errors += 1
                    continue

                # ДИАГНОСТИКА: проверим структуру индикаторов
                required_keys = ['ema5_history', 'ema8_history', 'current']
                missing_keys = [key for key in required_keys if key not in indicators_15m]
                if missing_keys:
                    logger.warning(f"{symbol}: отсутствуют ключи: {missing_keys}")
                    preparation_errors += 1
                    continue

                # Структура данных для ИИ
                pair_ai_data = {
                    'symbol': symbol,
                    'confidence': pair_data['confidence'],
                    'direction': pair_data['direction'],
                    'candles_15m': candles_15m[-config.AI_BULK_15M:],
                    'indicators_15m': indicators_15m
                }

                ai_input_data.append(pair_ai_data)
                logger.debug(f"{symbol}: подготовлено для ИИ")

            except Exception as e:
                preparation_errors += 1
                logger.error(f"Ошибка подготовки данных для ИИ {symbol}: {e}")
                continue

        if not ai_input_data:
            logger.error("НЕТ ДАННЫХ ДЛЯ ИИ АНАЛИЗА!")
            logger.error(f"Ошибок подготовки: {preparation_errors}")
            return []

        logger.info(f"Подготовлено {len(ai_input_data)} пар для ИИ из {len(signal_pairs)} (ошибок: {preparation_errors})")

        # Размер данных
        try:
            json_data = json.dumps(ai_input_data, separators=(',', ':'))
            data_size = len(json_data)
            logger.info(f"Размер данных для ИИ: {data_size/1024:.1f} KB")
        except Exception as e:
            logger.error(f"Ошибка сериализации данных для ИИ: {e}")
            return []

        # ИИ анализ
        logger.info("Отправляем данные в ИИ для анализа")
        selected_pairs = await ai_select_pairs(ai_input_data)

        elapsed = time.time() - start_time

        logger.info(f"РЕЗУЛЬТАТЫ ЭТАПА 2:")
        logger.info(f"Отправлено в ИИ: {len(ai_input_data)} пар")
        logger.info(f"Ошибок подготовки: {preparation_errors}")
        logger.info(f"Выбрано ИИ: {len(selected_pairs)} пар")
        logger.info(f"Время: {elapsed:.1f}сек")

        if selected_pairs:
            logger.info(f"Пары для детального анализа: {', '.join(selected_pairs)}")
        else:
            logger.warning("ИИ не выбрал пары для детального анализа")

        return selected_pairs

    async def stage3_detailed_analysis(self, selected_pairs: List[str]) -> List[Dict]:
        """ЭТАП 3: Детальный анализ каждой пары"""
        start_time = time.time()
        logger.info(f"ЭТАП 3: Детальный анализ {len(selected_pairs)} пар")

        if not selected_pairs:
            logger.warning("Нет пар для детального анализа")
            return []

        final_signals = []

        # Анализируем каждую пару
        for i, symbol in enumerate(selected_pairs):
            logger.info(f"Анализ {symbol} ({i+1}/{len(selected_pairs)})")

            try:
                # Получаем полные данные
                klines_5m_task = fetch_klines(symbol, '5', config.FINAL_5M)
                klines_15m_task = fetch_klines(symbol, '15', config.FINAL_15M)

                klines_5m, klines_15m = await asyncio.gather(klines_5m_task, klines_15m_task)

                if (not klines_5m or len(klines_5m) < 100 or
                        not klines_15m or len(klines_15m) < 50):
                    logger.warning(f"{symbol}: недостаточно данных (5м: {len(klines_5m) if klines_5m else 0}, 15м: {len(klines_15m) if klines_15m else 0})")
                    continue

                # Рассчитываем индикаторы
                indicators_5m = calculate_ai_indicators(klines_5m, config.FINAL_INDICATORS)
                indicators_15m = calculate_ai_indicators(klines_15m, config.FINAL_INDICATORS)

                if not indicators_5m or not indicators_15m:
                    logger.warning(f"{symbol}: ошибка расчета индикаторов")
                    continue

                # ИИ анализ
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

                    logger.info(f"{symbol}: {analysis['signal']} ({analysis['confidence']}%)")
                    if entry and stop and profit:
                        risk_reward = round(abs(profit - entry) / abs(entry - stop), 2) if entry != stop else 0
                        logger.info(f"Вход: {entry:.4f} | Стоп: {stop:.4f} | Профит: {profit:.4f} | R/R: 1:{risk_reward}")
                else:
                    logger.info(f"{symbol}: {analysis['signal']} ({analysis['confidence']}%) - не прошел фильтр")

            except Exception as e:
                logger.error(f"Ошибка анализа {symbol}: {e}")
                continue

        elapsed = time.time() - start_time

        logger.info(f"РЕЗУЛЬТАТЫ ЭТАПА 3:")
        logger.info(f"Анализировано: {len(selected_pairs)} пар")
        logger.info(f"Финальных сигналов: {len(final_signals)}")
        logger.info(f"Время: {elapsed:.1f}сек")

        if final_signals:
            logger.info("Торговые сигналы:")
            for signal in final_signals:
                logger.info(f"{signal['symbol']}: {signal['signal']} ({signal['confidence']}%)")

        return final_signals

    async def run_full_cycle(self) -> Dict[str, Any]:
        """Полный цикл работы бота"""
        cycle_start = time.time()

        logger.info("ЗАПУСК ЦИКЛА АНАЛИЗА")
        logger.info(f"Время запуска: {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"DeepSeek API: {'Доступен' if has_api_key else 'Недоступен'}")

        try:
            # ЭТАП 1
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

            # ЭТАП 2
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

            # ЭТАП 3
            stage3_start = time.time()
            final_signals = await self.stage3_detailed_analysis(selected_pairs)
            stage3_time = time.time() - stage3_start

            total_time = time.time() - cycle_start

            # Результат
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
                'signals': final_signals,
                'api_available': has_api_key
            }

            # Финальное логирование
            logger.info("ЗАВЕРШЕНИЕ ЦИКЛА АНАЛИЗА")
            logger.info(f"Пайплайн: {self.processed_pairs} -> {len(signal_pairs)} -> {len(selected_pairs)} -> {len(final_signals)}")
            logger.info(f"Время: общее {total_time:.1f}с (фильтр: {stage1_time:.1f}с | ИИ: {stage2_time:.1f}с | анализ: {stage3_time:.1f}с)")
            logger.info(f"Скорость: {self.processed_pairs / stage1_time:.0f} пар/сек")

            return result

        except Exception as e:
            logger.error(f"Критическая ошибка цикла: {e}")
            import traceback
            logger.error(f"Стек ошибки: {traceback.format_exc()}")
            return {
                'result': 'ERROR',
                'error': str(e),
                'total_time': time.time() - cycle_start
            }

    async def cleanup(self):
        """Очистка ресурсов"""
        logger.info("Очистка ресурсов")
        await cleanup_api()


async def main():
    """Главная функция с диагностикой"""
    print("СКАЛЬПИНГОВЫЙ БОТ")
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("ЛОГИКА РАБОТЫ:")
    print("ЭТАП 1: Фильтр пар с сигналами (15м данные)")
    print("ЭТАП 2: ИИ отбор (все пары + индикаторы одним запросом)")
    print("ЭТАП 3: Детальный анализ (полные данные 5м+15м, уровни)")
    print()
    print(f"DeepSeek ИИ: {'Доступен' if has_api_key else 'Недоступен (fallback режим)'}")
    print("=" * 70)

    if not has_api_key:
        print("ВНИМАНИЕ: DeepSeek API недоступен, будет использован fallback режим")
        print()

    # Включаем DEBUG логирование для диагностики
    logging.getLogger('__main__').setLevel(logging.DEBUG)
    logging.getLogger('func_trade').setLevel(logging.DEBUG)

    bot = ScalpingBot()

    try:
        # Запуск цикла
        result = await bot.run_full_cycle()

        # Вывод результата
        print(f"\nИТОГОВЫЙ РЕЗУЛЬТАТ:")
        print(f"Статус: {result['result']}")
        print(f"ИИ доступность: {'Да' if result.get('api_available') else 'Нет'}")

        if 'timing' in result:
            t = result['timing']
            print(f"Время: {t['total']}сек")
            print(f"├─ Фильтрация: {t['stage1_filter']}сек")
            print(f"├─ ИИ отбор: {t['stage2_ai_bulk']}сек")
            print(f"└─ Детальный анализ: {t['stage3_detailed']}сек")

        if 'stats' in result:
            s = result['stats']
            print(f"Статистика: {s['pairs_scanned']} -> {s['signal_pairs_found']} -> {s['ai_selected']} -> {s['final_signals']}")
            print(f"Производительность: {s['processing_speed']} пар/сек")

        if result.get('signals'):
            print(f"\nТОРГОВЫЕ СИГНАЛЫ:")
            for signal in result['signals']:
                print(f"{signal['symbol']}: {signal['signal']} ({signal['confidence']}%)")
                if signal.get('entry_price'):
                    print(f"  Вход: {signal['entry_price']:.4f} | Стоп: {signal.get('stop_loss', 0):.4f} | Профит: {signal.get('take_profit', 0):.4f}")

        # Сохраняем результат
        with open('bot_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nРезультат сохранен в bot_result.json")

    except KeyboardInterrupt:
        print("\nОстановлено пользователем")
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
    finally:
        await bot.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nПрограмма остановлена")
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback
        print(traceback.format_exc())
