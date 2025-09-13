# main_optimized.py - Максимально оптимизированная версия для скорости

import asyncio
import json
import logging
import time
import math
import numpy as np
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
import aiohttp
from collections import defaultdict
import weakref

# Импорт оптимизированной конфигурации
from config import optimized_config as config

# Остальные импорты
from func_async import get_klines_async, get_usdt_trading_pairs
from deepseek import deep_seek_selection, deep_seek_analysis, cleanup_http_client
from func_trade import detect_instruction_based_signals, calculate_indicators_by_instruction

# Минимальное логирование для скорости
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Глобальные кэши для скорости
_pairs_cache = None
_pairs_cache_time = 0
_indicators_cache = {}
_klines_cache = {}


class SpeedOptimizedAnalyzer:
    """Максимально оптимизированный анализатор"""

    def __init__(self):
        self.session_start = time.time()
        self.processed_pairs = set()
        self.session = None
        self.connection_pool = None
        logger.warning("Скоростной анализатор запущен")

    async def setup_session(self):
        """Настройка переиспользуемой сессии"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=config.exchange.API_TIMEOUT)
            connector = aiohttp.TCPConnector(
                limit=config.exchange.MAX_CONNECTIONS,
                limit_per_host=config.exchange.MAX_KEEPALIVE_CONNECTIONS,
                keepalive_timeout=config.exchange.KEEPALIVE_TIMEOUT,
                enable_cleanup_closed=True
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'Connection': 'keep-alive'}
            )

    async def get_cached_pairs(self) -> List[str]:
        """Кэшированное получение списка пар"""
        global _pairs_cache, _pairs_cache_time

        current_time = time.time()
        if (_pairs_cache is None or
                current_time - _pairs_cache_time > config.performance.CACHE_PAIRS_LIST_SECONDS):

            _pairs_cache = await get_usdt_trading_pairs()
            _pairs_cache_time = current_time

            # Предварительная фильтрация по объему если включена
            if config.performance.PREFILTER_BY_VOLUME:
                _pairs_cache = await self.prefilter_by_volume(_pairs_cache)

            logger.warning(f"Кэш пар обновлен: {len(_pairs_cache)} пар")

        return _pairs_cache

    async def prefilter_by_volume(self, pairs: List[str]) -> List[str]:
        """Предварительная фильтрация по объему (быстрая)"""
        if not config.performance.PREFILTER_BY_VOLUME:
            return pairs

        # Получаем минимальные данные для фильтрации
        filtered_pairs = []

        # Обрабатываем большими батчами для скорости
        batch_size = 100
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            tasks = []

            for symbol in batch:
                task = self.quick_volume_check(symbol)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for symbol, result in zip(batch, results):
                if isinstance(result, bool) and result:
                    filtered_pairs.append(symbol)

        logger.warning(f"Фильтрация по объему: {len(filtered_pairs)}/{len(pairs)} пар")
        return filtered_pairs

    async def quick_volume_check(self, symbol: str) -> bool:
        """Быстрая проверка объема (только последние 10 свечей)"""
        try:
            # Минимальные данные для проверки объема
            candles = await get_klines_async(symbol, "5", limit=10)
            if not candles or len(candles) < 5:
                return False

            # Быстрая оценка объема
            recent_volumes = []
            for c in candles[-5:]:
                volume_usd = float(c[5]) * float(c[4])  # volume * price
                recent_volumes.append(volume_usd)

            avg_volume_per_hour = sum(recent_volumes) * 12  # 5-минутки * 12 = час
            daily_volume_estimate = avg_volume_per_hour * 24

            return daily_volume_estimate > config.trading.MIN_LIQUIDITY_VOLUME

        except Exception:
            return False

    def passes_quick_filters(self, symbol: str, candles: List) -> bool:
        """Быстрые фильтры (без сложных вычислений)"""
        if not candles or len(candles) < 20:
            return False

        # Быстрая проверка волатильности
        if config.performance.PREFILTER_BY_VOLATILITY:
            closes = [float(c[4]) for c in candles[-10:]]
            price_range = (max(closes) - min(closes)) / closes[-1]

            # Отсеиваем слишком стабильные или слишком волатильные
            if price_range < 0.005 or price_range > 0.05:  # 0.5% - 5%
                return False

        return True

    async def ultra_fast_scan_pair(self, symbol: str) -> Optional[Dict]:
        """Ультра-быстрое сканирование пары"""
        try:
            # Минимальные данные для первичной оценки
            candles_5m = await get_klines_async(
                symbol,
                config.timeframe.ENTRY_TF,
                limit=config.timeframe.CANDLES_5M_QUICK
            )

            if not self.passes_quick_filters(symbol, candles_5m):
                return None

            # Получаем 15m только если 5m прошли фильтры
            candles_15m = await get_klines_async(
                symbol,
                config.timeframe.CONTEXT_TF,
                limit=config.timeframe.CANDLES_15M_QUICK
            )

            if not candles_15m:
                return None

            # Быстрый технический анализ
            signal_result = detect_instruction_based_signals(candles_5m, candles_15m)

            if (signal_result['signal'] == 'NO_SIGNAL' or
                    signal_result['confidence'] < config.trading.MIN_CONFIDENCE):
                return None

            # Возвращаем минимальную структуру для ИИ
            return {
                'pair': symbol,
                'signal_type': signal_result['signal'],
                'confidence': signal_result['confidence'],
                'pattern_type': signal_result.get('pattern_type', 'UNKNOWN'),
                'entry_price': float(candles_5m[-1][4]),
                'validation_score': signal_result.get('validation_score', '0/5'),

                # Минимальные данные для ИИ
                'candles_5m_mini': candles_5m[-config.timeframe.CANDLES_FOR_AI_SELECTION:],
                'candles_15m_mini': candles_15m[-config.timeframe.CANDLES_FOR_CONTEXT:],
                'indicators_mini': self.extract_key_indicators(
                    signal_result.get('indicators', {})
                )
            }

        except Exception as e:
            # Минимальное логирование
            if "timeout" in str(e).lower():
                logger.error(f"Таймаут {symbol}")
            return None

    def extract_key_indicators(self, indicators: Dict) -> Dict:
        """Извлечение только ключевых индикаторов для ИИ"""
        if not indicators:
            return {}

        return {
            'ema5_current': indicators.get('ema5', [])[-1] if indicators.get('ema5') else 0,
            'ema8_current': indicators.get('ema8', [])[-1] if indicators.get('ema8') else 0,
            'ema20_current': indicators.get('ema20', [])[-1] if indicators.get('ema20') else 0,
            'rsi_current': indicators.get('rsi_current', 50),
            'volume_ratio': indicators.get('volume_ratio', 1.0),
            'atr_current': indicators.get('atr_current', 0),
            'macd_histogram_current': (
                indicators.get('macd_histogram', [])[-1]
                if indicators.get('macd_histogram') else 0
            )
        }

    async def hyper_speed_mass_scan(self) -> List[Dict]:
        """Гиперскоростное массовое сканирование"""
        start_time = time.time()

        await self.setup_session()

        logger.warning("ГИПЕРСКОРОСТНОЕ СКАНИРОВАНИЕ ЗАПУЩЕНО")

        try:
            pairs = await self.get_cached_pairs()
            if not pairs:
                logger.error("Пары не получены")
                return []

            logger.warning(f"Сканируем {len(pairs)} пар")

            # Максимально агрессивная параллельная обработка
            promising_signals = []

            # Обрабатываем ОГРОМНЫМИ батчами
            batch_size = config.processing.BATCH_SIZE
            semaphore = asyncio.Semaphore(config.processing.MAX_CONCURRENT_REQUESTS)

            async def bounded_scan(pair):
                async with semaphore:
                    return await self.ultra_fast_scan_pair(pair)

            # Запускаем все задачи параллельно
            all_tasks = []
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                batch_tasks = [bounded_scan(pair) for pair in batch]
                all_tasks.extend(batch_tasks)

            # Ждем ВСЕ результаты сразу (максимальный параллелизм)
            results = await asyncio.gather(*all_tasks, return_exceptions=True)

            # Быстрая обработка результатов
            for result in results:
                if isinstance(result, dict):
                    promising_signals.append(result)

            # Быстрая сортировка по уверенности
            promising_signals.sort(key=lambda x: x['confidence'], reverse=True)

            execution_time = time.time() - start_time
            logger.warning(
                f"СКАНИРОВАНИЕ: {len(promising_signals)} сигналов "
                f"за {execution_time:.1f}сек ({len(pairs) / execution_time:.0f} пар/сек)"
            )

            return promising_signals

        except Exception as e:
            logger.error(f"Критическая ошибка сканирования: {e}")
            return []

    async def cleanup(self):
        """Очистка ресурсов"""
        if self.session:
            await self.session.close()


class TurboAISelector:
    """Турбо ИИ селектор для минимального времени отклика"""

    def __init__(self):
        self.selection_count = 0

    def prepare_ultra_compact_data(self, signals: List[Dict]) -> str:
        """Ультра-компактная подготовка данных для ИИ"""
        # Берем только ТОП сигналы
        top_signals = signals[:config.ai.MAX_PAIRS_TO_AI]

        compact_data = []
        for s in top_signals:
            # Только самые критичные данные
            compact = {
                'pair': s['pair'],
                'signal': s['signal_type'],
                'confidence': s['confidence'],
                'pattern': s['pattern_type'],
                'price': round(s['entry_price'], 6),

                # Только последние 5 свечей каждого таймфрейма
                'c5m': s['candles_5m_mini'][-5:] if s.get('candles_5m_mini') else [],
                'c15m': s['candles_15m_mini'][-3:] if s.get('candles_15m_mini') else [],

                # Только ключевые индикаторы
                'ema': [
                    s['indicators_mini'].get('ema5_current', 0),
                    s['indicators_mini'].get('ema8_current', 0),
                    s['indicators_mini'].get('ema20_current', 0)
                ],
                'rsi': s['indicators_mini'].get('rsi_current', 50),
                'vol': round(s['indicators_mini'].get('volume_ratio', 1.0), 2)
            }
            compact_data.append(compact)

        return json.dumps(compact_data, separators=(',', ':'))  # Минимальный JSON

    async def turbo_select_pairs(self, signals: List[Dict]) -> List[str]:
        """Турбо-отбор пар с минимальным временем ответа"""
        if not signals:
            return []

        self.selection_count += 1
        start_time = time.time()

        try:
            # Ультра-компактные данные
            compact_data = self.prepare_ultra_compact_data(signals)

            # Сверх-быстрый запрос к ИИ
            ai_response = await deep_seek_selection(
                data=compact_data,
                prompt=None  # Используем кэшированный промпт
            )

            # Быстрый парсинг ответа
            selected_pairs = self.parse_ai_response(ai_response)

            execution_time = time.time() - start_time
            logger.warning(
                f"ИИ отбор #{self.selection_count}: {len(selected_pairs)} пар "
                f"за {execution_time:.1f}сек"
            )

            return selected_pairs[:config.ai.MAX_SELECTED_PAIRS]

        except Exception as e:
            logger.error(f"Ошибка ИИ отбора: {e}")
            # Фаллбек - берем топ по уверенности
            return [s['pair'] for s in signals[:3]]

    def parse_ai_response(self, response: str) -> List[str]:
        """Быстрый парсинг ответа ИИ"""
        try:
            # Ищем JSON в ответе
            import re
            json_match = re.search(r'\{"pairs":\s*\[(.*?)\]\}', response)
            if json_match:
                json_data = json.loads(json_match.group(0))
                return json_data.get('pairs', [])

            # Альтернативный поиск списка пар
            pairs_match = re.findall(r'[A-Z]{2,10}USDT', response)
            return list(set(pairs_match))[:3]

        except Exception:
            return []


async def turbo_detailed_analysis(pair: str, signal_data: Dict) -> Optional[Dict]:
    """Турбо детальный анализ только для финалистов"""
    try:
        start_time = time.time()

        # Получаем детальные данные только для финалистов
        candles_5m_detailed = await get_klines_async(
            pair, "5",
            limit=config.timeframe.DETAILED_CANDLES_5M
        )
        candles_15m_detailed = await get_klines_async(
            pair, "15",
            limit=config.timeframe.DETAILED_CANDLES_15M
        )

        if not candles_5m_detailed or not candles_15m_detailed:
            return None

        # Полные индикаторы
        detailed_indicators = calculate_indicators_by_instruction(candles_5m_detailed)

        # Подготовка данных для детального ИИ анализа
        detailed_data = {
            'pair': pair,
            'signal_type': signal_data['signal_type'],
            'confidence': signal_data['confidence'],
            'pattern_type': signal_data['pattern_type'],

            # Детальные таймфреймы
            'timeframes': {
                '5m_detailed': [
                    {
                        'timestamp': int(c[0]),
                        'open': float(c[1]),
                        'high': float(c[2]),
                        'low': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5])
                    } for c in candles_5m_detailed[-50:]  # Последние 50 свечей
                ],
                '15m_context': [
                    {
                        'timestamp': int(c[0]),
                        'open': float(c[1]),
                        'high': float(c[2]),
                        'low': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5])
                    } for c in candles_15m_detailed[-20:]  # Последние 20 свечей
                ]
            },

            # Полные индикаторы (только последние значения для скорости)
            'technical_indicators': {
                'ema_system': {
                    'ema5': detailed_indicators.get('ema5', [])[-10:],
                    'ema8': detailed_indicators.get('ema8', [])[-10:],
                    'ema20': detailed_indicators.get('ema20', [])[-10:]
                },
                'momentum': {
                    'rsi': detailed_indicators.get('rsi', [])[-10:],
                    'macd_line': detailed_indicators.get('macd_line', [])[-10:],
                    'macd_signal': detailed_indicators.get('macd_signal', [])[-10:],
                    'macd_histogram': detailed_indicators.get('macd_histogram', [])[-10:]
                },
                'volatility': {
                    'atr_current': detailed_indicators.get('atr_current', 0),
                    'atr_values': detailed_indicators.get('atr', [])[-10:],
                    'bb_upper': detailed_indicators.get('bb_upper', [])[-10:],
                    'bb_lower': detailed_indicators.get('bb_lower', [])[-10:]
                },
                'volume': {
                    'volume_ratio': detailed_indicators.get('volume_ratio', 1.0),
                    'volume_sma': detailed_indicators.get('volume_sma', [])[-10:]
                }
            }
        }

        # Компактная сериализация
        json_data = json.dumps(detailed_data, separators=(',', ':'))

        # Детальный ИИ анализ
        analysis_result = await deep_seek_analysis(data=json_data)

        execution_time = time.time() - start_time
        logger.warning(f"Детальный анализ {pair}: {execution_time:.1f}сек")

        return {
            'pair': pair,
            'analysis': analysis_result,
            'execution_time': execution_time,
            'data_points': len(candles_5m_detailed) + len(candles_15m_detailed)
        }

    except Exception as e:
        logger.error(f"Ошибка детального анализа {pair}: {e}")
        return None


class HyperSpeedTradingBot:
    """Гиперскоростной торговый бот"""

    def __init__(self):
        self.analyzer = SpeedOptimizedAnalyzer()
        self.selector = TurboAISelector()
        self.cycle_count = 0
        self.total_pairs_processed = 0

    async def run_hyper_speed_cycle(self) -> Dict[str, Any]:
        """Один цикл гиперскоростной работы"""
        cycle_start = time.time()
        self.cycle_count += 1

        logger.warning(f"=== ЦИКЛ #{self.cycle_count} ЗАПУСК ===")

        try:
            # ЭТАП 1: Гиперскоростное сканирование
            scan_start = time.time()
            signals = await self.analyzer.hyper_speed_mass_scan()
            scan_time = time.time() - scan_start

            if not signals:
                logger.warning("Сигналы не найдены")
                return {
                    'cycle': self.cycle_count,
                    'result': 'NO_SIGNALS',
                    'scan_time': scan_time,
                    'total_time': time.time() - cycle_start
                }

            self.total_pairs_processed += len(signals)

            # ЭТАП 2: Турбо ИИ отбор
            selection_start = time.time()
            selected_pairs = await self.selector.turbo_select_pairs(signals)
            selection_time = time.time() - selection_start

            if not selected_pairs:
                logger.warning("ИИ не отобрал пары")
                return {
                    'cycle': self.cycle_count,
                    'result': 'NO_SELECTION',
                    'scan_time': scan_time,
                    'selection_time': selection_time,
                    'total_time': time.time() - cycle_start
                }

            # ЭТАП 3: Детальный анализ финалистов (параллельно)
            analysis_start = time.time()

            # Получаем данные выбранных сигналов
            selected_signals = {s['pair']: s for s in signals if s['pair'] in selected_pairs}

            # Параллельный детальный анализ
            analysis_tasks = []
            for pair in selected_pairs:
                if pair in selected_signals:
                    task = turbo_detailed_analysis(pair, selected_signals[pair])
                    analysis_tasks.append(task)

            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            analysis_time = time.time() - analysis_start

            # Фильтруем успешные результаты
            successful_analyses = []
            for result in analysis_results:
                if isinstance(result, dict) and result is not None:
                    successful_analyses.append(result)

            total_time = time.time() - cycle_start

            # Результат цикла
            cycle_result = {
                'cycle': self.cycle_count,
                'result': 'SUCCESS',
                'timing': {
                    'scan_time': round(scan_time, 2),
                    'selection_time': round(selection_time, 2),
                    'analysis_time': round(analysis_time, 2),
                    'total_time': round(total_time, 2)
                },
                'data': {
                    'signals_found': len(signals),
                    'pairs_selected': len(selected_pairs),
                    'analyses_completed': len(successful_analyses)
                },
                'selected_pairs': selected_pairs,
                'analyses': successful_analyses,
                'performance': {
                    'pairs_per_second': round(len(signals) / scan_time, 1),
                    'total_pairs_processed': self.total_pairs_processed
                }
            }

            logger.warning(
                f"ЦИКЛ #{self.cycle_count} ЗАВЕРШЕН: "
                f"{len(successful_analyses)} анализов за {total_time:.1f}сек "
                f"({len(signals) / scan_time:.0f} пар/сек)"
            )

            return cycle_result

        except Exception as e:
            logger.error(f"Критическая ошибка цикла: {e}")
            return {
                'cycle': self.cycle_count,
                'result': 'ERROR',
                'error': str(e),
                'total_time': time.time() - cycle_start
            }

    async def run_continuous(self, max_cycles: int = 100, delay_between_cycles: float = 1.0):
        """Непрерывная работа с минимальными задержками"""
        logger.warning(f"ЗАПУСК НЕПРЕРЫВНОЙ РАБОТЫ: {max_cycles} циклов")

        results = []

        try:
            for cycle_num in range(max_cycles):
                cycle_result = await self.run_hyper_speed_cycle()
                results.append(cycle_result)

                # Статистика каждые 10 циклов
                if cycle_num % 10 == 0 and cycle_num > 0:
                    avg_time = sum(r['timing']['total_time'] for r in results[-10:]) / 10
                    logger.warning(f"Последние 10 циклов: среднее время {avg_time:.1f}сек")

                # Минимальная задержка между циклами
                if delay_between_cycles > 0:
                    await asyncio.sleep(delay_between_cycles)

        except KeyboardInterrupt:
            logger.warning("Остановка по Ctrl+C")

        finally:
            await self.cleanup()

        return results

    async def cleanup(self):
        """Очистка ресурсов"""
        await self.analyzer.cleanup()
        await cleanup_http_client()


# Главная функция запуска
async def main():
    """Главная функция с максимальной скоростью"""
    print("🚀 ГИПЕРСКОРОСТНОЙ ТОРГОВЫЙ БОТ")
    print(f"⚡ Конфигурация: {config.ai.SELECTION_TIMEOUT}с отбор, {config.processing.BATCH_SIZE} батч")
    print("=" * 60)

    bot = HyperSpeedTradingBot()

    try:
        # Запускаем один тестовый цикл
        print("Тестовый цикл...")
        test_result = await bot.run_hyper_speed_cycle()

        print(f"\n📊 РЕЗУЛЬТАТ ТЕСТОВОГО ЦИКЛА:")
        print(f"   Время сканирования: {test_result['timing']['scan_time']}сек")
        print(f"   Время ИИ отбора: {test_result['timing']['selection_time']}сек")
        print(f"   Время анализа: {test_result['timing']['analysis_time']}сек")
        print(f"   Общее время: {test_result['timing']['total_time']}сек")
        print(f"   Найдено сигналов: {test_result['data']['signals_found']}")
        print(f"   Отобрано пар: {test_result['data']['pairs_selected']}")
        print(f"   Скорость: {test_result['performance']['pairs_per_second']} пар/сек")

        if test_result['selected_pairs']:
            print(f"   ТОП пары: {', '.join(test_result['selected_pairs'])}")

        # Показываем анализы
        if test_result.get('analyses'):
            print(f"\n📈 ДЕТАЛЬНЫЕ АНАЛИЗЫ:")
            for analysis in test_result['analyses']:
                print(f"   {analysis['pair']}: {analysis['execution_time']:.1f}сек")
                if 'analysis' in analysis:
                    # Краткий вывод анализа
                    analysis_text = analysis['analysis'][:200] + "..." if len(analysis['analysis']) > 200 else analysis[
                        'analysis']
                    print(f"      {analysis_text}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        await bot.cleanup()


# Функция для демонстрации производительности
async def performance_demo():
    """Демонстрация максимальной производительности"""
    print("⚡ ДЕМО ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 50)

    # Показываем настройки оптимизации
    print("🔧 ОПТИМИЗАЦИИ:")
    summary = config.get_speed_summary()
    for category, settings in summary.items():
        print(f"   {category.upper()}:")
        for setting, value in settings.items():
            print(f"      {setting}: {value}")

    print("\n📈 ОЖИДАЕМЫЙ ПРИРОСТ:")
    gains = config.estimate_performance_gain()
    for metric, gain in gains.items():
        print(f"   {metric}: {gain}")

    # Запускаем быстрый тест
    print("\n🚀 БЫСТРЫЙ ТЕСТ...")
    bot = HyperSpeedTradingBot()

    start_time = time.time()
    result = await bot.run_hyper_speed_cycle()
    end_time = time.time()

    print(f"✅ Тест завершен за {end_time - start_time:.1f}сек")
    print(f"   Производительность: {result['performance']['pairs_per_second']} пар/сек")

    await bot.cleanup()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        asyncio.run(performance_demo())
    else:
        asyncio.run(main())