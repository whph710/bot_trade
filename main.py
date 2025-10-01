"""
Оптимизированный скальпинговый бот с поддержкой multiple AI providers
ИНТЕГРИРОВАН С: Enhanced Validator, Market Data, Correlations, Volume Profile, AI Advanced
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from config import config, has_ai_available
from func_async import get_trading_pairs, fetch_klines, batch_fetch_klines, cleanup as cleanup_api, get_optimized_session
from func_trade import calculate_basic_indicators, calculate_ai_indicators, check_basic_signal
from ai_router import ai_router

# НОВЫЕ ИМПОРТЫ
from func_enhanced_validator import (
    EnhancedSignalValidator,
    validate_signals_batch,
    batch_quick_market_check,
    get_validation_statistics
)

# Упрощенное логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class DataCache:
    """Кеш для хранения и переиспользования рыночных данных"""

    def __init__(self):
        self.klines_cache = {}
        self.indicators_cache = {}

    def cache_klines(self, symbol: str, interval: str, klines: List):
        """Кеширование свечных данных"""
        if symbol not in self.klines_cache:
            self.klines_cache[symbol] = {}
        self.klines_cache[symbol][interval] = klines

    def get_klines(self, symbol: str, interval: str, limit: int) -> Optional[List]:
        """Получение кешированных данных с обрезкой по лимиту"""
        cached = self.klines_cache.get(symbol, {}).get(interval)
        if cached and len(cached) >= limit:
            return cached[-limit:]
        return None

    def cache_indicators(self, symbol: str, interval: str, indicators: Dict):
        """Кеширование индикаторов"""
        if symbol not in self.indicators_cache:
            self.indicators_cache[symbol] = {}
        self.indicators_cache[symbol][interval] = indicators

    def get_indicators(self, symbol: str, interval: str) -> Optional[Dict]:
        """Получение кешированных индикаторов"""
        return self.indicators_cache.get(symbol, {}).get(interval)

    def clear(self):
        """Очистка кеша"""
        self.klines_cache.clear()
        self.indicators_cache.clear()

    def clear_symbol(self, symbol: str):
        """Очистка данных конкретного символа"""
        if symbol in self.klines_cache:
            del self.klines_cache[symbol]
        if symbol in self.indicators_cache:
            del self.indicators_cache[symbol]


class OptimizedScalpingBot:
    """Оптимизированный скальпинговый бот с расширенной валидацией"""

    def __init__(self):
        self.processed_pairs = 0
        self.session_start = time.time()
        self.cache = DataCache()
        self.validation_data = {}
        self.enhanced_validator = None  # Инициализируем позже

    async def initialize_validator(self):
        """Инициализация расширенного валидатора"""
        if not self.enhanced_validator:
            session = await get_optimized_session()
            self.enhanced_validator = EnhancedSignalValidator(session, ai_router)
            logger.info("✅ Enhanced validator initialized")

    def validate_klines_data(self, klines: List, min_length: int = 10) -> bool:
        """Улучшенная валидация свечных данных"""
        if not klines or len(klines) < min_length:
            return False

        try:
            # Проверяем первые 3 свечи как образец
            for i, candle in enumerate(klines[:3]):
                if not isinstance(candle, list) or len(candle) < 6:
                    logger.debug(f"Свеча {i} имеет неправильную структуру")
                    return False

                try:
                    timestamp = int(candle[0])
                    open_price = float(candle[1])
                    high_price = float(candle[2])
                    low_price = float(candle[3])
                    close_price = float(candle[4])
                    volume = float(candle[5])

                    if any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
                        return False

                    if high_price < max(open_price, close_price) or low_price > min(open_price, close_price):
                        return False

                    if volume < 0:
                        return False

                except (ValueError, IndexError):
                    return False

            return True

        except Exception as e:
            logger.debug(f"Ошибка валидации свечных данных: {e}")
            return False

    async def load_initial_data(self, pairs: List[str]) -> Dict[str, bool]:
        """Предзагрузка всех необходимых данных"""
        logger.info(f"Предзагрузка данных для {len(pairs)} пар...")

        requests = []
        for pair in pairs:
            requests.extend([
                {'symbol': pair, 'interval': '15', 'limit': max(config.AI_BULK_15M, config.FINAL_15M)},
                {'symbol': pair, 'interval': '5', 'limit': config.FINAL_5M}
            ])

        results = await batch_fetch_klines(requests)

        loaded_pairs = set()
        for result in results:
            if result.get('success') and result.get('klines'):
                symbol = result['symbol']
                klines = result['klines']

                if not self.validate_klines_data(klines, 15):
                    continue

                if len(klines) >= 100:
                    self.cache.cache_klines(symbol, '5', klines)
                else:
                    self.cache.cache_klines(symbol, '15', klines)
                loaded_pairs.add(symbol)

        logger.info(f"Загружены данные по {len(loaded_pairs)} парам")
        return {pair: pair in loaded_pairs for pair in pairs}

    def calculate_and_cache_indicators(self, symbol: str, interval: str, klines: List, history_length: int) -> Optional[Dict]:
        """Расчет и кеширование индикаторов"""
        cached = self.cache.get_indicators(symbol, interval)
        if cached:
            return cached

        if not self.validate_klines_data(klines, 20):
            return None

        try:
            if history_length > 20:
                indicators = calculate_ai_indicators(klines, history_length)
            else:
                indicators = calculate_basic_indicators(klines)

            if indicators:
                self.cache.cache_indicators(symbol, interval, indicators)

            return indicators
        except Exception as e:
            logger.debug(f"Ошибка расчета индикаторов для {symbol} {interval}: {e}")
            return None

    def calculate_fresh_indicators(self, symbol: str, interval: str, klines: List, history_length: int) -> Optional[Dict]:
        """Расчет индикаторов БЕЗ кеширования (для свежих данных)"""
        if not self.validate_klines_data(klines, 20):
            return None

        try:
            if history_length > 20:
                indicators = calculate_ai_indicators(klines, history_length)
            else:
                indicators = calculate_basic_indicators(klines)

            return indicators
        except Exception as e:
            logger.debug(f"Ошибка расчета свежих индикаторов для {symbol} {interval}: {e}")
            return None

    async def stage1_filter_signals(self) -> List[Dict]:
        """ЭТАП 1: Фильтрация пар с сигналами + QUICK MARKET CHECKS"""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("ЭТАП 1: Фильтрация пар с сигналами + Quick Checks")
        logger.info("=" * 60)

        pairs = await get_trading_pairs()
        if not pairs:
            logger.error("Не удалось получить торговые пары")
            return []

        # НОВОЕ: Быстрая проверка market conditions
        logger.info(f"📊 Выполняем quick market checks для {len(pairs)} пар...")
        session = await get_optimized_session()
        quick_checks = await batch_quick_market_check(session, pairs[:200], max_concurrent=20)

        # Фильтруем только tradeable пары
        tradeable_pairs = [p for p, check in quick_checks.items() if check.get('tradeable', False)]
        filtered_out = len(quick_checks) - len(tradeable_pairs)

        logger.info(f"✅ Quick checks: {len(tradeable_pairs)} tradeable, {filtered_out} filtered out")

        # Предзагружаем данные только для tradeable пар
        loaded_data = await self.load_initial_data(tradeable_pairs)
        available_pairs = [pair for pair, loaded in loaded_data.items() if loaded]

        pairs_with_signals = []
        processed = 0
        errors = 0

        logger.info(f"🔍 Обработка {len(available_pairs)} пар...")

        for symbol in available_pairs:
            klines_15m = self.cache.get_klines(symbol, '15', config.QUICK_SCAN_15M)
            if not klines_15m or not self.validate_klines_data(klines_15m, 20):
                errors += 1
                continue

            try:
                indicators = self.calculate_and_cache_indicators(symbol, '15', klines_15m, 20)
                if not indicators:
                    errors += 1
                    continue

                signal_check = check_basic_signal(indicators)
                if signal_check['signal'] and signal_check['confidence'] >= config.MIN_CONFIDENCE:
                    pair_data = {
                        'symbol': symbol,
                        'confidence': signal_check['confidence'],
                        'direction': signal_check['direction'],
                        'base_indicators': indicators
                    }
                    pairs_with_signals.append(pair_data)

                processed += 1

            except Exception as e:
                logger.debug(f"Ошибка обработки {symbol}: {e}")
                errors += 1
                continue

        pairs_with_signals.sort(key=lambda x: x['confidence'], reverse=True)

        elapsed = time.time() - start_time
        self.processed_pairs = processed

        logger.info(f"✅ ЭТАП 1 завершен за {elapsed:.1f}с")
        logger.info(f"   Обработано: {processed} пар")
        logger.info(f"   Сигналы: {len(pairs_with_signals)}")
        logger.info(f"   Отфильтровано: {filtered_out}")
        logger.info(f"   Ошибки: {errors}")

        return pairs_with_signals

    async def stage2_ai_bulk_select(self, signal_pairs: List[Dict]) -> List[str]:
        """ЭТАП 2: AI отбор пар"""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"ЭТАП 2: AI анализ {len(signal_pairs)} пар")
        logger.info("=" * 60)

        if not signal_pairs:
            return []

        ai_input_data = []
        preparation_errors = 0

        for pair_data in signal_pairs:
            symbol = pair_data['symbol']

            try:
                candles_15m = self.cache.get_klines(symbol, '15', config.AI_BULK_15M)
                if not candles_15m or not self.validate_klines_data(candles_15m, 20):
                    preparation_errors += 1
                    continue

                indicators_15m = self.calculate_and_cache_indicators(
                    symbol, '15', candles_15m, config.AI_INDICATORS_HISTORY
                )
                if not indicators_15m:
                    preparation_errors += 1
                    continue

                pair_ai_data = {
                    'symbol': symbol,
                    'confidence': pair_data['confidence'],
                    'direction': pair_data['direction'],
                    'candles_15m': candles_15m,
                    'indicators_15m': indicators_15m
                }
                ai_input_data.append(pair_ai_data)

            except Exception as e:
                logger.debug(f"Ошибка подготовки данных для {symbol}: {e}")
                preparation_errors += 1
                continue

        if not ai_input_data:
            logger.error("Нет данных для AI анализа")
            return []

        logger.info(f"🤖 Отправка {len(ai_input_data)} пар в AI (ошибок подготовки: {preparation_errors})")

        selected_pairs = await ai_router.select_pairs(ai_input_data)

        elapsed = time.time() - start_time
        logger.info(f"✅ ЭТАП 2 завершен за {elapsed:.1f}с")
        logger.info(f"   Выбрано: {len(selected_pairs)} пар")

        return selected_pairs

    async def stage3_detailed_analysis(self, selected_pairs: List[str]) -> List[Dict]:
        """ЭТАП 3: Детальный анализ с ПЕРЕЗАГРУЗКОЙ свежих данных"""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"ЭТАП 3: Детальный анализ {len(selected_pairs)} пар (СВЕЖИЕ ДАННЫЕ)")
        logger.info("=" * 60)

        if not selected_pairs:
            return []

        final_signals = []

        for symbol in selected_pairs:
            try:
                logger.debug(f"📥 {symbol}: Загрузка свежих данных...")
                self.cache.clear_symbol(symbol)

                klines_5m = await fetch_klines(symbol, '5', config.FINAL_5M)
                klines_15m = await fetch_klines(symbol, '15', config.FINAL_15M)

                if not klines_5m or not klines_15m:
                    logger.warning(f"{symbol}: Не удалось загрузить свежие данные")
                    continue

                if not self.validate_klines_data(klines_5m, 20) or not self.validate_klines_data(klines_15m, 20):
                    logger.warning(f"{symbol}: Свежие данные не прошли валидацию")
                    continue

                logger.debug(f"🔢 {symbol}: Расчет индикаторов...")
                indicators_5m = self.calculate_fresh_indicators(symbol, '5', klines_5m, config.FINAL_INDICATORS)
                indicators_15m = self.calculate_fresh_indicators(symbol, '15', klines_15m, config.FINAL_INDICATORS)

                if not indicators_5m or not indicators_15m:
                    logger.warning(f"{symbol}: Ошибка расчета индикаторов")
                    continue

                # Сохраняем для валидации
                self.validation_data[symbol] = {
                    'klines_5m': klines_5m[-100:],
                    'klines_15m': klines_15m[-50:],
                    'indicators_5m': indicators_5m,
                    'indicators_15m': indicators_15m,
                    'data_timestamp': datetime.now().isoformat(),
                    'data_freshness': 'FRESH'
                }

                logger.debug(f"🤖 {symbol}: AI анализ...")
                analysis = await ai_router.analyze_pair(symbol, klines_5m, klines_15m, indicators_5m, indicators_15m)

                if analysis['signal'] != 'NO_SIGNAL' and analysis['confidence'] >= config.MIN_CONFIDENCE:
                    analysis['data_freshness'] = 'FRESH'
                    analysis['data_timestamp'] = datetime.now().isoformat()
                    final_signals.append(analysis)
                    logger.info(f"✅ {symbol}: {analysis['signal']} {analysis['confidence']}% (FRESH DATA)")

            except Exception as e:
                logger.error(f"Ошибка анализа {symbol}: {e}")
                continue

        elapsed = time.time() - start_time
        logger.info(f"✅ ЭТАП 3 завершен за {elapsed:.1f}с")
        logger.info(f"   Сигналы: {len(final_signals)}")

        return final_signals

    async def stage4_final_validation(self, preliminary_signals: List[Dict]) -> Dict[str, List[Dict]]:
        """ЭТАП 4: Финальная валидация с РАСШИРЕННЫМИ проверками"""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"ЭТАП 4: Расширенная валидация {len(preliminary_signals)} сигналов")
        logger.info("=" * 60)

        if not preliminary_signals:
            return {'validated': [], 'rejected': []}

        try:
            # Инициализируем валидатор
            await self.initialize_validator()

            # Подготавливаем данные
            candles_data = {}
            for signal in preliminary_signals:
                symbol = signal['symbol']
                if symbol in self.validation_data:
                    candles_data[symbol] = {
                        '1h': self.validation_data[symbol]['klines_5m'],
                        '4h': self.validation_data[symbol]['klines_15m']
                    }

            # Загружаем BTC данные
            logger.info("📊 Загрузка BTC данных для корреляционного анализа...")
            btc_candles_1h = await fetch_klines('BTCUSDT', '5', 168)

            # Пакетная валидация
            logger.info("🔍 Запуск расширенной валидации...")
            validation_result = await validate_signals_batch(
                self.enhanced_validator,
                preliminary_signals,
                candles_data,
                btc_candles_1h,
                None  # sector_candles опционально
            )

            validated_signals = validation_result['validated']
            rejected_signals = validation_result['rejected']

            # Статистика
            all_results = [s.get('validation', {}) for s in validated_signals]
            stats = get_validation_statistics(all_results)

            elapsed = time.time() - start_time
            logger.info(f"✅ ЭТАП 4 завершен за {elapsed:.1f}с")
            logger.info(f"   Подтверждено: {len(validated_signals)}")
            logger.info(f"   Отклонено: {len(rejected_signals)}")
            logger.info(f"   Approval rate: {stats['approval_rate']}%")
            logger.info(f"   Avg confidence change: {stats['avg_confidence_change']:+.1f}")

            return {
                'validated': validated_signals,
                'rejected': rejected_signals,
                'validation_stats': stats
            }

        except Exception as e:
            logger.error(f"Ошибка расширенной валидации: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'validated': [], 'rejected': []}

    async def run_full_cycle(self) -> Dict[str, Any]:
        """Полный цикл работы бота"""
        cycle_start = time.time()

        logger.info("=" * 80)
        logger.info("🚀 ЗАПУСК ПОЛНОГО ЦИКЛА АНАЛИЗА")
        logger.info("=" * 80)

        # Статус AI
        ai_status = ai_router.get_status()
        logger.info(f"🤖 AI провайдеры: {[k for k, v in ai_status['providers_available'].items() if v]}")
        logger.info(f"📊 Этапы: отбор-{ai_status['effective_providers']['selection']}, "
                    f"анализ-{ai_status['effective_providers']['analysis']}, "
                    f"валидация-{ai_status['effective_providers']['validation']}")

        try:
            self.cache.clear()

            # ЭТАП 1
            signal_pairs = await self.stage1_filter_signals()
            if not signal_pairs:
                return {
                    'result': 'NO_SIGNAL_PAIRS',
                    'total_time': time.time() - cycle_start,
                    'pairs_scanned': self.processed_pairs,
                    'message': 'Нет пар с торговыми сигналами'
                }

            # ЭТАП 2
            selected_pairs = await self.stage2_ai_bulk_select(signal_pairs)
            if not selected_pairs:
                return {
                    'result': 'NO_AI_SELECTION',
                    'total_time': time.time() - cycle_start,
                    'signal_pairs': len(signal_pairs),
                    'pairs_scanned': self.processed_pairs,
                    'message': 'AI не выбрал подходящих пар'
                }

            # ЭТАП 3
            preliminary_signals = await self.stage3_detailed_analysis(selected_pairs)
            if not preliminary_signals:
                return {
                    'result': 'NO_PRELIMINARY_SIGNALS',
                    'total_time': time.time() - cycle_start,
                    'pairs_scanned': self.processed_pairs,
                    'signal_pairs': len(signal_pairs),
                    'ai_selected': len(selected_pairs),
                    'message': 'Детальный анализ не выявил качественных сигналов'
                }

            # ЭТАП 4
            validation_result = await self.stage4_final_validation(preliminary_signals)
            validated_signals = validation_result['validated']
            rejected_signals = validation_result['rejected']

            # Результат
            total_time = time.time() - cycle_start
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if validated_signals:
                result_type = 'SUCCESS'
            elif rejected_signals:
                result_type = 'NO_VALIDATED_SIGNALS_WITH_REJECTED'
            else:
                result_type = 'NO_VALIDATED_SIGNALS'

            final_result = {
                'timestamp': timestamp,
                'result': result_type,
                'total_time': round(total_time, 2),
                'ai_status': ai_status,
                'data_freshness': 'STAGE3_FRESH_DATA',
                'stats': {
                    'pairs_scanned': self.processed_pairs,
                    'signal_pairs_found': len(signal_pairs),
                    'ai_selected': len(selected_pairs),
                    'preliminary_signals': len(preliminary_signals),
                    'validated_signals': len(validated_signals),
                    'rejected_signals': len(rejected_signals),
                    'processing_speed': round(self.processed_pairs / total_time, 1) if total_time > 0 else 0
                },
                'validated_signals': validated_signals,
                'rejected_signals': rejected_signals if rejected_signals else None,
                'validation_stats': validation_result.get('validation_stats', {}),
                'ai_providers': ai_status['providers_available']
            }

            # Сохраняем
            filename = f'bot_result_{timestamp}.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False, default=str)

            logger.info("=" * 80)
            logger.info(f"✅ ЦИКЛ ЗАВЕРШЕН: {self.processed_pairs}->{len(signal_pairs)}->{len(selected_pairs)}->{len(preliminary_signals)}->{len(validated_signals)}")
            logger.info(f"⏱️  Время: {total_time:.1f}с, скорость: {self.processed_pairs/total_time:.0f} пар/сек")
            logger.info(f"📁 Результат сохранен: {filename}")
            logger.info("=" * 80)

            return final_result

        except Exception as e:
            logger.error(f"Критическая ошибка цикла: {e}")
            import traceback
            logger.error(f"Трассировка: {traceback.format_exc()}")
            return {
                'result': 'ERROR',
                'error': str(e),
                'total_time': time.time() - cycle_start
            }

    async def cleanup(self):
        """Очистка ресурсов"""
        self.cache.clear()
        await cleanup_api()


async def main():
    """Главная функция"""
    print("=" * 80)
    print("🚀 ОПТИМИЗИРОВАННЫЙ СКАЛЬПИНГОВЫЙ БОТ v3.0 (ENHANCED VALIDATION)")
    print("=" * 80)
    print(f"Запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    ai_status = ai_router.get_status()
    print(f"🤖 Доступные AI: {[k for k, v in ai_status['providers_available'].items() if v]}")
    print(f"📊 Этапы: {ai_status['effective_providers']['selection']}/{ai_status['effective_providers']['analysis']}/{ai_status['effective_providers']['validation']}")
    print("=" * 80)
    print("✨ НОВОЕ: Enhanced Validator с Market Data + Correlations + Volume Profile + AI Advanced")
    print("=" * 80)
    print()

    bot = OptimizedScalpingBot()

    try:
        result = await bot.run_full_cycle()

        # Компактный вывод
        print()
        print("=" * 80)
        print(f"📊 РЕЗУЛЬТАТ: {result['result']}")
        print(f"⏱️  Время: {result.get('total_time', 0):.1f}сек")
        print(f"🔍 Свежесть данных: {result.get('data_freshness', 'UNKNOWN')}")
        print("=" * 80)

        if 'stats' in result:
            s = result['stats']
            print(f"\n📈 Пайплайн: {s['pairs_scanned']}->{s['signal_pairs_found']}->{s['ai_selected']}->{s['preliminary_signals']}->{s['validated_signals']}")
            print(f"⚡ Скорость: {s['processing_speed']} пар/сек")

        if 'validation_stats' in result and result['validation_stats']:
            vs = result['validation_stats']
            print(f"\n🎯 Валидация:")
            print(f"   Approval rate: {vs.get('approval_rate', 0)}%")
            print(f"   Avg confidence change: {vs.get('avg_confidence_change', 0):+.1f}")

        if result.get('validated_signals'):
            print(f"\n✅ ПОДТВЕРЖДЕННЫЕ СИГНАЛЫ ({len(result['validated_signals'])}):")
            for signal in result['validated_signals']:
                rr = signal.get('risk_reward_ratio', 'N/A')
                duration = signal.get('hold_duration_minutes', 'N/A')
                confidence = signal.get('confidence', 0)
                orig_conf = signal.get('original_confidence', confidence)
                adjustment = confidence - orig_conf
                freshness = signal.get('data_freshness', 'UNKNOWN')
                print(f"  {signal['symbol']}: {signal['signal']} ({orig_conf}→{confidence}% {adjustment:+d}) R/R:1:{rr} {duration}мин [{freshness}]")

        elif result.get('rejected_signals'):
            rejected = result['rejected_signals']
            print(f"\n❌ ОТКЛОНЕННЫЕ СИГНАЛЫ ({len(rejected)}):")
            for signal in rejected[:5]:  # Показываем топ-5
                entry = signal.get('entry_price', 0)
                confidence = signal.get('confidence', 0)
                freshness = signal.get('data_freshness', 'UNKNOWN')
                reason = signal.get('rejection_reason', 'Unknown')
                print(f"  {signal['symbol']}: {signal['signal']} ({confidence}%) [{freshness}]")
                print(f"    Вход: {entry:.6f} | Причина: {reason}")
        else:
            print("\n⚠️  Сигналов не найдено")

        print("\n" + "=" * 80)

    except KeyboardInterrupt:
        print("\n⚠️  Остановлено пользователем")
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        import traceback
        logger.error(f"Трассировка: {traceback.format_exc()}")
    finally:
        await bot.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Программа остановлена")