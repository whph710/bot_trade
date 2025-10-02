"""
Торговый бот - основной модуль
Без кэширования, универсальные таймфреймы
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from typing import List, Dict, Any

from config import config, has_ai_available
from func_async import get_trading_pairs, fetch_klines, batch_fetch_klines, cleanup as cleanup_api, get_optimized_session
from func_trade import calculate_basic_indicators, calculate_ai_indicators, check_basic_signal, validate_candles
from ai_router import ai_router
from func_enhanced_validator import (
    EnhancedSignalValidator,
    validate_signals_batch,
    batch_quick_market_check,
    get_validation_statistics
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class TradingBot:
    """Торговый бот без кэширования"""

    def __init__(self):
        self.processed_pairs = 0
        self.session_start = time.time()
        self.validation_data = {}
        self.enhanced_validator = None

    async def initialize_validator(self):
        """Инициализация валидатора"""
        if not self.enhanced_validator:
            session = await get_optimized_session()
            self.enhanced_validator = EnhancedSignalValidator(session, ai_router)
            logger.info("Enhanced validator initialized")

    async def load_candles_batch(self, pairs: List[str], interval: str, limit: int) -> Dict[str, List]:
        """Массовая загрузка свечей"""
        requests = [{'symbol': pair, 'interval': interval, 'limit': limit} for pair in pairs]
        results = await batch_fetch_klines(requests)

        candles_map = {}
        for result in results:
            if result.get('success') and result.get('klines'):
                symbol = result['symbol']
                klines = result['klines']
                if validate_candles(klines, 20):
                    candles_map[symbol] = klines

        return candles_map

    async def stage1_filter_signals(self) -> List[Dict]:
        """
        STAGE 1: Фильтрация пар с сигналами + Quick Market Checks
        Используется ДЛИННЫЙ таймфрейм для структурного анализа
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"STAGE 1: Signal filtering on {config.TIMEFRAME_LONG_NAME} timeframe")
        logger.info("=" * 60)

        # Получаем список пар
        pairs = await get_trading_pairs()
        if not pairs:
            logger.error("Failed to get trading pairs")
            return []

        # Quick market checks
        logger.info(f"Running quick market checks for {len(pairs)} pairs...")
        session = await get_optimized_session()
        quick_checks = await batch_quick_market_check(session, pairs, max_concurrent=20)

        tradeable_pairs = [p for p, check in quick_checks.items() if check.get('tradeable', False)]
        filtered_count = len(quick_checks) - len(tradeable_pairs)

        logger.info(f"Quick checks: {len(tradeable_pairs)} tradeable, {filtered_count} filtered out")

        # Загружаем свечи ДЛИННОГО таймфрейма
        candles_map = await self.load_candles_batch(
            tradeable_pairs,
            config.TIMEFRAME_LONG,
            config.QUICK_SCAN_CANDLES
        )

        logger.info(f"Loaded candles for {len(candles_map)} pairs")

        # Обрабатываем пары
        pairs_with_signals = []
        processed = 0

        for symbol, candles in candles_map.items():
            try:
                indicators = calculate_basic_indicators(candles)
                if not indicators:
                    continue

                signal_check = check_basic_signal(indicators)

                if signal_check['signal'] and signal_check['confidence'] >= config.MIN_CONFIDENCE:
                    pairs_with_signals.append({
                        'symbol': symbol,
                        'confidence': signal_check['confidence'],
                        'direction': signal_check['direction'],
                        'base_indicators': indicators
                    })

                processed += 1

            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue

        pairs_with_signals.sort(key=lambda x: x['confidence'], reverse=True)

        elapsed = time.time() - start_time
        self.processed_pairs = processed

        logger.info(f"STAGE 1 completed in {elapsed:.1f}s")
        logger.info(f"  Processed: {processed} pairs")
        logger.info(f"  Signals found: {len(pairs_with_signals)}")
        logger.info(f"  Filtered out: {filtered_count}")

        return pairs_with_signals

    async def stage2_ai_bulk_select(self, signal_pairs: List[Dict]) -> List[str]:
        """
        STAGE 2: AI отбор лучших пар
        Используется ДЛИННЫЙ таймфрейм
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"STAGE 2: AI selection from {len(signal_pairs)} pairs")
        logger.info("=" * 60)

        if not signal_pairs:
            return []

        # Загружаем свечи для AI анализа
        symbols = [p['symbol'] for p in signal_pairs]
        candles_map = await self.load_candles_batch(
            symbols,
            config.TIMEFRAME_LONG,
            config.AI_BULK_CANDLES
        )

        # Подготавливаем данные для AI
        ai_input_data = []

        for pair_data in signal_pairs:
            symbol = pair_data['symbol']

            if symbol not in candles_map:
                continue

            candles = candles_map[symbol]
            indicators = calculate_ai_indicators(candles, config.AI_INDICATORS_HISTORY)

            if not indicators:
                continue

            ai_input_data.append({
                'symbol': symbol,
                'confidence': pair_data['confidence'],
                'direction': pair_data['direction'],
                'candles_15m': candles,  # Имя ключа для совместимости с промптами
                'indicators_15m': indicators
            })

        if not ai_input_data:
            logger.error("No data prepared for AI")
            return []

        logger.info(f"Sending {len(ai_input_data)} pairs to AI")

        selected_pairs = await ai_router.select_pairs(ai_input_data)

        elapsed = time.time() - start_time
        logger.info(f"STAGE 2 completed in {elapsed:.1f}s")
        logger.info(f"  Selected: {len(selected_pairs)} pairs")

        return selected_pairs

    async def stage3_detailed_analysis(self, selected_pairs: List[str]) -> List[Dict]:
        """
        STAGE 3: Детальный анализ на обоих таймфреймах
        Загружаем КОРОТКИЙ и ДЛИННЫЙ таймфреймы
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"STAGE 3: Detailed analysis of {len(selected_pairs)} pairs")
        logger.info("=" * 60)

        if not selected_pairs:
            return []

        final_signals = []

        for symbol in selected_pairs:
            try:
                logger.debug(f"Analyzing {symbol}...")

                # Загружаем ОБА таймфрейма
                klines_short = await fetch_klines(symbol, config.TIMEFRAME_SHORT, config.FINAL_SHORT_CANDLES)
                klines_long = await fetch_klines(symbol, config.TIMEFRAME_LONG, config.FINAL_LONG_CANDLES)

                if not klines_short or not klines_long:
                    logger.warning(f"{symbol}: Failed to load candles")
                    continue

                if not validate_candles(klines_short, 20) or not validate_candles(klines_long, 20):
                    logger.warning(f"{symbol}: Invalid candles")
                    continue

                # Рассчитываем индикаторы
                indicators_short = calculate_ai_indicators(klines_short, config.FINAL_INDICATORS_HISTORY)
                indicators_long = calculate_ai_indicators(klines_long, config.FINAL_INDICATORS_HISTORY)

                if not indicators_short or not indicators_long:
                    logger.warning(f"{symbol}: Indicators calculation failed")
                    continue

                # Сохраняем для валидации
                self.validation_data[symbol] = {
                    'klines_short': klines_short[-100:],
                    'klines_long': klines_long[-50:],
                    'indicators_short': indicators_short,
                    'indicators_long': indicators_long,
                    'timestamp': datetime.now().isoformat()
                }

                # AI анализ (используем старые имена ключей для совместимости)
                analysis = await ai_router.analyze_pair(
                    symbol,
                    klines_short,  # data_5m для промпта
                    klines_long,   # data_15m для промпта
                    indicators_short,  # indicators_5m
                    indicators_long    # indicators_15m
                )

                if analysis['signal'] != 'NO_SIGNAL' and analysis['confidence'] >= config.MIN_CONFIDENCE:
                    analysis['timestamp'] = datetime.now().isoformat()
                    final_signals.append(analysis)
                    logger.info(f"Signal found: {symbol} {analysis['signal']} {analysis['confidence']}%")

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue

        elapsed = time.time() - start_time
        logger.info(f"STAGE 3 completed in {elapsed:.1f}s")
        logger.info(f"  Signals: {len(final_signals)}")

        return final_signals

    async def stage4_final_validation(self, preliminary_signals: List[Dict]) -> Dict[str, Any]:
        """
        STAGE 4: Расширенная валидация
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"STAGE 4: Enhanced validation of {len(preliminary_signals)} signals")
        logger.info("=" * 60)

        if not preliminary_signals:
            return {'validated': [], 'rejected': []}

        try:
            await self.initialize_validator()

            # Подготовка данных
            candles_data = {}
            for signal in preliminary_signals:
                symbol = signal['symbol']
                if symbol in self.validation_data:
                    candles_data[symbol] = {
                        '1h': self.validation_data[symbol]['klines_short'],
                        '4h': self.validation_data[symbol]['klines_long']
                    }

            # Загружаем BTC для корреляций
            logger.info("Loading BTC data for correlation analysis...")
            btc_candles = await fetch_klines('BTCUSDT', config.TIMEFRAME_SHORT, config.FINAL_SHORT_CANDLES)

            # Валидация
            logger.info("Running enhanced validation...")
            validation_result = await validate_signals_batch(
                self.enhanced_validator,
                preliminary_signals,
                candles_data,
                btc_candles,
                None
            )

            validated = validation_result['validated']
            rejected = validation_result['rejected']

            # Статистика
            all_results = [s.get('validation', {}) for s in validated]
            stats = get_validation_statistics(all_results)

            elapsed = time.time() - start_time
            logger.info(f"STAGE 4 completed in {elapsed:.1f}s")
            logger.info(f"  Approved: {len(validated)}")
            logger.info(f"  Rejected: {len(rejected)}")
            logger.info(f"  Approval rate: {stats['approval_rate']}%")

            return {
                'validated': validated,
                'rejected': rejected,
                'validation_stats': stats
            }

        except Exception as e:
            logger.error(f"Validation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'validated': [], 'rejected': []}

    async def run_full_cycle(self) -> Dict[str, Any]:
        """Полный цикл работы"""
        cycle_start = time.time()

        logger.info("=" * 80)
        logger.info("STARTING FULL ANALYSIS CYCLE")
        logger.info("=" * 80)

        ai_status = ai_router.get_status()
        logger.info(f"AI providers: {[k for k, v in ai_status['providers_available'].items() if v]}")
        logger.info(f"Timeframes: {config.TIMEFRAME_SHORT_NAME} (detail) + {config.TIMEFRAME_LONG_NAME} (structure)")

        try:
            # STAGE 1
            signal_pairs = await self.stage1_filter_signals()
            if not signal_pairs:
                return {
                    'result': 'NO_SIGNAL_PAIRS',
                    'total_time': time.time() - cycle_start,
                    'pairs_scanned': self.processed_pairs,
                    'message': 'No pairs with signals'
                }

            # STAGE 2
            selected_pairs = await self.stage2_ai_bulk_select(signal_pairs)
            if not selected_pairs:
                return {
                    'result': 'NO_AI_SELECTION',
                    'total_time': time.time() - cycle_start,
                    'signal_pairs': len(signal_pairs),
                    'pairs_scanned': self.processed_pairs,
                    'message': 'AI did not select any pairs'
                }

            # STAGE 3
            preliminary_signals = await self.stage3_detailed_analysis(selected_pairs)
            if not preliminary_signals:
                return {
                    'result': 'NO_PRELIMINARY_SIGNALS',
                    'total_time': time.time() - cycle_start,
                    'pairs_scanned': self.processed_pairs,
                    'signal_pairs': len(signal_pairs),
                    'ai_selected': len(selected_pairs),
                    'message': 'No signals from detailed analysis'
                }

            # STAGE 4
            validation_result = await self.stage4_final_validation(preliminary_signals)
            validated = validation_result['validated']
            rejected = validation_result['rejected']

            # Результат
            total_time = time.time() - cycle_start
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            result_type = 'SUCCESS' if validated else 'NO_VALIDATED_SIGNALS'

            final_result = {
                'timestamp': timestamp,
                'result': result_type,
                'total_time': round(total_time, 2),
                'timeframes': f"{config.TIMEFRAME_SHORT_NAME}/{config.TIMEFRAME_LONG_NAME}",
                'stats': {
                    'pairs_scanned': self.processed_pairs,
                    'signal_pairs_found': len(signal_pairs),
                    'ai_selected': len(selected_pairs),
                    'preliminary_signals': len(preliminary_signals),
                    'validated_signals': len(validated),
                    'rejected_signals': len(rejected),
                    'processing_speed': round(self.processed_pairs / total_time, 1)
                },
                'validated_signals': validated,
                'rejected_signals': rejected if rejected else None,
                'validation_stats': validation_result.get('validation_stats', {})
            }

            # Сохраняем
            filename = f'bot_result_{timestamp}.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False, default=str)

            logger.info("=" * 80)
            logger.info(f"CYCLE COMPLETE: {self.processed_pairs}->{len(signal_pairs)}->{len(selected_pairs)}->{len(preliminary_signals)}->{len(validated)}")
            logger.info(f"Time: {total_time:.1f}s, speed: {self.processed_pairs/total_time:.0f} pairs/sec")
            logger.info(f"Result saved: {filename}")
            logger.info("=" * 80)

            return final_result

        except Exception as e:
            logger.error(f"Critical cycle error: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
    print("=" * 80)
    print("TRADING BOT v3.0 (1H/4H STRATEGY)")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    ai_status = ai_router.get_status()
    print(f"AI providers: {[k for k, v in ai_status['providers_available'].items() if v]}")
    print(f"Timeframes: {config.TIMEFRAME_SHORT_NAME}/{config.TIMEFRAME_LONG_NAME}")
    print("=" * 80)

    bot = TradingBot()

    try:
        result = await bot.run_full_cycle()

        print()
        print("=" * 80)
        print(f"RESULT: {result['result']}")
        print(f"Time: {result.get('total_time', 0):.1f}s")
        print("=" * 80)

        if 'stats' in result:
            s = result['stats']
            print(f"\nPipeline: {s['pairs_scanned']}->{s['signal_pairs_found']}->{s['ai_selected']}->{s['preliminary_signals']}->{s['validated_signals']}")
            print(f"Speed: {s['processing_speed']} pairs/sec")

        if result.get('validated_signals'):
            print(f"\nVALIDATED SIGNALS ({len(result['validated_signals'])}):")
            for sig in result['validated_signals']:
                conf = sig.get('confidence', 0)
                orig = sig.get('original_confidence', conf)
                adj = conf - orig
                rr = sig.get('risk_reward_ratio', 'N/A')
                print(f"  {sig['symbol']}: {sig['signal']} ({orig}->{conf}% {adj:+d}) R/R:1:{rr}")

        elif result.get('rejected_signals'):
            print(f"\nREJECTED SIGNALS ({len(result['rejected_signals'])}):")
            for sig in result['rejected_signals'][:5]:
                print(f"  {sig['symbol']}: {sig['signal']} - {sig.get('rejection_reason', 'Unknown')}")

        print("\n" + "=" * 80)

    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        await bot.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram stopped")