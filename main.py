"""
Торговый бот - ФИНАЛЬНАЯ ВЕРСИЯ

Ключевые улучшения:
1. Все данные собираются в Stage 3
2. Stage 4 только валидирует
3. Компактный JSON на выходе
4. Адаптивные TP levels и hold time
5. Всегда возвращает JSON даже если rejected
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
    validate_signals_batch_improved,
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
    """Торговый бот - финальная версия"""

    def __init__(self):
        self.processed_pairs = 0
        self.session_start = time.time()
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
        """STAGE 1: Фильтрация + Quick Market Checks"""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"STAGE 1: Signal filtering on {config.TIMEFRAME_LONG_NAME} timeframe")
        logger.info("=" * 60)

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

        # Загружаем свечи
        candles_map = await self.load_candles_batch(
            tradeable_pairs,
            config.TIMEFRAME_LONG,
            config.QUICK_SCAN_CANDLES
        )

        logger.info(f"Loaded candles for {len(candles_map)} pairs")

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

        return pairs_with_signals

    async def stage2_ai_bulk_select(self, signal_pairs: List[Dict]) -> List[str]:
        """STAGE 2: AI отбор лучших пар"""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"STAGE 2: AI selection from {len(signal_pairs)} pairs")
        logger.info("=" * 60)

        if not signal_pairs:
            return []

        symbols = [p['symbol'] for p in signal_pairs]
        candles_map = await self.load_candles_batch(
            symbols,
            config.TIMEFRAME_LONG,
            config.AI_BULK_CANDLES
        )

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
                'candles_15m': candles,
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

    async def stage3_detailed_analysis_with_data(self, selected_pairs: List[str]) -> List[Dict]:
        """
        STAGE 3: ПОЛНЫЙ анализ с ВСЕМИ данными

        КРИТИЧНО: Собираются ВСЕ данные для AI:
        - Свечи 1H и 4H
        - Индикаторы
        - Market data (funding, OI, orderbook, taker volume)
        - BTC данные для корреляций
        - Volume Profile
        - AI анализы (OrderFlow, SMC)
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"STAGE 3: Detailed analysis + DATA COLLECTION for {len(selected_pairs)} pairs")
        logger.info("=" * 60)

        if not selected_pairs:
            return []

        await self.initialize_validator()

        # Загружаем BTC для корреляций
        logger.info("Loading BTC data for correlations...")
        btc_candles_1h = await fetch_klines('BTCUSDT', config.TIMEFRAME_SHORT, config.FINAL_SHORT_CANDLES)
        btc_candles_4h = await fetch_klines('BTCUSDT', config.TIMEFRAME_LONG, config.FINAL_LONG_CANDLES)

        final_signals = []

        for symbol in selected_pairs:
            try:
                logger.debug(f"Analyzing {symbol}...")

                # 1. Загружаем свечи
                klines_1h = await fetch_klines(symbol, config.TIMEFRAME_SHORT, config.FINAL_SHORT_CANDLES)
                klines_4h = await fetch_klines(symbol, config.TIMEFRAME_LONG, config.FINAL_LONG_CANDLES)

                if not klines_1h or not klines_4h:
                    logger.warning(f"{symbol}: Failed to load candles")
                    continue

                if not validate_candles(klines_1h, 20) or not validate_candles(klines_4h, 20):
                    logger.warning(f"{symbol}: Invalid candles")
                    continue

                # 2. Рассчитываем индикаторы
                indicators_1h = calculate_ai_indicators(klines_1h, config.FINAL_INDICATORS_HISTORY)
                indicators_4h = calculate_ai_indicators(klines_4h, config.FINAL_INDICATORS_HISTORY)

                if not indicators_1h or not indicators_4h:
                    logger.warning(f"{symbol}: Indicators calculation failed")
                    continue

                current_price = float(klines_1h[-1][4]) if klines_1h else 0

                # 3. СОБИРАЕМ ВСЕ РАСШИРЕННЫЕ ДАННЫЕ
                logger.debug(f"{symbol}: Collecting market data...")

                from func_market_data import MarketDataCollector
                from func_correlation import get_comprehensive_correlation_analysis
                from func_volume_profile import calculate_volume_profile_for_candles, analyze_volume_profile
                from ai_advanced_analysis import get_ai_orderflow_analysis, get_ai_smc_patterns

                collector = MarketDataCollector(await get_optimized_session())

                # Market snapshot с адаптивным стаканом
                market_snapshot = await collector.get_market_snapshot(symbol, current_price)

                # Correlation analysis
                corr_analysis = await get_comprehensive_correlation_analysis(
                    symbol,
                    klines_1h,
                    btc_candles_1h,
                    'UNKNOWN',
                    None
                )

                # Volume Profile (на 4H данных)
                vp_data = calculate_volume_profile_for_candles(klines_4h, num_bins=50)
                vp_analysis = analyze_volume_profile(vp_data, current_price) if vp_data else None

                # AI OrderFlow Analysis
                orderflow_ai = None
                if market_snapshot['orderbook']:
                    prices_recent = [float(c[4]) for c in klines_1h[-20:]]
                    orderflow_ai = await get_ai_orderflow_analysis(
                        ai_router,
                        symbol,
                        market_snapshot['orderbook'],
                        prices_recent
                    )

                # AI Smart Money Concepts
                smc_ai = await get_ai_smc_patterns(
                    ai_router,
                    symbol,
                    klines_1h,
                    current_price
                )

                # 4. ФОРМИРУЕМ ПОЛНЫЙ ПАКЕТ ДАННЫХ ДЛЯ AI
                comprehensive_data = {
                    'symbol': symbol,
                    'candles_1h': klines_1h,
                    'candles_4h': klines_4h,
                    'indicators_1h': indicators_1h,
                    'indicators_4h': indicators_4h,
                    'current_price': current_price,
                    'market_data': market_snapshot,
                    'correlation_data': corr_analysis,
                    'volume_profile': vp_data,
                    'vp_analysis': vp_analysis,
                    'orderflow_ai': orderflow_ai,
                    'smc_ai': smc_ai,
                    'btc_candles_1h': btc_candles_1h,
                    'btc_candles_4h': btc_candles_4h
                }

                # 5. AI АНАЛИЗ С ПОЛНЫМИ ДАННЫМИ
                logger.debug(f"{symbol}: Running AI analysis with comprehensive data...")

                analysis = await ai_router.analyze_pair_comprehensive(
                    symbol,
                    comprehensive_data
                )

                if analysis['signal'] != 'NO_SIGNAL' and analysis['confidence'] >= config.MIN_CONFIDENCE:
                    # Сохраняем ВСЕ данные для Stage 4
                    analysis['comprehensive_data'] = comprehensive_data
                    analysis['timestamp'] = datetime.now().isoformat()

                    final_signals.append(analysis)
                    logger.info(f"Signal found: {symbol} {analysis['signal']} {analysis['confidence']}%")

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        elapsed = time.time() - start_time
        logger.info(f"STAGE 3 completed in {elapsed:.1f}s")
        logger.info(f"  Signals with full data: {len(final_signals)}")

        return final_signals

    async def stage4_final_validation(self, preliminary_signals: List[Dict]) -> Dict[str, Any]:
        """
        STAGE 4: ТОЛЬКО валидация (БЕЗ сбора новых данных)

        Используются данные из Stage 3 (comprehensive_data)
        Всегда возвращает JSON с levels даже для rejected
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"STAGE 4: Final validation of {len(preliminary_signals)} signals")
        logger.info("=" * 60)

        if not preliminary_signals:
            return {'validated': [], 'rejected': []}

        try:
            await self.initialize_validator()

            logger.info("Running validation with Stage 3 data...")

            # Валидация БЕЗ сбора новых данных
            validation_result = await validate_signals_batch_improved(
                self.enhanced_validator,
                preliminary_signals
            )

            validated = validation_result['validated']
            rejected = validation_result['rejected']

            # Очищаем comprehensive_data из финального ответа (она огромная)
            for signal in validated + rejected:
                if 'comprehensive_data' in signal:
                    del signal['comprehensive_data']

            elapsed = time.time() - start_time
            logger.info(f"STAGE 4 completed in {elapsed:.1f}s")
            logger.info(f"  Approved: {len(validated)}")
            logger.info(f"  Rejected: {len(rejected)}")

            return {
                'validated': validated,
                'rejected': rejected
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

        try:
            # STAGE 1
            signal_pairs = await self.stage1_filter_signals()
            if not signal_pairs:
                return {
                    'result': 'NO_SIGNAL_PAIRS',
                    'total_time': time.time() - cycle_start,
                    'pairs_scanned': self.processed_pairs
                }

            # STAGE 2
            selected_pairs = await self.stage2_ai_bulk_select(signal_pairs)
            if not selected_pairs:
                return {
                    'result': 'NO_AI_SELECTION',
                    'total_time': time.time() - cycle_start,
                    'signal_pairs': len(signal_pairs),
                    'pairs_scanned': self.processed_pairs
                }

            # STAGE 3 (с ПОЛНЫМ сбором данных)
            preliminary_signals = await self.stage3_detailed_analysis_with_data(selected_pairs)
            if not preliminary_signals:
                return {
                    'result': 'NO_PRELIMINARY_SIGNALS',
                    'total_time': time.time() - cycle_start,
                    'pairs_scanned': self.processed_pairs,
                    'signal_pairs': len(signal_pairs),
                    'ai_selected': len(selected_pairs)
                }

            # STAGE 4 (только валидация)
            validation_result = await self.stage4_final_validation(preliminary_signals)
            validated = validation_result['validated']
            rejected = validation_result['rejected']

            # КОМПАКТНЫЙ финальный результат
            total_time = time.time() - cycle_start
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            result_type = 'SUCCESS' if validated else 'NO_VALIDATED_SIGNALS'

            # Форматируем КОМПАКТНЫЙ ответ
            compact_signals = []
            for sig in validated:
                compact_signal = {
                    'symbol': sig['symbol'],
                    'signal': sig['signal'],
                    'confidence': sig['confidence'],
                    'entry_price': sig['entry_price'],
                    'stop_loss': sig['stop_loss'],
                    'take_profit_levels': sig.get('take_profit_levels', []),
                    'hold_time_hours': sig.get('hold_time_hours', {'min': 4, 'max': 48}),
                    'risk_reward_ratio': sig.get('risk_reward_ratio', 0),
                    'analysis': sig.get('analysis', '')
                }
                compact_signals.append(compact_signal)

            # Статистика валидации
            all_validation_results = [sig.get('validation', {}) for sig in validated + rejected if sig.get('validation')]
            validation_stats = get_validation_statistics(all_validation_results)

            final_result = {
                'timestamp': timestamp,
                'result': result_type,
                'total_time': round(total_time, 1),
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
                'validated_signals': compact_signals,
                'rejected_signals': None,  # Не включаем rejected в финальный JSON
                'validation_stats': validation_stats
            }

            # Сохраняем
            filename = f'bot_result_{timestamp}.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)

            logger.info("=" * 80)
            logger.info(f"CYCLE COMPLETE: {len(validated)} signals validated")
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
    print("TRADING BOT v4.0 (FINAL - COMPACT OUTPUT)")
    print("=" * 80)

    bot = TradingBot()

    try:
        result = await bot.run_full_cycle()

        print()
        print("=" * 80)
        print(f"RESULT: {result['result']}")
        print(f"Time: {result.get('total_time', 0):.1f}s")

        stats = result.get('stats', {})
        print(f"Pairs scanned: {stats.get('pairs_scanned', 0)}")
        print(f"Speed: {stats.get('processing_speed', 0):.1f} pairs/sec")
        print("=" * 80)

        if result.get('validated_signals'):
            signals = result['validated_signals']
            print(f"\n✅ VALIDATED SIGNALS ({len(signals)}):")
            print("=" * 80)

            for sig in signals:
                print(f"\n{sig['symbol']}: {sig['signal']} (Confidence: {sig['confidence']}%)")
                print(f"  Entry: ${sig['entry_price']}")
                print(f"  Stop:  ${sig['stop_loss']}")

                tps = sig['take_profit_levels']
                if len(tps) >= 3:
                    print(f"  TP1:   ${tps[0]} (Conservative)")
                    print(f"  TP2:   ${tps[1]} (Base Target)")
                    print(f"  TP3:   ${tps[2]} (Extended)")
                elif len(tps) == 2:
                    print(f"  TP1:   ${tps[0]}")
                    print(f"  TP2:   ${tps[1]}")
                elif len(tps) == 1:
                    print(f"  TP:    ${tps[0]}")

                hold = sig['hold_time_hours']
                print(f"  Hold:  {hold['min']}-{hold['max']} hours")
                print(f"  R/R:   1:{sig['risk_reward_ratio']}")

                # Краткий анализ (первые 150 символов)
                analysis = sig.get('analysis', '')
                if len(analysis) > 150:
                    analysis = analysis[:150] + "..."
                print(f"  Analysis: {analysis}")

        # Статистика валидации
        if result.get('validation_stats'):
            vstats = result['validation_stats']
            print(f"\n📊 VALIDATION STATISTICS:")
            print(f"  Approval rate: {vstats.get('approval_rate', 0):.1f}%")
            print(f"  Avg confidence change: {vstats.get('avg_confidence_change', 0):+.1f}")

            if vstats.get('top_rejection_reasons'):
                print(f"  Top rejection reasons:")
                for reason in vstats['top_rejection_reasons'][:3]:
                    print(f"    - {reason}")

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        await bot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())