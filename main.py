"""
Trading Bot v5.0 - Production Ready - CLEAN SHUTDOWN
"""

import asyncio
import logging
import time
import json
import sys
from datetime import datetime
from typing import List, Dict, Any
import pytz

from config import config
from func_async import get_trading_pairs, fetch_klines, batch_fetch_klines, cleanup as cleanup_api, get_optimized_session
from func_trade import calculate_basic_indicators, calculate_ai_indicators, check_basic_signal, validate_candles
from ai_router import ai_router
from simple_validator import validate_signals_simple, calculate_validation_stats, check_trading_hours

# Configure logging with rotating file handler
from logging.handlers import RotatingFileHandler

log_filename = "bot_trading.log"

# Create rotating file handler (max 10MB, keep 5 backup files)
file_handler = RotatingFileHandler(
    log_filename,
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)


class TradingBot:
    """Trading bot with unified analysis"""

    def __init__(self):
        self.processed_pairs = 0
        self.signal_pairs_count = 0
        self.ai_selected_count = 0
        self.analyzed_count = 0
        self.session_start = time.time()

    def _print_time_info(self):
        """Вывести текущее время в разных таймзонах"""
        utc_now = datetime.now(pytz.UTC)
        perm_tz = pytz.timezone('Asia/Yekaterinburg')
        perm_now = utc_now.astimezone(perm_tz)

        logger.info(f"⏰ UTC: {utc_now.strftime('%H:%M:%S')} | Пермь: {perm_now.strftime('%H:%M:%S')}")

    async def load_candles_batch(self, pairs: List[str], interval: str, limit: int) -> Dict[str, List]:
        """Batch load candles"""
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
        """Stage 1: Base signal filtering"""
        start_time = time.time()
        logger.info("STAGE 1: Signal filtering on %s", config.TIMEFRAME_LONG_NAME)

        pairs = await get_trading_pairs()
        if not pairs:
            logger.error("Failed to get trading pairs")
            return []

        logger.info("Found %d trading pairs, loading candles...", len(pairs))
        candles_map = await self.load_candles_batch(pairs, config.TIMEFRAME_LONG, config.QUICK_SCAN_CANDLES)
        logger.info("Loaded candles for %d pairs", len(candles_map))

        if not candles_map:
            logger.error("No valid candles loaded!")
            return []

        pairs_with_signals = []

        for symbol, candles in candles_map.items():
            try:
                self.processed_pairs += 1

                indicators = calculate_basic_indicators(candles)
                if not indicators:
                    logger.debug("%s: No indicators", symbol)
                    continue

                signal_check = check_basic_signal(indicators)

                if signal_check['signal'] and signal_check['confidence'] >= config.MIN_CONFIDENCE:
                    pairs_with_signals.append({
                        'symbol': symbol,
                        'confidence': signal_check['confidence'],
                        'direction': signal_check['direction'],
                        'base_indicators': indicators
                    })
                    logger.debug("%s: Signal %s conf=%d", symbol, signal_check['direction'], signal_check['confidence'])

            except Exception as e:
                logger.debug("Error processing %s: %s", symbol, e)
                continue

        pairs_with_signals.sort(key=lambda x: x['confidence'], reverse=True)

        self.signal_pairs_count = len(pairs_with_signals)
        elapsed = time.time() - start_time

        logger.info("Stage 1 completed in %.1fs", elapsed)
        logger.info("Processed: %d, Signals: %d", self.processed_pairs, len(pairs_with_signals))

        return pairs_with_signals

    async def stage2_ai_select(self, signal_pairs: List[Dict]) -> List[str]:
        """Stage 2: AI selection"""
        start_time = time.time()
        logger.info("STAGE 2: %s selection from %d pairs", config.STAGE2_PROVIDER.upper(), len(signal_pairs))

        if not signal_pairs:
            logger.warning("No signal pairs to select from")
            return []

        symbols = [p['symbol'] for p in signal_pairs]
        candles_map = await self.load_candles_batch(symbols, config.TIMEFRAME_LONG, config.AI_BULK_CANDLES)

        ai_input_data = []

        for pair_data in signal_pairs:
            symbol = pair_data['symbol']
            if symbol not in candles_map:
                logger.debug("%s: No candles for AI selection", symbol)
                continue

            candles = candles_map[symbol]
            indicators = calculate_ai_indicators(candles, config.AI_INDICATORS_HISTORY)

            if not indicators:
                logger.debug("%s: No indicators for AI selection", symbol)
                continue

            ai_input_data.append({
                'symbol': symbol,
                'confidence': pair_data['confidence'],
                'direction': pair_data['direction'],
                'candles_15m': candles,
                'indicators_15m': indicators
            })

        if not ai_input_data:
            logger.error("No data prepared for AI selection")
            return []

        logger.info("Sending %d pairs to %s", len(ai_input_data), config.STAGE2_PROVIDER.upper())
        selected_pairs = await ai_router.select_pairs(ai_input_data)

        self.ai_selected_count = len(selected_pairs)
        elapsed = time.time() - start_time

        logger.info("Stage 2 completed in %.1fs", elapsed)
        logger.info("Selected: %d pairs", len(selected_pairs))

        if not selected_pairs:
            logger.warning("AI selected 0 pairs - check AI selection logic")

        return selected_pairs

    async def stage3_unified_analysis(self, selected_pairs: List[str]) -> List[Dict]:
        """Stage 3: Unified analysis - FULLY OPTIMIZED"""
        start_time = time.time()
        logger.info("STAGE 3: %s unified analysis for %d pairs", config.STAGE3_PROVIDER.upper(), len(selected_pairs))

        if not selected_pairs:
            logger.warning("No pairs to analyze")
            return []

        logger.info("Loading BTC data (parallel)...")
        btc_candles_1h, btc_candles_4h = await asyncio.gather(
            fetch_klines('BTCUSDT', config.TIMEFRAME_SHORT, config.FINAL_SHORT_CANDLES),
            fetch_klines('BTCUSDT', config.TIMEFRAME_LONG, config.FINAL_LONG_CANDLES)
        )

        if not btc_candles_1h or not btc_candles_4h:
            logger.error("Failed to load BTC candles!")
            return []

        final_signals = []

        for symbol in selected_pairs:
            try:
                logger.info("Analyzing %s...", symbol)

                klines_1h, klines_4h = await asyncio.gather(
                    fetch_klines(symbol, config.TIMEFRAME_SHORT, config.FINAL_SHORT_CANDLES),
                    fetch_klines(symbol, config.TIMEFRAME_LONG, config.FINAL_LONG_CANDLES)
                )

                if not klines_1h or not klines_4h:
                    logger.warning("%s: Failed to load candles", symbol)
                    continue

                if not validate_candles(klines_1h, 20) or not validate_candles(klines_4h, 20):
                    logger.warning("%s: Invalid candles", symbol)
                    continue

                indicators_1h = calculate_ai_indicators(klines_1h, config.FINAL_INDICATORS_HISTORY)
                indicators_4h = calculate_ai_indicators(klines_4h, config.FINAL_INDICATORS_HISTORY)

                if not indicators_1h or not indicators_4h:
                    logger.warning("%s: Indicators failed", symbol)
                    continue

                current_price = float(klines_1h[-1][4])

                logger.debug("%s: Collecting market data...", symbol)

                from func_market_data import MarketDataCollector
                from func_correlation import get_comprehensive_correlation_analysis
                from func_volume_profile import calculate_volume_profile_for_candles, analyze_volume_profile

                collector = MarketDataCollector(await get_optimized_session())
                market_snapshot = await collector.get_market_snapshot(symbol, current_price)

                corr_analysis = await get_comprehensive_correlation_analysis(
                    symbol,
                    klines_1h,
                    btc_candles_1h,
                    'UNKNOWN',
                    None
                )

                vp_data = calculate_volume_profile_for_candles(klines_4h, num_bins=50)
                vp_analysis = analyze_volume_profile(vp_data, current_price) if vp_data else None

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
                    'btc_candles_1h': btc_candles_1h,
                    'btc_candles_4h': btc_candles_4h
                }

                logger.info("%s: Running unified analysis...", symbol)

                analysis = await ai_router.analyze_pair_comprehensive(symbol, comprehensive_data)

                signal_type = analysis.get('signal', 'NO_SIGNAL')
                confidence = analysis.get('confidence', 0)
                rejection = analysis.get('rejection_reason')

                if signal_type != 'NO_SIGNAL' and confidence >= config.MIN_CONFIDENCE:
                    analysis['comprehensive_data'] = comprehensive_data
                    analysis['timestamp'] = datetime.now().isoformat()

                    final_signals.append(analysis)

                    tp_levels = analysis.get('take_profit_levels', [0, 0, 0])
                    logger.info("[SIGNAL] %s %s %d%%", symbol, signal_type, confidence)
                    logger.info("  Entry: $%.2f, Stop: $%.2f", analysis['entry_price'], analysis['stop_loss'])
                    logger.info("  TP: $%.2f / $%.2f / $%.2f", tp_levels[0], tp_levels[1], tp_levels[2])
                else:
                    logger.info("[SKIPPED] %s - %s", symbol, rejection if rejection else 'weak setup')

            except Exception as e:
                logger.error("Error analyzing %s: %s", symbol, e, exc_info=True)
                continue

        self.analyzed_count = len(final_signals)
        elapsed = time.time() - start_time

        logger.info("Stage 3 completed in %.1fs", elapsed)
        logger.info("Signals generated: %d", len(final_signals))

        return final_signals

    async def stage4_validation(self, preliminary_signals: List[Dict]) -> Dict[str, Any]:
        """Stage 4: Signal validation"""
        start_time = time.time()
        logger.info("STAGE 4: %s validation of %d signals", config.STAGE4_PROVIDER.upper(), len(preliminary_signals))

        if not preliminary_signals:
            logger.warning("No signals to validate")
            return {'validated': [], 'rejected': []}

        validation_result = await validate_signals_simple(ai_router, preliminary_signals)

        if validation_result.get('validation_skipped_reason'):
            logger.warning(validation_result['validation_skipped_reason'])
            return validation_result

        validated = validation_result['validated']
        rejected = validation_result['rejected']

        elapsed = time.time() - start_time
        logger.info("Stage 4 completed in %.1fs", elapsed)
        logger.info("Approved: %d, Rejected: %d", len(validated), len(rejected))

        return validation_result

    async def run_full_cycle(self) -> Dict[str, Any]:
        """Run full trading cycle"""
        cycle_start = time.time()

        logger.info("=" * 60)
        logger.info("STARTING FULL CYCLE")
        self._print_time_info()
        logger.info("=" * 60)

        try:
            signal_pairs = await self.stage1_filter_signals()
            if not signal_pairs:
                total_time = time.time() - cycle_start
                logger.warning("Stage 1 found no signal pairs (processed %d pairs)", self.processed_pairs)
                return {
                    'result': 'NO_SIGNAL_PAIRS',
                    'total_time': total_time,
                    'stats': {
                        'pairs_scanned': self.processed_pairs,
                        'signal_pairs_found': 0,
                        'ai_selected': 0,
                        'analyzed': 0,
                        'validated_signals': 0,
                        'rejected_signals': 0,
                        'processing_speed': round(self.processed_pairs / total_time, 1) if total_time > 0 else 0
                    }
                }

            selected_pairs = await self.stage2_ai_select(signal_pairs)
            if not selected_pairs:
                total_time = time.time() - cycle_start
                logger.warning("Stage 2 selected no pairs from %d candidates", len(signal_pairs))
                return {
                    'result': 'NO_AI_SELECTION',
                    'total_time': total_time,
                    'stats': {
                        'pairs_scanned': self.processed_pairs,
                        'signal_pairs_found': self.signal_pairs_count,
                        'ai_selected': 0,
                        'analyzed': 0,
                        'validated_signals': 0,
                        'rejected_signals': 0,
                        'processing_speed': round(self.processed_pairs / total_time, 1) if total_time > 0 else 0
                    }
                }

            preliminary_signals = await self.stage3_unified_analysis(selected_pairs)
            if not preliminary_signals:
                total_time = time.time() - cycle_start
                logger.warning("Stage 3 produced no signals from %d pairs", len(selected_pairs))
                return {
                    'result': 'NO_ANALYSIS_SIGNALS',
                    'total_time': total_time,
                    'stats': {
                        'pairs_scanned': self.processed_pairs,
                        'signal_pairs_found': self.signal_pairs_count,
                        'ai_selected': self.ai_selected_count,
                        'analyzed': 0,
                        'validated_signals': 0,
                        'rejected_signals': 0,
                        'processing_speed': round(self.processed_pairs / total_time, 1) if total_time > 0 else 0
                    }
                }

            validation_result = await self.stage4_validation(preliminary_signals)
            validated = validation_result['validated']
            rejected = validation_result['rejected']

            if validation_result.get('validation_skipped_reason'):
                total_time = time.time() - cycle_start
                return {
                    'result': 'VALIDATION_SKIPPED',
                    'reason': validation_result['validation_skipped_reason'],
                    'total_time': total_time,
                    'stats': {
                        'pairs_scanned': self.processed_pairs,
                        'signal_pairs_found': self.signal_pairs_count,
                        'ai_selected': self.ai_selected_count,
                        'analyzed': self.analyzed_count,
                        'validated_signals': 0,
                        'rejected_signals': 0,
                        'processing_speed': round(self.processed_pairs / total_time, 1) if total_time > 0 else 0
                    }
                }

            total_time = time.time() - cycle_start
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            result_type = 'SUCCESS' if validated else 'NO_VALIDATED_SIGNALS'

            validation_stats = calculate_validation_stats(validated, rejected)

            final_result = {
                'timestamp': timestamp,
                'result': result_type,
                'total_time': round(total_time, 1),
                'timeframes': f"{config.TIMEFRAME_SHORT_NAME}/{config.TIMEFRAME_LONG_NAME}",
                'ai_config': {
                    'stage2': config.STAGE2_PROVIDER,
                    'stage3': config.STAGE3_PROVIDER,
                    'stage4': config.STAGE4_PROVIDER
                },
                'stats': {
                    'pairs_scanned': self.processed_pairs,
                    'signal_pairs_found': self.signal_pairs_count,
                    'ai_selected': self.ai_selected_count,
                    'analyzed': self.analyzed_count,
                    'validated_signals': len(validated),
                    'rejected_signals': len(rejected),
                    'processing_speed': round(self.processed_pairs / total_time, 1) if total_time > 0 else 0
                },
                'validated_signals': validated,
                'rejected_signals': rejected,
                'validation_stats': validation_stats
            }

            filename = f'bot_result_{timestamp}.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)

            logger.info("CYCLE COMPLETE: %d validated, %d rejected", len(validated), len(rejected))
            logger.info("Result saved: %s", filename)

            return final_result

        except Exception as e:
            logger.error("Critical cycle error: %s", e, exc_info=True)
            return {
                'result': 'ERROR',
                'error': str(e),
                'total_time': time.time() - cycle_start,
                'stats': {
                    'pairs_scanned': self.processed_pairs,
                    'signal_pairs_found': self.signal_pairs_count,
                    'ai_selected': self.ai_selected_count,
                    'analyzed': self.analyzed_count,
                    'validated_signals': 0,
                    'rejected_signals': 0,
                    'processing_speed': 0
                }
            }

    async def cleanup(self):
        """Cleanup resources"""
        await cleanup_api()


def print_clean_separator():
    """Чистый разделитель"""
    print()


async def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("TRADING BOT v5.0 - FULLY OPTIMIZED")
    print("=" * 60)
    print(f"Stage 1: Base indicators")
    print(f"Stage 2: {config.STAGE2_PROVIDER.upper()} selection")
    print(f"Stage 3: {config.STAGE3_PROVIDER.upper()} unified analysis")
    print(f"Stage 4: {config.STAGE4_PROVIDER.upper()} validation")
    print(f"Logs: {log_filename}")
    print("=" * 60 + "\n")

    time_allowed, time_reason = check_trading_hours()
    print(time_reason + "\n")

    bot = TradingBot()

    try:
        result = await bot.run_full_cycle()

        print_clean_separator()
        print("=" * 60)
        print("RESULT: " + result['result'])
        print("=" * 60)
        print(f"Time: {result.get('total_time', 0):.1f}s")

        stats = result.get('stats', {})
        print(f"Pairs scanned: {stats.get('pairs_scanned', 0)}")
        print(f"Signal pairs found: {stats.get('signal_pairs_found', 0)}")
        print(f"AI selected: {stats.get('ai_selected', 0)}")
        print(f"Analyzed: {stats.get('analyzed', 0)}")
        print(f"Validated: {stats.get('validated_signals', 0)}")
        print(f"Speed: {stats.get('processing_speed', 0):.1f} pairs/sec")
        print()

        if result.get('result') == 'VALIDATION_SKIPPED':
            print(f"⏰ {result.get('reason', 'Валидация пропущена')}")
            print()
            return

        if result.get('validated_signals'):
            signals = result['validated_signals']
            print(f"✅ VALIDATED SIGNALS ({len(signals)}):")

            for sig in signals:
                tp_levels = sig.get('take_profit_levels', [0, 0, 0])
                print(f"\n{sig['symbol']}: {sig['signal']} (Confidence: {sig['confidence']}%)")
                print(f"  Entry: ${sig['entry_price']:.2f}")
                print(f"  Stop:  ${sig['stop_loss']:.2f}")
                print(f"  TP1:   ${tp_levels[0]:.2f}")
                print(f"  TP2:   ${tp_levels[1]:.2f}")
                print(f"  TP3:   ${tp_levels[2]:.2f}")
                print(f"  R/R:   1:{sig.get('risk_reward_ratio', 0):.1f}")

        if result.get('rejected_signals'):
            rejected = result['rejected_signals']
            print(f"\n❌ REJECTED SIGNALS ({len(rejected)}):")

            for rej in rejected[:5]:  # Показываем только первые 5
                print(f"\n{rej['symbol']}: {rej.get('rejection_reason', 'Unknown')}")

        if result.get('validation_stats'):
            vstats = result['validation_stats']
            print(f"\n📊 VALIDATION STATS:")
            print(f"  Approval rate: {vstats.get('approval_rate', 0)}%")
            print(f"  Avg R/R: 1:{vstats.get('avg_risk_reward', 0):.1f}")

    except KeyboardInterrupt:
        # ✅ ЧИСТАЯ ОСТАНОВКА БЕЗ ЛИШНЕГО
        print("\n" + "=" * 60)
        print("⏹️  BOT STOPPED BY USER")
        print("=" * 60)
        logger.info("Bot stopped by user")
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ ERROR: {e}")
        print("=" * 60)
        logger.error("Unexpected error: %s", e, exc_info=True)
    finally:
        await bot.cleanup()
        print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("=" * 60)
        print("⏹️  BOT STOPPED")
        print("=" * 60)
        sys.exit(0)