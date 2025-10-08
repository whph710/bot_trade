"""
Trading Bot v5.0 - Production Ready
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from typing import List, Dict, Any

from config import config
from func_async import get_trading_pairs, fetch_klines, batch_fetch_klines, cleanup as cleanup_api, get_optimized_session
from func_trade import calculate_basic_indicators, calculate_ai_indicators, check_basic_signal, validate_candles
from ai_router import ai_router
from simple_validator import validate_signals_simple, calculate_validation_stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class TradingBot:
    """Trading bot with unified analysis"""

    def __init__(self):
        self.processed_pairs = 0
        self.session_start = time.time()

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
        logger.info("=" * 60)
        logger.info(f"STAGE 1: Signal filtering on {config.TIMEFRAME_LONG_NAME}")
        logger.info("=" * 60)

        pairs = await get_trading_pairs()
        if not pairs:
            logger.error("Failed to get trading pairs")
            return []

        candles_map = await self.load_candles_batch(pairs, config.TIMEFRAME_LONG, config.QUICK_SCAN_CANDLES)
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

        logger.info(f"Stage 1 completed in {elapsed:.1f}s")
        logger.info(f"Processed: {processed}, Signals: {len(pairs_with_signals)}")

        return pairs_with_signals

    async def stage2_ai_select(self, signal_pairs: List[Dict]) -> List[str]:
        """Stage 2: AI selection"""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"STAGE 2: {config.STAGE2_PROVIDER.upper()} selection from {len(signal_pairs)} pairs")
        logger.info("=" * 60)

        if not signal_pairs:
            return []

        symbols = [p['symbol'] for p in signal_pairs]
        candles_map = await self.load_candles_batch(symbols, config.TIMEFRAME_LONG, config.AI_BULK_CANDLES)

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
            logger.error("No data for AI selection")
            return []

        logger.info(f"Sending {len(ai_input_data)} pairs to {config.STAGE2_PROVIDER.upper()}")
        selected_pairs = await ai_router.select_pairs(ai_input_data)

        elapsed = time.time() - start_time
        logger.info(f"Stage 2 completed in {elapsed:.1f}s")
        logger.info(f"Selected: {len(selected_pairs)} pairs")

        return selected_pairs

    async def stage3_unified_analysis(self, selected_pairs: List[str]) -> List[Dict]:
        """Stage 3: Unified analysis"""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"STAGE 3: {config.STAGE3_PROVIDER.upper()} unified analysis for {len(selected_pairs)} pairs")
        logger.info("=" * 60)

        if not selected_pairs:
            return []

        logger.info("Loading BTC data...")
        btc_candles_1h = await fetch_klines('BTCUSDT', config.TIMEFRAME_SHORT, config.FINAL_SHORT_CANDLES)
        btc_candles_4h = await fetch_klines('BTCUSDT', config.TIMEFRAME_LONG, config.FINAL_LONG_CANDLES)

        final_signals = []

        for symbol in selected_pairs:
            try:
                logger.debug(f"Analyzing {symbol}...")

                klines_1h = await fetch_klines(symbol, config.TIMEFRAME_SHORT, config.FINAL_SHORT_CANDLES)
                klines_4h = await fetch_klines(symbol, config.TIMEFRAME_LONG, config.FINAL_LONG_CANDLES)

                if not klines_1h or not klines_4h:
                    logger.warning(f"{symbol}: Failed to load candles")
                    continue

                if not validate_candles(klines_1h, 20) or not validate_candles(klines_4h, 20):
                    logger.warning(f"{symbol}: Invalid candles")
                    continue

                indicators_1h = calculate_ai_indicators(klines_1h, config.FINAL_INDICATORS_HISTORY)
                indicators_4h = calculate_ai_indicators(klines_4h, config.FINAL_INDICATORS_HISTORY)

                if not indicators_1h or not indicators_4h:
                    logger.warning(f"{symbol}: Indicators failed")
                    continue

                current_price = float(klines_1h[-1][4])

                logger.debug(f"{symbol}: Collecting market data...")

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

                logger.info(f"{symbol}: Running unified analysis...")

                analysis = await ai_router.analyze_pair_comprehensive(symbol, comprehensive_data)

                signal_type = analysis.get('signal', 'NO_SIGNAL')
                confidence = analysis.get('confidence', 0)
                rejection = analysis.get('rejection_reason')

                if signal_type != 'NO_SIGNAL' and confidence >= config.MIN_CONFIDENCE:
                    analysis['comprehensive_data'] = comprehensive_data
                    analysis['timestamp'] = datetime.now().isoformat()

                    final_signals.append(analysis)

                    tp_levels = analysis.get('take_profit_levels', [0, 0, 0])
                    logger.info(f"[SIGNAL] {symbol} {signal_type} {confidence}%")
                    logger.info(f"  Entry: ${analysis['entry_price']}, Stop: ${analysis['stop_loss']}")
                    logger.info(f"  TP: ${tp_levels[0]} / ${tp_levels[1]} / ${tp_levels[2]}")
                else:
                    logger.info(f"[SKIPPED] {symbol} - {rejection if rejection else 'weak setup'}")

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue

        elapsed = time.time() - start_time
        logger.info(f"Stage 3 completed in {elapsed:.1f}s")
        logger.info(f"Signals generated: {len(final_signals)}")

        return final_signals

    async def stage4_validation(self, preliminary_signals: List[Dict]) -> Dict[str, Any]:
        """Stage 4: Signal validation"""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"STAGE 4: {config.STAGE4_PROVIDER.upper()} validation of {len(preliminary_signals)} signals")
        logger.info("=" * 60)

        if not preliminary_signals:
            return {'validated': [], 'rejected': []}

        validation_result = await validate_signals_simple(ai_router, preliminary_signals)

        validated = validation_result['validated']
        rejected = validation_result['rejected']

        elapsed = time.time() - start_time
        logger.info(f"Stage 4 completed in {elapsed:.1f}s")
        logger.info(f"Approved: {len(validated)}, Rejected: {len(rejected)}")

        return validation_result

    async def run_full_cycle(self) -> Dict[str, Any]:
        """Run full trading cycle"""
        cycle_start = time.time()

        logger.info("=" * 80)
        logger.info("STARTING FULL CYCLE")
        logger.info("=" * 80)

        try:
            signal_pairs = await self.stage1_filter_signals()
            if not signal_pairs:
                return {
                    'result': 'NO_SIGNAL_PAIRS',
                    'total_time': time.time() - cycle_start,
                    'pairs_scanned': self.processed_pairs
                }

            selected_pairs = await self.stage2_ai_select(signal_pairs)
            if not selected_pairs:
                return {
                    'result': 'NO_AI_SELECTION',
                    'total_time': time.time() - cycle_start,
                    'signal_pairs': len(signal_pairs),
                    'pairs_scanned': self.processed_pairs
                }

            preliminary_signals = await self.stage3_unified_analysis(selected_pairs)
            if not preliminary_signals:
                return {
                    'result': 'NO_ANALYSIS_SIGNALS',
                    'total_time': time.time() - cycle_start,
                    'pairs_scanned': self.processed_pairs,
                    'signal_pairs': len(signal_pairs),
                    'ai_selected': len(selected_pairs)
                }

            validation_result = await self.stage4_validation(preliminary_signals)
            validated = validation_result['validated']
            rejected = validation_result['rejected']

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
                    'signal_pairs_found': len(signal_pairs),
                    'ai_selected': len(selected_pairs),
                    'analyzed': len(preliminary_signals),
                    'validated_signals': len(validated),
                    'rejected_signals': len(rejected),
                    'processing_speed': round(self.processed_pairs / total_time, 1)
                },
                'validated_signals': validated,
                'rejected_signals': rejected,
                'validation_stats': validation_stats
            }

            filename = f'bot_result_{timestamp}.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)

            logger.info("=" * 80)
            logger.info(f"CYCLE COMPLETE: {len(validated)} validated, {len(rejected)} rejected")
            logger.info(f"Result saved: {filename}")
            logger.info("=" * 80)

            return final_result

        except Exception as e:
            logger.error(f"Critical cycle error: {e}")
            return {
                'result': 'ERROR',
                'error': str(e),
                'total_time': time.time() - cycle_start
            }

    async def cleanup(self):
        """Cleanup resources"""
        await cleanup_api()


async def main():
    """Main function"""
    print("=" * 80)
    print("TRADING BOT v5.0")
    print(f"Stage 1: Base indicators")
    print(f"Stage 2: {config.STAGE2_PROVIDER.upper()} selection")
    print(f"Stage 3: {config.STAGE3_PROVIDER.upper()} unified analysis")
    print(f"Stage 4: {config.STAGE4_PROVIDER.upper()} validation")
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
        print(f"AI selected: {stats.get('ai_selected', 0)}")
        print(f"Analyzed: {stats.get('analyzed', 0)}")
        print(f"Validated: {stats.get('validated_signals', 0)}")
        print(f"Speed: {stats.get('processing_speed', 0):.1f} pairs/sec")
        print("=" * 80)

        if result.get('validated_signals'):
            signals = result['validated_signals']
            print(f"\nVALIDATED SIGNALS ({len(signals)}):")
            print("=" * 80)

            for sig in signals:
                tp_levels = sig.get('take_profit_levels', [0, 0, 0])
                print(f"\n{sig['symbol']}: {sig['signal']} (Confidence: {sig['confidence']}%)")
                print(f"  Entry: ${sig['entry_price']}")
                print(f"  Stop:  ${sig['stop_loss']}")
                print(f"  TP1:   ${tp_levels[0]} (conservative)")
                print(f"  TP2:   ${tp_levels[1]} (target)")
                print(f"  TP3:   ${tp_levels[2]} (extended)")
                print(f"  R/R:   1:{sig.get('risk_reward_ratio', 0)}")
                print(f"  Hold:  {sig.get('hold_duration_minutes', 720) // 60}h")

                val_notes = sig.get('validation_notes', 'N/A')
                if len(val_notes) > 100:
                    val_notes = val_notes[:100] + "..."
                print(f"  Validation: {val_notes}")

        if result.get('rejected_signals'):
            rejected = result['rejected_signals']
            print(f"\nREJECTED SIGNALS ({len(rejected)}):")
            print("=" * 80)

            for rej in rejected:
                tp_levels = rej.get('take_profit_levels', [0, 0, 0])
                print(f"\n{rej['symbol']}: {rej.get('signal', 'UNKNOWN')}")
                print(f"  Entry: ${rej.get('entry_price', 0)}")
                print(f"  Stop:  ${rej.get('stop_loss', 0)}")
                print(f"  TP: ${tp_levels[0]} / ${tp_levels[1]} / ${tp_levels[2]}")
                print(f"  Rejection: {rej.get('rejection_reason', 'Unknown')}")

        if result.get('validation_stats'):
            vstats = result['validation_stats']
            print(f"\nVALIDATION STATS:")
            print(f"  Approval rate: {vstats.get('approval_rate', 0)}%")
            print(f"  Avg R/R: 1:{vstats.get('avg_risk_reward', 0)}")
            if vstats.get('top_rejection_reasons'):
                print(f"  Top rejections:")
                for reason in vstats['top_rejection_reasons']:
                    print(f"    - {reason}")

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        await bot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())