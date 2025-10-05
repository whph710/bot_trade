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
    """Упрощенный торговый бот"""

    def __init__(self):
        self.processed_pairs = 0
        self.session_start = time.time()

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
        """Stage 1: Базовая фильтрация"""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"STAGE 1: Signal filtering on {config.TIMEFRAME_LONG_NAME}")
        logger.info("=" * 60)

        pairs = await get_trading_pairs()
        if not pairs:
            logger.error("Failed to get trading pairs")
            return []

        candles_map = await self.load_candles_batch(
            pairs,
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
        logger.info(f"  Processed: {processed}, Signals: {len(pairs_with_signals)}")

        return pairs_with_signals

    async def stage2_deepseek_select(self, signal_pairs: List[Dict]) -> List[str]:
        """Stage 2: DeepSeek отбор"""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"STAGE 2: DeepSeek selection from {len(signal_pairs)} pairs")
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
            logger.error("No data for DeepSeek")
            return []

        logger.info(f"Sending {len(ai_input_data)} pairs to DeepSeek")
        selected_pairs = await ai_router.select_pairs(ai_input_data)

        elapsed = time.time() - start_time
        logger.info(f"STAGE 2 completed in {elapsed:.1f}s")
        logger.info(f"  Selected: {len(selected_pairs)} pairs")

        return selected_pairs

    async def stage3_claude_full_analysis(self, selected_pairs: List[str]) -> List[Dict]:
        """
        Stage 3: Claude ПОЛНЫЙ анализ + сбор ВСЕХ данных
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"STAGE 3: Claude analysis + full data collection for {len(selected_pairs)} pairs")
        logger.info("=" * 60)

        if not selected_pairs:
            return []

        # Загружаем BTC для корреляций
        logger.info("Loading BTC data...")
        btc_candles_1h = await fetch_klines('BTCUSDT', config.TIMEFRAME_SHORT, config.FINAL_SHORT_CANDLES)
        btc_candles_4h = await fetch_klines('BTCUSDT', config.TIMEFRAME_LONG, config.FINAL_LONG_CANDLES)

        final_signals = []

        for symbol in selected_pairs:
            try:
                logger.debug(f"Analyzing {symbol}...")

                # 1. Свечи
                klines_1h = await fetch_klines(symbol, config.TIMEFRAME_SHORT, config.FINAL_SHORT_CANDLES)
                klines_4h = await fetch_klines(symbol, config.TIMEFRAME_LONG, config.FINAL_LONG_CANDLES)

                if not klines_1h or not klines_4h:
                    logger.warning(f"{symbol}: Failed to load candles")
                    continue

                if not validate_candles(klines_1h, 20) or not validate_candles(klines_4h, 20):
                    logger.warning(f"{symbol}: Invalid candles")
                    continue

                # 2. Индикаторы
                indicators_1h = calculate_ai_indicators(klines_1h, config.FINAL_INDICATORS_HISTORY)
                indicators_4h = calculate_ai_indicators(klines_4h, config.FINAL_INDICATORS_HISTORY)

                if not indicators_1h or not indicators_4h:
                    logger.warning(f"{symbol}: Indicators failed")
                    continue

                current_price = float(klines_1h[-1][4])

                # 3. СОБИРАЕМ ВСЕ РАСШИРЕННЫЕ ДАННЫЕ
                logger.debug(f"{symbol}: Collecting extended data...")

                from func_market_data import MarketDataCollector
                from func_correlation import get_comprehensive_correlation_analysis
                from func_volume_profile import calculate_volume_profile_for_candles, analyze_volume_profile
                from ai_advanced_analysis import get_ai_orderflow_analysis, get_ai_smc_patterns

                collector = MarketDataCollector(await get_optimized_session())

                # Market data
                market_snapshot = await collector.get_market_snapshot(symbol, current_price)

                # Correlations
                corr_analysis = await get_comprehensive_correlation_analysis(
                    symbol,
                    klines_1h,
                    btc_candles_1h,
                    'UNKNOWN',
                    None
                )

                # Volume Profile
                vp_data = calculate_volume_profile_for_candles(klines_4h, num_bins=50)
                vp_analysis = analyze_volume_profile(vp_data, current_price) if vp_data else None

                # OrderFlow AI
                orderflow_ai = None
                try:
                    if market_snapshot.get('orderbook'):
                        prices_recent = [float(c[4]) for c in klines_1h[-20:]]
                        orderflow_ai = await get_ai_orderflow_analysis(
                            ai_router,
                            symbol,
                            market_snapshot['orderbook'],
                            prices_recent
                        )
                except Exception as e:
                    logger.debug(f"{symbol}: OrderFlow AI error: {e}")
                    orderflow_ai = None

                # SMC AI
                smc_ai = None
                try:
                    smc_ai = await get_ai_smc_patterns(
                        ai_router,
                        symbol,
                        klines_1h,
                        current_price
                    )
                except Exception as e:
                    logger.debug(f"{symbol}: SMC AI error: {e}")
                    smc_ai = None

                # 4. ФОРМИРУЕМ ПОЛНЫЙ ПАКЕТ
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

                # 5. CLAUDE АНАЛИЗ
                logger.debug(f"{symbol}: Claude analysis...")

                analysis = await ai_router.analyze_pair_comprehensive(
                    symbol,
                    comprehensive_data
                )

                logger.debug(f"{symbol}: Analysis result - signal={analysis.get('signal')}, confidence={analysis.get('confidence')}")

                if analysis['signal'] != 'NO_SIGNAL' and analysis['confidence'] >= config.MIN_CONFIDENCE:
                    # Сохраняем данные для Stage 4
                    analysis['comprehensive_data'] = comprehensive_data
                    analysis['timestamp'] = datetime.now().isoformat()

                    final_signals.append(analysis)
                    logger.info(f"Signal: {symbol} {analysis['signal']} {analysis['confidence']}%")
                else:
                    logger.info(f"Skipped: {symbol} - signal={analysis['signal']}, confidence={analysis['confidence']}")

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        elapsed = time.time() - start_time
        logger.info(f"STAGE 3 completed in {elapsed:.1f}s")
        logger.info(f"  Signals: {len(final_signals)}")

        return final_signals

    async def stage4_claude_validation(self, preliminary_signals: List[Dict]) -> Dict[str, Any]:
        """
        Stage 4: Claude валидация (упрощенная)
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"STAGE 4: Claude validation of {len(preliminary_signals)} signals")
        logger.info("=" * 60)

        if not preliminary_signals:
            return {'validated': [], 'rejected': []}

        # Используем упрощенный валидатор
        validation_result = await validate_signals_simple(ai_router, preliminary_signals)

        validated = validation_result['validated']
        rejected = validation_result['rejected']

        elapsed = time.time() - start_time
        logger.info(f"STAGE 4 completed in {elapsed:.1f}s")
        logger.info(f"  Approved: {len(validated)}, Rejected: {len(rejected)}")

        return validation_result

    async def run_full_cycle(self) -> Dict[str, Any]:
        """Полный цикл работы"""
        cycle_start = time.time()

        logger.info("=" * 80)
        logger.info("STARTING FULL CYCLE")
        logger.info("=" * 80)

        try:
            # STAGE 1: Базовая фильтрация
            signal_pairs = await self.stage1_filter_signals()
            if not signal_pairs:
                return {
                    'result': 'NO_SIGNAL_PAIRS',
                    'total_time': time.time() - cycle_start,
                    'pairs_scanned': self.processed_pairs
                }

            # STAGE 2: DeepSeek отбор
            selected_pairs = await self.stage2_deepseek_select(signal_pairs)
            if not selected_pairs:
                return {
                    'result': 'NO_DEEPSEEK_SELECTION',
                    'total_time': time.time() - cycle_start,
                    'signal_pairs': len(signal_pairs),
                    'pairs_scanned': self.processed_pairs
                }

            # STAGE 3: Claude анализ + сбор данных
            preliminary_signals = await self.stage3_claude_full_analysis(selected_pairs)
            if not preliminary_signals:
                return {
                    'result': 'NO_CLAUDE_SIGNALS',
                    'total_time': time.time() - cycle_start,
                    'pairs_scanned': self.processed_pairs,
                    'signal_pairs': len(signal_pairs),
                    'deepseek_selected': len(selected_pairs)
                }

            # STAGE 4: Claude валидация
            validation_result = await self.stage4_claude_validation(preliminary_signals)
            validated = validation_result['validated']
            rejected = validation_result['rejected']

            # Финальный результат
            total_time = time.time() - cycle_start
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            result_type = 'SUCCESS' if validated else 'NO_VALIDATED_SIGNALS'

            # Статистика валидации
            validation_stats = calculate_validation_stats(validated, rejected)

            final_result = {
                'timestamp': timestamp,
                'result': result_type,
                'total_time': round(total_time, 1),
                'timeframes': f"{config.TIMEFRAME_SHORT_NAME}/{config.TIMEFRAME_LONG_NAME}",
                'stats': {
                    'pairs_scanned': self.processed_pairs,
                    'signal_pairs_found': len(signal_pairs),
                    'deepseek_selected': len(selected_pairs),
                    'claude_analyzed': len(preliminary_signals),
                    'validated_signals': len(validated),
                    'rejected_signals': len(rejected),
                    'processing_speed': round(self.processed_pairs / total_time, 1)
                },
                'validated_signals': validated,
                'rejected_signals': rejected,
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
    print("TRADING BOT v4.1 (SIMPLIFIED)")
    print("Stage 1: Base indicators")
    print("Stage 2: DeepSeek selection")
    print("Stage 3: Claude full analysis + data collection")
    print("Stage 4: Claude validation")
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
        print(f"DeepSeek selected: {stats.get('deepseek_selected', 0)}")
        print(f"Claude analyzed: {stats.get('claude_analyzed', 0)}")
        print(f"Validated: {stats.get('validated_signals', 0)}")
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
                print(f"  TP:    ${sig['take_profit']}")
                print(f"  R/R:   1:{sig.get('risk_reward_ratio', 0)}")
                print(f"  Hold:  {sig.get('hold_duration_minutes', 720)//60}h")

                val_notes = sig.get('validation_notes', 'N/A')
                if len(val_notes) > 100:
                    val_notes = val_notes[:100] + "..."
                print(f"  Validation: {val_notes}")

                analysis = sig.get('analysis', '')
                if len(analysis) > 120:
                    analysis = analysis[:120] + "..."
                print(f"  Analysis: {analysis}")

        if result.get('rejected_signals'):
            rejected = result['rejected_signals']
            print(f"\n❌ REJECTED SIGNALS ({len(rejected)}):")
            for rej in rejected[:5]:
                print(f"  {rej['symbol']}: {rej.get('rejection_reason', 'Unknown')}")

        # Статистика валидации
        if result.get('validation_stats'):
            vstats = result['validation_stats']
            print(f"\n📊 VALIDATION STATS:")
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