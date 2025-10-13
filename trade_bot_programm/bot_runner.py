"""
Trading Bot Runner - обёртка для вызова бота как функции
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any
import logging

# Добавляем текущую директорию в PATH для импортов
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))

# Импорты из основного бота
from config import config
from func_async import get_trading_pairs, fetch_klines, batch_fetch_klines, cleanup as cleanup_api, \
    get_optimized_session
from func_trade import calculate_basic_indicators, calculate_ai_indicators, check_basic_signal, validate_candles
from ai_router import ai_router
from simple_validator import validate_signals_simple, calculate_validation_stats
from datetime import datetime
import pytz

# Настройка логирования для функции
logger = logging.getLogger(__name__)


class TradingBotRunner:
    """Класс для запуска торгового бота"""

    def __init__(self):
        self.processed_pairs = 0
        self.signal_pairs_count = 0
        self.ai_selected_count = 0
        self.analyzed_count = 0

    async def load_candles_batch(self, pairs: list[str], interval: str, limit: int) -> Dict[str, list]:
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

    async def stage1_filter_signals(self) -> list[Dict]:
        """Stage 1: Base signal filtering"""
        logger.info("STAGE 1: Signal filtering")

        pairs = await get_trading_pairs()
        if not pairs:
            logger.error("Failed to get trading pairs")
            return []

        logger.info(f"Found {len(pairs)} trading pairs")
        candles_map = await self.load_candles_batch(pairs, config.TIMEFRAME_LONG, config.QUICK_SCAN_CANDLES)
        logger.info(f"Loaded candles for {len(candles_map)} pairs")

        if not candles_map:
            return []

        pairs_with_signals = []

        for symbol, candles in candles_map.items():
            try:
                self.processed_pairs += 1
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

            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue

        pairs_with_signals.sort(key=lambda x: x['confidence'], reverse=True)
        self.signal_pairs_count = len(pairs_with_signals)

        logger.info(f"Stage 1: Processed {self.processed_pairs}, Signals {len(pairs_with_signals)}")
        return pairs_with_signals

    async def stage2_ai_select(self, signal_pairs: list[Dict]) -> list[str]:
        """Stage 2: AI selection"""
        logger.info(f"STAGE 2: {config.STAGE2_PROVIDER.upper()} selection")

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
            return []

        selected_pairs = await ai_router.select_pairs(ai_input_data)
        self.ai_selected_count = len(selected_pairs)

        logger.info(f"Stage 2: Selected {len(selected_pairs)} pairs")
        return selected_pairs

    async def stage3_unified_analysis(self, selected_pairs: list[str]) -> list[Dict]:
        """Stage 3: Unified analysis"""
        logger.info(f"STAGE 3: {config.STAGE3_PROVIDER.upper()} analysis")

        if not selected_pairs:
            return []

        # Load BTC data
        btc_candles_1h, btc_candles_4h = await asyncio.gather(
            fetch_klines('BTCUSDT', config.TIMEFRAME_SHORT, config.FINAL_SHORT_CANDLES),
            fetch_klines('BTCUSDT', config.TIMEFRAME_LONG, config.FINAL_LONG_CANDLES)
        )

        if not btc_candles_1h or not btc_candles_4h:
            logger.error("Failed to load BTC candles")
            return []

        final_signals = []

        for symbol in selected_pairs:
            try:
                logger.info(f"Analyzing {symbol}...")

                klines_1h, klines_4h = await asyncio.gather(
                    fetch_klines(symbol, config.TIMEFRAME_SHORT, config.FINAL_SHORT_CANDLES),
                    fetch_klines(symbol, config.TIMEFRAME_LONG, config.FINAL_LONG_CANDLES)
                )

                if not klines_1h or not klines_4h:
                    continue

                if not validate_candles(klines_1h, 20) or not validate_candles(klines_4h, 20):
                    continue

                indicators_1h = calculate_ai_indicators(klines_1h, config.FINAL_INDICATORS_HISTORY)
                indicators_4h = calculate_ai_indicators(klines_4h, config.FINAL_INDICATORS_HISTORY)

                if not indicators_1h or not indicators_4h:
                    continue

                current_price = float(klines_1h[-1][4])

                from func_market_data import MarketDataCollector
                from func_correlation import get_comprehensive_correlation_analysis
                from func_volume_profile import calculate_volume_profile_for_candles, analyze_volume_profile

                collector = MarketDataCollector(await get_optimized_session())
                market_snapshot = await collector.get_market_snapshot(symbol, current_price)

                corr_analysis = await get_comprehensive_correlation_analysis(
                    symbol, klines_1h, btc_candles_1h, 'UNKNOWN', None
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

                analysis = await ai_router.analyze_pair_comprehensive(symbol, comprehensive_data)

                signal_type = analysis.get('signal', 'NO_SIGNAL')
                confidence = analysis.get('confidence', 0)

                if signal_type != 'NO_SIGNAL' and confidence >= config.MIN_CONFIDENCE:
                    analysis['comprehensive_data'] = comprehensive_data
                    analysis['timestamp'] = datetime.now().isoformat()
                    final_signals.append(analysis)

                    tp_levels = analysis.get('take_profit_levels', [0, 0, 0])
                    logger.info(f"[SIGNAL] {symbol} {signal_type} {confidence}%")
                    logger.info(f"  Entry: ${analysis['entry_price']:.2f}, Stop: ${analysis['stop_loss']:.2f}")
                    logger.info(f"  TP: ${tp_levels[0]:.2f} / ${tp_levels[1]:.2f} / ${tp_levels[2]:.2f}")

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue

        self.analyzed_count = len(final_signals)
        logger.info(f"Stage 3: Generated {len(final_signals)} signals")
        return final_signals

    async def stage4_validation(self, preliminary_signals: list[Dict]) -> Dict[str, Any]:
        """Stage 4: Signal validation"""
        logger.info(f"STAGE 4: {config.STAGE4_PROVIDER.upper()} validation")

        if not preliminary_signals:
            return {'validated': [], 'rejected': []}

        validation_result = await validate_signals_simple(ai_router, preliminary_signals)

        if validation_result.get('validation_skipped_reason'):
            logger.warning(validation_result['validation_skipped_reason'])
            return validation_result

        validated = validation_result['validated']
        rejected = validation_result['rejected']

        logger.info(f"Stage 4: Approved {len(validated)}, Rejected {len(rejected)}")
        return validation_result

    async def run_cycle(self) -> Dict[str, Any]:
        """Запуск полного цикла бота"""
        import time
        cycle_start = time.time()

        logger.info("=" * 60)
        logger.info("STARTING TRADING BOT CYCLE")
        logger.info("=" * 60)

        try:
            # Stage 1
            signal_pairs = await self.stage1_filter_signals()
            if not signal_pairs:
                return {
                    'result': 'NO_SIGNAL_PAIRS',
                    'total_time': time.time() - cycle_start,
                    'stats': {
                        'pairs_scanned': self.processed_pairs,
                        'signal_pairs_found': 0,
                        'ai_selected': 0,
                        'analyzed': 0,
                        'validated_signals': 0,
                        'rejected_signals': 0
                    }
                }

            # Stage 2
            selected_pairs = await self.stage2_ai_select(signal_pairs)
            if not selected_pairs:
                return {
                    'result': 'NO_AI_SELECTION',
                    'total_time': time.time() - cycle_start,
                    'stats': {
                        'pairs_scanned': self.processed_pairs,
                        'signal_pairs_found': self.signal_pairs_count,
                        'ai_selected': 0,
                        'analyzed': 0,
                        'validated_signals': 0,
                        'rejected_signals': 0
                    }
                }

            # Stage 3
            preliminary_signals = await self.stage3_unified_analysis(selected_pairs)
            if not preliminary_signals:
                return {
                    'result': 'NO_ANALYSIS_SIGNALS',
                    'total_time': time.time() - cycle_start,
                    'stats': {
                        'pairs_scanned': self.processed_pairs,
                        'signal_pairs_found': self.signal_pairs_count,
                        'ai_selected': self.ai_selected_count,
                        'analyzed': 0,
                        'validated_signals': 0,
                        'rejected_signals': 0
                    }
                }

            # Stage 4
            validation_result = await self.stage4_validation(preliminary_signals)
            validated = validation_result['validated']
            rejected = validation_result['rejected']

            if validation_result.get('validation_skipped_reason'):
                return {
                    'result': 'VALIDATION_SKIPPED',
                    'reason': validation_result['validation_skipped_reason'],
                    'total_time': time.time() - cycle_start,
                    'stats': {
                        'pairs_scanned': self.processed_pairs,
                        'signal_pairs_found': self.signal_pairs_count,
                        'ai_selected': self.ai_selected_count,
                        'analyzed': self.analyzed_count,
                        'validated_signals': 0,
                        'rejected_signals': 0
                    }
                }

            # Final result
            total_time = time.time() - cycle_start
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            validation_stats = calculate_validation_stats(validated, rejected)

            result_type = 'SUCCESS' if validated else 'NO_VALIDATED_SIGNALS'

            return {
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

        except Exception as e:
            logger.error(f"Critical cycle error: {e}", exc_info=True)
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
                    'rejected_signals': 0
                }
            }

        finally:
            await cleanup_api()


async def run_trading_bot() -> Dict[str, Any]:
    """
    Главная функция для запуска торгового бота

    Returns:
        Dict с результатами торговли (формат bot_result_*.json)
    """
    bot = TradingBotRunner()
    result = await bot.run_cycle()
    return result


