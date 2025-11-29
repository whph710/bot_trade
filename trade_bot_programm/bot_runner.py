"""
Trading Bot Runner - REMOVED 1D CANDLES
Файл: trade_bot_programm/bot_runner.py
ИЗМЕНЕНИЯ:
- Полностью удалена загрузка 1D свечей
- Удалены candles_1d из всех структур данных
- Удалены indicators_1d
- Удалён флаг has_1d_data
- ИСПРАВЛЕНО: Удалён мусор на строке 382
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import time

CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))

from config import config
from func_async import get_trading_pairs, fetch_klines, batch_fetch_klines, cleanup as cleanup_api, get_optimized_session
from func_trade import calculate_basic_indicators, calculate_ai_indicators, check_basic_signal, validate_candles
from ai_router import AIRouter
from simple_validator import validate_signals_simple, calculate_validation_stats
from data_storage import storage
from logging_config import setup_module_logger

logger = setup_module_logger(__name__)

ai_router = AIRouter()


class TradingBotRunner:
    """Класс для запуска торгового бота"""

    def __init__(self, progress_callback: Optional[Callable] = None):
        self.processed_pairs = 0
        self.signal_pairs_count = 0
        self.ai_selected_count = 0
        self.analyzed_count = 0
        self.analysis_data_cache = {}
        self.ai_router = ai_router
        self.last_haiku_call_time = 0
        self.progress_callback = progress_callback
        self.cycle_start_time = 0
        self.stage_times = {}

    async def _send_progress(self, stage: str, message: str):
        """Отправить прогресс через callback если доступен"""
        if self.progress_callback:
            try:
                await self.progress_callback(stage, message)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    async def load_candles_batch(self, pairs: list[str], interval: str, limit: int) -> Dict[str, list]:
        """Batch load candles"""
        logger.debug(f"Loading candles: {len(pairs)} pairs, interval={interval}, limit={limit}")

        requests = [{'symbol': pair, 'interval': interval, 'limit': limit} for pair in pairs]
        results = await batch_fetch_klines(requests)

        candles_map = {}
        for result in results:
            if result.get('success') and result.get('klines'):
                symbol = result['symbol']
                klines = result['klines']
                if validate_candles(klines, 20):
                    candles_map[symbol] = klines

        logger.debug(f"Loaded candles: {len(candles_map)}/{len(pairs)} pairs")
        return candles_map

    async def stage1_filter_signals(self) -> list[Dict]:
        """Stage 1: Base signal filtering"""
        stage_start = time.time()

        logger.info("=" * 70)
        logger.info("STAGE 1: SIGNAL FILTERING")
        logger.info("=" * 70)

        pairs = await get_trading_pairs()
        if not pairs:
            logger.error("Failed to get trading pairs")
            return []

        logger.info(f"Found {len(pairs)} trading pairs")

        candles_map = await self.load_candles_batch(pairs, config.TIMEFRAME_LONG, config.QUICK_SCAN_CANDLES)
        logger.info(f"Loaded candles for {len(candles_map)} pairs")

        if not candles_map:
            logger.warning("No valid candles loaded")
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
                    logger.debug(f"Signal: {symbol} {signal_check['direction']} ({signal_check['confidence']}%)")

            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue

        pairs_with_signals.sort(key=lambda x: x['confidence'], reverse=True)
        self.signal_pairs_count = len(pairs_with_signals)

        self.stage_times['stage1'] = time.time() - stage_start

        logger.info(f"Stage 1 complete: {self.processed_pairs} scanned, {self.signal_pairs_count} signals found ({self.stage_times['stage1']:.1f}s)")
        return pairs_with_signals

    async def stage2_ai_select(self, signal_pairs: list[Dict]) -> list[str]:
        """Stage 2: AI pair selection (COMPACT multi-TF data, БЕЗ 1D)"""
        stage_start = time.time()

        logger.info("=" * 70)
        logger.info(f"STAGE 2: {config.STAGE2_PROVIDER.upper()} PAIR SELECTION (1H/4H ONLY)")
        logger.info("=" * 70)

        if not signal_pairs:
            logger.warning("No signal pairs to select from")
            return []

        symbols = [p['symbol'] for p in signal_pairs]

        logger.debug(f"Loading compact data: 1H({config.STAGE2_CANDLES_1H}), 4H({config.STAGE2_CANDLES_4H})")

        candles_1h_map = await self.load_candles_batch(symbols, config.TIMEFRAME_SHORT, config.STAGE2_CANDLES_1H)
        candles_4h_map = await self.load_candles_batch(symbols, config.TIMEFRAME_LONG, config.STAGE2_CANDLES_4H)

        ai_input_data = []

        for pair_data in signal_pairs:
            symbol = pair_data['symbol']

            if symbol not in candles_1h_map or symbol not in candles_4h_map:
                continue

            candles_1h = candles_1h_map[symbol]
            candles_4h = candles_4h_map[symbol]

            indicators_1h = calculate_ai_indicators(candles_1h, min(30, len(candles_1h)))
            indicators_4h = calculate_ai_indicators(candles_4h, min(30, len(candles_4h)))

            if not indicators_1h or not indicators_4h:
                continue

            ai_input_data.append({
                'symbol': symbol,
                'confidence': pair_data['confidence'],
                'direction': pair_data['direction'],
                'candles_1h': candles_1h[-30:],
                'candles_4h': candles_4h[-30:],
                'indicators_1h': {
                    'current': indicators_1h.get('current', {}),
                    'ema5': indicators_1h.get('ema5_history', [])[-30:],
                    'ema8': indicators_1h.get('ema8_history', [])[-30:],
                    'ema20': indicators_1h.get('ema20_history', [])[-30:],
                    'rsi': indicators_1h.get('rsi_history', [])[-30:],
                    'macd': indicators_1h.get('macd_histogram_history', [])[-30:],
                },
                'indicators_4h': {
                    'current': indicators_4h.get('current', {}),
                    'ema5': indicators_4h.get('ema5_history', [])[-30:],
                    'ema8': indicators_4h.get('ema8_history', [])[-30:],
                    'ema20': indicators_4h.get('ema20_history', [])[-30:],
                    'rsi': indicators_4h.get('rsi_history', [])[-30:],
                    'macd': indicators_4h.get('macd_histogram_history', [])[-30:],
                }
            })

        if not ai_input_data:
            logger.warning("No valid AI input data prepared")
            return []

        logger.info(f"Sending {len(ai_input_data)} pairs to AI (limit: {config.MAX_FINAL_PAIRS})")

        self.last_haiku_call_time = time.time()

        selected_pairs = await self.ai_router.select_pairs(
            ai_input_data,
            max_pairs=config.MAX_FINAL_PAIRS
        )
        self.ai_selected_count = len(selected_pairs)

        self.stage_times['stage2'] = time.time() - stage_start

        if selected_pairs:
            logger.info(f"Stage 2 complete: {self.ai_selected_count} pairs selected - {selected_pairs} ({self.stage_times['stage2']:.1f}s)")

            stage2_message = f"<b>Выбрано пар:</b> {self.ai_selected_count}\n\n"
            stage2_message += "\n".join([f"• {pair}" for pair in selected_pairs])
            await self._send_progress("Stage 2 Complete", stage2_message)
        else:
            logger.warning("Stage 2: No pairs selected by AI")

        return selected_pairs

    async def stage3_unified_analysis(self, selected_pairs: list[str]) -> list[Dict]:
        """Stage 3: Unified analysis (БЕЗ 1D)"""
        stage_start = time.time()

        logger.info("=" * 70)
        logger.info(f"STAGE 3: {config.STAGE3_PROVIDER.upper()} UNIFIED ANALYSIS (1H/4H ONLY)")
        logger.info("=" * 70)

        if not selected_pairs:
            logger.warning("No pairs for analysis")
            return []

        if config.CLAUDE_RATE_LIMIT_DELAY > 0 and self.last_haiku_call_time > 0:
            elapsed = time.time() - self.last_haiku_call_time
            required_delay = config.CLAUDE_RATE_LIMIT_DELAY

            if elapsed < required_delay:
                wait_time = required_delay - elapsed
                logger.info(f"Rate limit protection: waiting {wait_time:.1f}s before Stage 3")
                await asyncio.sleep(wait_time)

        # Load BTC candles (БЕЗ 1D)
        logger.debug(f"Loading BTC candles: 1H({config.STAGE3_CANDLES_1H}), 4H({config.STAGE3_CANDLES_4H})")

        btc_candles_1h, btc_candles_4h = await asyncio.gather(
            fetch_klines('BTCUSDT', config.TIMEFRAME_SHORT, config.STAGE3_CANDLES_1H),
            fetch_klines('BTCUSDT', config.TIMEFRAME_LONG, config.STAGE3_CANDLES_4H)
        )

        if not btc_candles_1h or not btc_candles_4h:
            logger.error("Failed to load BTC 1H/4H candles (critical)")
            return []

        logger.debug(f"BTC candles loaded: {len(btc_candles_1h)} (1H), {len(btc_candles_4h)} (4H)")

        final_signals = []
        rejected_signals = []

        for symbol in selected_pairs:
            try:
                logger.info(f"Analyzing {symbol}...")

                await self._send_progress("Stage 3 Analysis", f"🔍 Анализирую <b>{symbol}</b>...")

                # Load timeframes (БЕЗ 1D)
                klines_1h, klines_4h = await asyncio.gather(
                    fetch_klines(symbol, config.TIMEFRAME_SHORT, config.STAGE3_CANDLES_1H),
                    fetch_klines(symbol, config.TIMEFRAME_LONG, config.STAGE3_CANDLES_4H)
                )

                if not klines_1h or not klines_4h:
                    logger.warning(f"{symbol}: Missing 1H/4H data - SKIP")

                    rejected_signals.append({
                        'symbol': symbol,
                        'signal': 'NO_SIGNAL',
                        'rejection_reason': 'Missing 1H/4H data'
                    })

                    await self._send_progress(
                        "Stage 3 Analysis",
                        f"❌ <b>{symbol}</b>\n<i>Missing 1H/4H data</i>"
                    )
                    continue

                if not validate_candles(klines_1h, 20) or not validate_candles(klines_4h, 20):
                    logger.warning(f"{symbol}: 1H/4H candle validation failed - SKIP")

                    rejected_signals.append({
                        'symbol': symbol,
                        'signal': 'NO_SIGNAL',
                        'rejection_reason': 'Candle validation failed'
                    })

                    await self._send_progress(
                        "Stage 3 Analysis",
                        f"❌ <b>{symbol}</b>\n<i>Candle validation failed</i>"
                    )
                    continue

                indicators_1h = calculate_ai_indicators(klines_1h, config.FINAL_INDICATORS_HISTORY)
                indicators_4h = calculate_ai_indicators(klines_4h, config.FINAL_INDICATORS_HISTORY)

                if not indicators_1h or not indicators_4h:
                    logger.warning(f"{symbol}: 1H/4H indicators calculation failed - SKIP")

                    rejected_signals.append({
                        'symbol': symbol,
                        'signal': 'NO_SIGNAL',
                        'rejection_reason': 'Indicators calculation failed'
                    })

                    await self._send_progress(
                        "Stage 3 Analysis",
                        f"❌ <b>{symbol}</b>\n<i>Indicators calculation failed</i>"
                    )
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

                analysis = await self.ai_router.analyze_pair_comprehensive(symbol, comprehensive_data)

                signal_type = analysis.get('signal', 'NO_SIGNAL')
                confidence = analysis.get('confidence', 0)

                if signal_type != 'NO_SIGNAL' and confidence >= config.MIN_CONFIDENCE:
                    analysis['comprehensive_data'] = comprehensive_data
                    analysis['timestamp'] = datetime.now().isoformat()
                    final_signals.append(analysis)

                    self.analysis_data_cache[symbol] = comprehensive_data

                    tp_levels = analysis.get('take_profit_levels', [0, 0, 0])
                    logger.info(f"✓ SIGNAL: {symbol} {signal_type} (confidence: {confidence}%)")
                    logger.debug(f"  Entry: ${analysis['entry_price']:.2f} | Stop: ${analysis['stop_loss']:.2f}")
                    logger.debug(f"  TP: ${tp_levels[0]:.2f} / ${tp_levels[1]:.2f} / ${tp_levels[2]:.2f}")

                    signal_message = (
                        f"✅ <b>{symbol}</b> {signal_type}\n\n"
                        f"<b>Confidence:</b> {confidence}%\n"
                        f"<b>Entry:</b> ${analysis['entry_price']:.4f}\n"
                        f"<b>Stop:</b> ${analysis['stop_loss']:.4f}\n"
                        f"<b>TP1/2/3:</b> ${tp_levels[0]:.4f} / ${tp_levels[1]:.4f} / ${tp_levels[2]:.4f}"
                    )
                    await self._send_progress("Stage 3 Analysis", signal_message)

                else:
                    rejection_reason = analysis.get('rejection_reason', 'Low confidence')
                    logger.info(f"✗ NO_SIGNAL: {symbol} - {rejection_reason}")

                    rejected_signals.append({
                        'symbol': symbol,
                        'signal': 'NO_SIGNAL',
                        'rejection_reason': rejection_reason
                    })

                    rejection_message = (
                        f"❌ <b>{symbol}</b> NO_SIGNAL\n\n"
                        f"<i>{rejection_reason}</i>"
                    )
                    await self._send_progress("Stage 3 Analysis", rejection_message)

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}", exc_info=False)

                rejected_signals.append({
                    'symbol': symbol,
                    'signal': 'NO_SIGNAL',
                    'rejection_reason': f'Error: {str(e)[:100]}'
                })

                await self._send_progress(
                    "Stage 3 Analysis",
                    f"❌ <b>{symbol}</b>\n<i>Error: {str(e)[:100]}</i>"
                )
                continue

        self.analyzed_count = len(final_signals)

        self.stage_times['stage3'] = time.time() - stage_start

        logger.info(f"Stage 3 complete: {len(final_signals)} approved, {len(rejected_signals)} rejected ({self.stage_times['stage3']:.1f}s)")

        return final_signals, rejected_signals

    def _enrich_signal_with_analysis_data(self, signal: Dict) -> Dict:
        """Добавить данные анализа к сигналу"""
        symbol = signal.get('symbol')
        if symbol not in self.analysis_data_cache:
            return signal

        comp_data = self.analysis_data_cache[symbol]

        signal['analysis_data'] = {
            'candles_1h': comp_data.get('candles_1h', []),
            'candles_4h': comp_data.get('candles_4h', []),
            'indicators_1h': comp_data.get('indicators_1h', {}),
            'indicators_4h': comp_data.get('indicators_4h', {}),
            'current_price': comp_data.get('current_price', 0),
            'market_data': comp_data.get('market_data', {}),
            'correlation_data': comp_data.get('correlation_data', {}),
            'volume_profile': comp_data.get('volume_profile', {}),
            'vp_analysis': comp_data.get('vp_analysis', {})
        }

        return signal

    async def run_cycle(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Запуск полного цикла бота"""
        if progress_callback:
            self.progress_callback = progress_callback

        self.cycle_start_time = time.time()
        cycle_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        logger.info("╔" + "=" * 68 + "╗")
        logger.info("║" + " TRADING BOT CYCLE STARTED (1H/4H ONLY)".center(68) + "║")
        logger.info("╚" + "=" * 68 + "╝")

        try:
            signal_pairs = await self.stage1_filter_signals()

            if not signal_pairs:
                logger.warning("Pipeline stopped: No signal pairs found")
                total_time = time.time() - self.cycle_start_time
                return self._build_result('NO_SIGNAL_PAIRS', total_time, [], [])

            selected_pairs = await self.stage2_ai_select(signal_pairs)

            if not selected_pairs:
                logger.warning("Pipeline stopped: AI selected 0 pairs")
                total_time = time.time() - self.cycle_start_time
                return self._build_result('NO_AI_SELECTION', total_time, [], [])

            preliminary_signals, rejected_signals = await self.stage3_unified_analysis(selected_pairs)

            if not preliminary_signals:
                logger.warning("Pipeline stopped: No analysis signals generated")
                total_time = time.time() - self.cycle_start_time
                return self._build_result('NO_ANALYSIS_SIGNALS', total_time, [], rejected_signals)

            logger.info("=" * 70)
            logger.info("STAGE 4: VALIDATION (Fallback)")
            logger.info("=" * 70)

            validation_result = await validate_signals_simple(self.ai_router, preliminary_signals)

            validated = validation_result['validated']
            rejected = rejected_signals + validation_result['rejected']

            total_time = time.time() - self.cycle_start_time

            if validated:
                enriched_validated = [self._enrich_signal_with_analysis_data(sig) for sig in validated]

                for sig in enriched_validated:
                    storage.save_signal(sig, compress=True)
            else:
                enriched_validated = []

            result = self._build_result(
                'SUCCESS' if validated else 'NO_VALIDATED_SIGNALS',
                total_time,
                enriched_validated,
                rejected
            )

            storage.save_daily_statistics(result['stats'])

            if datetime.now().hour == 0:
                storage.cleanup_old_data(days_to_keep=90)

            logger.info("╔" + "=" * 68 + "╗")
            logger.info(f"║ CYCLE COMPLETE: {result['result']}".ljust(69) + "║")
            logger.info(f"║ Time: {total_time:.1f}s | Signals: {len(validated)} approved, {len(rejected)} rejected".ljust(69) + "║")
            logger.info("╚" + "=" * 68 + "╝")

            return result

        except Exception as e:
            logger.error(f"CRITICAL CYCLE ERROR: {e}", exc_info=True)
            total_time = time.time() - self.cycle_start_time
            return self._build_result('ERROR', total_time, [], [], error=str(e))

        finally:
            await cleanup_api()

    def _build_stats(self, total_time: float) -> Dict:
        """Построить статистику"""
        return {
            'pairs_scanned': self.processed_pairs,
            'signal_pairs_found': self.signal_pairs_count,
            'ai_selected': self.ai_selected_count,
            'analyzed': self.analyzed_count,
            'processing_speed': round(self.processed_pairs / total_time, 1) if total_time > 0 else 0,
            'total_time': round(total_time, 1),
            'timeframes': f"{config.TIMEFRAME_SHORT_NAME}/{config.TIMEFRAME_LONG_NAME}",
            'stage_times': {
                'stage1': round(self.stage_times.get('stage1', 0), 1),
                'stage2': round(self.stage_times.get('stage2', 0), 1),
                'stage3': round(self.stage_times.get('stage3', 0), 1)
            }
        }

    def _build_result(self, result_type: str, total_time: float, validated: list, rejected: list,
                      error: str = None) -> Dict:
        """Построить результат"""
        stats = self._build_stats(total_time)
        stats['validated_signals'] = len(validated)
        stats['rejected_signals'] = len(rejected)

        result = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'result': result_type,
            'stats': stats
        }

        if validated:
            result['validated_signals'] = validated

        if rejected:
            result['rejected_signals'] = rejected

        if error:
            result['error'] = error

        return result


async def run_trading_bot(progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Главная функция запуска бота"""
    bot = TradingBotRunner(progress_callback=progress_callback)
    result = await bot.run_cycle(progress_callback=progress_callback)
    return result