"""
Trading Bot Runner - FIXED: 1D данные для swing анализа
Файл: trade_bot_programm/bot_runner.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))

from config import config
from func_async import get_trading_pairs, fetch_klines, batch_fetch_klines, cleanup as cleanup_api, get_optimized_session
from func_trade import calculate_basic_indicators, calculate_ai_indicators, check_basic_signal, validate_candles
from ai_router import ai_router
from simple_validator import validate_signals_simple, calculate_validation_stats
from checkpoint_manager import CheckpointManager
from data_storage import storage
from logging_config import setup_module_logger

logger = setup_module_logger(__name__)


class TradingBotRunner:
    """Класс для запуска торгового бота"""

    def __init__(self):
        self.processed_pairs = 0
        self.signal_pairs_count = 0
        self.ai_selected_count = 0
        self.analyzed_count = 0
        self.analysis_data_cache = {}
        self.checkpoint_mgr = CheckpointManager()

    async def load_candles_batch(self, pairs: list[str], interval: str, limit: int) -> Dict[str, list]:
        """Batch load candles"""
        logger.debug(f"Loading candles for {len(pairs)} pairs (interval: {interval}, limit: {limit})")

        requests = [{'symbol': pair, 'interval': interval, 'limit': limit} for pair in pairs]
        results = await batch_fetch_klines(requests)

        candles_map = {}
        for result in results:
            if result.get('success') and result.get('klines'):
                symbol = result['symbol']
                klines = result['klines']
                if validate_candles(klines, 20):
                    candles_map[symbol] = klines

        logger.debug(f"Loaded candles for {len(candles_map)}/{len(pairs)} pairs")
        return candles_map

    async def stage1_filter_signals(self) -> list[Dict]:
        """Stage 1: Base signal filtering"""
        logger.info("=" * 70)
        logger.info("STAGE 1: Signal filtering")
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
                    logger.debug(f"Signal found: {symbol} {signal_check['direction']} ({signal_check['confidence']}%)")

            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue

        pairs_with_signals.sort(key=lambda x: x['confidence'], reverse=True)
        self.signal_pairs_count = len(pairs_with_signals)

        logger.info(f"Stage 1 complete: {self.processed_pairs} scanned, {self.signal_pairs_count} signals found")
        return pairs_with_signals

    async def stage2_ai_select(self, signal_pairs: list[Dict]) -> list[str]:
        """Stage 2: AI selection"""
        logger.info("=" * 70)
        logger.info(f"STAGE 2: {config.STAGE2_PROVIDER.upper()} pair selection")
        logger.info("=" * 70)

        if not signal_pairs:
            logger.warning("No signal pairs to select from")
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
            logger.warning("No valid AI input data prepared")
            return []

        logger.info(f"Sending {len(ai_input_data)} pairs to {config.STAGE2_PROVIDER} for selection")
        selected_pairs = await ai_router.select_pairs(ai_input_data)
        self.ai_selected_count = len(selected_pairs)

        if selected_pairs:
            logger.info(f"Stage 2 complete: {self.ai_selected_count} pairs selected")
            for pair in selected_pairs:
                logger.debug(f"  ✓ {pair}")
        else:
            logger.warning("Stage 2: No pairs selected by AI")

        return selected_pairs

    async def stage3_unified_analysis(self, selected_pairs: list[str]) -> list[Dict]:
        """
        Stage 3: Unified analysis
        FIXED: Загружаем 1D данные для SWING анализа
        """
        logger.info("=" * 70)
        logger.info(f"STAGE 3: {config.STAGE3_PROVIDER.upper()} unified analysis")
        logger.info("=" * 70)

        if not selected_pairs:
            logger.warning("No pairs for analysis")
            return []

        # КРИТИЧНО: Загружаем BTC candles ОДИН РАЗ (включая 1D!)
        logger.debug("Loading BTC candles for correlation analysis (ONCE: 1H/4H/1D)")
        btc_candles_1h, btc_candles_4h, btc_candles_1d = await asyncio.gather(
            fetch_klines('BTCUSDT', config.TIMEFRAME_SHORT, config.FINAL_SHORT_CANDLES),
            fetch_klines('BTCUSDT', config.TIMEFRAME_LONG, config.FINAL_LONG_CANDLES),
            fetch_klines('BTCUSDT', config.TIMEFRAME_HTF, config.FINAL_HTF_CANDLES)
        )

        if not btc_candles_1h or not btc_candles_4h:
            logger.error("Failed to load BTC candles")
            return []

        logger.debug(f"✓ BTC candles loaded: {len(btc_candles_1h)} (1H), {len(btc_candles_4h)} (4H), {len(btc_candles_1d) if btc_candles_1d else 0} (1D)")

        final_signals = []

        for symbol in selected_pairs:
            try:
                logger.info(f"Analyzing {symbol}...")

                # FIXED: Загружаем ВСЕ таймфреймы включая 1D
                klines_1h, klines_4h, klines_1d = await asyncio.gather(
                    fetch_klines(symbol, config.TIMEFRAME_SHORT, config.FINAL_SHORT_CANDLES),
                    fetch_klines(symbol, config.TIMEFRAME_LONG, config.FINAL_LONG_CANDLES),
                    fetch_klines(symbol, config.TIMEFRAME_HTF, config.FINAL_HTF_CANDLES)
                )

                if not klines_1h or not klines_4h:
                    logger.debug(f"{symbol}: Insufficient 1H/4H data")
                    continue

                # 1D данные опциональны для не-swing пар, но желательны
                if not klines_1d:
                    logger.warning(f"{symbol}: No 1D data available (reduced analysis quality)")

                if not validate_candles(klines_1h, 20) or not validate_candles(klines_4h, 20):
                    logger.debug(f"{symbol}: Candle validation failed")
                    continue

                indicators_1h = calculate_ai_indicators(klines_1h, config.FINAL_INDICATORS_HISTORY)
                indicators_4h = calculate_ai_indicators(klines_4h, config.FINAL_INDICATORS_HISTORY)

                # FIXED: Рассчитываем индикаторы для 1D если есть данные
                indicators_1d = None
                if klines_1d and validate_candles(klines_1d, 10):
                    indicators_1d = calculate_ai_indicators(klines_1d, min(30, len(klines_1d)))
                    logger.debug(f"{symbol}: 1D indicators calculated")
                else:
                    logger.debug(f"{symbol}: No 1D indicators (data unavailable or insufficient)")

                if not indicators_1h or not indicators_4h:
                    logger.debug(f"{symbol}: Indicators calculation failed")
                    continue

                current_price = float(klines_1h[-1][4])

                from func_market_data import MarketDataCollector
                from func_correlation import get_comprehensive_correlation_analysis
                from func_volume_profile import calculate_volume_profile_for_candles, analyze_volume_profile

                collector = MarketDataCollector(await get_optimized_session())
                market_snapshot = await collector.get_market_snapshot(symbol, current_price)

                # Correlation analysis с переиспользованием BTC candles
                corr_analysis = await get_comprehensive_correlation_analysis(
                    symbol, klines_1h, btc_candles_1h, 'UNKNOWN', None
                )

                vp_data = calculate_volume_profile_for_candles(klines_4h, num_bins=50)
                vp_analysis = analyze_volume_profile(vp_data, current_price) if vp_data else None

                # FIXED: Добавлены 1D данные в comprehensive_data
                comprehensive_data = {
                    'symbol': symbol,
                    'candles_1h': klines_1h,
                    'candles_4h': klines_4h,
                    'candles_1d': klines_1d if klines_1d else [],  # FIXED
                    'indicators_1h': indicators_1h,
                    'indicators_4h': indicators_4h,
                    'indicators_1d': indicators_1d if indicators_1d else {},  # FIXED
                    'current_price': current_price,
                    'market_data': market_snapshot,
                    'correlation_data': corr_analysis,
                    'volume_profile': vp_data,
                    'vp_analysis': vp_analysis,
                    'btc_candles_1h': btc_candles_1h,
                    'btc_candles_4h': btc_candles_4h,
                    'btc_candles_1d': btc_candles_1d if btc_candles_1d else []  # FIXED
                }

                analysis = await ai_router.analyze_pair_comprehensive(symbol, comprehensive_data)

                signal_type = analysis.get('signal', 'NO_SIGNAL')
                confidence = analysis.get('confidence', 0)

                if signal_type != 'NO_SIGNAL' and confidence >= config.MIN_CONFIDENCE:
                    analysis['comprehensive_data'] = comprehensive_data
                    analysis['timestamp'] = datetime.now().isoformat()
                    final_signals.append(analysis)

                    # Сохраняем данные анализа для последующей записи
                    self.analysis_data_cache[symbol] = comprehensive_data

                    tp_levels = analysis.get('take_profit_levels', [0, 0, 0])
                    logger.info(f"✓ SIGNAL GENERATED: {symbol} {signal_type} (confidence: {confidence}%)")
                    logger.debug(f"  Entry: ${analysis['entry_price']:.2f} | Stop: ${analysis['stop_loss']:.2f}")
                    logger.debug(f"  TP: ${tp_levels[0]:.2f} / ${tp_levels[1]:.2f} / ${tp_levels[2]:.2f}")
                else:
                    rejection_reason = analysis.get('rejection_reason', 'Low confidence')
                    logger.info(f"✗ NO_SIGNAL: {symbol} - {rejection_reason}")

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}", exc_info=False)
                continue

        self.analyzed_count = len(final_signals)
        logger.info(f"Stage 3 complete: {len(final_signals)} signals generated")
        return final_signals

    async def stage4_validation(self, preliminary_signals: list[Dict]) -> Dict[str, Any]:
        """Stage 4: Signal validation"""
        logger.info("=" * 70)
        logger.info(f"STAGE 4: {config.STAGE4_PROVIDER.upper()} signal validation")
        logger.info("=" * 70)

        if not preliminary_signals:
            logger.warning("No preliminary signals to validate")
            return {'validated': [], 'rejected': []}

        logger.info(f"Validating {len(preliminary_signals)} signals...")
        validation_result = await validate_signals_simple(ai_router, preliminary_signals)

        if validation_result.get('validation_skipped_reason'):
            logger.warning(f"Validation skipped: {validation_result['validation_skipped_reason']}")
            return validation_result

        validated = validation_result['validated']
        rejected = validation_result['rejected']

        for sig in validated:
            logger.info(
                f"✓ APPROVED: {sig['symbol']} {sig['signal']} (confidence: {sig['confidence']}%, R/R: {sig.get('risk_reward_ratio', 0):.1f})")

        for rej in rejected:
            logger.info(f"✗ REJECTED: {rej['symbol']} - {rej.get('rejection_reason', 'Unknown')}")

        logger.info(f"Stage 4 complete: {len(validated)} approved, {len(rejected)} rejected")
        return validation_result

    def _enrich_signal_with_analysis_data(self, signal: Dict) -> Dict:
        """Добавляет данные анализа к сигналу"""
        symbol = signal.get('symbol')
        if symbol not in self.analysis_data_cache:
            return signal

        comp_data = self.analysis_data_cache[symbol]

        signal['analysis_data'] = {
            'candles_1h': comp_data.get('candles_1h', []),
            'candles_4h': comp_data.get('candles_4h', []),
            'candles_1d': comp_data.get('candles_1d', []),  # FIXED
            'indicators_1h': comp_data.get('indicators_1h', {}),
            'indicators_4h': comp_data.get('indicators_4h', {}),
            'indicators_1d': comp_data.get('indicators_1d', {}),  # FIXED
            'current_price': comp_data.get('current_price', 0),
            'market_data': comp_data.get('market_data', {}),
            'correlation_data': comp_data.get('correlation_data', {}),
            'volume_profile': comp_data.get('volume_profile', {}),
            'vp_analysis': comp_data.get('vp_analysis', {})
        }

        return signal

    async def run_cycle(self) -> Dict[str, Any]:
        """
        Запуск полного цикла бота
        FIXED: С поддержкой 1D данных для swing анализа
        """
        import time
        cycle_start = time.time()
        cycle_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        print("\n" + "=" * 70)
        print("🚀 TRADING BOT CYCLE STARTED")
        print("=" * 70 + "\n")

        logger.info("╔" + "=" * 68 + "╗")
        logger.info("║" + " TRADING BOT CYCLE STARTED".center(68) + "║")
        logger.info("╚" + "=" * 68 + "╝")

        # Проверка recovery
        last_checkpoint = self.checkpoint_mgr.get_last_checkpoint()

        if last_checkpoint:
            logger.info("🔄 RECOVERY MODE: Resuming from checkpoint")
            return await self._resume_from_checkpoint(last_checkpoint)

        # Начать новый checkpoint
        self.checkpoint_mgr.start_checkpoint(cycle_id)

        try:
            # Stage 1
            signal_pairs = await self.stage1_filter_signals()
            self.checkpoint_mgr.save_stage(1, {'signal_pairs': signal_pairs})

            if not signal_pairs:
                logger.warning("Pipeline stopped: No signal pairs found")
                total_time = time.time() - cycle_start
                return self._build_result('NO_SIGNAL_PAIRS', total_time, [], [])

            # Stage 2
            selected_pairs = await self.stage2_ai_select(signal_pairs)
            self.checkpoint_mgr.save_stage(2, {'selected_pairs': selected_pairs})

            if not selected_pairs:
                logger.warning("Pipeline stopped: AI selected 0 pairs")
                total_time = time.time() - cycle_start
                return self._build_result('NO_AI_SELECTION', total_time, [], [])

            # Stage 3 (FIXED: с 1D данными)
            preliminary_signals = await self.stage3_unified_analysis(selected_pairs)
            self.checkpoint_mgr.save_stage(3, {'preliminary_signals': preliminary_signals})

            if not preliminary_signals:
                logger.warning("Pipeline stopped: No analysis signals generated")
                total_time = time.time() - cycle_start
                return self._build_result('NO_ANALYSIS_SIGNALS', total_time, [], [])

            # Stage 4
            validation_result = await self.stage4_validation(preliminary_signals)
            self.checkpoint_mgr.save_stage(4, {'validation_result': validation_result})

            validated = validation_result['validated']
            rejected = validation_result['rejected']

            if validation_result.get('validation_skipped_reason'):
                logger.warning(f"Execution stopped: {validation_result['validation_skipped_reason']}")
                total_time = time.time() - cycle_start
                result = {
                    'timestamp': cycle_id,
                    'result': 'VALIDATION_SKIPPED',
                    'reason': validation_result['validation_skipped_reason'],
                    'stats': self._build_stats(total_time)
                }
                self.checkpoint_mgr.clear_checkpoint()
                return result

            total_time = time.time() - cycle_start

            # Success - clear checkpoint
            self.checkpoint_mgr.clear_checkpoint()

            # Сохраняем данные в storage
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

            # Сохраняем дневную статистику
            storage.save_daily_statistics(result['stats'])

            # Cleanup раз в день в полночь
            if datetime.now().hour == 0:
                storage.cleanup_old_data(days_to_keep=90)

            logger.info("╔" + "=" * 68 + "╗")
            logger.info(f"║ CYCLE COMPLETE: {result['result']}".ljust(69) + "║")
            logger.info(
                f"║ Time: {total_time:.1f}s | Signals: {len(validated)} approved, {len(rejected)} rejected".ljust(
                    69) + "║")
            logger.info("╚" + "=" * 68 + "╝")

            return result

        except Exception as e:
            logger.error(f"CRITICAL CYCLE ERROR: {e}", exc_info=True)
            total_time = time.time() - cycle_start
            return self._build_result('ERROR', total_time, [], [], error=str(e))

        finally:
            await cleanup_api()

    async def _resume_from_checkpoint(self, checkpoint: Dict) -> Dict:
        """Возобновляет выполнение из чекпоинта"""
        last_stage = checkpoint.get('stage', 0)
        data = checkpoint.get('data', {})

        logger.info(f"Resuming from Stage {last_stage}")

        if last_stage >= 1:
            signal_pairs = data.get('stage1', {}).get('signal_pairs', [])
            logger.info(f"Stage 1: Loaded {len(signal_pairs)} pairs from checkpoint")
        else:
            signal_pairs = await self.stage1_filter_signals()
            self.checkpoint_mgr.save_stage(1, {'signal_pairs': signal_pairs})

        if last_stage >= 2:
            selected_pairs = data.get('stage2', {}).get('selected_pairs', [])
            logger.info(f"Stage 2: Loaded {len(selected_pairs)} pairs from checkpoint")
        else:
            selected_pairs = await self.stage2_ai_select(signal_pairs)
            self.checkpoint_mgr.save_stage(2, {'selected_pairs': selected_pairs})

        if last_stage >= 3:
            preliminary_signals = data.get('stage3', {}).get('preliminary_signals', [])
            logger.info(f"Stage 3: Loaded {len(preliminary_signals)} signals from checkpoint")
        else:
            preliminary_signals = await self.stage3_unified_analysis(selected_pairs)
            self.checkpoint_mgr.save_stage(3, {'preliminary_signals': preliminary_signals})

        # Всегда запускаем Stage 4
        validation_result = await self.stage4_validation(preliminary_signals)
        self.checkpoint_mgr.save_stage(4, {'validation_result': validation_result})

        # Success - clear checkpoint
        self.checkpoint_mgr.clear_checkpoint()

        validated = validation_result['validated']
        rejected = validation_result['rejected']

        enriched_validated = [self._enrich_signal_with_analysis_data(sig) for sig in validated]

        for sig in enriched_validated:
            storage.save_signal(sig, compress=True)

        result = self._build_result(
            'SUCCESS' if validated else 'NO_VALIDATED_SIGNALS',
            0,
            enriched_validated,
            rejected
        )

        storage.save_daily_statistics(result['stats'])

        return result

    def _build_stats(self, total_time: float) -> Dict:
        """Helper для построения stats"""
        return {
            'pairs_scanned': self.processed_pairs,
            'signal_pairs_found': self.signal_pairs_count,
            'ai_selected': self.ai_selected_count,
            'analyzed': self.analyzed_count,
            'processing_speed': round(self.processed_pairs / total_time, 1) if total_time > 0 else 0,
            'total_time': round(total_time, 1),
            'timeframes': f"{config.TIMEFRAME_SHORT_NAME}/{config.TIMEFRAME_LONG_NAME}/{config.TIMEFRAME_HTF_NAME}"
        }

    def _build_result(self, result_type: str, total_time: float, validated: list, rejected: list,
                      error: str = None) -> Dict:
        """Helper для построения result"""
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

        if error:
            result['error'] = error

        return result


async def run_trading_bot() -> Dict[str, Any]:
    """Главная функция для запуска торгового бота"""
    bot = TradingBotRunner()
    result = await bot.run_cycle()
    return result