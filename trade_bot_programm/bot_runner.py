"""
Trading Bot Runner - FIXED: AI Router инициализирован
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
from ai_router import AIRouter
from simple_validator import validate_signals_simple, calculate_validation_stats
from data_storage import storage
from logging_config import setup_module_logger

logger = setup_module_logger(__name__)

# FIXED: Глобальный экземпляр AI Router
ai_router = AIRouter()


class TradingBotRunner:
    """Класс для запуска торгового бота"""

    def __init__(self):
        self.processed_pairs = 0
        self.signal_pairs_count = 0
        self.ai_selected_count = 0
        self.analyzed_count = 0
        self.analysis_data_cache = {}
        # FIXED: Используем глобальный экземпляр
        self.ai_router = ai_router

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
        # FIXED: Используем self.ai_router
        selected_pairs = await self.ai_router.select_pairs(ai_input_data)
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
        FIXED: 1D данные опциональны, их отсутствие НЕ блокирует анализ
        """
        logger.info("=" * 70)
        logger.info(f"STAGE 3: {config.STAGE3_PROVIDER.upper()} unified analysis")
        logger.info("=" * 70)

        if not selected_pairs:
            logger.warning("No pairs for analysis")
            return []

        # Загружаем BTC candles ОДИН РАЗ (1H/4H обязательны, 1D опционально)
        logger.debug("Loading BTC candles for correlation analysis (1H/4H required, 1D optional)")
        btc_candles_1h, btc_candles_4h, btc_candles_1d = await asyncio.gather(
            fetch_klines('BTCUSDT', config.TIMEFRAME_SHORT, config.FINAL_SHORT_CANDLES),
            fetch_klines('BTCUSDT', config.TIMEFRAME_LONG, config.FINAL_LONG_CANDLES),
            fetch_klines('BTCUSDT', config.TIMEFRAME_HTF, config.FINAL_HTF_CANDLES)
        )

        if not btc_candles_1h or not btc_candles_4h:
            logger.error("Failed to load BTC 1H/4H candles (critical)")
            return []

        # 1D могут отсутствовать
        if not btc_candles_1d:
            logger.warning("⚠️ BTC 1D candles not available (non-critical)")
        else:
            logger.debug(f"✓ BTC candles loaded: {len(btc_candles_1h)} (1H), {len(btc_candles_4h)} (4H), {len(btc_candles_1d)} (1D)")

        final_signals = []

        for symbol in selected_pairs:
            try:
                logger.info(f"Analyzing {symbol}...")

                # Загружаем все таймфреймы (1D опционально)
                klines_1h, klines_4h, klines_1d = await asyncio.gather(
                    fetch_klines(symbol, config.TIMEFRAME_SHORT, config.FINAL_SHORT_CANDLES),
                    fetch_klines(symbol, config.TIMEFRAME_LONG, config.FINAL_LONG_CANDLES),
                    fetch_klines(symbol, config.TIMEFRAME_HTF, config.FINAL_HTF_CANDLES)
                )

                # 1H и 4H ОБЯЗАТЕЛЬНЫ
                if not klines_1h or not klines_4h:
                    logger.warning(f"{symbol}: Missing 1H/4H data (critical) - SKIP")
                    continue

                # 1D данные ОПЦИОНАЛЬНЫ
                has_1d_data = bool(klines_1d and validate_candles(klines_1d, 10))

                if not has_1d_data:
                    logger.info(f"{symbol}: No 1D data available (non-critical, analysis continues)")
                    klines_1d = []

                if not validate_candles(klines_1h, 20) or not validate_candles(klines_4h, 20):
                    logger.warning(f"{symbol}: 1H/4H candle validation failed - SKIP")
                    continue

                indicators_1h = calculate_ai_indicators(klines_1h, config.FINAL_INDICATORS_HISTORY)
                indicators_4h = calculate_ai_indicators(klines_4h, config.FINAL_INDICATORS_HISTORY)

                # 1D индикаторы опциональны
                indicators_1d = {}
                if has_1d_data:
                    try:
                        indicators_1d = calculate_ai_indicators(klines_1d, min(30, len(klines_1d)))
                        if indicators_1d:
                            logger.debug(f"{symbol}: ✓ 1D indicators calculated")
                    except Exception as e:
                        logger.debug(f"{symbol}: Failed to calculate 1D indicators (non-critical): {e}")
                        indicators_1d = {}

                if not indicators_1h or not indicators_4h:
                    logger.warning(f"{symbol}: 1H/4H indicators calculation failed - SKIP")
                    continue

                current_price = float(klines_1h[-1][4])

                from func_market_data import MarketDataCollector
                from func_correlation import get_comprehensive_correlation_analysis
                from func_volume_profile import calculate_volume_profile_for_candles, analyze_volume_profile

                collector = MarketDataCollector(await get_optimized_session())
                market_snapshot = await collector.get_market_snapshot(symbol, current_price)

                # Correlation analysis
                corr_analysis = await get_comprehensive_correlation_analysis(
                    symbol, klines_1h, btc_candles_1h, 'UNKNOWN', None
                )

                vp_data = calculate_volume_profile_for_candles(klines_4h, num_bins=50)
                vp_analysis = analyze_volume_profile(vp_data, current_price) if vp_data else None

                # FIXED: Помечаем доступность 1D данных
                comprehensive_data = {
                    'symbol': symbol,
                    'candles_1h': klines_1h,
                    'candles_4h': klines_4h,
                    'candles_1d': klines_1d,  # Может быть пустым []
                    'indicators_1h': indicators_1h,
                    'indicators_4h': indicators_4h,
                    'indicators_1d': indicators_1d,  # Может быть пустым {}
                    'has_1d_data': has_1d_data,  # НОВОЕ: флаг доступности
                    'current_price': current_price,
                    'market_data': market_snapshot,
                    'correlation_data': corr_analysis,
                    'volume_profile': vp_data,
                    'vp_analysis': vp_analysis,
                    'btc_candles_1h': btc_candles_1h,
                    'btc_candles_4h': btc_candles_4h,
                    'btc_candles_1d': btc_candles_1d if btc_candles_1d else []
                }

                # FIXED: Используем self.ai_router
                analysis = await self.ai_router.analyze_pair_comprehensive(symbol, comprehensive_data)

                signal_type = analysis.get('signal', 'NO_SIGNAL')
                confidence = analysis.get('confidence', 0)

                if signal_type != 'NO_SIGNAL' and confidence >= config.MIN_CONFIDENCE:
                    analysis['comprehensive_data'] = comprehensive_data
                    analysis['timestamp'] = datetime.now().isoformat()
                    final_signals.append(analysis)

                    # Сохраняем данные анализа
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
        # FIXED: Передаем self.ai_router
        validation_result = await validate_signals_simple(self.ai_router, preliminary_signals)

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
            'candles_1d': comp_data.get('candles_1d', []),
            'indicators_1h': comp_data.get('indicators_1h', {}),
            'indicators_4h': comp_data.get('indicators_4h', {}),
            'indicators_1d': comp_data.get('indicators_1d', {}),
            'has_1d_data': comp_data.get('has_1d_data', False),  # Флаг доступности
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
        FIXED: Убран checkpoint manager
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

        try:
            # Stage 1
            signal_pairs = await self.stage1_filter_signals()

            if not signal_pairs:
                logger.warning("Pipeline stopped: No signal pairs found")
                total_time = time.time() - cycle_start
                return self._build_result('NO_SIGNAL_PAIRS', total_time, [], [])

            # Stage 2
            selected_pairs = await self.stage2_ai_select(signal_pairs)

            if not selected_pairs:
                logger.warning("Pipeline stopped: AI selected 0 pairs")
                total_time = time.time() - cycle_start
                return self._build_result('NO_AI_SELECTION', total_time, [], [])

            # Stage 3
            preliminary_signals = await self.stage3_unified_analysis(selected_pairs)

            if not preliminary_signals:
                logger.warning("Pipeline stopped: No analysis signals generated")
                total_time = time.time() - cycle_start
                return self._build_result('NO_ANALYSIS_SIGNALS', total_time, [], [])

            # Stage 4
            validation_result = await self.stage4_validation(preliminary_signals)

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
                return result

            total_time = time.time() - cycle_start

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