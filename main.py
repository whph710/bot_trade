"""
Упрощенный но мощный скальпинг-бот
Фокус на проверенных прибыльных стратегиях
"""

import asyncio
import logging
import time
from typing import List, Dict, Any
from dataclasses import dataclass

# Импорты модулей
from config import config
from exchange import get_candles, get_usdt_pairs, cleanup as cleanup_exchange
from indicators import generate_trading_signal, calculate_all_indicators
from ai_client import select_best_pairs, analyze_pair_detailed, cleanup_ai, check_ai_health

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, config.system.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.system.LOG_FILE, encoding=config.system.ENCODING),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Торговый сигнал"""
    pair: str
    signal: str  # LONG/SHORT/NO_SIGNAL
    confidence: int
    pattern: str
    volume_ratio: float
    trend_strength: float
    candles_5m: List = None
    candles_15m: List = None
    indicators: Dict = None


class SimplifiedScalpingBot:
    """Упрощенный скальпинг-бот"""

    def __init__(self):
        self.start_time = time.time()
        logger.info("🚀 Simplified Scalping Bot Started")

    async def scan_pair(self, pair: str) -> TradingSignal:
        """Сканирование одной пары"""
        try:
            # Получаем данные с обоих таймфреймов
            candles_5m = await get_candles(
                pair,
                config.trading.ENTRY_TF,
                config.trading.CANDLES_ENTRY
            )
            candles_15m = await get_candles(
                pair,
                config.trading.HIGHER_TF,
                config.trading.CANDLES_HIGHER
            )

            if not candles_5m or not candles_15m:
                logger.debug(f"No data for {pair}")
                return None

            # Генерируем торговый сигнал
            signal_data = generate_trading_signal(candles_5m, candles_15m)

            if signal_data['signal'] == 'NO_SIGNAL':
                return None

            # Рассчитываем дополнительные индикаторы
            indicators = calculate_all_indicators(candles_5m)

            return TradingSignal(
                pair=pair,
                signal=signal_data['signal'],
                confidence=signal_data['confidence'],
                pattern=signal_data.get('pattern', 'UNKNOWN'),
                volume_ratio=indicators.get('volume_ratio', 1.0),
                trend_strength=indicators.get('trend', {}).get('strength', 0),
                candles_5m=candles_5m,
                candles_15m=candles_15m,
                indicators=indicators
            )

        except Exception as e:
            logger.error(f"Error scanning {pair}: {e}")
            return None

    async def mass_scan(self) -> List[TradingSignal]:
        """Массовое сканирование всех пар"""
        logger.info("🔍 Starting mass market scan...")
        start_time = time.time()

        try:
            # Получаем список активных пар
            pairs = await get_usdt_pairs()
            logger.info(f"Scanning {len(pairs)} USDT pairs")

            # Сканируем все пары параллельно (батчами для стабильности)
            signals = []
            batch_size = 20

            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} pairs")

                # Запускаем задачи параллельно
                tasks = [self.scan_pair(pair) for pair in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Собираем успешные результаты
                for result in results:
                    if isinstance(result, TradingSignal):
                        signals.append(result)

                # Небольшая пауза между батчами
                if i + batch_size < len(pairs):
                    await asyncio.sleep(0.1)

            # Сортируем по уверенности
            signals.sort(key=lambda x: x.confidence, reverse=True)

            scan_time = time.time() - start_time
            logger.info(f"✅ Scan complete: {len(signals)} signals found in {scan_time:.1f}s")

            # Показываем топ сигналы
            for i, signal in enumerate(signals[:10]):
                logger.info(f"  {i+1}. {signal.pair}: {signal.signal} {signal.pattern} ({signal.confidence}%)")

            return signals

        except Exception as e:
            logger.error(f"Mass scan error: {e}")
            return []

    async def ai_selection_phase(self, signals: List[TradingSignal]) -> List[str]:
        """Фаза ИИ отбора лучших пар"""
        if not signals:
            logger.warning("No signals for AI selection")
            return []

        logger.info(f"🤖 AI selecting from {len(signals)} signals...")

        # Подготавливаем данные для ИИ
        signal_data = []
        for signal in signals:
            signal_data.append({
                'pair': signal.pair,
                'signal': signal.signal,
                'confidence': signal.confidence,
                'pattern': signal.pattern,
                'volume_ratio': signal.volume_ratio,
                'trend_strength': signal.trend_strength
            })

        try:
            selected_pairs = await select_best_pairs(signal_data)
            logger.info(f"✅ AI selected {len(selected_pairs)} pairs: {selected_pairs}")
            return selected_pairs

        except Exception as e:
            logger.error(f"AI selection error: {e}")
            # Fallback - берем топ по confidence
            fallback = [s.pair for s in signals[:config.ai.MAX_SELECTED_PAIRS]]
            logger.info(f"Using fallback selection: {fallback}")
            return fallback

    async def detailed_analysis_phase(self, selected_pairs: List[str], all_signals: List[TradingSignal]) -> int:
        """Фаза детального анализа выбранных пар"""
        if not selected_pairs:
            logger.warning("No pairs for detailed analysis")
            return 0

        logger.info(f"📊 Starting detailed analysis of {len(selected_pairs)} pairs...")

        # Создаем словарь сигналов для быстрого доступа
        signals_dict = {signal.pair: signal for signal in all_signals}

        successful_analyses = 0

        for pair in selected_pairs:
            signal = signals_dict.get(pair)
            if not signal:
                logger.warning(f"Signal not found for {pair}")
                continue

            try:
                logger.info(f"Analyzing {pair}...")

                # Отправляем на детальный анализ ИИ
                analysis = await analyze_pair_detailed(
                    pair=pair,
                    candles_5m=signal.candles_5m,
                    candles_15m=signal.candles_15m,
                    indicators=signal.indicators
                )

                if analysis:
                    self._save_analysis(pair, analysis, signal)
                    successful_analyses += 1
                    logger.info(f"✅ {pair} analysis complete")
                else:
                    logger.error(f"❌ {pair} analysis failed")

                # Пауза между анализами
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error analyzing {pair}: {e}")

        return successful_analyses

    def _save_analysis(self, pair: str, analysis: str, signal: TradingSignal):
        """Сохранение результатов анализа"""
        try:
            with open(config.system.ANALYSIS_FILE, 'a', encoding=config.system.ENCODING) as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"PAIR: {pair}\n")
                f.write(f"TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"SIGNAL: {signal.signal}\n")
                f.write(f"PATTERN: {signal.pattern}\n")
                f.write(f"CONFIDENCE: {signal.confidence}%\n")
                f.write(f"VOLUME RATIO: {signal.volume_ratio:.2f}\n")
                f.write(f"TREND STRENGTH: {signal.trend_strength:.1f}\n")
                f.write(f"{'='*40}\n")
                f.write(f"{analysis}\n")
                f.write(f"{'='*80}\n")
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")

    async def run_full_cycle(self):
        """Полный цикл работы бота"""
        cycle_start = time.time()

        try:
            # Проверка ИИ
            logger.info("🔧 Checking AI connection...")
            if not await check_ai_health():
                logger.error("AI health check failed - aborting")
                return

            # Этап 1: Массовое сканирование
            signals = await self.mass_scan()
            if not signals:
                logger.info("No trading signals found")
                return

            # Этап 2: ИИ отбор
            selected_pairs = await self.ai_selection_phase(signals)
            if not selected_pairs:
                logger.info("AI selected no pairs")
                return

            # Этап 3: Детальный анализ
            successful_analyses = await self.detailed_analysis_phase(selected_pairs, signals)

            # Итоги
            cycle_time = time.time() - cycle_start
            total_time = time.time() - self.start_time

            logger.info(f"\n{'='*60}")
            logger.info(f"🏆 CYCLE COMPLETE")
            logger.info(f"⏱️  Cycle time: {cycle_time:.1f}s")
            logger.info(f"📊 Total signals found: {len(signals)}")
            logger.info(f"🤖 AI selected pairs: {len(selected_pairs)}")
            logger.info(f"✅ Successful analyses: {successful_analyses}")
            logger.info(f"📈 Success rate: {successful_analyses/len(selected_pairs)*100:.1f}%")
            logger.info(f"💾 Results saved to: {config.system.ANALYSIS_FILE}")
            logger.info(f"🕒 Total runtime: {total_time:.1f}s")
            logger.info(f"{'='*60}")

        except KeyboardInterrupt:
            logger.info("⏹️  Stopped by user")
        except Exception as e:
            logger.error(f"💥 Fatal error in cycle: {e}")
        finally:
            await self._cleanup()

    async def _cleanup(self):
        """Очистка ресурсов"""
        logger.info("🧹 Cleaning up resources...")
        try:
            await cleanup_ai()
            await cleanup_exchange()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def main():
    """Главная функция"""
    logger.info("🚀 Starting Simplified Scalping Bot...")

    # Проверка конфигурации
    logger.info("🔧 Validating configuration...")
    if not config.validate():
        logger.error("❌ Configuration validation failed - check your .env file")
        logger.error("💡 Make sure you have DEEPSEEK=your_api_key in .env file")
        return

    # Создание и запуск бота
    bot = SimplifiedScalpingBot()

    # Показываем настройки
    logger.info(f"📊 Configuration:")
    logger.info(f"   Timeframes: {config.trading.HIGHER_TF}m + {config.trading.ENTRY_TF}m")
    logger.info(f"   Min confidence: {config.trading.MIN_CONFIDENCE}%")
    logger.info(f"   Max pairs to AI: {config.ai.MAX_PAIRS_TO_AI}")
    logger.info(f"   Max selected: {config.ai.MAX_SELECTED_PAIRS}")
    logger.info(f"   API model: {config.ai.MODEL}")

    await bot.run_full_cycle()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹️  Application stopped by user")
    except Exception as e:
        logger.error(f"💥 Application crash: {e}")
    finally:
        logger.info("🏁 Application finished")