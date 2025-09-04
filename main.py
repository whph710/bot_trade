import asyncio
import json
import logging
import time
import math
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import datetime

# ИМПОРТЫ КОТОРЫЕ НЕЛЬЗЯ МЕНЯТЬ
from func_async import get_klines_async, get_usdt_trading_pairs
from deepseek import deep_seek_selection, deep_seek_analysis, cleanup_http_client

# НОВЫЙ УПРОЩЕННЫЙ ИМПОРТ
from func_trade import (
    detect_scalping_entry,
    calculate_simplified_indicators,
    prepare_ai_data,
    safe_float,
    safe_int,
    test_json_serialization
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scalping_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# УПРОЩЕННАЯ КОНФИГУРАЦИЯ
SCALPING_CONFIG = {
    'candles_for_scan': 50,
    'batch_size': 30,  # Уменьшен для стабильности
    'min_confidence': 75,
    'max_pairs_to_ai': 8,  # Уменьшено
    'forbidden_hours': [22, 23, 0, 1, 2, 3, 4, 5],
}


@dataclass
class SimpleScalpingSignal:
    """Упрощенный торговый сигнал"""
    pair: str
    signal_type: str
    confidence: int
    entry_price: float
    timestamp: int

    # Упрощенные метрики
    ema_signal: str
    rsi_value: float
    volume_spike: bool
    entry_reasons: List[str]

    # Данные для ИИ
    candles_data: List = None
    indicators_data: Dict = None


class SimplifiedScalpingAnalyzer:
    """Упрощенный анализатор - только 3 индикатора"""

    def __init__(self):
        self.session_start = time.time()
        logger.info("🚀 Упрощенный скальпинговый анализатор запущен (3 индикатора)")

    def is_trading_hours(self) -> bool:
        """Проверка торговых часов"""
        current_hour = datetime.datetime.utcnow().hour
        return current_hour not in SCALPING_CONFIG['forbidden_hours']

    async def quick_scan_pair(self, symbol: str) -> Optional[SimpleScalpingSignal]:
        """Быстрое сканирование одной пары"""
        try:
            # Получаем свечи
            candles = await get_klines_async(
                symbol,
                interval="5",
                limit=SCALPING_CONFIG['candles_for_scan']
            )

            if not candles or len(candles) < 30:
                return None

            # Используем упрощенную систему сигналов
            signal_result = detect_scalping_entry(candles)

            if signal_result['signal'] == 'NO_SIGNAL':
                return None

            # Создаем упрощенный сигнал
            try:
                entry_price = safe_float(candles[-1][4])
                confidence = safe_int(signal_result['confidence'])

                if entry_price <= 0 or confidence <= 0:
                    return None

                indicators = signal_result.get('indicators', {})

                return SimpleScalpingSignal(
                    pair=symbol,
                    signal_type=signal_result['signal'],
                    confidence=confidence,
                    entry_price=entry_price,
                    timestamp=int(time.time()),
                    ema_signal=str(indicators.get('ema_signal', 'NEUTRAL')),
                    rsi_value=safe_float(indicators.get('rsi_value', 50)),
                    volume_spike=bool(indicators.get('volume_spike', False)),
                    entry_reasons=signal_result.get('entry_reasons', []),
                    candles_data=candles[-20:],  # Только последние 20 свечей
                    indicators_data=indicators
                )

            except Exception as e:
                logger.warning(f"❌ Ошибка создания сигнала {symbol}: {e}")
                return None

        except Exception as e:
            logger.error(f"❌ Ошибка сканирования {symbol}: {e}")
            return None

    async def mass_scan_markets(self) -> List[SimpleScalpingSignal]:
        """Массовое сканирование рынков"""
        if not self.is_trading_hours():
            logger.warning("⏰ Неторговые часы - пропускаем сканирование")
            return []

        start_time = time.time()
        logger.info("🔍 ЭТАП 1: Быстрое сканирование (3 индикатора)")

        try:
            pairs = await get_usdt_trading_pairs()
            if not pairs:
                logger.error("❌ Не удалось получить список пар")
                return []

            # Берем только топ-100 пар для скорости
            pairs = pairs
            logger.info(f"📊 Сканируем топ-{len(pairs)} пар")

            promising_signals = []

            # Обрабатываем батчами
            for i in range(0, len(pairs), SCALPING_CONFIG['batch_size']):
                batch = pairs[i:i + SCALPING_CONFIG['batch_size']]

                tasks = [self.quick_scan_pair(pair) for pair in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Собираем результаты
                for result in results:
                    if isinstance(result, SimpleScalpingSignal):
                        promising_signals.append(result)
                    elif isinstance(result, Exception):
                        logger.debug(f"Исключение в батче: {result}")

                # Логируем прогресс
                processed = min(i + SCALPING_CONFIG['batch_size'], len(pairs))
                logger.info(f"⏳ Обработано: {processed}/{len(pairs)}")

                # Пауза между батчами
                await asyncio.sleep(0.2)

            # Сортируем по уверенности
            promising_signals.sort(key=lambda x: x.confidence, reverse=True)

            execution_time = time.time() - start_time
            logger.info(f"✅ ЭТАП 1 завершен: {len(promising_signals)} сигналов за {execution_time:.2f}сек")

            return promising_signals

        except Exception as e:
            logger.error(f"❌ Критическая ошибка сканирования: {e}")
            return []


class SimpleAISelector:
    """Упрощенный ИИ селектор"""

    def __init__(self):
        self.selection_prompt = self._load_prompt('prompt2.txt')

    def _load_prompt(self, filename: str) -> str:
        """Загрузка промпта"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error(f"❌ Файл {filename} не найден")
            return "Выбери лучшие пары для скальпинга. Верни JSON: {\"pairs\": [\"BTCUSDT\"]}"

    async def select_best_pairs(self, signals: List[SimpleScalpingSignal]) -> List[str]:
        """Первичный отбор через ИИ"""
        if not self.selection_prompt or not signals:
            return []

        logger.info(f"🤖 ЭТАП 2: ИИ отбор из {len(signals)} сигналов")

        try:
            # Ограничиваем количество
            top_signals = signals[:SCALPING_CONFIG['max_pairs_to_ai']]

            # Подготавливаем данные с помощью новой функции
            ai_data = self._prepare_simple_data(top_signals)

            # Проверяем JSON сериализацию
            if not test_json_serialization(ai_data):
                logger.error("❌ Данные не сериализуются в JSON")
                return []

            # Формируем запрос
            message = f"""{self.selection_prompt}

=== УПРОЩЕННЫЙ СКАЛЬПИНГ 15M ===
СИСТЕМА: 3 индикатора (EMA + RSI + Volume)
СИГНАЛОВ: {len(top_signals)}
УДЕРЖАНИЕ: 3-4 свечи

{json.dumps(ai_data, indent=2, ensure_ascii=False)}

ВАЖНО: Выбери максимум 3 лучших пары.
Верни JSON: {{"pairs": ["BTCUSDT", "ETHUSDT"]}}"""

            # Отправляем в ИИ
            ai_response = await deep_seek_selection(message)

            if not ai_response:
                logger.error("❌ ИИ не ответил")
                return []

            # Парсим ответ
            selected_pairs = self._parse_ai_response(ai_response)

            logger.info(f"✅ ЭТАП 2 завершен: ИИ выбрал {len(selected_pairs)} пар")
            return selected_pairs

        except Exception as e:
            logger.error(f"❌ Ошибка ИИ отбора: {e}")
            return []

    def _prepare_simple_data(self, signals: List[SimpleScalpingSignal]) -> Dict[str, Any]:
        """Подготовка упрощенных данных для ИИ"""
        prepared_signals = []

        for signal in signals:
            try:
                # Только самые важные данные
                signal_data = {
                    'pair': str(signal.pair),
                    'signal_type': str(signal.signal_type),
                    'confidence': safe_int(signal.confidence),
                    'entry_price': safe_float(signal.entry_price),

                    # 3 ключевых индикатора
                    'ema_signal': str(signal.ema_signal),
                    'rsi_value': safe_float(signal.rsi_value),
                    'volume_spike': bool(signal.volume_spike),

                    'entry_reasons': [str(r) for r in signal.entry_reasons],

                    # Последние 5 свечей для контекста
                    'last_5_candles': []
                }

                # Добавляем свечи если есть
                if signal.candles_data and len(signal.candles_data) >= 5:
                    for candle in signal.candles_data[-5:]:
                        if len(candle) >= 6:
                            signal_data['last_5_candles'].append({
                                'close': safe_float(candle[4]),
                                'volume': safe_float(candle[5])
                            })

                prepared_signals.append(signal_data)

            except Exception as e:
                logger.warning(f"Ошибка подготовки {signal.pair}: {e}")
                continue

        return {
            'signals_count': len(prepared_signals),
            'strategy': '3_indicators_scalping',
            'signals': prepared_signals
        }

    def _parse_ai_response(self, response: str) -> List[str]:
        """Парсинг ответа ИИ"""
        try:
            import re
            # Ищем JSON с парами
            json_match = re.search(r'\{[^}]*"pairs"[^}]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                pairs = data.get('pairs', [])
                return [str(pair) for pair in pairs if isinstance(pair, str)]
            return []
        except Exception as e:
            logger.error(f"Ошибка парсинга ИИ ответа: {e}")
            return []

    async def detailed_analysis(self, pair: str) -> Optional[str]:
        """Упрощенный детальный анализ"""
        logger.info(f"🔬 ЭТАП 3: Упрощенный анализ {pair}")

        try:
            # Получаем больше данных для анализа
            full_candles = await get_klines_async(pair, "15", limit=100)

            if not full_candles or len(full_candles) < 50:
                logger.error(f"❌ Недостаточно данных для {pair}")
                return None

            # Рассчитываем упрощенные индикаторы
            indicators = calculate_simplified_indicators(full_candles)

            if not indicators:
                logger.error(f"❌ Не удалось рассчитать индикаторы для {pair}")
                return None

            # Подготавливаем упрощенные данные для анализа
            analysis_data = {
                'pair': pair,
                'current_price': safe_float(full_candles[-1][4]),
                'timestamp': int(time.time()),

                # 3 ключевых индикатора
                'ema_fast': safe_float(indicators.get('ema_fast_value', 0)),
                'ema_slow': safe_float(indicators.get('ema_slow_value', 0)),
                'ema_signal': str(indicators.get('ema_signal', 'NEUTRAL')),
                'ema_diff_percent': safe_float(indicators.get('ema_diff_percent', 0)),

                'rsi_value': safe_float(indicators.get('rsi_value', 50)),
                'rsi_signal': str(indicators.get('rsi_signal', 'NEUTRAL')),

                'volume_spike': bool(indicators.get('volume_spike', False)),
                'volume_ratio': safe_float(indicators.get('volume_ratio', 1.0)),
                'volume_trend': str(indicators.get('volume_trend', 'NEUTRAL')),

                'signal_quality': safe_int(indicators.get('signal_quality', 0)),

                # Последние 10 свечей
                'recent_candles': [
                    {
                        'open': safe_float(c[1]),
                        'high': safe_float(c[2]),
                        'low': safe_float(c[3]),
                        'close': safe_float(c[4]),
                        'volume': safe_float(c[5])
                    } for c in full_candles[-10:]
                ]
            }

            # Проверяем JSON сериализацию
            if not test_json_serialization(analysis_data):
                logger.error(f"❌ Данные анализа {pair} не сериализуются")
                return None

            # Формируем запрос для анализа
            analysis_prompt = self._load_prompt('prompt.txt')
            message = f"""{analysis_prompt}

=== ДЕТАЛЬНЫЙ АНАЛИЗ УПРОЩЕННОЙ СИСТЕМЫ ===
ПАРА: {pair}
СИСТЕМА: EMA + RSI + Volume
ЦЕНА: {analysis_data['current_price']}

{json.dumps(analysis_data, indent=2, ensure_ascii=False)}

Дай конкретные рекомендации по сделке."""

            # Отправляем в ИИ
            analysis_result = await deep_seek_analysis(message)

            if analysis_result:
                self._save_analysis(pair, analysis_result)
                logger.info(f"✅ Анализ {pair} завершен")
                return analysis_result

            return None

        except Exception as e:
            logger.error(f"❌ Ошибка детального анализа {pair}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _save_analysis(self, pair: str, analysis: str):
        """Сохранение результата анализа"""
        try:
            with open('simple_scalping_analysis.log', 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"ПАРА: {pair}\n")
                f.write(f"ВРЕМЯ: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"СИСТЕМА: EMA + RSI + Volume\n")
                f.write(f"АНАЛИЗ:\n{analysis}\n")
                f.write(f"{'=' * 60}\n")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения: {e}")


async def main():
    """Главная функция упрощенного скальпингового бота"""
    logger.info("🚀 УПРОЩЕННЫЙ СКАЛЬПИНГОВЫЙ БОТ 15M - ЗАПУСК")
    logger.info("🎯 Система: EMA + RSI + Volume (3 индикатора)")
    logger.info("⚡ Стратегия: Удержание 3-4 свечи")

    # Инициализация компонентов
    analyzer = SimplifiedScalpingAnalyzer()
    ai_selector = SimpleAISelector()

    try:
        # ЭТАП 1: Быстрое сканирование
        promising_signals = await analyzer.mass_scan_markets()

        if not promising_signals:
            logger.info("ℹ️ Качественных сигналов не найдено")
            return

        logger.info(f"📈 Найдено {len(promising_signals)} перспективных сигналов")

        # Показываем топ-сигналы
        for i, signal in enumerate(promising_signals[:5], 1):
            logger.info(f"  {i}. {signal.pair}: {signal.signal_type} "
                        f"(уверенность: {signal.confidence}%, "
                        f"EMA: {signal.ema_signal}, RSI: {signal.rsi_value:.1f})")

        # ЭТАП 2: ИИ отбор
        selected_pairs = await ai_selector.select_best_pairs(promising_signals)

        if not selected_pairs:
            logger.info("ℹ️ ИИ не выбрал ни одной пары")
            return

        logger.info(f"🤖 ИИ выбрал {len(selected_pairs)} пар: {selected_pairs}")

        # ЭТАП 3: Детальный анализ
        successful_analyses = 0

        for pair in selected_pairs:
            logger.info(f"🔍 Анализирую {pair}...")

            analysis = await ai_selector.detailed_analysis(pair)

            if analysis:
                successful_analyses += 1
                logger.info(f"✅ {pair} - анализ завершен")
            else:
                logger.error(f"❌ {pair} - ошибка анализа")

            # Пауза между запросами
            await asyncio.sleep(2)

        # ИТОГИ
        logger.info(f"\n🎉 УПРОЩЕННЫЙ АНАЛИЗ ЗАВЕРШЕН!")
        logger.info(f"📊 Система: 3 индикатора (EMA + RSI + Volume)")
        logger.info(f"🎯 Найдено сигналов: {len(promising_signals)}")
        logger.info(f"🤖 ИИ выбрал: {len(selected_pairs)}")
        logger.info(f"📋 Успешных анализов: {successful_analyses}")
        logger.info(f"📁 Результаты: simple_scalping_analysis.log")

        # Очищаем HTTP клиент
        await cleanup_http_client()

    except KeyboardInterrupt:
        logger.info("⏹️ Остановка по запросу пользователя")
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await cleanup_http_client()


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🎯 УПРОЩЕННЫЙ СКАЛЬПИНГОВЫЙ БОТ")
    logger.info("📊 Система: EMA + RSI + Volume")
    logger.info("⚡ Удержание: 3-4 свечи на 15M")
    logger.info("=" * 80)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Программа остановлена")
    except Exception as e:
        logger.error(f"💥 Фатальная ошибка: {e}")
    finally:
        logger.info("🔚 Работа завершена")