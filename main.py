import asyncio
import json
import logging
import time
import math
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import re
import datetime

# ИМПОРТЫ КОТОРЫЕ НЕЛЬЗЯ МЕНЯТЬ
from func_async import get_klines_async, get_usdt_trading_pairs
from deepseek import deep_seek_selection, deep_seek_analysis, cleanup_http_client

# НОВЫЕ ИМПОРТЫ (заменяют старые из func_trade)
from func_trade import detect_scalping_signal, calculate_scalping_indicators

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

# КОНФИГУРАЦИЯ СКАЛЬПИНГА
SCALPING_CONFIG = {
    'candles_for_scan': 50,  # Уменьшено с 200 для скорости
    'candles_for_analysis': 16,  # Для первичного отбора ИИ
    'candles_for_detailed': 200,  # Для детального анализа
    'batch_size': 50,  # Размер батча
    'min_confidence': 70,  # Минимальная уверенность
    'max_pairs_to_ai': 10,  # Максимум пар для ИИ анализа
    'forbidden_hours': [22, 23, 0, 1, 2, 3, 4, 5],  # Низкая ликвидность UTC
}


def clean_value(value):
    """Очистка значений от NaN, Infinity и приведение к JSON-сериализуемым типам"""
    if isinstance(value, (np.integer, np.floating)):
        value = float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, np.ndarray):
        return [clean_value(x) for x in value.tolist()]

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value
    elif isinstance(value, dict):
        return {k: clean_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [clean_value(item) for item in value]
    else:
        return value


def safe_json_serialize(obj: Any) -> Any:
    """Безопасная сериализация для JSON с обработкой NaN"""
    return clean_value(obj)


@dataclass
class ScalpingSignal:
    """Упрощенный торговый сигнал для скальпинга"""
    pair: str
    signal_type: str  # 'LONG', 'SHORT', 'NO_SIGNAL'
    confidence: int
    entry_price: float
    timestamp: int

    # Ключевые метрики
    quality_score: int
    volatility_regime: str
    volume_confirmed: bool
    entry_reasons: List[str]

    # Для ИИ анализа
    candles_data: List = None
    indicators_data: Dict = None


class FastScalpingAnalyzer:
    """Быстрый анализатор для скальпинга 15M"""

    def __init__(self):
        self.session_start = time.time()
        logger.info("🚀 Быстрый скальпинговый анализатор запущен")

    def is_trading_hours(self) -> bool:
        """Проверка торговых часов"""
        current_hour = datetime.datetime.utcnow().hour
        return current_hour not in SCALPING_CONFIG['forbidden_hours']

    async def quick_scan_pair(self, symbol: str) -> Optional[ScalpingSignal]:
        """Быстрое сканирование одной пары"""
        try:
            # Получаем свечи для быстрого анализа
            candles = await get_klines_async(
                symbol,
                interval="15",
                limit=SCALPING_CONFIG['candles_for_scan']
            )

            if not candles or len(candles) < 30:
                return None

            # Определяем сигнал с помощью новой быстрой функции
            signal_result = detect_scalping_signal(candles)

            if signal_result['signal'] == 'NO_SIGNAL':
                return None

            # Создаем сигнал с проверкой значений
            try:
                entry_price = float(candles[-1][4])
                confidence = int(signal_result['confidence'])
                quality_score = int(signal_result.get('quality_score', 0))

                # Проверяем на валидность
                if math.isnan(entry_price) or math.isnan(confidence):
                    logger.warning(f"❌ NaN значения в {symbol}")
                    return None

            except (ValueError, TypeError) as e:
                logger.warning(f"❌ Ошибка конвертации значений {symbol}: {e}")
                return None

            return ScalpingSignal(
                pair=symbol,
                signal_type=signal_result['signal'],
                confidence=confidence,
                entry_price=entry_price,
                timestamp=int(time.time()),
                quality_score=quality_score,
                volatility_regime=signal_result.get('volatility_regime', 'MEDIUM'),
                volume_confirmed=bool(signal_result.get('indicators', {}).get('volume_spike', False)),
                entry_reasons=signal_result.get('entry_reasons', []),
                candles_data=candles[-SCALPING_CONFIG['candles_for_analysis']:],
                indicators_data=clean_value(signal_result.get('indicators', {}))  # Очищаем данные
            )

        except Exception as e:
            logger.error(f"❌ Ошибка сканирования {symbol}: {e}")
            return None

    async def mass_scan_markets(self) -> List[ScalpingSignal]:
        """Массовое сканирование рынков"""
        # if not self.is_trading_hours():
        #     logger.warning("⏰ Неторговые часы - пропускаем сканирование")
        #     return []

        start_time = time.time()
        logger.info("🔍 ЭТАП 1: Быстрое сканирование всех пар")

        try:
            # Получаем список пар
            pairs = await get_usdt_trading_pairs()
            if not pairs:
                logger.error("❌ Не удалось получить список пар")
                return []

            logger.info(f"📊 Сканируем {len(pairs)} пар")

            # Обрабатываем батчами
            promising_signals = []

            for i in range(0, len(pairs), SCALPING_CONFIG['batch_size']):
                batch = pairs[i:i + SCALPING_CONFIG['batch_size']]

                # Создаем задачи для батча
                tasks = [self.quick_scan_pair(pair) for pair in batch]

                # Выполняем параллельно
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Собираем результаты
                for result in results:
                    if isinstance(result, ScalpingSignal):
                        promising_signals.append(result)

                # Логируем прогресс
                processed = min(i + SCALPING_CONFIG['batch_size'], len(pairs))
                logger.info(f"⏳ Обработано: {processed}/{len(pairs)}")

                # Небольшая пауза между батчами
                if i + SCALPING_CONFIG['batch_size'] < len(pairs):
                    await asyncio.sleep(0.1)

            # Сортируем по уверенности
            promising_signals.sort(key=lambda x: x.confidence, reverse=True)

            execution_time = time.time() - start_time
            logger.info(f"✅ ЭТАП 1 завершен: {len(promising_signals)} сигналов за {execution_time:.2f}сек")

            return promising_signals

        except Exception as e:
            logger.error(f"❌ Критическая ошибка сканирования: {e}")
            return []


class AIScalpingSelector:
    """ИИ селектор для скальпинга"""

    def __init__(self):
        self.selection_prompt = self._load_prompt('prompt2.txt')
        self.analysis_prompt = self._load_prompt('prompt.txt')

    def _load_prompt(self, filename: str) -> str:
        """Загрузка промпта"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error(f"❌ Файл {filename} не найден")
            return ""

    def _prepare_signals_for_ai(self, signals: List[ScalpingSignal]) -> Dict[str, Any]:
        """Подготовка данных для ИИ анализа (16 свечей + индикаторы)"""
        prepared_data = []

        for signal in signals:
            # Берем только последние 16 свечей
            recent_candles = signal.candles_data[-16:] if signal.candles_data else []

            signal_data = {
                'pair': signal.pair,
                'signal_type': signal.signal_type,
                'confidence': int(signal.confidence),
                'entry_price': float(signal.entry_price),
                'quality_score': int(signal.quality_score),
                'volatility_regime': str(signal.volatility_regime),
                'volume_confirmed': bool(signal.volume_confirmed),
                'entry_reasons': [str(reason) for reason in signal.entry_reasons],

                # 16 последних свечей
                'recent_candles': [
                    {
                        'timestamp': int(c[0]),
                        'open': float(c[1]),
                        'high': float(c[2]),
                        'low': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5])
                    } for c in recent_candles
                ],

                # Ключевые индикаторы (последние 16 значений)
                'indicators': safe_json_serialize({
                    'tema3': signal.indicators_data.get('tema3_values', [])[-16:],
                    'tema5': signal.indicators_data.get('tema5_values', [])[-16:],
                    'tema8': signal.indicators_data.get('tema8_values', [])[-16:],
                    'rsi': signal.indicators_data.get('rsi_values', [])[-16:],
                    'stoch_k': signal.indicators_data.get('stoch_k', [])[-16:],
                    'stoch_d': signal.indicators_data.get('stoch_d', [])[-16:],
                    'macd_line': signal.indicators_data.get('macd_line', [])[-16:],
                    'macd_signal': signal.indicators_data.get('macd_signal', [])[-16:],
                    'atr': signal.indicators_data.get('atr_values', [])[-16:],

                    # Текущие значения
                    'current_rsi': signal.indicators_data.get('rsi_current', 50.0),
                    'current_atr': signal.indicators_data.get('atr_current', 0.0),
                    'tema_alignment': signal.indicators_data.get('tema_alignment', False),
                    'tema_slope': signal.indicators_data.get('tema_slope', 0.0),
                    'macd_crossover': signal.indicators_data.get('macd_crossover', 'NONE'),
                    'stoch_signal': signal.indicators_data.get('stoch_signal', 'NEUTRAL'),

                    # Объемы и уровни
                    'volume_ratio': signal.indicators_data.get('volume_ratio', 1.0),
                    'volume_strength': signal.indicators_data.get('volume_strength', 0),
                    'support_levels': signal.indicators_data.get('support_levels', []),
                    'resistance_levels': signal.indicators_data.get('resistance_levels', []),
                    'near_support': signal.indicators_data.get('near_support', False),
                    'near_resistance': signal.indicators_data.get('near_resistance', False),

                    # Микроструктура
                    'price_velocity': signal.indicators_data.get('price_velocity', 0.0),
                    'momentum_acceleration': signal.indicators_data.get('momentum_acceleration', 0.0),
                    'trend_strength': signal.indicators_data.get('trend_strength', 0)
                })
            }

            prepared_data.append(signal_data)

        return {
            'signals_count': len(prepared_data),
            'timeframe': '15m',
            'strategy': 'scalping_3_4_candles',
            'timestamp': int(time.time()),
            'signals': prepared_data
        }

    async def select_best_pairs(self, signals: List[ScalpingSignal]) -> List[str]:
        """Первичный отбор через ИИ (16 свечей)"""
        if not self.selection_prompt or not signals:
            return []

        logger.info(f"🤖 ЭТАП 2: ИИ отбор из {len(signals)} сигналов")

        try:
            # Ограничиваем количество сигналов для ИИ
            top_signals = signals[:SCALPING_CONFIG['max_pairs_to_ai']]

            # Подготавливаем данные
            ai_data = self._prepare_signals_for_ai(top_signals)

            # Формируем запрос
            message = f"""{self.selection_prompt}

=== СКАЛЬПИНГ 15M: ПЕРВИЧНЫЙ ОТБОР ===
КОЛИЧЕСТВО СИГНАЛОВ: {len(top_signals)}
УДЕРЖАНИЕ: 3-4 свечи
ДАННЫЕ: Последние 16 свечей + индикаторы

{json.dumps(ai_data, indent=2, ensure_ascii=False)}

ВАЖНО: Выбери максимум 3-5 лучших пар для скальпинга.
Верни JSON: {{"pairs": ["BTCUSDT", "ETHUSDT"]}}"""

            # Отправляем в ИИ для быстрого отбора
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
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def _parse_ai_response(self, response: str) -> List[str]:
        """Парсинг ответа ИИ"""
        try:
            # Ищем JSON с парами
            json_match = re.search(r'\{[^}]*"pairs"[^}]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return data.get('pairs', [])
            return []
        except:
            return []

    async def detailed_analysis(self, pair: str) -> Optional[str]:
        """Детальный анализ выбранной пары (200 свечей)"""
        if not self.analysis_prompt:
            return None

        logger.info(f"🔬 ЭТАП 3: Детальный анализ {pair}")

        try:
            # Получаем полные данные
            full_candles = await get_klines_async(pair, "15", limit=SCALPING_CONFIG['candles_for_detailed'])

            if not full_candles or len(full_candles) < 100:
                logger.error(f"❌ Недостаточно данных для {pair}")
                return None

            # Рассчитываем полные индикаторы
            full_indicators = calculate_scalping_indicators(full_candles)

            # Подготавливаем данные для детального анализа
            analysis_data = {
                'pair': pair,
                'timestamp': int(time.time()),
                'current_price': float(full_candles[-1][4]),

                # Полная история (200 свечей)
                'candles_count': len(full_candles),
                'last_20_candles': [
                    {
                        'open': float(c[1]),
                        'high': float(c[2]),
                        'low': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5])
                    } for c in full_candles[-20:]
                ],

                # Полные индикаторы
                'technical_analysis': safe_json_serialize({
                    'tema_trend': {
                        'current_alignment': full_indicators.get('tema_alignment', False),
                        'slope': full_indicators.get('tema_slope', 0),
                        'strength': full_indicators.get('trend_strength', 0)
                    },
                    'momentum': {
                        'rsi': full_indicators.get('rsi_current', 50),
                        'stoch_signal': full_indicators.get('stoch_signal', 'NEUTRAL'),
                        'macd_crossover': full_indicators.get('macd_crossover', 'NONE'),
                        'acceleration': full_indicators.get('momentum_acceleration', 0)
                    },
                    'volume': {
                        'spike_detected': full_indicators.get('volume_spike', False),
                        'ratio': full_indicators.get('volume_ratio', 1.0),
                        'strength': full_indicators.get('volume_strength', 0)
                    },
                    'volatility': {
                        'regime': full_indicators.get('volatility_regime', 'MEDIUM'),
                        'atr_current': full_indicators.get('atr_current', 0),
                        'price_velocity': full_indicators.get('price_velocity', 0)
                    },
                    'levels': {
                        'support': full_indicators.get('support_levels', []),
                        'resistance': full_indicators.get('resistance_levels', []),
                        'near_support': full_indicators.get('near_support', False),
                        'near_resistance': full_indicators.get('near_resistance', False)
                    }
                }),

                'signal_quality': full_indicators.get('signal_quality', 0)
            }

            # Формируем запрос для детального анализа
            message = f"""{self.analysis_prompt}

=== ДЕТАЛЬНЫЙ АНАЛИЗ СКАЛЬПИНГА ===
ПАРА: {pair}
СТРАТЕГИЯ: Удержание 3-4 свечи на 15M
ТЕКУЩАЯ ЦЕНА: {analysis_data['current_price']}

{json.dumps(analysis_data, indent=2, ensure_ascii=False)}

Проанализируй и дай конкретные рекомендации по торговле."""

            # Отправляем в ИИ для детального анализа
            analysis_result = await deep_seek_analysis(message)

            if analysis_result:
                # Сохраняем результат
                self._save_analysis(pair, analysis_result)
                logger.info(f"✅ Анализ {pair} завершен")
                return analysis_result

            return None

        except Exception as e:
            logger.error(f"❌ Ошибка детального анализа {pair}: {e}")
            return None

    def _save_analysis(self, pair: str, analysis: str):
        """Сохранение результата анализа"""
        try:
            with open('scalping_analysis.log', 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"ПАРА: {pair}\n")
                f.write(f"ВРЕМЯ: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"АНАЛИЗ:\n{analysis}\n")
                f.write(f"{'=' * 80}\n")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения: {e}")


async def main():
    """Главная функция оптимизированного скальпингового бота"""
    logger.info("🚀 СКАЛЬПИНГОВЫЙ БОТ 15M - ЗАПУСК")
    logger.info("🎯 Стратегия: Удержание 3-4 свечи")
    logger.info("⚡ Режим: Быстрый анализ + ИИ отбор")

    # Инициализация компонентов
    analyzer = FastScalpingAnalyzer()
    ai_selector = AIScalpingSelector()

    try:
        # ЭТАП 1: Быстрое сканирование всех пар
        promising_signals = await analyzer.mass_scan_markets()

        if not promising_signals:
            logger.info("ℹ️ Качественных сигналов не найдено")
            return

        logger.info(f"📈 Найдено {len(promising_signals)} перспективных сигналов")

        # ЭТАП 2: ИИ отбор лучших (16 свечей)
        selected_pairs = await ai_selector.select_best_pairs(promising_signals)

        if not selected_pairs:
            logger.info("ℹ️ ИИ не выбрал ни одной пары")
            return

        logger.info(f"🤖 ИИ выбрал {len(selected_pairs)} пар: {selected_pairs}")

        # ЭТАП 3: Детальный анализ каждой выбранной пары (200 свечей)
        successful_analyses = 0

        for pair in selected_pairs:
            analysis = await ai_selector.detailed_analysis(pair)

            if analysis:
                successful_analyses += 1
                logger.info(f"✅ {pair} - анализ завершен")
            else:
                logger.error(f"❌ {pair} - ошибка анализа")

            # Пауза между запросами к ИИ
            await asyncio.sleep(1)

        # ИТОГИ
        logger.info(f"\n🎉 АНАЛИЗ ЗАВЕРШЕН!")
        logger.info(f"📊 Отсканировано пар: много")
        logger.info(f"🎯 Найдено сигналов: {len(promising_signals)}")
        logger.info(f"🤖 ИИ выбрал: {len(selected_pairs)}")
        logger.info(f"📋 Успешных анализов: {successful_analyses}")
        logger.info(f"📁 Результаты: scalping_analysis.log")

        # Очищаем HTTP клиент при завершении
        await cleanup_http_client()

    except KeyboardInterrupt:
        logger.info("⏹️ Остановка по запросу пользователя")
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🎯 СКАЛЬПИНГОВЫЙ БОТ - ОПТИМИЗИРОВАННАЯ ВЕРСИЯ")
    logger.info("📊 Удержание: 3-4 свечи на 15M")
    logger.info("⚡ Быстрые индикаторы + ИИ анализ")
    logger.info("=" * 80)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Программа остановлена")
    except Exception as e:
        logger.error(f"💥 Фатальная ошибка: {e}")
    finally:
        logger.info("🔚 Работа завершена")