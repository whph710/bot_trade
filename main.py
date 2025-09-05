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
from func_async import get_klines_async, get_usdt_trading_pairs, filter_high_volume_pairs
from deepseek import deep_seek_selection, deep_seek_analysis, cleanup_http_client

# НОВЫЕ ИМПОРТЫ (заменяют старые из func_trade)
from func_trade import detect_scalping_signal, calculate_scalping_indicators

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scalping_5m_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ЭКСТРЕМАЛЬНАЯ КОНФИГУРАЦИЯ ДЛЯ 5M СКАЛЬПИНГА
SCALPING_CONFIG = {
    'candles_for_scan': 40,  # Уменьшено для 5M (40 свечей = 3.3 часа)
    'candles_for_analysis': 20,  # Для первичного отбора ИИ (20 свечей = 1.7 часа)
    'candles_for_detailed': 100,  # Для детального анализа (100 свечей = 8.3 часа)
    'batch_size': 30,  # Уменьшен для быстроты
    'min_confidence': 80,  # ПОВЫШЕН для качества 5M сигналов
    'max_pairs_to_ai': 8,  # Ограничено для скорости
    'min_volume_24h': 100_000_000,  # Минимум $100M оборота
    'forbidden_hours': [22, 23, 0, 1, 2, 3, 4],  # Низкая ликвидность UTC (сокращено)
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
    """Оптимизированный торговый сигнал для 5M скальпинга"""
    pair: str
    signal_type: str  # 'LONG', 'SHORT', 'NO_SIGNAL'
    confidence: int
    entry_price: float
    timestamp: int

    # Ключевые метрики для 5M
    quality_score: int
    volatility_regime: str
    volume_confirmed: bool
    momentum_strength: int
    entry_reasons: List[str]

    # Для ИИ анализа
    candles_data: List = None
    indicators_data: Dict = None


class Extreme5MAnalyzer:
    """Экстремальный анализатор для 5M скальпинга - максимальный профит"""

    def __init__(self):
        self.session_start = time.time()
        logger.info("🚀 ЭКСТРЕМАЛЬНЫЙ 5M СКАЛЬПИНГОВЫЙ АНАЛИЗАТОР ЗАПУЩЕН")
        logger.info("⚡ ЦЕЛЬ: МАКСИМАЛЬНЫЙ ПРОФИТ НА 5-МИНУТНЫХ СВЕЧАХ")

    def is_trading_hours(self) -> bool:
        """Проверка торговых часов (более гибкая для 5M)"""
        current_hour = datetime.datetime.utcnow().hour
        return current_hour not in SCALPING_CONFIG['forbidden_hours']

    async def quick_scan_pair(self, symbol: str) -> Optional[ScalpingSignal]:
        """Молниеносное сканирование одной пары для 5M"""
        try:
            # Получаем свечи для быстрого анализа
            candles = await get_klines_async(
                symbol,
                interval="5",  # 5-минутные свечи
                limit=SCALPING_CONFIG['candles_for_scan']
            )

            if not candles or len(candles) < 25:
                return None

            # Определяем сигнал с помощью экстремальной функции
            signal_result = detect_scalping_signal(candles)

            if signal_result['signal'] == 'NO_SIGNAL':
                return None

            # Создаем сигнал с проверкой значений
            try:
                entry_price = float(candles[-1][4])
                confidence = int(signal_result['confidence'])
                quality_score = int(signal_result.get('quality_score', 0))
                momentum_strength = int(signal_result.get('momentum_strength', 0))

                # Проверяем на валидность
                if math.isnan(entry_price) or math.isnan(confidence):
                    logger.warning(f"❌ NaN значения в {symbol}")
                    return None

                # Дополнительная фильтрация для 5M
                if confidence < SCALPING_CONFIG['min_confidence']:
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
                momentum_strength=momentum_strength,
                entry_reasons=signal_result.get('entry_reasons', []),
                candles_data=candles[-SCALPING_CONFIG['candles_for_analysis']:],
                indicators_data=clean_value(signal_result.get('indicators', {}))
            )

        except Exception as e:
            logger.error(f"❌ Ошибка сканирования {symbol}: {e}")
            return None

    async def mass_scan_markets(self) -> List[ScalpingSignal]:
        """Массовое сканирование рынков с предварительной фильтрацией"""
        # if not self.is_trading_hours():
        #     logger.warning("⏰ Неторговые часы - пропускаем сканирование")
        #     return []

        start_time = time.time()
        logger.info("🔍 ЭТАП 1: ЭКСТРЕМАЛЬНОЕ 5M СКАНИРОВАНИЕ")

        try:
            # Получаем список пар
            all_pairs = await get_usdt_trading_pairs()
            if not all_pairs:
                logger.error("❌ Не удалось получить список пар")
                return []

            # НОВИНКА: Предварительная фильтрация по объему
            high_volume_pairs = await filter_high_volume_pairs(all_pairs, SCALPING_CONFIG['min_volume_24h'])

            if not high_volume_pairs:
                logger.error("❌ Нет высоколиквидных пар")
                return []

            logger.info(f"📊 Сканируем {len(high_volume_pairs)} высоколиквидных пар (из {len(all_pairs)})")

            # Обрабатываем батчами
            promising_signals = []

            for i in range(0, len(high_volume_pairs), SCALPING_CONFIG['batch_size']):
                batch = high_volume_pairs[i:i + SCALPING_CONFIG['batch_size']]

                # Создаем задачи для батча
                tasks = [self.quick_scan_pair(pair) for pair in batch]

                # Выполняем параллельно
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Собираем результаты
                for result in results:
                    if isinstance(result, ScalpingSignal):
                        promising_signals.append(result)

                # Логируем прогресс
                processed = min(i + SCALPING_CONFIG['batch_size'], len(high_volume_pairs))
                logger.info(f"⏳ Обработано: {processed}/{len(high_volume_pairs)}")

                # Быстрая пауза между батчами
                if i + SCALPING_CONFIG['batch_size'] < len(high_volume_pairs):
                    await asyncio.sleep(0.05)  # Уменьшена для 5M

            # Сортируем по уверенности, затем по momentum
            promising_signals.sort(key=lambda x: (x.confidence, x.momentum_strength), reverse=True)

            execution_time = time.time() - start_time
            logger.info(f"✅ ЭТАП 1 завершен: {len(promising_signals)} сигналов за {execution_time:.2f}сек")

            # Логируем ТОП-5 сигналов
            if promising_signals:
                logger.info("🏆 ТОП-5 СИГНАЛОВ:")
                for i, signal in enumerate(promising_signals[:5], 1):
                    logger.info(
                        f"  {i}. {signal.pair} - {signal.signal_type} ({signal.confidence}% conf, {signal.momentum_strength}% momentum)")

            return promising_signals

        except Exception as e:
            logger.error(f"❌ Критическая ошибка сканирования: {e}")
            return []


class AI5MSelector:
    """ИИ селектор для 5M скальпинга"""

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
        """Подготовка данных для ИИ анализа (20 свечей + экстремальные индикаторы)"""
        prepared_data = []

        for signal in signals:
            # Берем только последние 20 свечей для 5M
            recent_candles = signal.candles_data[-20:] if signal.candles_data else []

            signal_data = {
                'pair': signal.pair,
                'signal_type': signal.signal_type,
                'confidence': int(signal.confidence),
                'entry_price': float(signal.entry_price),
                'quality_score': int(signal.quality_score),
                'volatility_regime': str(signal.volatility_regime),
                'volume_confirmed': bool(signal.volume_confirmed),
                'momentum_strength': int(signal.momentum_strength),
                'entry_reasons': [str(reason) for reason in signal.entry_reasons],

                # 20 последних 5M свечей
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

                # Экстремальные индикаторы (последние 20 значений)
                'indicators': safe_json_serialize({
                    # TSI - король momentum
                    'tsi_values': signal.indicators_data.get('tsi_values', [])[-20:],
                    'tsi_current': signal.indicators_data.get('tsi_current', 0.0),
                    'tsi_bullish': signal.indicators_data.get('tsi_bullish', False),
                    'tsi_bearish': signal.indicators_data.get('tsi_bearish', False),

                    # Linda Raschke MACD
                    'macd_line': signal.indicators_data.get('macd_line', [])[-20:],
                    'macd_signal': signal.indicators_data.get('macd_signal', [])[-20:],
                    'macd_histogram': signal.indicators_data.get('macd_histogram', [])[-20:],
                    'macd_crossover': signal.indicators_data.get('macd_crossover', 'NONE'),

                    # VWAP
                    'vwap_values': signal.indicators_data.get('vwap_values', [])[-20:],
                    'vwap_position': signal.indicators_data.get('vwap_position', 'BELOW'),
                    'vwap_distance': signal.indicators_data.get('vwap_distance', 0.0),

                    # DeMarker
                    'demarker_values': signal.indicators_data.get('demarker_values', [])[-20:],
                    'demarker_current': signal.indicators_data.get('demarker_current', 0.5),
                    'demarker_signal': signal.indicators_data.get('demarker_signal', 'NEUTRAL'),

                    # Быстрый RSI
                    'rsi_values': signal.indicators_data.get('rsi_values', [])[-20:],
                    'rsi_current': signal.indicators_data.get('rsi_current', 50.0),

                    # EMA тренд
                    'ema_fast': signal.indicators_data.get('ema_fast', [])[-20:],
                    'ema_slow': signal.indicators_data.get('ema_slow', [])[-20:],
                    'ema_trend': signal.indicators_data.get('ema_trend', 'NEUTRAL'),
                    'ema_distance': signal.indicators_data.get('ema_distance', 0.0),

                    # Волатильность
                    'atr_values': signal.indicators_data.get('atr_values', [])[-20:],
                    'atr_current': signal.indicators_data.get('atr_current', 0.0),

                    # Объемы (усиленный анализ)
                    'volume_ratio': signal.indicators_data.get('volume_ratio', 1.0),
                    'volume_strength': signal.indicators_data.get('volume_strength', 0),
                    'volume_acceleration': signal.indicators_data.get('volume_acceleration', 0.0),

                    # Моментум
                    'price_momentum': signal.indicators_data.get('price_momentum', 0.0)
                })
            }

            prepared_data.append(signal_data)

        return {
            'signals_count': len(prepared_data),
            'timeframe': '5m',  # ИЗМЕНЕНО на 5M
            'strategy': 'extreme_5m_scalping',  # ИЗМЕНЕНО
            'timestamp': int(time.time()),
            'signals': prepared_data
        }

    async def select_best_pairs(self, signals: List[ScalpingSignal]) -> List[str]:
        """Первичный отбор через ИИ (20 свечей 5M)"""
        if not self.selection_prompt or not signals:
            return []

        logger.info(f"🤖 ЭТАП 2: ИИ отбор из {len(signals)} 5M сигналов")

        try:
            # Ограничиваем количество сигналов для ИИ
            top_signals = signals[:SCALPING_CONFIG['max_pairs_to_ai']]

            # Подготавливаем данные
            ai_data = self._prepare_signals_for_ai(top_signals)

            # Формируем запрос
            message = f"""{self.selection_prompt}

=== ЭКСТРЕМАЛЬНЫЙ 5M СКАЛЬПИНГ: ПЕРВИЧНЫЙ ОТБОР ===
КОЛИЧЕСТВО СИГНАЛОВ: {len(top_signals)}
УДЕРЖАНИЕ: 1-2 свечи (5-10 минут)
ДАННЫЕ: Последние 20 свечей 5M + экстремальные индикаторы

{json.dumps(ai_data, indent=2, ensure_ascii=False)}

КРИТИЧЕСКИ ВАЖНО: Выбери максимум 3 САМЫХ ЛУЧШИХ пары для 5M скальпинга.
Учитывай TSI momentum, MACD пересечения, VWAP позицию, DeMarker экстремумы.
Верни JSON: {{"pairs": ["BTCUSDT", "ETHUSDT"]}}"""

            # Отправляем в ИИ для быстрого отбора
            ai_response = await deep_seek_selection(message)

            if not ai_response:
                logger.error("❌ ИИ не ответил")
                return []

            # Парсим ответ
            selected_pairs = self._parse_ai_response(ai_response)

            logger.info(f"✅ ЭТАП 2 завершен: ИИ выбрал {len(selected_pairs)} пар для 5M")
            if selected_pairs:
                logger.info(f"🎯 ВЫБРАННЫЕ ПАРЫ: {', '.join(selected_pairs)}")

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
        """Детальный анализ выбранной пары (100 свечей 5M)"""
        if not self.analysis_prompt:
            return None

        logger.info(f"🔬 ЭТАП 3: Детальный 5M анализ {pair}")

        try:
            # Получаем полные данные (100 свечей = 8.3 часа истории)
            full_candles = await get_klines_async(pair, "5", limit=SCALPING_CONFIG['candles_for_detailed'])

            if not full_candles or len(full_candles) < 50:
                logger.error(f"❌ Недостаточно 5M данных для {pair}")
                return None

            # Рассчитываем полные индикаторы
            full_indicators = calculate_scalping_indicators(full_candles)

            # Подготавливаем данные для детального анализа
            analysis_data = {
                'pair': pair,
                'timeframe': '5M',
                'timestamp': int(time.time()),
                'current_price': float(full_candles[-1][4]),

                # Полная история 5M свечей
                'candles_count': len(full_candles),
                'last_30_candles': [  # Увеличено для 5M
                    {
                        'open': float(c[1]),
                        'high': float(c[2]),
                        'low': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5])
                    } for c in full_candles[-30:]
                ],

                # Полные экстремальные индикаторы
                'technical_analysis': safe_json_serialize({
                    'tsi_analysis': {
                        'current_value': full_indicators.get('tsi_current', 0.0),
                        'bullish_momentum': full_indicators.get('tsi_bullish', False),
                        'bearish_momentum': full_indicators.get('tsi_bearish', False)
                    },
                    'macd_analysis': {
                        'crossover': full_indicators.get('macd_crossover', 'NONE'),
                        'current_histogram': full_indicators.get('macd_histogram', [])[-1] if full_indicators.get(
                            'macd_histogram') else 0
                    },
                    'vwap_analysis': {
                        'position': full_indicators.get('vwap_position', 'BELOW'),
                        'distance_percent': full_indicators.get('vwap_distance', 0.0),
                        'fair_value': full_indicators.get('vwap_values', [])[-1] if full_indicators.get(
                            'vwap_values') else 0
                    },
                    'demarker_analysis': {
                        'current_value': full_indicators.get('demarker_current', 0.5),
                        'signal': full_indicators.get('demarker_signal', 'NEUTRAL'),
                        'reversal_potential': 'HIGH' if full_indicators.get('demarker_current',
                                                                            0.5) < 0.3 or full_indicators.get(
                            'demarker_current', 0.5) > 0.7 else 'LOW'
                    },
                    'ema_trend': {
                        'direction': full_indicators.get('ema_trend', 'NEUTRAL'),
                        'strength_percent': full_indicators.get('ema_distance', 0.0)
                    },
                    'volume': {
                        'spike_detected': full_indicators.get('volume_spike', False),
                        'ratio': full_indicators.get('volume_ratio', 1.0),
                        'strength': full_indicators.get('volume_strength', 0),
                        'acceleration': full_indicators.get('volume_acceleration', 0.0)
                    },
                    'volatility': {
                        'regime': full_indicators.get('volatility_regime', 'MEDIUM'),
                        'atr_current': full_indicators.get('atr_current', 0),
                        'atr_percent': (
                                    full_indicators.get('atr_current', 0) / float(full_candles[-1][4]) * 100) if float(
                            full_candles[-1][4]) > 0 else 0
                    },
                    'momentum': {
                        'price_momentum': full_indicators.get('price_momentum', 0.0),
                        'strength': full_indicators.get('momentum_strength', 0)
                    }
                }),

                'signal_quality': full_indicators.get('signal_quality', 0),
                'recommended_hold_time': '5-15 minutes',  # Для 5M скальпинга
                'target_profit': '0.3-0.8%'  # Реалистичные цели для 5M
            }

            # Формируем запрос для детального анализа
            message = f"""{self.analysis_prompt}

=== ДЕТАЛЬНЫЙ 5M СКАЛЬПИНГ АНАЛИЗ ===
ПАРА: {pair}
СТРАТЕГИЯ: Удержание 1-2 свечи на 5M (5-10 минут)
ТЕКУЩАЯ ЦЕНА: {analysis_data['current_price']}
ЦЕЛЕВАЯ ПРИБЫЛЬ: 0.3-0.8% за 5-15 минут

{json.dumps(analysis_data, indent=2, ensure_ascii=False)}

КРИТИЧНО: Учти что это 5-минутный скальпинг:
- Спреды должны быть <0.05%
- Объем критически важен
- TSI momentum приоритетен
- VWAP как справедливая цена
- DeMarker для точек разворота

Проанализируй и дай точный план сделки с учётом микроструктуры 5M рынка."""

            # Отправляем в ИИ для детального анализа
            analysis_result = await deep_seek_analysis(message)

            if analysis_result:
                # Сохраняем результат
                self._save_analysis(pair, analysis_result)
                logger.info(f"✅ 5M анализ {pair} завершен")
                return analysis_result

            return None

        except Exception as e:
            logger.error(f"❌ Ошибка детального 5M анализа {pair}: {e}")
            return None

    def _save_analysis(self, pair: str, analysis: str):
        """Сохранение результата анализа"""
        try:
            with open('scalping_5m_analysis.log', 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"ПАРА: {pair} (5M СКАЛЬПИНГ)\n")
                f.write(f"ВРЕМЯ: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"АНАЛИЗ:\n{analysis}\n")
                f.write(f"{'=' * 80}\n")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения: {e}")


async def main():
    """Главная функция экстремального 5M скальпингового бота"""
    logger.info("🚀 ЭКСТРЕМАЛЬНЫЙ 5M СКАЛЬПИНГОВЫЙ БОТ - ЗАПУСК")
    logger.info("🎯 Стратегия: Удержание 1-2 свечи (5-10 минут)")
    logger.info("⚡ Режим: Экстремальные индикаторы + ИИ отбор")
    logger.info("💰 Цель: МАКСИМАЛЬНЫЙ ПРОФИТ на 5-минутках")

    # Инициализация компонентов
    analyzer = Extreme5MAnalyzer()
    ai_selector = AI5MSelector()

    try:
        # ЭТАП 1: Экстремальное сканирование всех высоколиквидных пар
        promising_signals = await analyzer.mass_scan_markets()

        if not promising_signals:
            logger.info("ℹ️ Качественных 5M сигналов не найдено")
            return

        logger.info(f"📈 Найдено {len(promising_signals)} перспективных 5M сигналов")

        # ЭТАП 2: ИИ отбор лучших (20 свечей 5M)
        selected_pairs = await ai_selector.select_best_pairs(promising_signals)

        if not selected_pairs:
            logger.info("ℹ️ ИИ не выбрал ни одной пары для 5M")
            return

        logger.info(f"🤖 ИИ выбрал {len(selected_pairs)} пар для 5M: {selected_pairs}")

        # ЭТАП 3: Детальный анализ каждой выбранной пары (100 свечей 5M)
        successful_analyses = 0

        for pair in selected_pairs:
            analysis = await ai_selector.detailed_analysis(pair)

            if analysis:
                successful_analyses += 1
                logger.info(f"✅ {pair} - 5M анализ завершен")
            else:
                logger.error(f"❌ {pair} - ошибка 5M анализа")

            # Короткая пауза между запросами к ИИ для 5M
            await asyncio.sleep(0.8)

        # ИТОГИ
        logger.info(f"\n🎉 5M АНАЛИЗ ЗАВЕРШЕН!")
        logger.info(f"📊 Отсканировано высоколиквидных пар: много")
        logger.info(f"🎯 Найдено 5M сигналов: {len(promising_signals)}")
        logger.info(f"🤖 ИИ выбрал для 5M: {len(selected_pairs)}")
        logger.info(f"📋 Успешных 5M анализов: {successful_analyses}")
        logger.info(f"📁 Результаты: scalping_5m_analysis.log")

        # Показываем ТОП сигналы
        if promising_signals:
            logger.info(f"\n🏆 ТОП-3 СИГНАЛА:")
            for i, signal in enumerate(promising_signals[:3], 1):
                reasons_str = ', '.join(signal.entry_reasons[:3])  # Первые 3 причины
                logger.info(f"  {i}. {signal.pair}: {signal.signal_type} "
                            f"(Conf: {signal.confidence}%, Mom: {signal.momentum_strength}%, Vol: {'✓' if signal.volume_confirmed else '✗'})")
                logger.info(f"     Причины: {reasons_str}")

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
    logger.info("🎯 ЭКСТРЕМАЛЬНЫЙ 5M СКАЛЬПИНГОВЫЙ БОТ")
    logger.info("📊 Удержание: 1-2 свечи (5-10 минут)")
    logger.info("⚡ TSI + MACD + VWAP + DeMarker + объемы")
    logger.info("💰 ЦЕЛЬ: МАКСИМАЛЬНЫЙ ПРОФИТ")
    logger.info("=" * 80)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Программа остановлена")
    except Exception as e:
        logger.error(f"💥 Фатальная ошибка: {e}")
    finally:
        logger.info("🔚 Работа завершена")