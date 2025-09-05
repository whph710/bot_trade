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

# КОНФИГУРАЦИЯ
SCALPING_CONFIG = {
    'candles_for_scan': 50,
    'batch_size': 30,
    'min_confidence': 75,
    'pairs_per_ai_batch': 50,  # Максимум пар в одной партии для ИИ
    'best_pairs_from_batch': 5,  # Сколько лучших выбираем из партии
    'final_pairs_count': 5,  # Итоговое количество пар для полного анализа
    'forbidden_hours': [22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
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
        logger.info("Упрощенный скальпинговый анализатор запущен (3 индикатора)")

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
                logger.warning(f"Ошибка создания сигнала {symbol}: {e}")
                return None

        except Exception as e:
            logger.error(f"Ошибка сканирования {symbol}: {e}")
            return None

    async def mass_scan_markets(self) -> List[SimpleScalpingSignal]:
        """Массовое сканирование рынков"""
        start_time = time.time()
        logger.info("ЭТАП 1: Быстрое сканирование всех USDT пар")

        try:
            pairs = await get_usdt_trading_pairs()
            if not pairs:
                logger.error("Не удалось получить список пар")
                return []

            logger.info(f"Сканируем {len(pairs)} USDT пар")

            signals_with_entries = []

            # Обрабатываем батчами
            for i in range(0, len(pairs), SCALPING_CONFIG['batch_size']):
                batch = pairs[i:i + SCALPING_CONFIG['batch_size']]

                tasks = [self.quick_scan_pair(pair) for pair in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Собираем результаты
                for result in results:
                    if isinstance(result, SimpleScalpingSignal):
                        signals_with_entries.append(result)
                    elif isinstance(result, Exception):
                        logger.debug(f"Исключение в батче: {result}")

                # Логируем прогресс
                processed = min(i + SCALPING_CONFIG['batch_size'], len(pairs))
                logger.info(f"Обработано: {processed}/{len(pairs)}")

                # Пауза между батчами
                await asyncio.sleep(0.2)

            # Сортируем по уверенности
            signals_with_entries.sort(key=lambda x: x.confidence, reverse=True)

            execution_time = time.time() - start_time
            logger.info(f"ЭТАП 1 завершен: {len(signals_with_entries)} сигналов за {execution_time:.2f}сек")

            return signals_with_entries

        except Exception as e:
            logger.error(f"Критическая ошибка сканирования: {e}")
            return []


class AIAnalyzer:
    """Анализатор с ИИ для отбора и анализа"""

    def __init__(self):
        self.quick_prompt = self._load_prompt('prompt2.txt')  # Быстрый анализ
        self.full_prompt = self._load_prompt('prompt.txt')  # Полный анализ

    def _load_prompt(self, filename: str) -> str:
        """Загрузка промпта"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error(f"Файл {filename} не найден")
            return "Анализируй торговые сигналы."

    def _prepare_signals_for_ai(self, signals: List[SimpleScalpingSignal]) -> Dict[str, Any]:
        """Подготовка данных сигналов для ИИ"""
        prepared_signals = []

        for signal in signals:
            try:
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

    def _parse_ai_pairs_response(self, response: str) -> List[str]:
        """Парсинг ответа ИИ с парами"""
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

    async def process_signals_in_batches(self, signals: List[SimpleScalpingSignal]) -> List[str]:
        """Обработка сигналов партиями через ИИ"""
        logger.info(f"ЭТАП 2: Обработка {len(signals)} сигналов партиями по {SCALPING_CONFIG['pairs_per_ai_batch']}")

        all_selected_pairs = []

        # Разбиваем на партии
        for i in range(0, len(signals), SCALPING_CONFIG['pairs_per_ai_batch']):
            batch = signals[i:i + SCALPING_CONFIG['pairs_per_ai_batch']]
            batch_num = (i // SCALPING_CONFIG['pairs_per_ai_batch']) + 1

            logger.info(f"Обрабатывается партия {batch_num}: {len(batch)} сигналов")

            try:
                # Подготавливаем данные для ИИ
                ai_data = self._prepare_signals_for_ai(batch)

                # Проверяем JSON сериализацию
                if not test_json_serialization(ai_data):
                    logger.error(f"Данные партии {batch_num} не сериализуются в JSON")
                    continue

                # Формируем запрос с быстрым промптом
                message = f"""{self.quick_prompt}

=== ПАРТИЯ {batch_num} ===
СИСТЕМА: 3 индикатора (EMA + RSI + Volume)
СИГНАЛОВ В ПАРТИИ: {len(batch)}

{json.dumps(ai_data, indent=2, ensure_ascii=False)}

ВАЖНО: Выбери максимум {SCALPING_CONFIG['best_pairs_from_batch']} лучших пар из этой партии.
Верни JSON: {{"pairs": ["BTCUSDT", "ETHUSDT"]}}"""

                # Отправляем в ИИ
                ai_response = await deep_seek_selection(message)

                if not ai_response:
                    logger.error(f"ИИ не ответил для партии {batch_num}")
                    continue

                # Парсим ответ
                selected_pairs = self._parse_ai_pairs_response(ai_response)

                if selected_pairs:
                    all_selected_pairs.extend(selected_pairs)
                    logger.info(f"Партия {batch_num}: выбрано {len(selected_pairs)} пар")
                else:
                    logger.info(f"Партия {batch_num}: пары не выбраны")

                # Пауза между партиями
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Ошибка обработки партии {batch_num}: {e}")
                continue

        logger.info(f"ЭТАП 2 завершен: всего отобрано {len(all_selected_pairs)} пар")
        return all_selected_pairs

    async def final_selection(self, pairs: List[str]) -> List[str]:
        """Финальный отбор лучших пар"""
        if len(pairs) <= SCALPING_CONFIG['final_pairs_count']:
            return pairs

        logger.info(f"ЭТАП 3: Финальный отбор из {len(pairs)} пар")

        try:
            # Получаем свежие данные для финального отбора
            fresh_signals = []

            for pair in pairs:
                try:
                    candles = await get_klines_async(pair, "5", limit=50)
                    if candles and len(candles) >= 30:
                        signal_result = detect_scalping_entry(candles)
                        if signal_result['signal'] != 'NO_SIGNAL':
                            entry_price = safe_float(candles[-1][4])
                            confidence = safe_int(signal_result['confidence'])
                            indicators = signal_result.get('indicators', {})

                            fresh_signals.append(SimpleScalpingSignal(
                                pair=pair,
                                signal_type=signal_result['signal'],
                                confidence=confidence,
                                entry_price=entry_price,
                                timestamp=int(time.time()),
                                ema_signal=str(indicators.get('ema_signal', 'NEUTRAL')),
                                rsi_value=safe_float(indicators.get('rsi_value', 50)),
                                volume_spike=bool(indicators.get('volume_spike', False)),
                                entry_reasons=signal_result.get('entry_reasons', []),
                                candles_data=candles[-20:],
                                indicators_data=indicators
                            ))
                except Exception as e:
                    logger.warning(f"Ошибка получения свежих данных для {pair}: {e}")
                    continue

            if not fresh_signals:
                logger.error("Нет свежих сигналов для финального отбора")
                return []

            # Подготавливаем данные для финального ИИ анализа
            ai_data = self._prepare_signals_for_ai(fresh_signals)

            if not test_json_serialization(ai_data):
                logger.error("Данные финального отбора не сериализуются в JSON")
                return []

            # Формируем запрос для финального отбора
            message = f"""{self.quick_prompt}

=== ФИНАЛЬНЫЙ ОТБОР ===
СИСТЕМА: 3 индикатора (EMA + RSI + Volume)
КАНДИДАТОВ: {len(fresh_signals)}

{json.dumps(ai_data, indent=2, ensure_ascii=False)}

ВАЖНО: Выбери ТОЛЬКО {SCALPING_CONFIG['final_pairs_count']} ЛУЧШИХ пар для полного анализа.
Верни JSON: {{"pairs": ["BTCUSDT", "ETHUSDT"]}}"""

            # Отправляем в ИИ
            ai_response = await deep_seek_selection(message)

            if not ai_response:
                logger.error("ИИ не ответил для финального отбора")
                return []

            # Парсим ответ
            final_pairs = self._parse_ai_pairs_response(ai_response)

            logger.info(f"ЭТАП 3 завершен: финально выбрано {len(final_pairs)} пар")
            return final_pairs

        except Exception as e:
            logger.error(f"Ошибка финального отбора: {e}")
            return []

    async def detailed_analysis(self, pair: str) -> Optional[str]:
        """Детальный анализ с полным промптом"""
        logger.info(f"ЭТАП 4: Детальный анализ {pair}")

        try:
            # Получаем больше данных для анализа
            full_candles = await get_klines_async(pair, "15", limit=100)

            if not full_candles or len(full_candles) < 50:
                logger.error(f"Недостаточно данных для {pair}")
                return None

            # Рассчитываем индикаторы
            indicators = calculate_simplified_indicators(full_candles)

            if not indicators:
                logger.error(f"Не удалось рассчитать индикаторы для {pair}")
                return None

            # Подготавливаем данные для анализа
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
                logger.error(f"Данные анализа {pair} не сериализуются")
                return None

            # Формируем запрос для полного анализа
            message = f"""{self.full_prompt}

=== ДЕТАЛЬНЫЙ АНАЛИЗ ===
ПАРА: {pair}
СИСТЕМА: EMA + RSI + Volume
ЦЕНА: {analysis_data['current_price']}

{json.dumps(analysis_data, indent=2, ensure_ascii=False)}

Дай конкретные торговые рекомендации."""

            # Отправляем в ИИ
            analysis_result = await deep_seek_analysis(message)

            if analysis_result:
                self._save_analysis(pair, analysis_result)
                logger.info(f"Анализ {pair} завершен")
                return analysis_result

            return None

        except Exception as e:
            logger.error(f"Ошибка детального анализа {pair}: {e}")
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
            logger.error(f"Ошибка сохранения: {e}")


async def main():
    """Главная функция с новой логикой"""
    logger.info("УПРОЩЕННЫЙ СКАЛЬПИНГОВЫЙ БОТ 15M - ЗАПУСК")
    logger.info("Система: EMA + RSI + Volume (3 индикатора)")

    # Инициализация компонентов
    analyzer = SimplifiedScalpingAnalyzer()
    ai_analyzer = AIAnalyzer()

    try:
        # ЭТАП 1: Сканирование всех USDT пар через индикаторы
        all_signals = await analyzer.mass_scan_markets()

        if not all_signals:
            logger.info("Качественных сигналов не найдено")
            return

        logger.info(f"Найдено {len(all_signals)} сигналов с торговыми входами")

        # ЭТАП 2: Обработка сигналов партиями через ИИ (быстрый анализ)
        selected_pairs = await ai_analyzer.process_signals_in_batches(all_signals)

        if not selected_pairs:
            logger.info("ИИ не выбрал ни одной пары из всех партий")
            return

        logger.info(f"Отобрано из всех партий: {selected_pairs}")

        # ЭТАП 3: Финальный отбор лучших пар (если их больше 5)
        final_pairs = await ai_analyzer.final_selection(selected_pairs)

        if not final_pairs:
            logger.info("Финальный отбор не дал результатов")
            return

        logger.info(f"Финально выбраны пары: {final_pairs}")

        # ЭТАП 4: Детальный анализ каждой финальной пары
        successful_analyses = 0

        for pair in final_pairs:
            logger.info(f"Анализирую {pair}...")

            analysis = await ai_analyzer.detailed_analysis(pair)

            if analysis:
                successful_analyses += 1
                logger.info(f"{pair} - анализ завершен")
            else:
                logger.error(f"{pair} - ошибка анализа")

            # Пауза между анализами
            await asyncio.sleep(2)

        # ИТОГИ
        logger.info("АНАЛИЗ ЗАВЕРШЕН")
        logger.info(f"Найдено сигналов: {len(all_signals)}")
        logger.info(f"Отобрано партиями: {len(selected_pairs)}")
        logger.info(f"Финально выбрано: {len(final_pairs)}")
        logger.info(f"Успешных анализов: {successful_analyses}")
        logger.info("Результаты: simple_scalping_analysis.log")

        # Очищаем HTTP клиент
        await cleanup_http_client()

    except KeyboardInterrupt:
        logger.info("Остановка по запросу пользователя")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await cleanup_http_client()


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("УПРОЩЕННЫЙ СКАЛЬПИНГОВЫЙ БОТ")
    logger.info("Система: EMA + RSI + Volume")
    logger.info("Удержание: 3-4 свечи на 15M")
    logger.info("=" * 80)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Программа остановлена")
    except Exception as e:
        logger.error(f"Фатальная ошибка: {e}")
    finally:
        logger.info("Работа завершена")