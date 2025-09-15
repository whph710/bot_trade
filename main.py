"""
Оптимизированный скальпинговый бот с кешированием данных и улучшенным логированием
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from config import config, has_api_key
from func_async import get_trading_pairs, fetch_klines, batch_fetch_klines, cleanup as cleanup_api
from func_trade import calculate_basic_indicators, calculate_ai_indicators, check_basic_signal
from deepseek import ai_select_pairs, ai_analyze_pair

# Упрощенное логирование без избыточной детализации
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class DataCache:
    """Кеш для хранения и переиспользования рыночных данных"""

    def __init__(self):
        self.klines_cache = {}  # {symbol: {interval: klines}}
        self.indicators_cache = {}  # {symbol: {interval: indicators}}

    def cache_klines(self, symbol: str, interval: str, klines: List):
        """Кеширование свечных данных"""
        if symbol not in self.klines_cache:
            self.klines_cache[symbol] = {}
        self.klines_cache[symbol][interval] = klines

    def get_klines(self, symbol: str, interval: str, limit: int) -> Optional[List]:
        """Получение кешированных данных с обрезкой по лимиту"""
        cached = self.klines_cache.get(symbol, {}).get(interval)
        if cached and len(cached) >= limit:
            return cached[-limit:]
        return None

    def cache_indicators(self, symbol: str, interval: str, indicators: Dict):
        """Кеширование индикаторов"""
        if symbol not in self.indicators_cache:
            self.indicators_cache[symbol] = {}
        self.indicators_cache[symbol][interval] = indicators

    def get_indicators(self, symbol: str, interval: str) -> Optional[Dict]:
        """Получение кешированных индикаторов"""
        return self.indicators_cache.get(symbol, {}).get(interval)

    def clear(self):
        """Очистка кеша"""
        self.klines_cache.clear()
        self.indicators_cache.clear()


class OptimizedScalpingBot:
    """Оптимизированный скальпинговый бот с кешированием и улучшенной производительностью"""

    def __init__(self):
        self.processed_pairs = 0
        self.session_start = time.time()
        self.cache = DataCache()
        # Данные для финальной валидации (сохраняем только необходимое)
        self.validation_data = {}

    async def load_initial_data(self, pairs: List[str]) -> Dict[str, Dict]:
        """Предзагрузка всех необходимых данных одним батчем"""
        logger.info(f"Предзагрузка данных для {len(pairs)} пар...")

        # Подготавливаем все запросы сразу
        requests = []
        for pair in pairs:
            requests.extend([
                {'symbol': pair, 'interval': '15', 'limit': max(config.AI_BULK_15M, config.FINAL_15M)},
                {'symbol': pair, 'interval': '5', 'limit': config.FINAL_5M}
            ])

        # Массовая загрузка
        results = await batch_fetch_klines(requests)

        # Кешируем все данные
        loaded_pairs = set()
        for result in results:
            if result.get('success'):
                symbol = result['symbol']
                # Определяем интервал из запроса
                klines = result['klines']
                if len(klines) >= 100:  # Это 5м данные
                    self.cache.cache_klines(symbol, '5', klines)
                    loaded_pairs.add(symbol)
                else:  # Это 15м данные
                    self.cache.cache_klines(symbol, '15', klines)
                    loaded_pairs.add(symbol)

        logger.info(f"Загружены данные по {len(loaded_pairs)} парам")
        return {pair: True for pair in loaded_pairs}

    def calculate_and_cache_indicators(self, symbol: str, interval: str, klines: List, history_length: int) -> Optional[Dict]:
        """Расчет и кеширование индикаторов"""
        # Проверяем кеш
        cached = self.cache.get_indicators(symbol, interval)
        if cached:
            return cached

        # Рассчитываем индикаторы
        if interval == '5' and history_length > 20:
            indicators = calculate_ai_indicators(klines, history_length)
        elif interval == '15' and history_length > 20:
            indicators = calculate_ai_indicators(klines, history_length)
        else:
            indicators = calculate_basic_indicators(klines)

        # Кешируем результат
        if indicators:
            self.cache.cache_indicators(symbol, interval, indicators)

        return indicators

    async def stage1_filter_signals(self) -> List[Dict]:
        """ЭТАП 1: Оптимизированная фильтрация пар"""
        start_time = time.time()
        logger.info("ЭТАП 1: Фильтрация пар с сигналами")

        # Получаем пары
        pairs = await get_trading_pairs()
        if not pairs:
            logger.error("Не удалось получить торговые пары")
            return []

        # Предзагружаем все данные
        await self.load_initial_data(pairs)

        pairs_with_signals = []
        processed = 0
        errors = 0

        logger.info(f"Обработка {len(pairs)} пар...")

        # Обрабатываем кешированные данные
        for symbol in pairs:
            klines_15m = self.cache.get_klines(symbol, '15', config.QUICK_SCAN_15M)
            if not klines_15m or len(klines_15m) < 20:
                errors += 1
                continue

            try:
                indicators = self.calculate_and_cache_indicators(symbol, '15', klines_15m, 20)
                if not indicators:
                    errors += 1
                    continue

                signal_check = check_basic_signal(indicators)
                if signal_check['signal'] and signal_check['confidence'] >= config.MIN_CONFIDENCE:
                    pair_data = {
                        'symbol': symbol,
                        'confidence': signal_check['confidence'],
                        'direction': signal_check['direction'],
                        'base_indicators': indicators
                    }
                    pairs_with_signals.append(pair_data)

                processed += 1

            except Exception as e:
                errors += 1
                continue

        # Сортируем по уверенности
        pairs_with_signals.sort(key=lambda x: x['confidence'], reverse=True)

        elapsed = time.time() - start_time
        self.processed_pairs = processed

        logger.info(f"ЭТАП 1 завершен: обработано {processed} пар, найдено {len(pairs_with_signals)} сигналов, ошибок {errors}, время {elapsed:.1f}с")
        return pairs_with_signals

    async def stage2_ai_bulk_select(self, signal_pairs: List[Dict]) -> List[str]:
        """ЭТАП 2: Оптимизированный ИИ отбор"""
        start_time = time.time()
        logger.info(f"ЭТАП 2: ИИ анализ {len(signal_pairs)} пар")

        if not signal_pairs:
            return []

        ai_input_data = []
        preparation_errors = 0

        logger.info("Подготовка данных для ИИ...")

        for pair_data in signal_pairs:
            symbol = pair_data['symbol']

            try:
                # Используем кешированные данные
                candles_15m = self.cache.get_klines(symbol, '15', config.AI_BULK_15M)
                if not candles_15m or len(candles_15m) < 20:
                    preparation_errors += 1
                    continue

                # Используем кешированные индикаторы или рассчитываем новые
                indicators_15m = self.calculate_and_cache_indicators(
                    symbol, '15', candles_15m, config.AI_INDICATORS_HISTORY
                )
                if not indicators_15m:
                    preparation_errors += 1
                    continue

                pair_ai_data = {
                    'symbol': symbol,
                    'confidence': pair_data['confidence'],
                    'direction': pair_data['direction'],
                    'candles_15m': candles_15m,
                    'indicators_15m': indicators_15m
                }
                ai_input_data.append(pair_ai_data)

            except Exception as e:
                preparation_errors += 1
                continue

        if not ai_input_data:
            logger.error("Нет данных для ИИ анализа")
            return []

        logger.info(f"Отправка {len(ai_input_data)} пар в ИИ (ошибок подготовки: {preparation_errors})")

        # ИИ анализ
        selected_pairs = await ai_select_pairs(ai_input_data)

        elapsed = time.time() - start_time
        logger.info(f"ЭТАП 2 завершен: выбрано {len(selected_pairs)} пар, время {elapsed:.1f}с")

        return selected_pairs

    async def stage3_detailed_analysis(self, selected_pairs: List[str]) -> List[Dict]:
        """ЭТАП 3: Оптимизированный детальный анализ"""
        start_time = time.time()
        logger.info(f"ЭТАП 3: Детальный анализ {len(selected_pairs)} пар")

        if not selected_pairs:
            return []

        final_signals = []

        logger.info("Анализ финальных пар...")

        for symbol in selected_pairs:
            try:
                # Используем кешированные данные
                klines_5m = self.cache.get_klines(symbol, '5', config.FINAL_5M)
                klines_15m = self.cache.get_klines(symbol, '15', config.FINAL_15M)

                if not klines_5m or not klines_15m:
                    # Загружаем недостающие данные
                    if not klines_5m:
                        klines_5m = await fetch_klines(symbol, '5', config.FINAL_5M)
                        if klines_5m:
                            self.cache.cache_klines(symbol, '5', klines_5m)
                    if not klines_15m:
                        klines_15m = await fetch_klines(symbol, '15', config.FINAL_15M)
                        if klines_15m:
                            self.cache.cache_klines(symbol, '15', klines_15m)

                if not klines_5m or not klines_15m:
                    continue

                # Используем кешированные индикаторы
                indicators_5m = self.calculate_and_cache_indicators(symbol, '5', klines_5m, config.FINAL_INDICATORS)
                indicators_15m = self.calculate_and_cache_indicators(symbol, '15', klines_15m, config.FINAL_INDICATORS)

                if not indicators_5m or not indicators_15m:
                    continue

                # Сохраняем данные для валидации (минимальный набор)
                self.validation_data[symbol] = {
                    'klines_5m': klines_5m[-100:],
                    'klines_15m': klines_15m[-50:],
                    'indicators_5m': indicators_5m,
                    'indicators_15m': indicators_15m
                }

                # ИИ анализ
                analysis = await ai_analyze_pair(symbol, klines_5m, klines_15m, indicators_5m, indicators_15m)

                if analysis['signal'] != 'NO_SIGNAL' and analysis['confidence'] >= config.MIN_CONFIDENCE:
                    final_signals.append(analysis)

            except Exception as e:
                logger.error(f"Ошибка анализа {symbol}: {e}")
                continue

        elapsed = time.time() - start_time
        logger.info(f"ЭТАП 3 завершен: получено {len(final_signals)} сигналов, время {elapsed:.1f}с")

        return final_signals

    async def stage4_final_validation(self, preliminary_signals: List[Dict]) -> List[Dict]:
        """ЭТАП 4: Оптимизированная финальная валидация"""
        start_time = time.time()
        logger.info(f"ЭТАП 4: Финальная валидация {len(preliminary_signals)} сигналов")

        if not preliminary_signals or not config.DEEPSEEK_API_KEY:
            if not config.DEEPSEEK_API_KEY:
                logger.warning("DeepSeek API недоступен для валидации")
            return preliminary_signals

        try:
            from deepseek import load_prompt, extract_json_from_text
            from openai import AsyncOpenAI

            # Используем минимально необходимые данные для валидации
            validation_payload = {
                'preliminary_signals': preliminary_signals,
                'market_data': {}
            }

            # Добавляем только ключевые данные
            for signal in preliminary_signals:
                symbol = signal['symbol']
                if symbol in self.validation_data:
                    data = self.validation_data[symbol]
                    validation_payload['market_data'][symbol] = {
                        'candles_5m': data['klines_5m'],
                        'candles_15m': data['klines_15m'],
                        'indicators_5m': data['indicators_5m'],
                        'indicators_15m': data['indicators_15m']
                    }

            logger.info("Отправка данных на финальную валидацию...")

            # Загружаем промпт и отправляем запрос
            prompt = load_prompt('prompt_validate.txt')
            client = AsyncOpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_URL)
            print(json.dumps(validation_payload, separators=(',', ':')))
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=config.DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(validation_payload, separators=(',', ':'))}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=3000,
                    temperature=0.3
                ),
                timeout=config.API_TIMEOUT
            )

            result_text = response.choices[0].message.content
            validation_result = extract_json_from_text(result_text)

            if validation_result:
                final_signals = validation_result.get('final_signals', [])
                rejected_count = len(validation_result.get('rejected_signals', []))

                elapsed = time.time() - start_time
                logger.info(f"ЭТАП 4 завершен: подтверждено {len(final_signals)} сигналов, отклонено {rejected_count}, время {elapsed:.1f}с")

                return final_signals
            else:
                logger.error("Ошибка парсинга результата валидации")
                return preliminary_signals

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Ошибка валидации: {e}, время {elapsed:.1f}с")
            return preliminary_signals

    async def run_full_cycle(self) -> Dict[str, Any]:
        """Оптимизированный полный цикл работы бота"""
        cycle_start = time.time()

        logger.info("ЗАПУСК ПОЛНОГО ЦИКЛА АНАЛИЗА")
        logger.info(f"DeepSeek API: {'Доступен' if has_api_key else 'Недоступен'}")

        try:
            # Очищаем кеш перед началом
            self.cache.clear()

            # ЭТАП 1: Фильтрация
            signal_pairs = await self.stage1_filter_signals()
            if not signal_pairs:
                return {
                    'result': 'NO_SIGNAL_PAIRS',
                    'total_time': time.time() - cycle_start,
                    'pairs_scanned': self.processed_pairs,
                    'message': 'Нет пар с торговыми сигналами'
                }

            # ЭТАП 2: ИИ отбор
            selected_pairs = await self.stage2_ai_bulk_select(signal_pairs)
            if not selected_pairs:
                return {
                    'result': 'NO_AI_SELECTION',
                    'total_time': time.time() - cycle_start,
                    'signal_pairs': len(signal_pairs),
                    'pairs_scanned': self.processed_pairs,
                    'message': 'ИИ не выбрал подходящих пар'
                }

            # ЭТАП 3: Детальный анализ
            preliminary_signals = await self.stage3_detailed_analysis(selected_pairs)
            if not preliminary_signals:
                return {
                    'result': 'NO_PRELIMINARY_SIGNALS',
                    'total_time': time.time() - cycle_start,
                    'message': 'Детальный анализ не выявил качественных сигналов'
                }

            # ЭТАП 4: Финальная валидация
            validated_signals = await self.stage4_final_validation(preliminary_signals)

            # Формируем финальный результат
            total_time = time.time() - cycle_start
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            final_result = {
                'timestamp': timestamp,
                'result': 'SUCCESS' if validated_signals else 'NO_VALIDATED_SIGNALS',
                'total_time': round(total_time, 2),
                'stats': {
                    'pairs_scanned': self.processed_pairs,
                    'signal_pairs_found': len(signal_pairs),
                    'ai_selected': len(selected_pairs),
                    'preliminary_signals': len(preliminary_signals),
                    'validated_signals': len(validated_signals),
                    'processing_speed': round(self.processed_pairs / total_time, 1) if total_time > 0 else 0
                },
                'validated_signals': validated_signals,
                'api_available': has_api_key
            }

            # Сохраняем результат
            filename = f'bot_result_{timestamp}.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"ЦИКЛ ЗАВЕРШЕН: {self.processed_pairs}->{len(signal_pairs)}->{len(selected_pairs)}->{len(preliminary_signals)}->{len(validated_signals)}, время {total_time:.1f}с, скорость {self.processed_pairs/total_time:.0f} пар/сек")

            return final_result

        except Exception as e:
            logger.error(f"Критическая ошибка цикла: {e}")
            return {
                'result': 'ERROR',
                'error': str(e),
                'total_time': time.time() - cycle_start
            }

    async def cleanup(self):
        """Очистка ресурсов"""
        self.cache.clear()
        await cleanup_api()


async def main():
    """Оптимизированная главная функция"""
    print("ОПТИМИЗИРОВАННЫЙ СКАЛЬПИНГОВЫЙ БОТ v2.1")
    print(f"Запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"DeepSeek ИИ: {'Доступен' if has_api_key else 'Недоступен'}")
    print("=" * 50)

    bot = OptimizedScalpingBot()

    try:
        result = await bot.run_full_cycle()

        # Компактный вывод результата
        print(f"\nРЕЗУЛЬТАТ: {result['result']}")
        print(f"Время: {result.get('total_time', 0)}сек")

        if 'stats' in result:
            s = result['stats']
            print(f"Пайплайн: {s['pairs_scanned']}->{s['signal_pairs_found']}->{s['ai_selected']}->{s['preliminary_signals']}->{s['validated_signals']}")
            print(f"Скорость: {s['processing_speed']} пар/сек")

        if result.get('validated_signals'):
            print(f"\nФИНАЛЬНЫЕ СИГНАЛЫ:")
            for signal in result['validated_signals']:
                rr = signal.get('risk_reward_ratio', 'N/A')
                duration = signal.get('hold_duration_minutes', 'N/A')
                print(f"{signal['symbol']}: {signal['signal']} ({signal['confidence']}%) R/R:1:{rr} {duration}мин")

    except KeyboardInterrupt:
        print("\nОстановлено пользователем")
    except Exception as e:
        logger.error(f"Ошибка: {e}")
    finally:
        await bot.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nПрограмма остановлена")