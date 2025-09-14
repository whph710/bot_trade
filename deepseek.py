"""
Исправленный ИИ клиент для DeepSeek + функция финальной валидации
"""

import asyncio
import json
import logging
import os
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from config import config

logger = logging.getLogger(__name__)


def load_prompt(filename: str) -> str:
    """Загрузка промпта с обязательной проверкой файла"""
    if not os.path.exists(filename):
        logger.error(f"Файл промпта {filename} не найден!")
        raise FileNotFoundError(f"Обязательный файл промпта {filename} отсутствует")

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                raise ValueError(f"Файл промпта {filename} пустой")
            logger.info(f"Промпт загружен: {filename}")
            return content
    except Exception as e:
        logger.error(f"Ошибка загрузки промпта {filename}: {e}")
        raise


def extract_json_from_text(text: str) -> Optional[Dict]:
    """Извлечение JSON из ответа ИИ"""
    if not text:
        return None

    try:
        # Убираем markdown блоки
        import re
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        # Ищем JSON
        start = text.find('{')
        if start == -1:
            logger.error(f"JSON не найден в тексте: {text[:100]}")
            return None

        # Считаем скобки
        brace_count = 0
        for i, char in enumerate(text[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = text[start:i+1]
                    return json.loads(json_str)

        # Если не нашли закрывающую скобку
        return json.loads(text[start:])

    except Exception as e:
        logger.error(f"Ошибка извлечения JSON: {e}")
        return None


def smart_fallback_selection(pairs_data: List[Dict], max_pairs: int = 3) -> List[str]:
    """Умный fallback отбор без ИИ"""
    logger.info("Используется fallback отбор (без ИИ)")

    if not pairs_data:
        return []

    def calculate_score(pair_data: Dict) -> float:
        base_confidence = pair_data.get('confidence', 0)
        direction = pair_data.get('direction', 'NONE')
        base_indicators = pair_data.get('base_indicators', {})

        score = base_confidence

        # Бонус за четкое направление
        if direction in ['LONG', 'SHORT']:
            score += 15

        # Анализ индикаторов
        volume_ratio = base_indicators.get('volume_ratio', 1.0)
        rsi = base_indicators.get('rsi', 50)
        macd_hist = abs(base_indicators.get('macd_histogram', 0))

        if volume_ratio > 1.5:
            score += 20
        elif volume_ratio > 1.2:
            score += 10

        if 35 <= rsi <= 65:
            score += 15

        if macd_hist > 0.001:
            score += 12

        return score

    # Сортируем по оценке
    scored_pairs = [(pair, calculate_score(pair)) for pair in pairs_data]
    sorted_pairs = sorted(scored_pairs, key=lambda x: x[1], reverse=True)

    selected = []
    for pair, score in sorted_pairs[:max_pairs]:
        if score >= config.MIN_CONFIDENCE:
            selected.append(pair['symbol'])
            logger.info(f"Fallback выбрал {pair['symbol']} (оценка: {score:.1f})")

    return selected


def create_fallback_validation(preliminary_signals: List[Dict]) -> List[Dict]:
    """Fallback валидация без ИИ - простая проверка R/R"""
    logger.info("Используется fallback валидация (без ИИ)")

    validated_signals = []

    for signal in preliminary_signals:
        entry = signal.get('entry_price', 0)
        stop = signal.get('stop_loss', 0)
        profit = signal.get('take_profit', 0)

        if entry > 0 and stop > 0 and profit > 0:
            # Рассчитываем R/R
            risk = abs(entry - stop)
            reward = abs(profit - entry)

            if risk > 0:
                rr_ratio = round(reward / risk, 2)

                # Принимаем сигналы с R/R >= 1.5
                if rr_ratio >= 1.5:
                    # Добавляем дополнительные поля
                    validated_signal = signal.copy()
                    validated_signal['risk_reward_ratio'] = rr_ratio
                    validated_signal['hold_duration_minutes'] = 30  # Стандартное время
                    validated_signal['validation_notes'] = f'Fallback валидация: R/R {rr_ratio}'
                    validated_signal['action'] = 'APPROVED'
                    validated_signal['market_conditions'] = 'Автоматическая оценка'
                    validated_signal['key_levels'] = f'Вход: {entry}, Стоп: {stop}, Профит: {profit}'

                    validated_signals.append(validated_signal)
                    logger.info(f"Fallback подтвердил {signal['symbol']} (R/R: {rr_ratio})")
                else:
                    logger.info(f"Fallback отклонил {signal['symbol']} (R/R: {rr_ratio} < 1.5)")
            else:
                logger.info(f"Fallback отклонил {signal['symbol']} (некорректный риск)")
        else:
            logger.info(f"Fallback отклонил {signal['symbol']} (некорректные уровни)")

    return validated_signals


async def ai_select_pairs(pairs_data: List[Dict]) -> List[str]:
    """ИИ отбор пар"""
    logger.info(f"ИИ отбор: анализ {len(pairs_data)} пар")

    if not config.DEEPSEEK_API_KEY:
        logger.warning("DeepSeek API недоступен")
        return smart_fallback_selection(pairs_data, config.MAX_FINAL_PAIRS)

    if not pairs_data:
        logger.warning("Нет данных для ИИ анализа")
        return []

    try:
        # Ограничиваем количество пар
        if len(pairs_data) > config.MAX_BULK_PAIRS:
            pairs_data = sorted(pairs_data, key=lambda x: x.get('confidence', 0), reverse=True)[:config.MAX_BULK_PAIRS]

        # Подготавливаем данные для ИИ
        full_market_data = {}

        for item in pairs_data:
            symbol = item['symbol']

            # Проверяем наличие данных
            if 'indicators_15m' not in item or 'candles_15m' not in item:
                logger.warning(f"{symbol}: отсутствуют данные для ИИ")
                continue

            candles_15m = item.get('candles_15m', [])
            indicators_15m = item.get('indicators_15m', {})

            if not candles_15m or not indicators_15m:
                continue

            # Готовим данные
            full_market_data[symbol] = {
                'base_signal': {
                    'direction': item.get('direction', 'NONE'),
                    'confidence': item.get('confidence', 0)
                },
                'candles_15m': candles_15m[-32:],
                'indicators': {
                    'ema5': indicators_15m.get('ema5_history', [])[-32:],
                    'ema8': indicators_15m.get('ema8_history', [])[-32:],
                    'ema20': indicators_15m.get('ema20_history', [])[-32:],
                    'rsi': indicators_15m.get('rsi_history', [])[-32:],
                    'macd_histogram': indicators_15m.get('macd_histogram_history', [])[-32:],
                    'volume_ratio': indicators_15m.get('volume_ratio_history', [])[-32:]
                },
                'current_state': indicators_15m.get('current', {})
            }

        if not full_market_data:
            logger.error("НЕТ ДАННЫХ ДЛЯ ИИ! Переключаемся на fallback")
            return smart_fallback_selection(pairs_data, config.MAX_FINAL_PAIRS)

        # Размер данных
        json_data = json.dumps(full_market_data, separators=(',', ':'))
        data_size = len(json_data)
        logger.info(f"Размер данных для ИИ: {data_size/1024:.1f} KB")

        # ИИ клиент
        client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_URL
        )

        prompt = load_prompt(config.SELECTION_PROMPT)
        logger.info(f"Отправляем запрос к ИИ: {len(full_market_data)} пар")

        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json_data}
                ],
                response_format={"type": "json_object"},
                max_tokens=config.AI_MAX_TOKENS_SELECT,
                temperature=config.AI_TEMPERATURE_SELECT
            ),
            timeout=config.API_TIMEOUT
        )

        result_text = response.choices[0].message.content
        logger.info(f"ИИ ответ получен: {len(result_text)} символов")

        # Парсим JSON
        json_result = extract_json_from_text(result_text)

        if json_result:
            selected_pairs = json_result.get('selected_pairs', [])

            if selected_pairs:
                logger.info(f"ИИ выбрал {len(selected_pairs)} пар: {selected_pairs}")
                return selected_pairs[:config.MAX_FINAL_PAIRS]
            else:
                logger.info("ИИ не выбрал пары")
                return []
        else:
            logger.error("Не удалось извлечь JSON из ответа ИИ")
            return smart_fallback_selection(pairs_data, config.MAX_FINAL_PAIRS)

    except asyncio.TimeoutError:
        logger.error(f"Таймаут ИИ запроса ({config.API_TIMEOUT}с)")
        return smart_fallback_selection(pairs_data, config.MAX_FINAL_PAIRS)

    except Exception as e:
        logger.error(f"Ошибка ИИ отбора: {e}")
        return smart_fallback_selection(pairs_data, config.MAX_FINAL_PAIRS)


async def ai_analyze_pair(symbol: str, data_5m: List, data_15m: List,
                          indicators_5m: Dict, indicators_15m: Dict) -> Dict:
    """Детальный ИИ анализ пары"""
    logger.info(f"ИИ анализ {symbol}: {len(data_5m)} свечей 5м, {len(data_15m)} свечей 15м")

    if not config.DEEPSEEK_API_KEY:
        logger.warning(f"DeepSeek API недоступен для анализа {symbol}")
        return create_fallback_analysis(symbol, indicators_5m)

    try:
        current_price = indicators_5m.get('current', {}).get('price', 0)
        atr_5m = indicators_5m.get('current', {}).get('atr', 0)

        if current_price <= 0:
            logger.error(f"{symbol}: некорректная цена ({current_price})")
            return create_fallback_analysis(symbol, indicators_5m)

        # Подготавливаем данные
        analysis_data = {
            'symbol': symbol,
            'current_price': current_price,
            'timeframes': {
                '5m': {
                    'candles': data_5m[-100:],
                    'indicators': {
                        'ema5': indicators_5m.get('ema5_history', [])[-100:],
                        'ema8': indicators_5m.get('ema8_history', [])[-100:],
                        'ema20': indicators_5m.get('ema20_history', [])[-100:],
                        'rsi': indicators_5m.get('rsi_history', [])[-100:],
                        'macd_histogram': indicators_5m.get('macd_histogram_history', [])[-100:],
                        'volume_ratio': indicators_5m.get('volume_ratio_history', [])[-100:]
                    }
                },
                '15m': {
                    'candles': data_15m[-50:],
                    'indicators': {
                        'ema5': indicators_15m.get('ema5_history', [])[-50:],
                        'ema8': indicators_15m.get('ema8_history', [])[-50:],
                        'ema20': indicators_15m.get('ema20_history', [])[-50:],
                        'rsi': indicators_15m.get('rsi_history', [])[-50:],
                        'macd_histogram': indicators_15m.get('macd_histogram_history', [])[-50:]
                    }
                }
            },
            'current_state': {
                'price': current_price,
                'atr': atr_5m,
                'trend_5m': 'UP' if indicators_5m.get('current', {}).get('ema5', 0) > indicators_5m.get('current', {}).get('ema20', 0) else 'DOWN',
                'trend_15m': 'UP' if indicators_15m.get('current', {}).get('ema5', 0) > indicators_15m.get('current', {}).get('ema20', 0) else 'DOWN',
                'rsi_5m': indicators_5m.get('current', {}).get('rsi', 50),
                'rsi_15m': indicators_15m.get('current', {}).get('rsi', 50),
                'volume_ratio': indicators_5m.get('current', {}).get('volume_ratio', 1.0),
                'macd_momentum': indicators_5m.get('current', {}).get('macd_histogram', 0)
            }
        }

        client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_URL
        )

        prompt = load_prompt(config.ANALYSIS_PROMPT)

        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(analysis_data, separators=(',', ':'))}
                ],
                response_format={"type": "json_object"},
                max_tokens=config.AI_MAX_TOKENS_ANALYZE,
                temperature=config.AI_TEMPERATURE_ANALYZE
            ),
            timeout=config.API_TIMEOUT
        )

        result_text = response.choices[0].message.content
        json_result = extract_json_from_text(result_text)

        if json_result:
            signal = json_result.get('signal', 'NO_SIGNAL').upper()
            confidence = max(0, min(100, int(json_result.get('confidence', 0))))
            entry_price = float(json_result.get('entry_price', current_price))
            stop_loss = float(json_result.get('stop_loss', 0))
            take_profit = float(json_result.get('take_profit', 0))
            analysis = json_result.get('analysis', 'Анализ от ИИ')

            # Валидация уровней
            if signal in ['LONG', 'SHORT'] and entry_price > 0:
                if stop_loss <= 0:
                    stop_loss = entry_price * 0.98 if signal == 'LONG' else entry_price * 1.02
                if take_profit <= 0:
                    risk = abs(entry_price - stop_loss)
                    take_profit = entry_price + risk * 2 if signal == 'LONG' else entry_price - risk * 2

            result = {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'analysis': analysis,
                'ai_generated': True
            }

            logger.info(f"{symbol}: {signal} ({confidence}%) Вход: {entry_price:.4f}")
            return result
        else:
            logger.error(f"Не удалось извлечь JSON из анализа {symbol}")
            return create_fallback_analysis(symbol, indicators_5m)

    except asyncio.TimeoutError:
        logger.error(f"Таймаут анализа {symbol}")
        return create_fallback_analysis(symbol, indicators_5m)
    except Exception as e:
        logger.error(f"Ошибка ИИ анализа {symbol}: {e}")
        return create_fallback_analysis(symbol, indicators_5m)


async def ai_final_validation(preliminary_signals: List[Dict], market_data: Dict) -> List[Dict]:
    """НОВАЯ ФУНКЦИЯ: Финальная валидация сигналов с ИИ"""
    logger.info(f"ИИ валидация {len(preliminary_signals)} сигналов")

    if not config.DEEPSEEK_API_KEY:
        logger.warning("DeepSeek API недоступен для валидации")
        return create_fallback_validation(preliminary_signals)

    if not preliminary_signals:
        logger.warning("Нет сигналов для валидации")
        return []

    try:
        # Подготавливаем данные для валидации
        validation_data = {
            'preliminary_signals': preliminary_signals,
            'market_data': market_data
        }

        # Размер данных
        json_data = json.dumps(validation_data, separators=(',', ':'))
        data_size = len(json_data)
        logger.info(f"Размер данных для валидации: {data_size/1024:.1f} KB")

        # ИИ клиент
        client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_URL
        )

        prompt = load_prompt('prompt_validate.txt')
        logger.info("Отправляем данные на финальную валидацию")

        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json_data}
                ],
                response_format={"type": "json_object"},
                max_tokens=3000,  # Больше токенов для детального ответа
                temperature=0.3   # Низкая температура для точности
            ),
            timeout=config.API_TIMEOUT
        )

        result_text = response.choices[0].message.content
        logger.info(f"Получен ответ валидации: {len(result_text)} символов")

        # Парсим JSON
        validation_result = extract_json_from_text(result_text)

        if validation_result:
            final_signals = validation_result.get('final_signals', [])
            rejected_signals = validation_result.get('rejected_signals', [])

            logger.info(f"Валидация завершена: подтверждено {len(final_signals)}, отклонено {len(rejected_signals)}")

            # Логируем результаты
            for signal in final_signals:
                action = signal.get('action', 'UNKNOWN')
                duration = signal.get('hold_duration_minutes', 'N/A')
                rr_ratio = signal.get('risk_reward_ratio', 'N/A')
                logger.info(f"{signal['symbol']}: {action} R/R:{rr_ratio} Время:{duration}мин")

            for rejected in rejected_signals:
                logger.info(f"{rejected['symbol']}: ОТКЛОНЕН - {rejected['reason']}")

            return final_signals
        else:
            logger.error("Не удалось извлечь результат валидации")
            return create_fallback_validation(preliminary_signals)

    except asyncio.TimeoutError:
        logger.error(f"Таймаут валидации ({config.API_TIMEOUT}с)")
        return create_fallback_validation(preliminary_signals)
    except Exception as e:
        logger.error(f"Ошибка ИИ валидации: {e}")
        return create_fallback_validation(preliminary_signals)


def create_fallback_analysis(symbol: str, indicators_5m: Dict) -> Dict:
    """Fallback анализ без ИИ"""
    current_price = indicators_5m.get('current', {}).get('price', 0)

    return {
        'symbol': symbol,
        'signal': 'NO_SIGNAL',
        'confidence': 0,
        'entry_price': current_price,
        'stop_loss': 0,
        'take_profit': 0,
        'analysis': 'ИИ анализ недоступен - fallback режим',
        'ai_generated': False
    }