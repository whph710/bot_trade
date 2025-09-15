"""
Оптимизированный ИИ клиент для DeepSeek с улучшенной обработкой данных
"""

import asyncio
import json
import logging
import os
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from config import config

logger = logging.getLogger(__name__)

# Кеш для промптов
_prompts_cache = {}

def load_prompt_cached(filename: str) -> str:
    """Загрузка промпта с кешированием"""
    if filename in _prompts_cache:
        return _prompts_cache[filename]

    if not os.path.exists(filename):
        logger.error(f"Файл промпта {filename} не найден!")
        raise FileNotFoundError(f"Обязательный файл промпта {filename} отсутствует")

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                raise ValueError(f"Файл промпта {filename} пустой")

            _prompts_cache[filename] = content
            logger.info(f"Промпт закеширован: {filename}")
            return content
    except Exception as e:
        logger.error(f"Ошибка загрузки промпта {filename}: {e}")
        raise

# Алиас для совместимости
load_prompt = load_prompt_cached

def extract_json_optimized(text: str) -> Optional[Dict]:
    """Оптимизированное извлечение JSON из ответа ИИ"""
    if not text or len(text) < 10:
        return None

    try:
        # Быстрое удаление markdown блоков
        if '```json' in text:
            start = text.find('```json') + 7
            end = text.find('```', start)
            if end != -1:
                text = text[start:end].strip()
        elif '```' in text:
            text = text.replace('```', '').strip()

        # Поиск JSON объекта
        start_idx = text.find('{')
        if start_idx == -1:
            return None

        # Быстрый подсчет скобок
        brace_count = 0
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = text[start_idx:i+1]
                    return json.loads(json_str)

        # Если не нашли закрывающую скобку, пробуем весь остаток
        return json.loads(text[start_idx:])

    except json.JSONDecodeError as e:
        logger.error(f"Ошибка парсинга JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Общая ошибка извлечения JSON: {e}")
        return None

# Алиас для совместимости
extract_json_from_text = extract_json_optimized

def create_optimized_fallback_selection(pairs_data: List[Dict], max_pairs: int = 3) -> List[str]:
    """Оптимизированный fallback отбор без ИИ"""
    if not pairs_data:
        return []

    # Быстрая сортировка с lambda функцией
    def quick_score(pair_data: Dict) -> float:
        base_confidence = pair_data.get('confidence', 0)
        direction = pair_data.get('direction', 'NONE')
        indicators = pair_data.get('base_indicators', {})

        score = base_confidence

        # Бонусы за качественные сигналы
        if direction in ['LONG', 'SHORT']:
            score += 15

        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            score += 20
        elif volume_ratio > 1.2:
            score += 10

        rsi = indicators.get('rsi', 50)
        if 35 <= rsi <= 65:
            score += 15

        return score

    # Быстрая сортировка и фильтрация
    scored_pairs = [(pair, quick_score(pair)) for pair in pairs_data]
    top_pairs = sorted(scored_pairs, key=lambda x: x[1], reverse=True)[:max_pairs]

    selected = []
    for pair, score in top_pairs:
        if score >= config.MIN_CONFIDENCE:
            selected.append(pair['symbol'])

    logger.info(f"Fallback выбрал {len(selected)} пар")
    return selected

# Алиас для совместимости
smart_fallback_selection = create_optimized_fallback_selection

def create_optimized_fallback_validation(preliminary_signals: List[Dict]) -> List[Dict]:
    """Оптимизированная fallback валидация без ИИ"""
    validated_signals = []

    for signal in preliminary_signals:
        entry = signal.get('entry_price', 0)
        stop = signal.get('stop_loss', 0)
        profit = signal.get('take_profit', 0)

        if entry > 0 and stop > 0 and profit > 0:
            risk = abs(entry - stop)
            reward = abs(profit - entry)

            if risk > 0:
                rr_ratio = round(reward / risk, 2)
                if rr_ratio >= 1.5:
                    validated_signal = signal.copy()
                    validated_signal.update({
                        'risk_reward_ratio': rr_ratio,
                        'hold_duration_minutes': 30,
                        'validation_notes': f'Fallback валидация: R/R {rr_ratio}',
                        'action': 'APPROVED',
                        'market_conditions': 'Автоматическая оценка',
                        'key_levels': f'Вход: {entry}, Стоп: {stop}, Профит: {profit}'
                    })
                    validated_signals.append(validated_signal)

    logger.info(f"Fallback валидация: подтверждено {len(validated_signals)} сигналов")
    return validated_signals

# Алиас для совместимости
create_fallback_validation = create_optimized_fallback_validation

async def ai_select_pairs_optimized(pairs_data: List[Dict]) -> List[str]:
    """Оптимизированный ИИ отбор пар"""
    if not config.DEEPSEEK_API_KEY:
        return create_optimized_fallback_selection(pairs_data, config.MAX_FINAL_PAIRS)

    if not pairs_data:
        return []

    try:
        # Ограничиваем количество пар для оптимизации
        if len(pairs_data) > config.MAX_BULK_PAIRS:
            pairs_data = sorted(pairs_data, key=lambda x: x.get('confidence', 0), reverse=True)[:config.MAX_BULK_PAIRS]

        # Подготавливаем минимально необходимые данные
        compact_data = {}
        for item in pairs_data:
            symbol = item['symbol']

            if 'indicators_15m' not in item or 'candles_15m' not in item:
                continue

            candles_15m = item.get('candles_15m', [])
            indicators_15m = item.get('indicators_15m', {})

            if not candles_15m or not indicators_15m:
                continue

            # Компактная структура данных
            compact_data[symbol] = {
                'base_signal': {
                    'direction': item.get('direction', 'NONE'),
                    'confidence': item.get('confidence', 0)
                },
                'candles_15m': candles_15m[-30:],  # Уменьшено для скорости
                'indicators': {
                    'ema5': indicators_15m.get('ema5_history', [])[-30:],
                    'ema8': indicators_15m.get('ema8_history', [])[-30:],
                    'ema20': indicators_15m.get('ema20_history', [])[-30:],
                    'rsi': indicators_15m.get('rsi_history', [])[-30:],
                    'macd_histogram': indicators_15m.get('macd_histogram_history', [])[-30:],
                    'volume_ratio': indicators_15m.get('volume_ratio_history', [])[-30:]
                },
                'current_state': indicators_15m.get('current', {})
            }

        if not compact_data:
            return create_optimized_fallback_selection(pairs_data, config.MAX_FINAL_PAIRS)

        # Создаем ИИ клиент
        client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_URL
        )

        prompt = load_prompt_cached(config.SELECTION_PROMPT)
        json_payload = json.dumps(compact_data, separators=(',', ':'))

        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json_payload}
                ],
                response_format={"type": "json_object"},
                max_tokens=config.AI_MAX_TOKENS_SELECT,
                temperature=config.AI_TEMPERATURE_SELECT
            ),
            timeout=config.API_TIMEOUT
        )

        result_text = response.choices[0].message.content
        json_result = extract_json_optimized(result_text)

        if json_result:
            selected_pairs = json_result.get('selected_pairs', [])
            if selected_pairs:
                logger.info(f"ИИ выбрал {len(selected_pairs)} пар")
                return selected_pairs[:config.MAX_FINAL_PAIRS]

        logger.info("ИИ не выбрал пары")
        return []

    except asyncio.TimeoutError:
        logger.error("Таймаут ИИ запроса")
        return create_optimized_fallback_selection(pairs_data, config.MAX_FINAL_PAIRS)
    except Exception as e:
        logger.error(f"Ошибка ИИ отбора: {e}")
        return create_optimized_fallback_selection(pairs_data, config.MAX_FINAL_PAIRS)

# Алиас для совместимости
ai_select_pairs = ai_select_pairs_optimized

async def ai_analyze_pair_optimized(symbol: str, data_5m: List, data_15m: List,
                                   indicators_5m: Dict, indicators_15m: Dict) -> Dict:
    """Оптимизированный детальный ИИ анализ пары"""
    if not config.DEEPSEEK_API_KEY:
        return create_fallback_analysis(symbol, indicators_5m)

    try:
        current_price = indicators_5m.get('current', {}).get('price', 0)
        if current_price <= 0:
            return create_fallback_analysis(symbol, indicators_5m)

        # Компактная подготовка данных для анализа
        analysis_data = {
            'symbol': symbol,
            'current_price': current_price,
            'timeframes': {
                '5m': {
                    'candles': data_5m[-80:],  # Уменьшено для скорости
                    'indicators': {
                        'ema5': indicators_5m.get('ema5_history', [])[-80:],
                        'ema8': indicators_5m.get('ema8_history', [])[-80:],
                        'ema20': indicators_5m.get('ema20_history', [])[-80:],
                        'rsi': indicators_5m.get('rsi_history', [])[-80:],
                        'macd_histogram': indicators_5m.get('macd_histogram_history', [])[-80:],
                        'volume_ratio': indicators_5m.get('volume_ratio_history', [])[-80:]
                    }
                },
                '15m': {
                    'candles': data_15m[-40:],  # Уменьшено для скорости
                    'indicators': {
                        'ema5': indicators_15m.get('ema5_history', [])[-40:],
                        'ema8': indicators_15m.get('ema8_history', [])[-40:],
                        'ema20': indicators_15m.get('ema20_history', [])[-40:],
                        'rsi': indicators_15m.get('rsi_history', [])[-40:],
                        'macd_histogram': indicators_15m.get('macd_histogram_history', [])[-40:]
                    }
                }
            },
            'current_state': {
                'price': current_price,
                'atr': indicators_5m.get('current', {}).get('atr', 0),
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

        prompt = load_prompt_cached(config.ANALYSIS_PROMPT)

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
        json_result = extract_json_optimized(result_text)

        if json_result:
            signal = json_result.get('signal', 'NO_SIGNAL').upper()
            confidence = max(0, min(100, int(json_result.get('confidence', 0))))
            entry_price = float(json_result.get('entry_price', current_price))
            stop_loss = float(json_result.get('stop_loss', 0))
            take_profit = float(json_result.get('take_profit', 0))
            analysis = json_result.get('analysis', 'Анализ от ИИ')

            # Быстрая валидация уровней
            if signal in ['LONG', 'SHORT'] and entry_price > 0:
                if stop_loss <= 0:
                    stop_loss = entry_price * (0.98 if signal == 'LONG' else 1.02)
                if take_profit <= 0:
                    risk = abs(entry_price - stop_loss)
                    take_profit = entry_price + (risk * 2 if signal == 'LONG' else -risk * 2)

            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'analysis': analysis,
                'ai_generated': True
            }
        else:
            return create_fallback_analysis(symbol, indicators_5m)

    except asyncio.TimeoutError:
        logger.error(f"Таймаут анализа {symbol}")
        return create_fallback_analysis(symbol, indicators_5m)
    except Exception as e:
        logger.error(f"Ошибка ИИ анализа {symbol}: {e}")
        return create_fallback_analysis(symbol, indicators_5m)

# Алиас для совместимости
ai_analyze_pair = ai_analyze_pair_optimized

async def ai_final_validation_optimized(preliminary_signals: List[Dict], market_data: Dict) -> List[Dict]:
    """Оптимизированная финальная валидация сигналов с ИИ"""
    if not config.DEEPSEEK_API_KEY:
        return create_optimized_fallback_validation(preliminary_signals)

    if not preliminary_signals:
        return []

    try:
        # Компактная подготовка данных для валидации
        validation_data = {
            'preliminary_signals': preliminary_signals,
            'market_data': market_data
        }

        client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_URL
        )

        prompt = load_prompt_cached('prompt_validate.txt')

        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(validation_data, separators=(',', ':'))}
                ],
                response_format={"type": "json_object"},
                max_tokens=3000,
                temperature=0.3
            ),
            timeout=config.API_TIMEOUT
        )

        result_text = response.choices[0].message.content
        validation_result = extract_json_optimized(result_text)

        if validation_result:
            final_signals = validation_result.get('final_signals', [])
            rejected_count = len(validation_result.get('rejected_signals', []))

            logger.info(f"Валидация завершена: подтверждено {len(final_signals)}, отклонено {rejected_count}")
            return final_signals
        else:
            return create_optimized_fallback_validation(preliminary_signals)

    except asyncio.TimeoutError:
        logger.error("Таймаут валидации")
        return create_optimized_fallback_validation(preliminary_signals)
    except Exception as e:
        logger.error(f"Ошибка ИИ валидации: {e}")
        return create_optimized_fallback_validation(preliminary_signals)

# Алиас для совместимости
ai_final_validation = ai_final_validation_optimized

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