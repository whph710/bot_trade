"""
Исправленный ИИ клиент для DeepSeek
Передача полных данных на этапе 2, структурированные ответы
"""

import asyncio
import json
import logging
import re
from typing import List, Dict
from openai import AsyncOpenAI
from config import config

logger = logging.getLogger(__name__)

# Кэш промптов
_prompts_cache = {}


def load_prompt(filename: str) -> str:
    """Загрузка промпта с кэшированием"""
    if filename not in _prompts_cache:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                _prompts_cache[filename] = f.read()
        except FileNotFoundError:
            # Дефолтный промпт если файл не найден
            if 'select' in filename:
                _prompts_cache[filename] = """Ты эксперт-трейдер. Анализируешь данные по нескольким торговым парам одновременно.

Получаешь для каждой пары:
- 32 свечи 15-минутного таймфрейма
- 32 значения каждого индикатора
- Текущие показатели

ТВОЯ ЗАДАЧА: Выбрать 3-5 лучших пар для детального анализа.

КРИТЕРИИ ОТБОРА:
1. Четкий тренд (EMA выравнивание)
2. Momentum (RSI + MACD активность) 
3. Объем подтверждает движение
4. Свечные паттерны поддерживают направление
5. Синхронизация индикаторов

СТРОГО ВЕРНИ JSON:
{"selected_pairs": ["SYMBOL1", "SYMBOL2", "SYMBOL3"]}

Если нет подходящих пар: {"selected_pairs": []}"""
            else:
                _prompts_cache[filename] = """Ты профессиональный трейдер-скальпер. 

Получаешь ПОЛНЫЕ данные по одной торговой паре:
- Свечи 5м и 15м
- Все индикаторы с историей
- Текущую рыночную ситуацию

ТВОЯ ЗАДАЧА: Дать точный торговый сигнал с уровнями.

АНАЛИЗИРУЙ:
1. Тренд 15м (общее направление)
2. Сигнал 5м (точка входа)
3. Индикаторы (подтверждение)
4. Уровни поддержки/сопротивления
5. Волатильность (ATR)

РАСЧЕТ УРОВНЕЙ:
- Вход: текущая цена + коррекция
- Стоп-лосс: ATR * 1.5 или ключевой уровень
- Тейк-профит: риск/доходность 1:2

СТРОГО ВЕРНИ JSON:
{
  "signal": "LONG/SHORT/NO_SIGNAL",
  "confidence": 85,
  "entry_price": 43250.5,
  "stop_loss": 43100.0,
  "take_profit": 43550.0,
  "analysis": "краткое обоснование"
}"""

    return _prompts_cache[filename]


async def ai_select_pairs(pairs_data: List[Dict]) -> List[str]:
    """
    ИИ отбор лучших пар - передаем ВСЕ данные одним запросом
    Получает полные данные по всем парам с сигналами
    """
    if not config.DEEPSEEK_API_KEY or not pairs_data:
        # Фаллбек - берем топ по базовой уверенности
        sorted_pairs = sorted(pairs_data, key=lambda x: x.get('confidence', 0), reverse=True)
        return [pair['symbol'] for pair in sorted_pairs[:5]]

    try:
        # Подготавливаем ПОЛНЫЕ данные для ИИ - все пары одним запросом
        full_market_data = {}

        for item in pairs_data:
            symbol = item['symbol']

            # Берем полные данные 15м + индикаторы
            candles_15m = item.get('candles_15m', [])
            indicators_15m = item.get('indicators_15m', {})

            # Подготавливаем структурированные данные
            full_market_data[symbol] = {
                'base_signal': {
                    'direction': item.get('direction', 'NONE'),
                    'confidence': item.get('confidence', 0)
                },
                'candles_15m': candles_15m[-32:],  # Последние 32 свечи
                'indicators': {
                    'ema5': indicators_15m.get('ema5_history', [])[-32:],
                    'ema8': indicators_15m.get('ema8_history', [])[-32:],
                    'ema20': indicators_15m.get('ema20_history', [])[-32:],
                    'rsi': indicators_15m.get('rsi_history', [])[-32:],
                    'macd_histogram': indicators_15m.get('macd_histogram_history', [])[-32:],
                    'volume_ratio': indicators_15m.get('volume_ratio_history', [])[-32:]
                },
                'current': indicators_15m.get('current', {})
            }

        client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_URL
        )

        prompt = load_prompt(config.SELECTION_PROMPT)

        # Отправляем все данные одним запросом
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(full_market_data, separators=(',', ':'))}
                ],
                max_tokens=1000,
                temperature=0.3
            ),
            timeout=config.API_TIMEOUT
        )

        result = response.choices[0].message.content
        logger.info(f"ИИ отбор получил {len(pairs_data)} пар, ответ: {len(result)} символов")

        # Парсим JSON ответ
        try:
            # Ищем JSON в ответе
            json_match = re.search(r'\{[^}]*"selected_pairs"[^}]*\}', result)
            if json_match:
                json_data = json.loads(json_match.group(0))
                pairs = json_data.get('selected_pairs', [])
                return pairs[:config.MAX_FINAL_PAIRS]

            # Альтернативный парсинг
            json_match = re.search(r'\{[^}]*"pairs"[^}]*\}', result)
            if json_match:
                json_data = json.loads(json_match.group(0))
                pairs = json_data.get('pairs', [])
                return pairs[:config.MAX_FINAL_PAIRS]

        except Exception as parse_error:
            logger.error(f"Ошибка парсинга JSON: {parse_error}")

        # Фаллбек - ищем символы в тексте
        symbols = re.findall(r'[A-Z]{3,10}USDT', result)
        return list(set(symbols))[:config.MAX_FINAL_PAIRS]

    except Exception as e:
        logger.error(f"Ошибка ИИ отбора: {e}")
        # Фаллбек
        sorted_pairs = sorted(pairs_data, key=lambda x: x.get('confidence', 0), reverse=True)
        return [pair['symbol'] for pair in sorted_pairs[:3]]


async def ai_analyze_pair(symbol: str, data_5m: List, data_15m: List,
                          indicators_5m: Dict, indicators_15m: Dict) -> Dict:
    """
    Детальный ИИ анализ конкретной пары
    Возвращает структурированный JSON с торговым сигналом и уровнями
    """
    if not config.DEEPSEEK_API_KEY:
        return {
            'symbol': symbol,
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'entry_price': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'analysis': 'ИИ недоступен'
        }

    try:
        # Подготовка ПОЛНЫХ данных для детального анализа
        current_price = indicators_5m.get('current', {}).get('price', 0)
        atr_5m = indicators_5m.get('current', {}).get('atr', 0)

        analysis_data = {
            'symbol': symbol,
            'current_price': current_price,
            'timeframes': {
                '5m': {
                    'candles': data_5m[-50:],  # Больше данных для анализа
                    'indicators': {
                        'ema5': indicators_5m.get('ema5_history', [])[-50:],
                        'ema8': indicators_5m.get('ema8_history', [])[-50:],
                        'ema20': indicators_5m.get('ema20_history', [])[-50:],
                        'rsi': indicators_5m.get('rsi_history', [])[-50:],
                        'macd_histogram': indicators_5m.get('macd_histogram_history', [])[-50:],
                        'volume_ratio': indicators_5m.get('volume_ratio_history', [])[-50:]
                    }
                },
                '15m': {
                    'candles': data_15m[-30:],  # 15м для контекста
                    'indicators': {
                        'ema5': indicators_15m.get('ema5_history', [])[-30:],
                        'ema8': indicators_15m.get('ema8_history', [])[-30:],
                        'ema20': indicators_15m.get('ema20_history', [])[-30:],
                        'rsi': indicators_15m.get('rsi_history', [])[-30:],
                        'macd_histogram': indicators_15m.get('macd_histogram_history', [])[-30:]
                    }
                }
            },
            'current_state': {
                'price': current_price,
                'atr': atr_5m,
                'trend_5m': 'UP' if indicators_5m.get('current', {}).get('ema5', 0) > indicators_5m.get('current', {}).get('ema20', 0) else 'DOWN',
                'trend_15m': 'UP' if indicators_15m.get('current', {}).get('ema5', 0) > indicators_15m.get('current', {}).get('ema20', 0) else 'DOWN',
                'volume_ratio': indicators_5m.get('current', {}).get('volume_ratio', 1.0)
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
                max_tokens=2000,
                temperature=0.7
            ),
            timeout=config.API_TIMEOUT
        )

        result = response.choices[0].message.content
        logger.info(f"ИИ анализ {symbol}: получен ответ {len(result)} символов")

        # Парсим JSON ответ
        try:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result)
            if json_match:
                json_data = json.loads(json_match.group(0))

                # Проверяем обязательные поля
                signal = json_data.get('signal', 'NO_SIGNAL')
                confidence = int(json_data.get('confidence', 0))
                entry_price = float(json_data.get('entry_price', current_price))
                stop_loss = float(json_data.get('stop_loss', 0))
                take_profit = float(json_data.get('take_profit', 0))

                return {
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': confidence,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'analysis': json_data.get('analysis', result),
                    'trend_alignment': analysis_data['current_state']['trend_5m'] == analysis_data['current_state']['trend_15m'],
                    'volume_confirmation': analysis_data['current_state']['volume_ratio'] > 1.2
                }
        except Exception as parse_error:
            logger.error(f"Ошибка парсинга JSON анализа {symbol}: {parse_error}")

        # Фаллбек парсинг
        signal = 'NO_SIGNAL'
        confidence = 0

        if 'LONG' in result.upper():
            signal = 'LONG'
        elif 'SHORT' in result.upper():
            signal = 'SHORT'

        # Ищем числовую уверенность
        confidence_match = re.search(r'(\d{1,3})%?', result)
        if confidence_match:
            confidence = int(confidence_match.group(1))
            confidence = min(100, max(0, confidence))

        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'entry_price': current_price,
            'stop_loss': 0,
            'take_profit': 0,
            'analysis': result,
            'trend_alignment': analysis_data['current_state']['trend_5m'] == analysis_data['current_state']['trend_15m'],
            'volume_confirmation': analysis_data['current_state']['volume_ratio'] > 1.2
        }

    except Exception as e:
        logger.error(f"Ошибка ИИ анализа {symbol}: {e}")
        return {
            'symbol': symbol,
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'entry_price': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'analysis': f'Ошибка анализа: {str(e)}'
        }