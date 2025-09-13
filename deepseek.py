"""
Упрощенный ИИ клиент для DeepSeek
Убрано дублирование, оставлены только 2 функции
"""

import asyncio
import json
import logging
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
                _prompts_cache[
                    filename] = "Выбери 3-5 лучших торговых пар из предоставленных данных. Верни JSON: {\"pairs\": [\"PAIR1\", \"PAIR2\", \"PAIR3\"]}"
            else:
                _prompts_cache[
                    filename] = "Проанализируй торговые данные и дай рекомендацию LONG/SHORT/NO_SIGNAL с уверенностью 0-100."

    return _prompts_cache[filename]


async def ai_select_pairs(market_data: List[Dict]) -> List[str]:
    """
    ИИ отбор лучших пар для торговли
    Получает данные 15m + индикаторы, возвращает 3-5 пар
    """
    if not config.DEEPSEEK_API_KEY or not market_data:
        # Фаллбек - берем топ по базовой уверенности
        sorted_pairs = sorted(market_data, key=lambda x: x.get('confidence', 0), reverse=True)
        return [pair['symbol'] for pair in sorted_pairs[:5]]

    try:
        # Компактная подготовка данных для ИИ
        compact_data = []
        for item in market_data:
            symbol = item['symbol']
            indicators = item.get('indicators', {}).get('current', {})

            compact_data.append({
                'symbol': symbol,
                'price': indicators.get('price', 0),
                'ema_alignment': 'UP' if indicators.get('ema5', 0) > indicators.get('ema20', 0) else 'DOWN',
                'rsi': indicators.get('rsi', 50),
                'macd': indicators.get('macd_histogram', 0),
                'volume_ratio': indicators.get('volume_ratio', 1.0),
                'confidence': item.get('confidence', 0)
            })

        # Берем только топ для экономии токенов
        top_data = compact_data[:config.MAX_PAIRS_TO_AI]

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
        logger.info(f"ИИ анализ {symbol}: {len(result)} символов")

        # Парсим сигнал из ответа
        signal = 'NO_SIGNAL'
        confidence = 0

        if 'LONG' in result.upper():
            signal = 'LONG'
        elif 'SHORT' in result.upper():
            signal = 'SHORT'

        # Ищем числовую уверенность
        import re
        confidence_match = re.search(r'(\d{1,3})%?', result)
        if confidence_match:
            confidence = int(confidence_match.group(1))
            confidence = min(100, max(0, confidence))

        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'analysis': result,
            'trend_alignment': analysis_data['current']['trend_5m'] == analysis_data['current']['trend_15m'],
            'volume_confirmation': analysis_data['current']['volume_ratio'] > 1.2
        }

    except Exception as e:
        logger.error(f"Ошибка ИИ анализа {symbol}: {e}")
        return {
            'symbol': symbol,
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'analysis': f'Ошибка анализа: {str(e)}'
        }
        base_url = config.DEEPSEEK_URL
    )

    prompt = load_prompt(config.SELECTION_PROMPT)

    response = await asyncio.wait_for(
    client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(top_data, separators=(',', ':'))}
        ],
        max_tokens=500,
        temperature=0.3
    ),
    timeout = config.API_TIMEOUT

)

result = response.choices[0].message.content
logger.info(f"ИИ отбор ответ: {result[:100]}...")

# Парсим ответ
try:
    # Ищем JSON в ответе
    import re

    json_match = re.search(r'\{[^}]*"pairs"[^}]*\}', result)
    if json_match:
        json_data = json.loads(json_match.group(0))
        pairs = json_data.get('pairs', [])
        return pairs[:config.MAX_FINAL_PAIRS]
except:
    pass

# Альтернативный парсинг - ищем символы
symbols = re.findall(r'[A-Z]{3,10}USDT', result)
return list(set(symbols))[:config.MAX_FINAL_PAIRS]

except Exception as e:
logger.error(f"Ошибка ИИ отбора: {e}")
# Фаллбек
sorted_pairs = sorted(market_data, key=lambda x: x.get('confidence', 0), reverse=True)
return [pair['symbol'] for pair in sorted_pairs[:3]]


async def ai_analyze_pair(symbol: str, data_5m: List, data_15m: List,
                          indicators_5m: Dict, indicators_15m: Dict) -> Dict:
    """
    Детальный ИИ анализ конкретной пары
    Получает полные данные 5m + 15m + индикаторы, возвращает торговый сигнал
    """
    if not config.DEEPSEEK_API_KEY:
        return {
            'symbol': symbol,
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'analysis': 'ИИ недоступен'
        }

    try:
        # Подготовка полных данных для детального анализа
        analysis_data = {
            'symbol': symbol,
            'timeframes': {
                '5m': {
                    'recent_candles': data_5m[-20:] if len(data_5m) >= 20 else data_5m,  # Последние 20 свечей
                    'indicators': {
                        'ema5': indicators_5m.get('ema5_history', [])[-20:],
                        'ema8': indicators_5m.get('ema8_history', [])[-20:],
                        'ema20': indicators_5m.get('ema20_history', [])[-20:],
                        'rsi': indicators_5m.get('rsi_history', [])[-20:],
                        'macd_histogram': indicators_5m.get('macd_histogram_history', [])[-20:],
                        'volume_ratio': indicators_5m.get('volume_ratio_history', [])[-20:]
                    }
                },
                '15m': {
                    'recent_candles': data_15m[-10:] if len(data_15m) >= 10 else data_15m,  # Последние 10 свечей  
                    'indicators': {
                        'ema5': indicators_15m.get('ema5_history', [])[-10:],
                        'ema8': indicators_15m.get('ema8_history', [])[-10:],
                        'ema20': indicators_15m.get('ema20_history', [])[-10:],
                        'rsi': indicators_15m.get('rsi_history', [])[-10:],
                        'macd_histogram': indicators_15m.get('macd_histogram_history', [])[-10:]
                    }
                }
            },
            'current': {
                'price': indicators_5m.get('current', {}).get('price', 0),
                'trend_5m': 'UP' if indicators_5m.get('current', {}).get('ema5', 0) > indicators_5m.get('current',
                                                                                                        {}).get('ema20',
                                                                                                                0) else 'DOWN',
                'trend_15m': 'UP' if indicators_15m.get('current', {}).get('ema5', 0) > indicators_15m.get('current',
                                                                                                           {}).get(
                    'ema20', 0) else 'DOWN',
                'volume_ratio': indicators_5m.get('current', {}).get('volume_ratio', 1.0),
                'atr': indicators_5m.get('current', {}).get('atr', 0)
            }
        }

        client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,