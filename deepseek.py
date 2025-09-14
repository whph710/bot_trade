"""
ИСПРАВЛЕННЫЙ ИИ клиент для DeepSeek
Устранены все критические ошибки:
- Детальная проверка API ключа
- Улучшенное логирование с диагностикой
- Более надежный fallback
- Исправлена проблема "нет данных для ИИ анализа"
"""

import asyncio
import json
import logging
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from config import config

# Настройка логирования с временными метками
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Кэш промптов
_prompts_cache = {}


def validate_api_key() -> tuple[bool, str]:
    """Валидация API ключа DeepSeek"""
    if not config.DEEPSEEK_API_KEY:
        return False, "API ключ не найден в переменных окружения"

    key = config.DEEPSEEK_API_KEY.strip()

    if len(key) < 20:
        return False, f"API ключ слишком короткий ({len(key)} символов)"

    if not key.startswith('sk-'):
        return False, f"API ключ должен начинаться с 'sk-' (текущий: {key[:10]}...)"

    return True, "API ключ прошел валидацию"


def load_prompt(filename: str) -> str:
    """Загрузка промпта с кэшированием"""
    if filename not in _prompts_cache:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                _prompts_cache[filename] = f.read()
                logger.info(f"✅ Загружен промпт: {filename}")
        except FileNotFoundError:
            logger.warning(f"⚠️ Файл промпта {filename} не найден, используется встроенный")
            # Дефолтные промпты с ПРИНУДИТЕЛЬНЫМ JSON
            if 'select' in filename:
                _prompts_cache[filename] = """Ты эксперт-трейдер с 10-летним опытом скальпинга криптовалют.

ПОЛУЧАЕШЬ: Данные по торговым парам с базовыми сигналами.
Для каждой пары: 32 свечи 15м + полная история индикаторов + текущие значения.

ЗАДАЧА: Выбрать 3-5 лучших пар для скальпинга.

КРИТЕРИИ ОТБОРА:
1. ЧЕТКИЙ ТРЕНД: EMA выравнивание держится 10+ свечей
2. СИЛЬНЫЙ MOMENTUM: RSI активен (не в крайностях) + MACD гистограмма растет
3. ОБЪЕМ ПОДТВЕРЖДАЕТ: Volume ratio > 1.2 последние свечи
4. СТАБИЛЬНОСТЬ ПАТТЕРНА: Индикаторы синхронизированы, нет хаотичности
5. СВЕЧНОЙ АНАЛИЗ: Последние 5-8 свечей поддерживают направление

ДОПОЛНИТЕЛЬНО:
- Избегай пары с противоречивыми сигналами
- Приоритет парам с усиливающимся momentum
- Ищи синхронизацию всех индикаторов

ОТВЕТ СТРОГО В JSON БЕЗ ДОПОЛНИТЕЛЬНОГО ТЕКСТА:
{
  "selected_pairs": ["BTCUSDT", "ETHUSDT"],
  "reasoning": "краткое обоснование"
}

Если качественных пар нет:
{
  "selected_pairs": [],
  "reasoning": "нет четких сигналов"
}"""
            else:
                _prompts_cache[filename] = """Ты профессиональный скальпер с экспертизой в мультитаймфреймовом анализе.

ПОЛУЧАЕШЬ: Полные данные по одной торговой паре:
- 200 свечей 5м + 100 свечей 15м
- Полную историю всех индикаторов  
- Текущее состояние рынка

ЗАДАЧА: Дать точный торговый сигнал с конкретными уровнями.

МЕТОДОЛОГИЯ:
1. КОНТЕКСТ 15М: Основной тренд, ключевые уровни
2. ТОЧКА ВХОДА 5М: Оптимальный момент входа
3. ИНДИКАТОРЫ: EMA система, RSI momentum, MACD подтверждение
4. СВЕЧНЫЕ ПАТТЕРНЫ: Анализ последних формаций
5. ВОЛАТИЛЬНОСТЬ: ATR для расчета стопов

РАСЧЕТ УРОВНЕЙ:
- ВХОД: Текущая цена ± коррекция на лучшее исполнение
- СТОП: Больший из (ATR × 1.5) или (ключевой уровень + буфер)  
- ПРОФИТ: Риск/доходность минимум 1:1.5

ВАЛИДАЦИЯ (все условия):
✓ Тренд 15м поддерживает направление
✓ Свеча 5м подтвердила движение
✓ Объем выше среднего
✓ RSI не в экстремуме
✓ MACD гистограмма в нужном направлении

ОТВЕТ СТРОГО В JSON БЕЗ ДОПОЛНИТЕЛЬНОГО ТЕКСТА:
{
  "signal": "LONG",
  "confidence": 85,
  "entry_price": 43250.50,
  "stop_loss": 43100.00,
  "take_profit": 43475.75,
  "analysis": "Краткое обоснование сигнала"
}

Если сигнал слабый: {"signal": "NO_SIGNAL", "confidence": 0, "analysis": "причина"}"""

    return _prompts_cache[filename]


def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    ИСПРАВЛЕННАЯ функция извлечения JSON из текста ИИ
    """
    if not text or not isinstance(text, str):
        logger.error("❌ Пустой или некорректный текст от ИИ")
        return None

    try:
        # Убираем возможные markdown блоки
        import re
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        # Ищем начало JSON
        start = text.find('{')
        if start == -1:
            logger.error(f"❌ JSON не найден в тексте: {text[:100]}...")
            return None

        # Считаем скобки для нахождения конца JSON
        brace_count = 0
        for i, char in enumerate(text[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = text[start:i+1]
                    try:
                        result = json.loads(json_str)
                        logger.debug(f"✅ JSON успешно извлечен и распарсен")
                        return result
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ JSON decode error: {e}")
                        logger.debug(f"Проблемный JSON: {json_str[:200]}...")
                        return None

        # Если не нашли закрывающую скобку, пробуем парсить как есть
        try:
            result = json.loads(text[start:])
            logger.debug(f"✅ JSON распарсен после обрезки")
            return result
        except:
            logger.error(f"❌ Не удалось распарсить JSON: {text[start:100]}...")
            return None

    except Exception as e:
        logger.error(f"❌ Ошибка извлечения JSON: {e}")
        return None


def smart_fallback_selection(pairs_data: List[Dict], max_pairs: int = 3) -> List[str]:
    """
    УЛУЧШЕННАЯ fallback логика без ИИ
    """
    logger.info("🔄 Используется умный fallback отбор пар (без ИИ)")

    if not pairs_data:
        logger.warning("⚠️ Нет данных для fallback отбора")
        return []

    def calculate_comprehensive_score(pair_data: Dict) -> float:
        """Комплексная оценка пары"""
        base_confidence = pair_data.get('confidence', 0)
        direction = pair_data.get('direction', 'NONE')
        base_indicators = pair_data.get('base_indicators', {})

        score = base_confidence

        # Бонус за четкое направление
        if direction in ['LONG', 'SHORT']:
            score += 15

        # Анализ базовых индикаторов
        volume_ratio = base_indicators.get('volume_ratio', 1.0)
        rsi = base_indicators.get('rsi', 50)
        macd_hist = abs(base_indicators.get('macd_histogram', 0))

        # Бонусы за качественные показатели
        if volume_ratio > 1.5:
            score += 20
        elif volume_ratio > 1.2:
            score += 10

        # RSI в рабочем диапазоне
        if 35 <= rsi <= 65:
            score += 15
        elif 25 <= rsi <= 75:
            score += 8

        # MACD активность
        if macd_hist > 0.001:
            score += 12

        return score

    # Сортируем по комплексной оценке
    scored_pairs = []
    for pair in pairs_data:
        score = calculate_comprehensive_score(pair)
        scored_pairs.append((pair, score))
        logger.debug(f"   {pair['symbol']}: оценка {score:.1f}")

    sorted_pairs = sorted(scored_pairs, key=lambda x: x[1], reverse=True)

    # Берем топ пары
    selected = []
    for pair, score in sorted_pairs[:max_pairs]:
        if score >= config.MIN_CONFIDENCE:
            selected.append(pair['symbol'])
            logger.info(f"✅ Fallback выбрал {pair['symbol']} (оценка: {score:.1f})")

    logger.info(f"📊 Fallback отобрал {len(selected)} из {len(pairs_data)} пар")
    return selected


async def ai_select_pairs(pairs_data: List[Dict]) -> List[str]:
    """
    ИСПРАВЛЕННЫЙ ИИ отбор пар с детальной диагностикой
    """
    logger.info(f"🤖 ИИ отбор: начинаем анализ {len(pairs_data)} пар")

    # Валидация API ключа
    api_valid, api_message = validate_api_key()
    if not api_valid:
        logger.warning(f"⚠️ {api_message}")
        logger.info("🔄 Переключаемся на fallback режим")
        return smart_fallback_selection(pairs_data, config.MAX_FINAL_PAIRS)

    logger.info(f"✅ {api_message}")

    if not pairs_data:
        logger.warning("⚠️ Нет данных для ИИ анализа")
        return []

    try:
        # Ограничиваем количество пар для одного запроса
        if len(pairs_data) > config.MAX_BULK_PAIRS:
            logger.info(f"📊 Ограничиваем до {config.MAX_BULK_PAIRS} пар для ИИ анализа")
            pairs_data = sorted(pairs_data, key=lambda x: x.get('confidence', 0), reverse=True)[:config.MAX_BULK_PAIRS]

        # Подготавливаем ПОЛНЫЕ данные для ИИ
        logger.info("🔄 Подготавливаем данные для ИИ...")
        full_market_data = {}

        for item in pairs_data:
            symbol = item['symbol']

            # Проверяем наличие необходимых данных
            if 'indicators_15m' not in item:
                logger.warning(f"⚠️ {symbol}: отсутствуют индикаторы 15м")
                continue

            candles_15m = item.get('candles_15m', [])
            indicators_15m = item.get('indicators_15m', {})

            if not candles_15m:
                logger.warning(f"⚠️ {symbol}: отсутствуют свечи 15м")
                continue

            if not indicators_15m:
                logger.warning(f"⚠️ {symbol}: отсутствуют индикаторы")
                continue

            # СОХРАНЯЕМ ПОЛНЫЙ ОБЪЕМ ДАННЫХ
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

            logger.debug(f"✅ {symbol}: данные подготовлены для ИИ")

        if not full_market_data:
            logger.error("❌ НЕТ ПОДГОТОВЛЕННЫХ ДАННЫХ ДЛЯ ИИ!")
            logger.error("Возможные причины:")
            logger.error("1. Данные pairs_data не содержат indicators_15m")
            logger.error("2. Данные pairs_data не содержат candles_15m")
            logger.error("3. Ошибка в stage2_ai_bulk_select при передаче данных")
            logger.error("Переключаемся на fallback...")
            return smart_fallback_selection(pairs_data, config.MAX_FINAL_PAIRS)

        # Подсчитываем размер данных
        json_data = json.dumps(full_market_data, separators=(',', ':'))
        data_size = len(json_data)
        logger.info(f"📊 Размер данных для ИИ: {data_size:,} байт ({data_size/1024:.1f} KB)")

        if data_size > 500000:  # Больше 500KB
            logger.warning(f"⚠️ Большой объем данных: {data_size/1024:.0f}KB")

        # ИСПРАВЛЕННЫЙ клиент с правильным URL
        logger.info(f"🔗 Подключаемся к DeepSeek API: {config.DEEPSEEK_URL}")
        client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_URL
        )

        prompt = load_prompt(config.SELECTION_PROMPT)
        logger.info(f"🤖 Отправляем запрос к ИИ: {len(full_market_data)} пар")

        # ИСПРАВЛЕННЫЙ запрос с принудительным JSON
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
        logger.info(f"📨 ИИ ответ получен: {len(result_text)} символов")
        logger.debug(f"Начало ответа ИИ: {result_text[:200]}...")

        # ИСПРАВЛЕННЫЙ парсинг JSON
        json_result = extract_json_from_text(result_text)

        if json_result:
            selected_pairs = json_result.get('selected_pairs', [])
            reasoning = json_result.get('reasoning', 'Нет обоснования')

            if selected_pairs:
                logger.info(f"✅ ИИ выбрал {len(selected_pairs)} пар: {selected_pairs}")
                logger.info(f"💭 Обоснование ИИ: {reasoning}")
                return selected_pairs[:config.MAX_FINAL_PAIRS]
            else:
                logger.info(f"⚠️ ИИ не выбрал пары. Причина: {reasoning}")
                return []
        else:
            logger.error("❌ Не удалось извлечь JSON из ответа ИИ")
            logger.error(f"🔍 Полный ответ ИИ: {result_text}")
            logger.info("🔄 Переключаемся на fallback...")
            return smart_fallback_selection(pairs_data, config.MAX_FINAL_PAIRS)

    except asyncio.TimeoutError:
        logger.error(f"⏰ Таймаут ИИ запроса ({config.API_TIMEOUT}с)")
        logger.info("🔄 Переключаемся на fallback...")
        return smart_fallback_selection(pairs_data, config.MAX_FINAL_PAIRS)

    except Exception as e:
        logger.error(f"💥 Критическая ошибка ИИ отбора: {e}")
        logger.error(f"🔍 Тип ошибки: {type(e).__name__}")

        # Детальная диагностика ошибки
        if "401" in str(e) or "authentication" in str(e).lower():
            logger.error("🔑 Ошибка аутентификации - проверьте API ключ")
        elif "403" in str(e) or "forbidden" in str(e).lower():
            logger.error("🚫 Доступ запрещен - проверьте права API ключа")
        elif "429" in str(e) or "rate limit" in str(e).lower():
            logger.error("🚦 Превышен лимит запросов - попробуйте позже")
        elif "connection" in str(e).lower():
            logger.error("🌐 Проблема с подключением к API")

        logger.info("🔄 Переключаемся на fallback...")
        return smart_fallback_selection(pairs_data, config.MAX_FINAL_PAIRS)


async def ai_analyze_pair(symbol: str, data_5m: List, data_15m: List,
                          indicators_5m: Dict, indicators_15m: Dict) -> Dict:
    """
    ИСПРАВЛЕННЫЙ детальный ИИ анализ конкретной пары
    """
    logger.info(f"🔍 ИИ анализ {symbol}: {len(data_5m)} свечей 5м, {len(data_15m)} свечей 15м")

    # Валидация API ключа
    api_valid, api_message = validate_api_key()
    if not api_valid:
        logger.warning(f"⚠️ DeepSeek API недоступен для анализа {symbol}: {api_message}")
        return create_fallback_analysis(symbol, indicators_5m)

    try:
        current_price = indicators_5m.get('current', {}).get('price', 0)
        atr_5m = indicators_5m.get('current', {}).get('atr', 0)

        if current_price <= 0:
            logger.error(f"❌ {symbol}: некорректная текущая цена ({current_price})")
            return create_fallback_analysis(symbol, indicators_5m)

        # ПОЛНЫЕ данные для детального анализа
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

        # ИСПРАВЛЕННЫЙ запрос с принудительным JSON
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
        logger.debug(f"📨 ИИ анализ {symbol}: получен ответ {len(result_text)} символов")

        # ИСПРАВЛЕННЫЙ парсинг JSON
        json_result = extract_json_from_text(result_text)

        if json_result:
            # Валидация и обработка ответа
            signal = json_result.get('signal', 'NO_SIGNAL').upper()
            confidence = max(0, min(100, int(json_result.get('confidence', 0))))
            entry_price = float(json_result.get('entry_price', current_price))
            stop_loss = float(json_result.get('stop_loss', 0))
            take_profit = float(json_result.get('take_profit', 0))
            analysis = json_result.get('analysis', 'Анализ от ИИ')

            # Дополнительная валидация уровней
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
                'trend_alignment': analysis_data['current_state']['trend_5m'] == analysis_data['current_state']['trend_15m'],
                'volume_confirmation': analysis_data['current_state']['volume_ratio'] > 1.2,
                'ai_generated': True
            }

            logger.info(f"✅ {symbol}: {signal} ({confidence}%) Вход: {entry_price:.4f}")
            return result
        else:
            logger.error(f"❌ Не удалось извлечь JSON из анализа {symbol}")
            logger.debug(f"🔍 Ответ ИИ: {result_text}")
            return create_fallback_analysis(symbol, indicators_5m)

    except asyncio.TimeoutError:
        logger.error(f"⏰ Таймаут анализа {symbol} ({config.API_TIMEOUT}с)")
        return create_fallback_analysis(symbol, indicators_5m)
    except Exception as e:
        logger.error(f"💥 Ошибка ИИ анализа {symbol}: {e}")
        return create_fallback_analysis(symbol, indicators_5m)


def create_fallback_analysis(symbol: str, indicators_5m: Dict) -> Dict:
    """Создание fallback анализа без ИИ"""
    current_price = indicators_5m.get('current', {}).get('price', 0)

    return {
        'symbol': symbol,
        'signal': 'NO_SIGNAL',
        'confidence': 0,
        'entry_price': current_price,
        'stop_loss': 0,
        'take_profit': 0,
        'analysis': 'ИИ анализ недоступен - используется fallback режим',
        'trend_alignment': False,
        'volume_confirmation': False,
        'ai_generated': False
    }