"""
AI модуль для продвинутого анализа
Order Flow, Smart Money Concepts, Volume Profile interpretation
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class AIAdvancedAnalyzer:
    """AI-driven анализ сложных рыночных паттернов"""

    def __init__(self, ai_router):
        """
        Args:
            ai_router: экземпляр AIRouter из твоего бота
        """
        self.ai_router = ai_router

    async def analyze_orderbook_flow(
            self,
            symbol: str,
            orderbook_data: Dict,
            recent_price_action: List[float]
    ) -> Dict:
        """
        AI анализ Order Flow из стакана заявок

        Args:
            symbol: торговая пара
            orderbook_data: данные стакана (bids, asks)
            recent_price_action: последние 10-20 цен для контекста

        Returns:
            {
                'orderflow_direction': 'BULLISH' / 'BEARISH' / 'NEUTRAL',
                'absorption_detected': True / False,
                'spoofing_risk': 'HIGH' / 'LOW',
                'key_levels': [49800, 50200],
                'confidence_adjustment': +10 / -10 / 0,
                'reasoning': 'детальное объяснение'
            }
        """
        if not orderbook_data or not orderbook_data.get('bids') or not orderbook_data.get('asks'):
            return {
                'orderflow_direction': 'UNKNOWN',
                'absorption_detected': False,
                'spoofing_risk': 'UNKNOWN',
                'key_levels': [],
                'confidence_adjustment': 0,
                'reasoning': 'No orderbook data available'
            }

        try:
            # Форматируем данные для AI
            bids = orderbook_data['bids'][:20]
            asks = orderbook_data['asks'][:20]
            mid_price = orderbook_data['mid_price']

            # Подготавливаем промпт
            prompt = f"""Ты institutional orderflow trader с опытом tape reading. Проанализируй стакан заявок для {symbol}.

ТЕКУЩИЙ СТАКАН:
Средняя цена: {mid_price}

TOP 20 BIDS (покупки):
{self._format_orderbook_side(bids, 'BID')}

TOP 20 ASKS (продажи):
{self._format_orderbook_side(asks, 'ASK')}

НЕДАВНЕЕ ДВИЖЕНИЕ ЦЕН (последние свечи):
{recent_price_action[-10:]}

ЗАДАЧИ:
1. **ABSORPTION ZONES**: Есть ли крупные "стены" заявок (>3x средний размер)?
   - Где именно? На каких уровнях?
   - Это реальное накопление или fake wall (spoofing)?

2. **BID/ASK IMBALANCE**: Кто доминирует - покупатели или продавцы?
   - Считай суммарный объем в топ-10 уровнях

3. **KEY LEVELS**: Определи критические уровни где сосредоточены заявки

4. **SPOOFING DETECTION**: Признаки фейковых заявок:
   - Крупные заявки далеко от цены
   - Симметричные "красивые" числа
   - Слишком идеальное распределение

5. **ORDERFLOW DIRECTION**: Куда скорее всего пойдет цена?
   - BULLISH: сильный bid support, weak asks
   - BEARISH: weak bids, strong ask resistance
   - NEUTRAL: balanced

6. **CONFIDENCE ADJUSTMENT**: 
   - +10 если очень четкий orderflow в одну сторону
   - -10 если противоречивые сигналы или spoofing
   - 0 если нейтрально

Верни ТОЛЬКО JSON (без markdown):
{{
  "orderflow_direction": "BULLISH|BEARISH|NEUTRAL",
  "absorption_detected": true|false,
  "absorption_zones": ["49750-49800: strong bids 500k USDT"],
  "spoofing_risk": "HIGH|MEDIUM|LOW",
  "key_levels": [49800, 50200],
  "bid_ask_dominance": "BIDS|ASKS|BALANCED",
  "confidence_adjustment": -10 to +10,
  "reasoning": "краткое объяснение ключевых наблюдений (2-3 предложения)"
}}"""

            # Отправляем в AI
            response = await self.ai_router.call_ai(
                prompt=prompt,
                stage='analysis',
                max_tokens=1500
            )

            # Парсим ответ
            result = self._safe_parse_json(response)

            if not result:
                return self._get_fallback_orderbook_analysis()

            return result

        except Exception as e:
            logger.error(f"Ошибка AI анализа orderbook для {symbol}: {e}")
            return self._get_fallback_orderbook_analysis()

    async def detect_smart_money_concepts(
            self,
            symbol: str,
            candles_1h: List[List],
            current_price: float
    ) -> Dict:
        """
        AI поиск Smart Money Concepts паттернов

        Args:
            symbol: торговая пара
            candles_1h: свечи 1H (последние 50)
            current_price: текущая цена

        Returns:
            {
                'order_blocks': [{'index': 45, 'zone': [49000, 49200], 'type': 'bullish'}],
                'fair_value_gaps': [...],
                'liquidity_sweeps': [...],
                'break_of_structure': {...},
                'confidence_boost': +15 / 0,
                'reasoning': '...'
            }
        """
        if not candles_1h or len(candles_1h) < 20:
            return {
                'order_blocks': [],
                'fair_value_gaps': [],
                'liquidity_sweeps': [],
                'break_of_structure': None,
                'confidence_boost': 0,
                'reasoning': 'Insufficient candle data for SMC analysis'
            }

        try:
            # Подготавливаем структурированные данные свечей
            candle_summary = []
            for i, c in enumerate(candles_1h[-50:]):
                o, h, l, cl, v = float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])
                candle_summary.append({
                    'index': i,
                    'open': o,
                    'high': h,
                    'low': l,
                    'close': cl,
                    'volume': v,
                    'type': 'bullish' if cl > o else 'bearish',
                    'body_size': abs(cl - o),
                    'wick_upper': h - max(o, cl),
                    'wick_lower': min(o, cl) - l
                })

            prompt = f"""Ты Smart Money Concepts (SMC) эксперт. Проанализируй 50 свечей 1H для {symbol}.

ТЕКУЩАЯ ЦЕНА: {current_price}

ПОСЛЕДНИЕ 20 СВЕЧЕЙ (для контекста, полные данные есть):
{json.dumps(candle_summary[-20:], indent=2)}

ЗАДАЧИ - найди институциональные паттерны:

1. **ORDER BLOCKS** - зоны институционального интереса:
   - Последняя противоположная свеча ПЕРЕД сильным импульсом
   - Если цена резко выросла, найди последнюю bearish свечу перед этим
   - Зона OB = [low, high] этой свечи
   - Пример: "Свеча #42 (bearish) перед импульсом вверх на #43-45"

2. **FAIR VALUE GAPS (FVG)** - незаполненные ценовые разрывы:
   - Когда high свечи [i-1] НЕ пересекается с low свечи [i+1]
   - Это зона "недоторгованности" которую цена стремится заполнить
   - Ищи FVG которые ЕЩЕ НЕ заполнены

3. **LIQUIDITY SWEEPS** - ложные пробои для сбора стопов:
   - Пробой недавнего swing high/low
   - НО затем БЫСТРЫЙ возврат (в 1-2 свечах)
   - Классический stop hunt паттерн

4. **BREAK OF STRUCTURE (BOS)** - пробой структуры:
   - Для uptrend: пробой последнего swing high
   - Для downtrend: пробой последнего swing low
   - Это подтверждение продолжения тренда

ОЦЕНКА СИЛЫ:
- Если найдено 2+ паттерна которые все указывают в одну сторону: +15 к confidence
- Если Order Block близко к текущей цене (<2%): +10
- Если паттерны противоречивые: 0

Верни ТОЛЬКО JSON:
{{
  "order_blocks": [
    {{"candle_index": 42, "zone": [49100, 49300], "type": "bullish", "distance_pct": 1.5}}
  ],
  "fair_value_gaps": [
    {{"start_candle": 38, "zone": [48800, 49000], "filled": false}}
  ],
  "liquidity_sweeps": [
    {{"candle_index": 45, "swept_level": 50200, "direction": "upward"}}
  ],
  "break_of_structure": {{"detected": true, "type": "bullish", "level": 50100}},
  "patterns_alignment": "BULLISH|BEARISH|MIXED",
  "confidence_boost": 0 to +15,
  "reasoning": "краткое резюме найденных паттернов (3-4 предложения)"
}}"""

            response = await self.ai_router.call_ai(
                prompt=prompt,
                stage='analysis',
                max_tokens=2000
            )

            result = self._safe_parse_json(response)

            if not result:
                return self._get_fallback_smc_analysis()

            return result

        except Exception as e:
            logger.error(f"Ошибка AI SMC анализа для {symbol}: {e}")
            return self._get_fallback_smc_analysis()

    async def interpret_correlation_anomaly(
            self,
            symbol: str,
            symbol_change: float,
            btc_change: float,
            correlation: float,
            sector_data: Optional[Dict] = None
    ) -> Dict:
        """
        AI интерпретация аномалий корреляции

        Args:
            symbol: торговая пара
            symbol_change: % изменение символа за период
            btc_change: % изменение BTC за период
            correlation: историческая корреляция
            sector_data: опционально - данные по сектору

        Returns:
            {
                'interpretation': 'STRENGTH' / 'WEAKNESS' / 'NOISE',
                'expected_continuation': True / False,
                'time_horizon': 'SHORT' / 'MEDIUM',
                'confidence_adjustment': +10 / -10 / 0,
                'reasoning': '...'
            }
        """
        try:
            sector_context = ""
            if sector_data and sector_data.get('sector'):
                sector_context = f"""
СЕКТОР: {sector_data['sector']}
Тренд сектора: {sector_data.get('sector_trend', 'UNKNOWN')}
Позиция {symbol}: {sector_data.get('symbol_vs_sector', 'UNKNOWN')}
"""

            prompt = f"""Ты chief analyst криптофонда. Объясни аномальное поведение {symbol} относительно BTC.

ДАННЫЕ:
BTC изменение (1H): {btc_change:+.2f}%
{symbol} изменение (1H): {symbol_change:+.2f}%
Историческая корреляция: {correlation:.3f}

{sector_context}

КОНТЕКСТ:
Обычно {symbol} движется {'вместе с' if correlation > 0 else 'против'} BTC (корреляция {correlation:.2f}).
Но сейчас {symbol} показывает {'сильное' if abs(symbol_change) > abs(btc_change) else 'слабое'} движение.

АНАЛИЗ:
1. **ЧТО ПРОИСХОДИТ?**
   - Это STRENGTH (символ сильнее рынка) или WEAKNESS (отстает)?
   - Decoupling от BTC - это хорошо или плохо?

2. **СЕКТОР** (если есть данные):
   - Весь сектор движется так или только {symbol}?
   - Если только {symbol} - это индивидуальная сила или риск?

3. **ИСТОРИЧЕСКИЕ АНАЛОГИИ**:
   - Что обычно происходит после такого расхождения?
   - Цена возвращается к корреляции или продолжает decoupling?

4. **ВРЕМЕННОЙ ГОРИЗОНТ**:
   - SHORT (1-4 часа) - временная аномалия
   - MEDIUM (4-24 часа) - начало нового тренда

5. **ВЕРДИКТ**:
   - BULLISH для {symbol}: +10 к confidence
   - BEARISH для {symbol}: -10 к confidence  
   - NOISE (шум): 0

Верни ТОЛЬКО JSON:
{{
  "interpretation": "STRENGTH|WEAKNESS|NOISE",
  "is_decoupling": true|false,
  "expected_continuation": true|false,
  "time_horizon": "SHORT|MEDIUM",
  "verdict": "BULLISH|BEARISH|NEUTRAL",
  "confidence_adjustment": -10 to +10,
  "reasoning": "объяснение что происходит и почему (3-4 предложения)"
}}"""

            response = await self.ai_router.call_ai(
                prompt=prompt,
                stage='analysis',
                max_tokens=1500
            )

            result = self._safe_parse_json(response)

            if not result:
                return {
                    'interpretation': 'UNKNOWN',
                    'expected_continuation': False,
                    'time_horizon': 'SHORT',
                    'confidence_adjustment': 0,
                    'reasoning': 'AI analysis failed, using neutral interpretation'
                }

            return result

        except Exception as e:
            logger.error(f"Ошибка AI интерпретации корреляции для {symbol}: {e}")
            return {
                'interpretation': 'UNKNOWN',
                'expected_continuation': False,
                'time_horizon': 'SHORT',
                'confidence_adjustment': 0,
                'reasoning': f'Error: {str(e)}'
            }

    async def interpret_volume_profile(
            self,
            symbol: str,
            volume_profile_data: Dict,
            current_price: float,
            price_direction: str
    ) -> Dict:
        """
        AI интерпретация Volume Profile

        Args:
            symbol: торговая пара
            volume_profile_data: {poc: price, value_area: [low, high], profile: {price: volume}}
            current_price: текущая цена
            price_direction: 'UP' / 'DOWN' / 'FLAT'

        Returns:
            {
                'poc_significance': 'STRONG_MAGNET' / 'WEAK' / 'EXPIRED',
                'value_area_position': 'ABOVE' / 'INSIDE' / 'BELOW',
                'expected_behavior': 'PULLBACK_TO_POC' / 'BREAKOUT' / 'RANGING',
                'confidence_adjustment': +8 / 0,
                'reasoning': '...'
            }
        """
        try:
            poc = volume_profile_data.get('poc', 0)
            value_area = volume_profile_data.get('value_area', [0, 0])

            poc_distance_pct = abs((current_price - poc) / current_price * 100) if current_price > 0 else 0

            # Позиция относительно Value Area
            if current_price > value_area[1]:
                va_position = 'ABOVE'
            elif current_price < value_area[0]:
                va_position = 'BELOW'
            else:
                va_position = 'INSIDE'

            prompt = f"""Ты специалист по Volume Profile анализу. Интерпретируй ситуацию для {symbol}.

ДАННЫЕ:
Текущая цена: {current_price}
POC (Point of Control): {poc}
Value Area: [{value_area[0]}, {value_area[1]}]
Дистанция до POC: {poc_distance_pct:.2f}%
Позиция относительно VA: {va_position}
Направление движения: {price_direction}

КОНТЕКСТ:
POC - уровень где торговался максимальный объем (это "магнит" для цены).
Value Area - зона где прошло 70% объема торгов.

АНАЛИЗ:
1. **ЗНАЧИМОСТЬ POC**:
   - Если цена близко к POC (<1%) - это сильный магнит?
   - Если далеко (>3%) - POC еще актуален или уже "протух"?

2. **ПОЗИЦИЯ ОТНОСИТЕЛЬНО VALUE AREA**:
   - ABOVE VA: цена перекуплена или это сила?
   - BELOW VA: перепродана или слабость?
   - INSIDE VA: нормальная торговая зона

3. **ОЖИДАЕМОЕ ПОВЕДЕНИЕ**:
   - PULLBACK_TO_POC: цена вернется к POC
   - BREAKOUT: цена продолжит движение от VA
   - RANGING: будет торговаться в VA

4. **ТОРГОВЫЕ РЕКОМЕНДАЦИИ**:
   - Если цена далеко от POC и движется к нему: +8 к confidence (хороший setup)
   - Если в LVN (Low Volume Node) зоне: 0 (быстрое прохождение)
   - Если протух POC: 0

Верни ТОЛЬКО JSON:
{{
  "poc_significance": "STRONG_MAGNET|WEAK|EXPIRED",
  "poc_distance_assessment": "CLOSE|MODERATE|FAR",
  "value_area_position": "{va_position}",
  "market_condition": "OVEREXTENDED|NORMAL|UNDEREXTENDED",
  "expected_behavior": "PULLBACK_TO_POC|BREAKOUT|RANGING",
  "optimal_entry_zone": "price level or 'current'",
  "confidence_adjustment": 0 to +8,
  "reasoning": "интерпретация ситуации (2-3 предложения)"
}}"""

            response = await self.ai_router.call_ai(
                prompt=prompt,
                stage='analysis',
                max_tokens=1200
            )

            result = self._safe_parse_json(response)

            if not result:
                return {
                    'poc_significance': 'UNKNOWN',
                    'value_area_position': va_position,
                    'expected_behavior': 'RANGING',
                    'confidence_adjustment': 0,
                    'reasoning': 'AI analysis failed'
                }

            return result

        except Exception as e:
            logger.error(f"Ошибка AI интерпретации VP для {symbol}: {e}")
            return {
                'poc_significance': 'UNKNOWN',
                'value_area_position': 'UNKNOWN',
                'expected_behavior': 'RANGING',
                'confidence_adjustment': 0,
                'reasoning': f'Error: {str(e)}'
            }

    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================

    def _format_orderbook_side(self, orders: List[List[float]], side: str) -> str:
        """Форматирование стакана для промпта"""
        lines = []
        for i, (price, size) in enumerate(orders[:20]):
            size_indicator = '█' * min(int(size / 1000), 20)  # Визуальная индикация размера
            lines.append(f"{i + 1}. ${price:,.2f} | {size:,.0f} | {size_indicator}")
        return '\n'.join(lines)

    def _safe_parse_json(self, response: str) -> Optional[Dict]:
        """Безопасный парсинг JSON из AI ответа"""
        try:
            # Удаляем markdown если есть
            response = response.strip()
            if response.startswith('```'):
                # Извлекаем JSON из markdown блока
                lines = response.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith('```'):
                        if in_json:
                            break
                        in_json = True
                        continue
                    if in_json:
                        json_lines.append(line)
                response = '\n'.join(json_lines)

            # Парсим JSON
            result = json.loads(response)
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Ошибка парсинга JSON от AI: {e}")
            logger.debug(f"Ответ AI: {response[:500]}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка парсинга: {e}")
            return None

    def _get_fallback_orderbook_analysis(self) -> Dict:
        """Fallback если AI не отвечает"""
        return {
            'orderflow_direction': 'NEUTRAL',
            'absorption_detected': False,
            'spoofing_risk': 'UNKNOWN',
            'key_levels': [],
            'confidence_adjustment': 0,
            'reasoning': 'AI analysis unavailable, using neutral assessment'
        }

    def _get_fallback_smc_analysis(self) -> Dict:
        """Fallback для SMC анализа"""
        return {
            'order_blocks': [],
            'fair_value_gaps': [],
            'liquidity_sweeps': [],
            'break_of_structure': None,
            'confidence_boost': 0,
            'reasoning': 'AI analysis unavailable'
        }


# ==================== ИНТЕГРАЦИОННЫЕ ФУНКЦИИ ====================

async def get_ai_orderflow_analysis(
        ai_router,
        symbol: str,
        orderbook_data: Dict,
        recent_prices: List[float]
) -> Dict:
    """
    Удобная функция для получения AI анализа Order Flow

    Args:
        ai_router: твой AIRouter
        symbol: торговая пара
        orderbook_data: данные стакана
        recent_prices: последние цены

    Returns:
        Результат анализа с confidence_adjustment
    """
    analyzer = AIAdvancedAnalyzer(ai_router)
    return await analyzer.analyze_orderbook_flow(symbol, orderbook_data, recent_prices)


async def get_ai_smc_patterns(
        ai_router,
        symbol: str,
        candles_1h: List[List],
        current_price: float
) -> Dict:
    """
    Удобная функция для получения AI анализа Smart Money Concepts

    Args:
        ai_router: твой AIRouter
        symbol: торговая пара
        candles_1h: свечи 1H
        current_price: текущая цена

    Returns:
        Найденные SMC паттерны с confidence_boost
    """
    analyzer = AIAdvancedAnalyzer(ai_router)
    return await analyzer.detect_smart_money_concepts(symbol, candles_1h, current_price)


async def get_ai_correlation_interpretation(
        ai_router,
        symbol: str,
        symbol_change: float,
        btc_change: float,
        correlation: float,
        sector_data: Optional[Dict] = None
) -> Dict:
    """
    Удобная функция для AI интерпретации корреляций

    Args:
        ai_router: твой AIRouter
        symbol: торговая пара
        symbol_change: % изменение
        btc_change: % изменение BTC
        correlation: корреляция
        sector_data: опционально - данные сектора

    Returns:
        Интерпретация аномалии
    """
    analyzer = AIAdvancedAnalyzer(ai_router)
    return await analyzer.interpret_correlation_anomaly(
        symbol, symbol_change, btc_change, correlation, sector_data
    )


async def get_ai_volume_profile_interpretation(
        ai_router,
        symbol: str,
        vp_data: Dict,
        current_price: float,
        price_direction: str
) -> Dict:
    """
    Удобная функция для AI интерпретации Volume Profile

    Args:
        ai_router: твой AIRouter
        symbol: торговая пара
        vp_data: данные Volume Profile
        current_price: текущая цена
        price_direction: направление

    Returns:
        Интерпретация VP
    """
    analyzer = AIAdvancedAnalyzer(ai_router)
    return await analyzer.interpret_volume_profile(
        symbol, vp_data, current_price, price_direction
    )