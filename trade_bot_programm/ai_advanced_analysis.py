"""
AI advanced analysis module - FIXED: 1D данные опциональны
"""

import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AIAdvancedAnalyzer:
    """AI-driven unified analysis"""

    def __init__(self, ai_router):
        self.ai_router = ai_router

    async def analyze_comprehensive_unified(
            self,
            symbol: str,
            comprehensive_data: Dict
    ) -> Dict:
        """Unified analysis in one request"""
        try:
            candles_1h = comprehensive_data.get('candles_1h', [])
            candles_4h = comprehensive_data.get('candles_4h', [])
            candles_1d = comprehensive_data.get('candles_1d', [])  # Может быть пустым
            indicators_1h = comprehensive_data.get('indicators_1h', {})
            indicators_4h = comprehensive_data.get('indicators_4h', {})
            indicators_1d = comprehensive_data.get('indicators_1d', {})  # Может быть пустым
            has_1d_data = comprehensive_data.get('has_1d_data', False)
            current_price = comprehensive_data.get('current_price', 0)
            market_data = comprehensive_data.get('market_data', {})

            orderbook_data = market_data.get('orderbook', {})
            orderbook_formatted = None

            if orderbook_data and orderbook_data.get('bids') and orderbook_data.get('asks'):
                bids = orderbook_data['bids'][:20]
                asks = orderbook_data['asks'][:20]
                orderbook_formatted = {
                    'bids': [[float(p), float(s)] for p, s in bids],
                    'asks': [[float(p), float(s)] for p, s in asks],
                    'mid_price': orderbook_data.get('mid_price', current_price),
                    'spread_pct': orderbook_data.get('spread_pct', 0)
                }

            candles_1h_formatted = []
            for i, c in enumerate(candles_1h[-50:]):
                try:
                    o, h, l, cl, v = float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])
                    candles_1h_formatted.append({
                        'index': i,
                        'open': o,
                        'high': h,
                        'low': l,
                        'close': cl,
                        'volume': v,
                        'type': 'bullish' if cl > o else 'bearish'
                    })
                except (IndexError, ValueError):
                    continue

            # КРИТИЧНО: Даже если индикаторы 1D не рассчитаны, ОТПРАВЛЯЕМ СВЕЧИ AI
            # AI должен видеть price action на 1D даже без индикаторов
            unified_data = {
                'symbol': symbol,
                'current_price': current_price,
                'has_1d_data': has_1d_data,  # Флаг доступности
                'timeframes': {
                    '1h': {
                        'candles': candles_1h[-80:],
                        'indicators': {
                            'ema5': indicators_1h.get('ema5_history', [])[-80:],
                            'ema8': indicators_1h.get('ema8_history', [])[-80:],
                            'ema20': indicators_1h.get('ema20_history', [])[-80:],
                            'rsi': indicators_1h.get('rsi_history', [])[-80:],
                            'macd_histogram': indicators_1h.get('macd_histogram_history', [])[-80:],
                            'volume_ratio': indicators_1h.get('volume_ratio_history', [])[-80:]
                        }
                    },
                    '4h': {
                        'candles': candles_4h[-40:],
                        'indicators': {
                            'ema5': indicators_4h.get('ema5_history', [])[-40:],
                            'ema8': indicators_4h.get('ema8_history', [])[-40:],
                            'ema20': indicators_4h.get('ema20_history', [])[-40:],
                            'rsi': indicators_4h.get('rsi_history', [])[-40:],
                            'macd_histogram': indicators_4h.get('macd_histogram_history', [])[-40:]
                        }
                    },
                    '1d': {
                        # ВСЕГДА отправляем свечи (даже если мало для индикаторов)
                        'candles': candles_1d[-30:] if candles_1d else [],
                        'indicators': {
                            # Индикаторы могут быть пустыми, это нормально
                            'ema5': indicators_1d.get('ema5_history', [])[-30:] if indicators_1d else [],
                            'ema8': indicators_1d.get('ema8_history', [])[-30:] if indicators_1d else [],
                            'ema20': indicators_1d.get('ema20_history', [])[-30:] if indicators_1d else [],
                            'rsi': indicators_1d.get('rsi_history', [])[-30:] if indicators_1d else [],
                            'macd_histogram': indicators_1d.get('macd_histogram_history', [])[-30:] if indicators_1d else []
                        }
                    }
                },
                'current_state': {
                    'price': current_price,
                    'atr': indicators_1h.get('current', {}).get('atr', 0),
                    'rsi_1h': indicators_1h.get('current', {}).get('rsi', 50),
                    'rsi_4h': indicators_4h.get('current', {}).get('rsi', 50),
                    'rsi_1d': indicators_1d.get('current', {}).get('rsi', 50) if indicators_1d else None,
                    'volume_ratio': indicators_1h.get('current', {}).get('volume_ratio', 1.0),
                    'macd_momentum': indicators_1h.get('current', {}).get('macd_histogram', 0)
                },
                'orderbook': orderbook_formatted,
                'candles_for_smc': candles_1h_formatted,
                'market_context': self._extract_market_context(comprehensive_data)
            }

            prompt = self._create_unified_prompt(symbol, unified_data)

            logger.debug(f"AI call for {symbol}: prompt length={len(prompt)} chars, has_1d_data={has_1d_data}")

            response = await self.ai_router.call_ai(
                prompt=prompt,
                stage='analysis',
                max_tokens=4000,
                temperature=0.7
            )

            logger.debug(f"AI response for {symbol}: {len(response)} chars")

            if not response or response == "{}":
                logger.error(f"Empty AI response for {symbol}")
                return self._fallback_unified_analysis(symbol, current_price)

            result = self._safe_parse_json(response)

            if not result:
                logger.warning(f"Failed to parse analysis for {symbol}")
                logger.debug(f"Response preview: {response[:500]}")
                return self._fallback_unified_analysis(symbol, current_price)

            return self._validate_and_format_result(result, symbol, current_price)

        except Exception as e:
            logger.error(f"Unified analysis error for {symbol}: {e}", exc_info=True)
            return self._fallback_unified_analysis(symbol, current_price)

    def _create_unified_prompt(self, symbol: str, data: Dict) -> str:
        """Create unified prompt"""

        orderbook_str = "No orderbook data"
        if data.get('orderbook'):
            ob = data['orderbook']
            orderbook_str = f"""ORDERBOOK:
Mid: ${ob['mid_price']:,.2f}, Spread: {ob['spread_pct']:.4f}%
TOP 10 BIDS: {self._format_orderbook_levels(ob['bids'][:10])}
TOP 10 ASKS: {self._format_orderbook_levels(ob['asks'][:10])}"""

        smc_candles_str = json.dumps(data.get('candles_for_smc', [])[-20:], indent=2) if data.get('candles_for_smc') else "No SMC data"
        market_ctx = data.get('market_context', {})
        has_1d_data = data.get('has_1d_data', False)

        # Информация о доступности 1D
        candles_1d = data['timeframes']['1d']['candles']
        indicators_1d = data['timeframes']['1d']['indicators']

        if has_1d_data and candles_1d:
            data_availability = f"1D DATA: Available ({len(candles_1d)} candles)"
            if any(indicators_1d.values()):
                data_availability += " with indicators"
            else:
                data_availability += " WITHOUT indicators (insufficient history)"
        else:
            data_availability = "1D DATA: NOT AVAILABLE (use 4H as major timeframe)"

        prompt = f"""You are an institutional trader. Task: FULL analysis of {symbol} in ONE response.

{data_availability}

DATA:
Current Price: ${data['current_price']:,.2f}
ATR (1H): {data['current_state']['atr']:.2f}
RSI 1H: {data['current_state']['rsi_1h']:.1f}
RSI 4H: {data['current_state']['rsi_4h']:.1f}
RSI 1D: {data['current_state'].get('rsi_1d', 'N/A')}
Volume Ratio: {data['current_state']['volume_ratio']:.2f}x
MACD Momentum: {data['current_state']['macd_momentum']:.4f}

MARKET:
Funding: {market_ctx.get('funding_rate', 0):.4f}%
OI Trend: {market_ctx.get('oi_trend', 'UNKNOWN')}
Spread: {market_ctx.get('spread_pct', 0):.4f}%
Buy Pressure: {market_ctx.get('buy_pressure', 0.5):.2%}

{orderbook_str}

SMC CANDLES (last 20 on 1H):
{smc_candles_str}

1D CANDLES (for price action analysis):
{json.dumps(candles_1d[-20:], indent=2) if candles_1d else "NOT AVAILABLE"}

CRITICAL INSTRUCTION:
- If 1D data unavailable or indicators empty → USE 4H as major timeframe
- DO NOT reject signal due to missing 1D data
- Analyze price action from 1D candles even without indicators
- If no 1D candles → base major trend on 4H EMA9/EMA21

TASK - perform 3 ANALYSES IN ONE:

PART 1: ORDERFLOW (from orderbook)
1. BID/ASK IMBALANCE: Sum top-10 bid vs ask volume
2. ABSORPTION ZONES: Large walls (>3x avg size)?
3. SPOOFING: Fake orders (round numbers far from price)?
4. KEY LEVELS: Liquidity concentration

OUTPUT:
- orderflow_direction: BULLISH/BEARISH/NEUTRAL
- spoofing_risk: HIGH/MEDIUM/LOW
- confidence_adjustment: -10 to +10

PART 2: SMART MONEY CONCEPTS (1H)
1. ORDER BLOCKS: Last opposite candle BEFORE impulse
2. FAIR VALUE GAPS: Price gaps (high[i-1] not touching low[i+1])
3. LIQUIDITY SWEEPS: False breakouts with quick reversal
4. BREAK OF STRUCTURE: Trend structure break

OUTPUT:
- order_blocks: [{{"level": 43100, "type": "bullish"}}]
- fair_value_gaps: [{{"zone": [43000, 43200]}}]
- patterns_alignment: BULLISH/BEARISH/MIXED
- confidence_boost: 0 to +15

PART 3: FINAL DECISION
COMPRESSED SPRING setup:
- 4H: consolidation pattern (triangle/flag)
- 1H: Order Block for precise entry
- Confluence: all factors (trend, RSI, volume, orderflow, SMC)

ENTRY:
LONG: entry = resistance + ATR*0.2 OR OB_50%, stop = swing_low - 0.3%
SHORT: entry = support - ATR*0.2 OR OB_50%, stop = swing_high + 0.3%

TAKE PROFIT (3 levels):
risk = |entry - stop|
STRONG setup (vol >200%, clear OB): [1.8, 3.0, 5.0]
MEDIUM (vol 150-200%): [1.5, 2.5, 4.0]
WEAK (vol 120-150%): [1.3, 2.0, 3.0]

LONG: tp = [entry + risk*mult1, entry + risk*mult2, entry + risk*mult3]
SHORT: tp = [entry - risk*mult1, entry - risk*mult2, entry - risk*mult3]

R/R CHECK: (tp2 - entry)/risk >= 2.0

CONFIDENCE (base 50):
+15: compression <2.5%
+15: ATR squeeze
+12: clear pattern
+10: Order Block
+12: volume spike >150%
+8: SMC alignment
+8: orderflow aligned

ADAPT FOR MISSING 1D:
- If has_1d_data == false: use 4H for major trend
- If 1D candles present but no indicators: analyze price action only
- DO NOT subtract confidence for missing 1D data

MINIMUM: 70

REJECT if:
- confidence <70
- R/R <2.0
- contradictions
- no clear OB
- BUT NOT because 1D unavailable!

OUTPUT (JSON only, no markdown):
{{
  "orderflow_analysis": {{
    "orderflow_direction": "BULLISH",
    "absorption_zones": ["49750-49800: strong bids 500k"],
    "spoofing_risk": "LOW",
    "key_levels": [49800, 50200],
    "confidence_adjustment": 8
  }},
  "smc_analysis": {{
    "order_blocks": [{{"level": 43100, "type": "bullish", "zone": [43050, 43150]}}],
    "fair_value_gaps": [{{"zone": [42800, 43000], "filled": false}}],
    "liquidity_sweeps": [],
    "patterns_alignment": "BULLISH",
    "confidence_boost": 12
  }},
  "signal": "LONG",
  "confidence": 82,
  "entry_price": 43251.25,
  "stop_loss": 42980.50,
  "take_profit_levels": [43892.75, 44500.00, 45200.00],
  "analysis": "Bull flag 4H compression 1.8%. OB 1H $43100. Volume spike 170%. Orderflow: strong bid support. SMC: bullish OB confirmed. R/R 2.4:1. Hold 4-24h. (1D data unavailable, using 4H major trend)",
  "rejection_reason": null
}}

If REJECTING:
{{
  "signal": "NO_SIGNAL",
  "confidence": 45,
  "entry_price": 0,
  "stop_loss": 0,
  "take_profit_levels": [],
  "analysis": "Reason for rejection (NOT missing 1D data!)",
  "rejection_reason": "Weak compression 4.5%, no clear OB, funding against position",
  "orderflow_analysis": {{}},
  "smc_analysis": {{}}
}}

CRITICAL:
- take_profit_levels ALWAYS array of 3 numbers
- If NO_SIGNAL, still fill orderflow_analysis and smc_analysis
- analysis max 150 words
- NEVER reject because has_1d_data == false
"""
        return prompt

    def _format_orderbook_levels(self, orders: List[List[float]]) -> str:
        """Format orderbook levels"""
        lines = []
        for i, (price, size) in enumerate(orders[:10]):
            lines.append(f"{i+1}. ${price:,.2f} | {size:,.0f}")
        return '\n'.join(lines)

    def _extract_market_context(self, data: Dict) -> Dict:
        """Extract market context"""
        market_data = data.get('market_data', {})
        return {
            'funding_rate': market_data.get('funding_rate', {}).get('funding_rate', 0) * 100 if market_data.get('funding_rate') else 0,
            'oi_trend': market_data.get('open_interest', {}).get('oi_trend', 'UNKNOWN') if market_data.get('open_interest') else 'UNKNOWN',
            'spread_pct': market_data.get('orderbook', {}).get('spread_pct', 0) if market_data.get('orderbook') else 0,
            'buy_pressure': market_data.get('taker_volume', {}).get('buy_pressure', 0.5) if market_data.get('taker_volume') else 0.5
        }

    def _validate_and_format_result(self, result: Dict, symbol: str, current_price: float) -> Dict:
        """Validate and format result"""
        try:
            signal = str(result.get('signal', 'NO_SIGNAL')).upper()
            confidence = max(0, min(100, int(float(result.get('confidence', 0)))))

            entry_price = float(result.get('entry_price', 0) or 0)
            stop_loss = float(result.get('stop_loss', 0) or 0)

            tp_levels_raw = result.get('take_profit_levels', [])

            if isinstance(tp_levels_raw, list) and len(tp_levels_raw) >= 3:
                take_profit_levels = [float(tp) for tp in tp_levels_raw[:3]]
            elif isinstance(tp_levels_raw, list) and len(tp_levels_raw) > 0:
                base_tp = float(tp_levels_raw[0])
                take_profit_levels = [base_tp, base_tp * 1.1, base_tp * 1.2]
            else:
                if signal in ['LONG', 'SHORT'] and entry_price > 0 and stop_loss > 0:
                    risk = abs(entry_price - stop_loss)
                    if signal == 'LONG':
                        take_profit_levels = [
                            round(entry_price + risk * 1.5, 2),
                            round(entry_price + risk * 2.5, 2),
                            round(entry_price + risk * 4.0, 2)
                        ]
                    else:
                        take_profit_levels = [
                            round(entry_price - risk * 1.5, 2),
                            round(entry_price - risk * 2.5, 2),
                            round(entry_price - risk * 4.0, 2)
                        ]
                else:
                    take_profit_levels = [0, 0, 0]

            analysis = str(result.get('analysis', 'AI unified analysis'))
            rejection_reason = result.get('rejection_reason')

            orderflow = result.get('orderflow_analysis', {})
            smc = result.get('smc_analysis', {})

            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'entry_price': entry_price if entry_price > 0 else current_price,
                'stop_loss': stop_loss,
                'take_profit_levels': take_profit_levels,
                'analysis': analysis,
                'rejection_reason': rejection_reason,
                'orderflow_analysis': orderflow,
                'smc_analysis': smc,
                'ai_generated': True,
                'stage': 3
            }

        except Exception as e:
            logger.error(f"Error validating result for {symbol}: {e}")
            return self._fallback_unified_analysis(symbol, current_price)

    def _fallback_unified_analysis(self, symbol: str, current_price: float) -> Dict:
        """Fallback if AI failed"""
        return {
            'symbol': symbol,
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'entry_price': current_price,
            'stop_loss': 0,
            'take_profit_levels': [0, 0, 0],
            'analysis': 'Unified analysis failed',
            'rejection_reason': 'AI analysis unavailable',
            'orderflow_analysis': {},
            'smc_analysis': {},
            'ai_generated': False,
            'stage': 3
        }

    def _safe_parse_json(self, response: str) -> Optional[Dict]:
        """Safe JSON parsing with better error reporting"""
        try:
            response = response.strip()

            if not response:
                logger.error("Empty response from AI")
                return None

            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                if end != -1:
                    response = response[start:end].strip()
            elif '```' in response:
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

            start_idx = response.find('{')
            if start_idx == -1:
                logger.error(f"No JSON object found in response: {response[:200]}")
                return None

            brace_count = 0
            for i, char in enumerate(response[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = response[start_idx:i + 1]
                        parsed = json.loads(json_str)
                        logger.debug(f"Successfully parsed JSON: {len(json_str)} chars")
                        return parsed

            logger.error(f"Unmatched braces in response: {response[:300]}")
            return None

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error at line {e.lineno}, col {e.colno}: {e.msg}")
            logger.debug(f"Response preview: {response[:500]}")
            return None
        except Exception as e:
            logger.error(f"Parsing error: {e}")
            return None


async def get_unified_analysis(
        ai_router,
        symbol: str,
        comprehensive_data: Dict
) -> Dict:
    """Convenience function for unified analysis"""
    analyzer = AIAdvancedAnalyzer(ai_router)
    return await analyzer.analyze_comprehensive_unified(symbol, comprehensive_data)

            candles_1h = comprehensive_data.get('candles_1h', [])
            candles_4h = comprehensive_data.get('candles_4h', [])
            candles_1d = comprehensive_data.get('candles_1d', [])  # Может быть пустым
            indicators_1h = comprehensive_data.get('indicators_1h', {})
            indicators_4h = comprehensive_data.get('indicators_4h', {})
            indicators_1d = comprehensive_data.get('indicators_1d', {})  # Может быть пустым
            has_1d_data = comprehensive_data.get('has_1d_data', False)
            current_price = comprehensive_data.get('current_price', 0)
            market_data = comprehensive_data.get('market_data', {})

            orderbook_data = market_data.get('orderbook', {})
            orderbook_formatted = None

            if orderbook_data and orderbook_data.get('bids') and orderbook_data.get('asks'):
                bids = orderbook_data['bids'][:20]
                asks = orderbook_data['asks'][:20]
                orderbook_formatted = {
                    'bids': [[float(p), float(s)] for p, s in bids],
                    'asks': [[float(p), float(s)] for p, s in asks],
                    'mid_price': orderbook_data.get('mid_price', current_price),
                    'spread_pct': orderbook_data.get('spread_pct', 0)
                }

            candles_1h_formatted = []
            for i, c in enumerate(candles_1h[-50:]):
                try:
                    o, h, l, cl, v = float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])
                    candles_1h_formatted.append({
                        'index': i,
                        'open': o,
                        'high': h,
                        'low': l,
                        'close': cl,
                        'volume': v,
                        'type': 'bullish' if cl > o else 'bearish'
                    })
                except (IndexError, ValueError):
                    continue

            # КРИТИЧНО: Даже если индикаторы 1D не рассчитаны, ОТПРАВЛЯЕМ СВЕЧИ AI
            # AI должен видеть price action на 1D даже без индикаторов
            unified_data = {
                'symbol': symbol,
                'current_price': current_price,
                'has_1d_data': has_1d_data,  # Флаг доступности
                'timeframes': {
                    '1h': {
                        'candles': candles_1h[-80:],
                        'indicators': {
                            'ema5': indicators_1h.get('ema5_history', [])[-80:],
                            'ema8': indicators_1h.get('ema8_history', [])[-80:],
                            'ema20': indicators_1h.get('ema20_history', [])[-80:],
                            'rsi': indicators_1h.get('rsi_history', [])[-80:],
                            'macd_histogram': indicators_1h.get('macd_histogram_history', [])[-80:],
                            'volume_ratio': indicators_1h.get('volume_ratio_history', [])[-80:]
                        }
                    },
                    '4h': {
                        'candles': candles_4h[-40:],
                        'indicators': {
                            'ema5': indicators_4h.get('ema5_history', [])[-40:],
                            'ema8': indicators_4h.get('ema8_history', [])[-40:],
                            'ema20': indicators_4h.get('ema20_history', [])[-40:],
                            'rsi': indicators_4h.get('rsi_history', [])[-40:],
                            'macd_histogram': indicators_4h.get('macd_histogram_history', [])[-40:]
                        }
                    },
                    '1d': {
                        # ВСЕГДА отправляем свечи (даже если мало для индикаторов)
                        'candles': candles_1d[-30:] if candles_1d else [],
                        'indicators': {
                            # Индикаторы могут быть пустыми, это нормально
                            'ema5': indicators_1d.get('ema5_history', [])[-30:] if indicators_1d else [],
                            'ema8': indicators_1d.get('ema8_history', [])[-30:] if indicators_1d else [],
                            'ema20': indicators_1d.get('ema20_history', [])[-30:] if indicators_1d else [],
                            'rsi': indicators_1d.get('rsi_history', [])[-30:] if indicators_1d else [],
                            'macd_histogram': indicators_1d.get('macd_histogram_history', [])[-30:] if indicators_1d else []
                        }
                    }
                },
                'current_state': {
                    'price': current_price,
                    'atr': indicators_1h.get('current', {}).get('atr', 0),
                    'rsi_1h': indicators_1h.get('current', {}).get('rsi', 50),
                    'rsi_4h': indicators_4h.get('current', {}).get('rsi', 50),
                    'rsi_1d': indicators_1d.get('current', {}).get('rsi', 50) if indicators_1d else None,
                    'volume_ratio': indicators_1h.get('current', {}).get('volume_ratio', 1.0),
                    'macd_momentum': indicators_1h.get('current', {}).get('macd_histogram', 0)
                },
                'orderbook': orderbook_formatted,
                'candles_for_smc': candles_1h_formatted,
                'market_context': self._extract_market_context(comprehensive_data)
            }

            prompt = self._create_unified_prompt(symbol, unified_data)

            logger.debug(f"AI call for {symbol}: prompt length={len(prompt)} chars, has_1d_data={has_1d_data}")

            response = await self.ai_router.call_ai(
                prompt=prompt,
                stage='analysis',
                max_tokens=4000,
                temperature=0.7
            )

            logger.debug(f"AI response for {symbol}: {len(response)} chars")

            if not response or response == "{}":
                logger.error(f"Empty AI response for {symbol}")
                return self._fallback_unified_analysis(symbol, current_price)

            result = self._safe_parse_json(response)

            if not result:
                logger.warning(f"Failed to parse analysis for {symbol}")
                logger.debug(f"Response preview: {response[:500]}")
                return self._fallback_unified_analysis(symbol, current_price)

            return self._validate_and_format_result(result, symbol, current_price)

        except Exception as e:
            logger.error(f"Unified analysis error for {symbol}: {e}", exc_info=True)
            return self._fallback_unified_analysis(symbol, current_price)

    def _create_unified_prompt(self, symbol: str, data: Dict) -> str:
        """Create unified prompt"""

        orderbook_str = "No orderbook data"
        if data.get('orderbook'):
            ob = data['orderbook']
            orderbook_str = f"""ORDERBOOK:
Mid: ${ob['mid_price']:,.2f}, Spread: {ob['spread_pct']:.4f}%
TOP 10 BIDS: {self._format_orderbook_levels(ob['bids'][:10])}
TOP 10 ASKS: {self._format_orderbook_levels(ob['asks'][:10])}"""

        smc_candles_str = json.dumps(data.get('candles_for_smc', [])[-20:], indent=2) if data.get('candles_for_smc') else "No SMC data"
        market_ctx = data.get('market_context', {})
        has_1d_data = data.get('has_1d_data', False)

        # Информация о доступности 1D
        candles_1d = data['timeframes']['1d']['candles']
        indicators_1d = data['timeframes']['1d']['indicators']

        if has_1d_data and candles_1d:
            data_availability = f"1D DATA: Available ({len(candles_1d)} candles)"
            if any(indicators_1d.values()):
                data_availability += " with indicators"
            else:
                data_availability += " WITHOUT indicators (insufficient history)"
        else:
            data_availability = "1D DATA: NOT AVAILABLE (use 4H as major timeframe)"

        prompt = f"""You are an institutional trader. Task: FULL analysis of {symbol} in ONE response.

{data_availability}

DATA:
Current Price: ${data['current_price']:,.2f}
ATR (1H): {data['current_state']['atr']:.2f}
RSI 1H: {data['current_state']['rsi_1h']:.1f}
RSI 4H: {data['current_state']['rsi_4h']:.1f}
RSI 1D: {data['current_state'].get('rsi_1d', 'N/A')}
Volume Ratio: {data['current_state']['volume_ratio']:.2f}x
MACD Momentum: {data['current_state']['macd_momentum']:.4f}

MARKET:
Funding: {market_ctx.get('funding_rate', 0):.4f}%
OI Trend: {market_ctx.get('oi_trend', 'UNKNOWN')}
Spread: {market_ctx.get('spread_pct', 0):.4f}%
Buy Pressure: {market_ctx.get('buy_pressure', 0.5):.2%}

{orderbook_str}

SMC CANDLES (last 20 on 1H):
{smc_candles_str}

1D CANDLES (for price action analysis):
{json.dumps(candles_1d[-20:], indent=2) if candles_1d else "NOT AVAILABLE"}

CRITICAL INSTRUCTION:
- If 1D data unavailable or indicators empty → USE 4H as major timeframe
- DO NOT reject signal due to missing 1D data
- Analyze price action from 1D candles even without indicators
- If no 1D candles → base major trend on 4H EMA9/EMA21

TASK - perform 3 ANALYSES IN ONE:

[... остальной промпт как в prompt_analyze.txt ...]

OUTPUT (JSON only, no markdown):
{{
  "orderflow_analysis": {{...}},
  "smc_analysis": {{...}},
  "signal": "LONG",
  "confidence": 82,
  "entry_price": 43251.25,
  "stop_loss": 42980.50,
  "take_profit_levels": [43892.75, 44500.00, 45200.00],
  "analysis": "Analysis text (mention 1D data availability)",
  "rejection_reason": null
}}

If REJECTING:
{{
  "signal": "NO_SIGNAL",
  "confidence": 45,
  "entry_price": 0,
  "stop_loss": 0,
  "take_profit_levels": [],
  "analysis": "Reason for rejection (DO NOT reject due to missing 1D data!)",
  "rejection_reason": "Actual reason (NOT '1D data unavailable')",
  "orderflow_analysis": {{}},
  "smc_analysis": {{}}
}}

CRITICAL:
- take_profit_levels ALWAYS array of 3 numbers
- If NO_SIGNAL, still fill orderflow_analysis and smc_analysis
- analysis max 150 words
- NEVER reject because has_1d_data == false
"""
        return prompt