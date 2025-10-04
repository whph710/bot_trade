"""
Расширенный валидатор - ФИНАЛЬНАЯ ВЕРСИЯ

Критические изменения:
1. Всегда возвращает JSON с levels (даже если rejected)
2. 3 уровня TP адаптивно
3. Время удержания от/до адаптивно
4. Работает только с данными из Stage 3
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class EnhancedSignalValidator:
    """Комплексный валидатор"""

    def __init__(self, session, ai_router):
        self.session = session
        self.ai_router = ai_router

    async def validate_signal_from_stage3_data(
            self,
            signal: Dict,
            comprehensive_data: Dict
    ) -> Dict:
        """
        Валидация с данными из Stage 3
        ВСЕГДА возвращает JSON с levels
        """
        symbol = signal['symbol']
        signal_direction = signal['signal']
        original_confidence = signal.get('confidence', 0)

        logger.info(f"Validating {symbol} {signal_direction} (confidence: {original_confidence})")

        market_data = comprehensive_data.get('market_data', {})
        corr_data = comprehensive_data.get('correlation_data', {})
        vp_analysis = comprehensive_data.get('vp_analysis', {})
        orderflow_ai = comprehensive_data.get('orderflow_ai', {})
        smc_ai = comprehensive_data.get('smc_ai', {})

        adjustments = {
            'funding_rate': 0,
            'open_interest': 0,
            'spread': 0,
            'orderbook_imbalance': 0,
            'taker_volume': 0,
            'btc_correlation': 0,
            'sector_analysis': 0,
            'orderflow_ai': 0,
            'smc_patterns': 0,
            'volume_profile': 0,
            'key_levels': 0
        }

        blocking_factors = []
        validation_details = []

        try:
            # 1. FUNDING RATE
            if market_data.get('funding_rate'):
                funding_rate = market_data['funding_rate'].get('funding_rate', 0)

                if signal_direction == 'LONG' and funding_rate > 0.001:
                    adjustments['funding_rate'] = -15
                    validation_details.append(f"Funding: High funding {funding_rate*100:.3f}% - longs overleveraged")
                elif signal_direction == 'SHORT' and funding_rate < -0.001:
                    adjustments['funding_rate'] = -15
                    validation_details.append(f"Funding: Negative funding {funding_rate*100:.3f}% - shorts overleveraged")
                elif abs(funding_rate) < 0.0003:
                    adjustments['funding_rate'] = 0
                    validation_details.append(f"Funding: Funding rate {funding_rate*100:.3f}% - neutral leverage")
                else:
                    adjustments['funding_rate'] = 0
                    validation_details.append(f"Funding: Funding rate {funding_rate*100:.3f}% - acceptable")

            # 2. OPEN INTEREST
            if market_data.get('open_interest'):
                oi_trend = market_data['open_interest'].get('oi_trend', 'STABLE')
                oi_change = market_data['open_interest'].get('oi_change_24h', 0)

                candles_1h = comprehensive_data.get('candles_1h', [])
                if len(candles_1h) >= 10:
                    from func_correlation import determine_trend, extract_prices_from_candles
                    prices = extract_prices_from_candles(candles_1h)
                    price_direction = determine_trend(prices, window=10)
                else:
                    price_direction = 'FLAT'

                if price_direction == 'UP' and oi_trend == 'GROWING':
                    if signal_direction == 'LONG':
                        adjustments['open_interest'] = +12
                        validation_details.append(f"OI: Price rising + OI growing (+{oi_change:.1f}%) = strong bullish trend, new buyers")
                    else:
                        adjustments['open_interest'] = -10
                        validation_details.append(f"OI: Price rising + OI growing (+{oi_change:.1f}%), SHORT risky")

                elif price_direction == 'DOWN' and oi_trend == 'GROWING':
                    if signal_direction == 'SHORT':
                        adjustments['open_interest'] = +12
                        validation_details.append(f"OI: Price falling + OI growing (+{oi_change:.1f}%) = strong bearish trend")
                    else:
                        adjustments['open_interest'] = -10
                        validation_details.append(f"OI: Price falling + OI growing, LONG risky")

                elif price_direction == 'UP' and oi_trend == 'DECLINING':
                    adjustments['open_interest'] = -8
                    validation_details.append(f"OI: Price rising + OI declining = weak rally (short covering)")

                else:
                    adjustments['open_interest'] = +3
                    validation_details.append(f"OI: Price {price_direction} + OI {oi_trend} = moderate bullish")

            # 3. SPREAD
            if market_data.get('orderbook'):
                spread_pct = market_data['orderbook'].get('spread_pct', 0)

                if spread_pct > 0.15:
                    blocking_factors.append(f"SPREAD TOO WIDE: {spread_pct:.3f}%")
                    validation_details.append(f"Spread: Spread {spread_pct:.3f}% TOO WIDE - illiquid")
                elif spread_pct > 0.08:
                    adjustments['spread'] = -5
                    validation_details.append(f"Spread: Spread {spread_pct:.3f}% wide but acceptable")
                else:
                    adjustments['spread'] = 0
                    validation_details.append(f"Spread: Spread {spread_pct:.3f}% tight, good liquidity")

            # 4. ORDERBOOK IMBALANCE
            if market_data.get('orderbook'):
                bids = market_data['orderbook'].get('bids', [])
                asks = market_data['orderbook'].get('asks', [])

                if bids and asks:
                    bid_volume = sum(size for price, size in bids[:10])
                    ask_volume = sum(size for price, size in asks[:10])

                    if ask_volume > 0:
                        ratio = bid_volume / ask_volume

                        if signal_direction == 'LONG':
                            if ratio > 1.5:
                                adjustments['orderbook_imbalance'] = +10
                                validation_details.append(f"Orderbook: Bid/Ask ratio {ratio:.2f} - strong buy pressure")
                            elif ratio < 0.67:
                                adjustments['orderbook_imbalance'] = -10
                                validation_details.append(f"Orderbook: Bid/Ask ratio {ratio:.2f} - strong sell pressure in orderbook")
                            else:
                                adjustments['orderbook_imbalance'] = 0
                                validation_details.append(f"Orderbook: Bid/Ask ratio {ratio:.2f} - balanced orderbook")

                        elif signal_direction == 'SHORT':
                            if ratio < 0.67:
                                adjustments['orderbook_imbalance'] = +10
                                validation_details.append(f"Orderbook: Bid/Ask ratio {ratio:.2f} - strong sell pressure")
                            elif ratio > 1.5:
                                adjustments['orderbook_imbalance'] = -10
                                validation_details.append(f"Orderbook: Bid/Ask ratio {ratio:.2f} - strong buy pressure against SHORT")
                            else:
                                adjustments['orderbook_imbalance'] = 0
                                validation_details.append(f"Orderbook: Bid/Ask ratio {ratio:.2f} - balanced orderbook")

            # 5. TAKER VOLUME
            if market_data.get('taker_volume'):
                buy_pressure = market_data['taker_volume'].get('buy_pressure', 0.5)

                if signal_direction == 'LONG':
                    if buy_pressure > 0.60:
                        adjustments['taker_volume'] = +8
                        validation_details.append(f"Taker: Buy pressure {buy_pressure*100:.1f}% - aggressive buyers dominating")
                    elif buy_pressure < 0.40:
                        adjustments['taker_volume'] = -8
                        validation_details.append(f"Taker: Buy pressure {buy_pressure*100:.1f}% - aggressive sellers dominating")
                    else:
                        adjustments['taker_volume'] = 0
                        validation_details.append(f"Taker: Buy pressure {buy_pressure*100:.1f}% - balanced")

                elif signal_direction == 'SHORT':
                    if buy_pressure < 0.40:
                        adjustments['taker_volume'] = +8
                        validation_details.append(f"Taker: Buy pressure {buy_pressure*100:.1f}% - sellers dominating")
                    elif buy_pressure > 0.60:
                        adjustments['taker_volume'] = -8
                        validation_details.append(f"Taker: Buy pressure {buy_pressure*100:.1f}% - buyers dominating against SHORT")
                    else:
                        adjustments['taker_volume'] = 0
                        validation_details.append(f"Taker: Buy pressure {buy_pressure*100:.1f}% - balanced")

            # 6. BTC CORRELATION
            if corr_data:
                if corr_data.get('should_block_signal'):
                    blocking_factors.append(corr_data['btc_alignment']['reasoning'])
                    validation_details.append(f"BTC Corr: {corr_data['btc_alignment']['reasoning']}")
                else:
                    adjustments['btc_correlation'] = corr_data.get('total_confidence_adjustment', 0)
                    btc_corr = corr_data.get('btc_correlation', {})
                    validation_details.append(f"BTC Corr: {btc_corr.get('reasoning', 'No BTC correlation data')}")

            # 7. SECTOR ANALYSIS
            if corr_data and corr_data.get('sector_analysis'):
                sector_adj = corr_data['sector_analysis'].get('confidence_adjustment', 0)
                adjustments['sector_analysis'] = sector_adj
                validation_details.append(f"Sector: {corr_data['sector_analysis'].get('reasoning', 'No sector data')}")
            else:
                validation_details.append("Sector: No sector data")

            # 8. VOLUME PROFILE
            if vp_analysis:
                vp_adj = vp_analysis.get('total_confidence_adjustment', 0)
                adjustments['volume_profile'] = vp_adj

                vp_data = comprehensive_data.get('volume_profile', {})
                poc = vp_data.get('poc', 0)
                va = [vp_data.get('value_area_low', 0), vp_data.get('value_area_high', 0)]
                validation_details.append(f"VP: POC {poc}, VA [{va[0]}-{va[1]}]")

            # 9. KEY LEVELS
            from func_volume_profile import calculate_round_numbers_proximity
            current_price = comprehensive_data.get('current_price', 0)
            if current_price > 0:
                round_analysis = calculate_round_numbers_proximity(current_price)
                adjustments['key_levels'] = round_analysis.get('confidence_adjustment', 0)
                validation_details.append(f"Key Levels: {round_analysis.get('reasoning', 'No key levels')}")

            # 10. ORDERFLOW AI
            if orderflow_ai:
                adjustments['orderflow_ai'] = orderflow_ai.get('confidence_adjustment', 0)
                validation_details.append(f"OrderFlow AI: {orderflow_ai.get('reasoning', 'No orderflow AI')}")

                if orderflow_ai.get('spoofing_risk') == 'HIGH':
                    validation_details.append("OrderFlow AI: High spoofing risk detected")

            # 11. SMC PATTERNS
            if smc_ai:
                adjustments['smc_patterns'] = smc_ai.get('confidence_boost', 0)
                validation_details.append(f"SMC AI: {smc_ai.get('reasoning', 'No SMC analysis')}")

            # ФИНАЛЬНЫЙ РАСЧЕТ
            total_adjustment = sum(adjustments.values())
            final_confidence = original_confidence + total_adjustment
            final_confidence = max(0, min(100, final_confidence))

            MIN_CONFIDENCE = 70
            approved = final_confidence >= MIN_CONFIDENCE and len(blocking_factors) == 0

            # LEVELS (ВСЕГДА)
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)

            take_profit_levels = self._calculate_take_profit_levels(
                signal,
                comprehensive_data,
                approved
            )

            hold_time = self._calculate_hold_time(
                signal,
                comprehensive_data,
                approved
            )

            validation_summary = self._create_summary(
                symbol, signal_direction, original_confidence, final_confidence,
                adjustments, blocking_factors, approved
            )

            result = {
                'approved': approved,
                'final_confidence': final_confidence,
                'original_confidence': original_confidence,