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

            # Расчет R/R
            risk = abs(entry_price - stop_loss) if entry_price > 0 and stop_loss > 0 else 0
            if risk > 0 and take_profit_levels:
                tp_base = take_profit_levels[1] if len(take_profit_levels) >= 2 else take_profit_levels[0]
                reward = abs(tp_base - entry_price)
                risk_reward_ratio = round(reward / risk, 2)
            else:
                risk_reward_ratio = 0

            validation_summary = self._create_summary(
                symbol, signal_direction, original_confidence, final_confidence,
                adjustments, blocking_factors, approved
            )

            result = {
                'approved': approved,
                'final_confidence': final_confidence,
                'original_confidence': original_confidence,
                'total_adjustment': total_adjustment,
                'adjustments': adjustments,
                'blocking_factors': blocking_factors,
                'validation_details': validation_details,
                'validation_summary': validation_summary,
                'take_profit_levels': take_profit_levels,
                'hold_time_hours': hold_time,
                'risk_reward_ratio': risk_reward_ratio,
                'market_data': {
                    'symbol': symbol,
                    'funding_rate': market_data.get('funding_rate'),
                    'open_interest': market_data.get('open_interest'),
                    'orderbook': market_data.get('orderbook'),
                    'taker_volume': market_data.get('taker_volume'),
                    'timestamp': market_data.get('timestamp')
                },
                'correlation_data': corr_data,
                'volume_profile_data': comprehensive_data.get('volume_profile')
            }

            return result

        except Exception as e:
            logger.error(f"Validation error for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Fallback - всё равно возвращаем JSON с levels
            return {
                'approved': False,
                'final_confidence': 0,
                'original_confidence': original_confidence,
                'total_adjustment': 0,
                'adjustments': adjustments,
                'blocking_factors': [f"Validation error: {str(e)}"],
                'validation_details': [f"Error: {str(e)}"],
                'validation_summary': f"❌ {symbol} VALIDATION ERROR",
                'take_profit_levels': self._calculate_take_profit_levels(signal, comprehensive_data, False),
                'hold_time_hours': {'min': 4, 'max': 48},
                'risk_reward_ratio': 0
            }

    def _calculate_take_profit_levels(
            self,
            signal: Dict,
            comprehensive_data: Dict,
            approved: bool
    ) -> List[float]:
        """
        Рассчитать 3 уровня TP адаптивно

        TP1: Консервативный (R/R ~1.5)
        TP2: Базовый таргет (R/R ~2.5)
        TP3: Расширенный (R/R ~4.0)
        """
        try:
            entry = signal.get('entry_price', 0)
            stop = signal.get('stop_loss', 0)
            signal_direction = signal.get('signal', 'LONG')

            if entry == 0 or stop == 0:
                return []

            risk = abs(entry - stop)

            if risk == 0:
                return []

            # Определяем множители на основе силы сетапа
            confidence = signal.get('confidence', 0)
            vp_analysis = comprehensive_data.get('vp_analysis', {})
            smc_ai = comprehensive_data.get('smc_ai', {})

            # Базовые множители
            multipliers = [1.5, 2.5, 4.0]  # Консервативный, Базовый, Расширенный

            # Корректируем на основе confidence
            if confidence >= 85:
                # Сильный сетап - можем быть агрессивнее
                multipliers = [1.8, 3.0, 5.0]
            elif confidence <= 75:
                # Слабый сетап - более консервативно
                multipliers = [1.3, 2.0, 3.0]

            # Корректируем на основе Volume Profile
            if vp_analysis and vp_analysis.get('total_confidence_adjustment', 0) > 0:
                # VP поддерживает - увеличиваем таргеты
                multipliers = [m * 1.15 for m in multipliers]

            # Корректируем на основе SMC
            if smc_ai and smc_ai.get('confidence_boost', 0) > 15:
                # Сильные SMC паттерны
                multipliers = [m * 1.1 for m in multipliers]

            # Рассчитываем уровни
            if signal_direction == 'LONG':
                tp_levels = [
                    round(entry + (risk * m), 2)
                    for m in multipliers
                ]
            else:  # SHORT
                tp_levels = [
                    round(entry - (risk * m), 2)
                    for m in multipliers
                ]

            # Проверяем на адекватность (не должны быть <= 0)
            tp_levels = [tp for tp in tp_levels if tp > 0]

            # Проверяем на ближайшие уровни S/R
            vp_data = comprehensive_data.get('volume_profile', {})
            if vp_data:
                tp_levels = self._adjust_tp_for_key_levels(
                    tp_levels,
                    vp_data,
                    signal_direction
                )

            return tp_levels[:3]  # Максимум 3 уровня

        except Exception as e:
            logger.error(f"Error calculating TP levels: {e}")
            return []

    def _adjust_tp_for_key_levels(
            self,
            tp_levels: List[float],
            vp_data: Dict,
            signal_direction: str
    ) -> List[float]:
        """Корректировка TP на основе ключевых уровней VP"""
        try:
            poc = vp_data.get('poc', 0)
            va_high = vp_data.get('value_area_high', 0)
            va_low = vp_data.get('value_area_low', 0)
            hvn_zones = vp_data.get('hvn_zones', [])

            adjusted = []

            for tp in tp_levels:
                # Проверяем близость к POC
                if poc > 0 and abs(tp - poc) / poc < 0.005:  # В пределах 0.5%
                    if signal_direction == 'LONG':
                        tp = poc - (poc * 0.003)  # Чуть ниже POC
                    else:
                        tp = poc + (poc * 0.003)  # Чуть выше POC

                # Проверяем HVN зоны
                for hvn_low, hvn_high in hvn_zones:
                    hvn_center = (hvn_low + hvn_high) / 2
                    if abs(tp - hvn_center) / tp < 0.005:
                        if signal_direction == 'LONG':
                            tp = hvn_low - (hvn_low * 0.003)
                        else:
                            tp = hvn_high + (hvn_high * 0.003)

                adjusted.append(round(tp, 2))

            return adjusted

        except Exception as e:
            logger.error(f"Error adjusting TP for key levels: {e}")
            return tp_levels

    def _calculate_hold_time(
            self,
            signal: Dict,
            comprehensive_data: Dict,
            approved: bool
    ) -> Dict:
        """
        Рассчитать адаптивное время удержания (от/до в часах)

        Returns:
            {'min': 4, 'max': 48}
        """
        try:
            confidence = signal.get('confidence', 0)

            # Извлекаем volume ratio
            indicators_1h = comprehensive_data.get('indicators_1h', {})
            current_state = indicators_1h.get('current', {})
            volume_ratio = current_state.get('volume_ratio', 1.0)

            # Базовое время
            if volume_ratio > 2.0:
                # Сильный импульс - быстрое движение
                min_hold = 2
                max_hold = 12
            elif volume_ratio > 1.5:
                # Средний импульс
                min_hold = 4
                max_hold = 24
            else:
                # Слабый импульс - медленное движение
                min_hold = 8
                max_hold = 48

            # Корректируем на основе confidence
            if confidence >= 85:
                # Высокий confidence - может быть быстрее
                max_hold = int(max_hold * 0.8)
            elif confidence <= 75:
                # Низкий confidence - даем больше времени
                max_hold = int(max_hold * 1.2)

            # Ограничения
            min_hold = max(1, min(min_hold, 12))
            max_hold = max(min_hold + 4, min(max_hold, 72))

            return {
                'min': min_hold,
                'max': max_hold
            }

        except Exception as e:
            logger.error(f"Error calculating hold time: {e}")
            return {'min': 4, 'max': 48}

    def _create_summary(
            self,
            symbol: str,
            direction: str,
            orig_conf: int,
            final_conf: int,
            adjustments: Dict,
            blocking: List,
            approved: bool
    ) -> str:
        """Создать краткую сводку валидации"""

        if not approved:
            if blocking:
                return f"❌ {symbol} {direction} REJECTED: {blocking[0]}"
            else:
                return f"❌ {symbol} {direction} REJECTED: Confidence {final_conf} < 70"

        # Топ-3 adjustment факторов
        top_adjustments = sorted(
            [(k, v) for k, v in adjustments.items() if v != 0],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]

        adj_str = ", ".join([f"{k}: {v:+d}" for k, v in top_adjustments])

        return f"✅ {symbol} {direction} APPROVED: {orig_conf}->{final_conf} ({adj_str})"


# ==================== BATCH VALIDATION ====================

async def validate_signals_batch_improved(
        validator: EnhancedSignalValidator,
        preliminary_signals: List[Dict]
) -> Dict:
    """
    Batch валидация сигналов

    Args:
        validator: экземпляр EnhancedSignalValidator
        preliminary_signals: список сигналов с comprehensive_data

    Returns:
        {'validated': [...], 'rejected': [...]}
    """
    validated = []
    rejected = []

    for signal in preliminary_signals:
        try:
            comprehensive_data = signal.get('comprehensive_data', {})

            if not comprehensive_data:
                logger.warning(f"No comprehensive_data for {signal['symbol']}, skipping")
                continue

            # Валидация
            validation_result = await validator.validate_signal_from_stage3_data(
                signal,
                comprehensive_data
            )

            # Формируем финальный сигнал
            final_signal = {
                'symbol': signal['symbol'],
                'signal': signal['signal'],
                'confidence': validation_result['final_confidence'],
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal.get('take_profit', 0),  # Старый single TP (для совместимости)
                'analysis': signal.get('analysis', ''),
                'ai_generated': signal.get('ai_generated', True),
                'timestamp': signal.get('timestamp'),
                'original_confidence': validation_result['original_confidence'],
                'validation': validation_result
            }

            # Добавляем новые поля
            final_signal['take_profit_levels'] = validation_result.get('take_profit_levels', [])
            final_signal['hold_time_hours'] = validation_result.get('hold_time_hours', {'min': 4, 'max': 48})
            final_signal['risk_reward_ratio'] = validation_result.get('risk_reward_ratio', 0)

            if validation_result['approved']:
                validated.append(final_signal)
                logger.info(f"✅ {signal['symbol']} validated: {validation_result['validation_summary']}")
            else:
                rejected.append(final_signal)
                logger.info(f"❌ {signal['symbol']} rejected: {validation_result['validation_summary']}")

        except Exception as e:
            logger.error(f"Error validating {signal.get('symbol', 'UNKNOWN')}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    return {
        'validated': validated,
        'rejected': rejected
    }


async def batch_quick_market_check(
        session,
        symbols: List[str],
        max_concurrent: int = 20
) -> Dict[str, Dict]:
    """
    Быстрая проверка рыночных условий (spread check)

    Returns:
        {symbol: {'tradeable': bool, 'reason': str}}
    """
    from func_market_data import MarketDataCollector

    collector = MarketDataCollector(session)
    results = {}

    # Создаем семафор для ограничения concurrent запросов
    semaphore = asyncio.Semaphore(max_concurrent)

    async def check_single(symbol: str):
        async with semaphore:
            try:
                # Только orderbook для проверки spread
                orderbook = await collector.get_orderbook(symbol, depth=10)

                if not orderbook:
                    return symbol, {'tradeable': False, 'reason': 'No orderbook data'}

                spread_pct = orderbook.get('spread_pct', 999)

                if spread_pct > 0.15:
                    return symbol, {'tradeable': False, 'reason': f'Wide spread {spread_pct:.3f}%'}

                return symbol, {'tradeable': True, 'reason': 'Good liquidity'}

            except Exception as e:
                logger.debug(f"Quick check error {symbol}: {e}")
                return symbol, {'tradeable': False, 'reason': 'Check failed'}

    # Запускаем все проверки параллельно
    tasks = [check_single(sym) for sym in symbols]
    check_results = await asyncio.gather(*tasks)

    for symbol, result in check_results:
        results[symbol] = result

    return results


def get_validation_statistics(validation_results: List[Dict]) -> Dict:
    """
    Собрать статистику по валидации

    Args:
        validation_results: список validation объектов из validated/rejected signals

    Returns:
        {
            'total': int,
            'approved': int,
            'rejected': int,
            'approval_rate': float,
            'avg_confidence_change': float,
            'top_rejection_reasons': [...]
        }
    """
    if not validation_results:
        return {
            'total': 0,
            'approved': 0,
            'rejected': 0,
            'approval_rate': 0.0,
            'avg_confidence_change': 0.0,
            'top_rejection_reasons': []
        }

    total = len(validation_results)
    approved = sum(1 for v in validation_results if v.get('approved', False))
    rejected = total - approved

    approval_rate = (approved / total * 100) if total > 0 else 0.0

    # Средние изменение confidence
    conf_changes = [
        v.get('total_adjustment', 0)
        for v in validation_results
    ]
    avg_change = sum(conf_changes) / len(conf_changes) if conf_changes else 0.0

    # Топ причин отклонения
    rejection_reasons = []
    for v in validation_results:
        if not v.get('approved', False):
            reasons = v.get('blocking_factors', [])
            rejection_reasons.extend(reasons)

    # Подсчитываем частоту
    from collections import Counter
    reason_counts = Counter(rejection_reasons)
    top_reasons = [reason for reason, count in reason_counts.most_common(5)]

    return {
        'total': total,
        'approved': approved,
        'rejected': rejected,
        'approval_rate': round(approval_rate, 1),
        'avg_confidence_change': round(avg_change, 1),
        'top_rejection_reasons': top_reasons
    }