"""
Расширенный валидатор - УЛУЧШЕННАЯ ВЕРСИЯ

Изменения:
- Работает с данными из Stage 3 (не собирает новые)
- Всегда возвращает JSON даже если не прошел валидацию
- Фокус на критичных факторах
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class EnhancedSignalValidator:
    """Комплексный валидатор (использует данные из Stage 3)"""

    def __init__(self, session, ai_router):
        self.session = session
        self.ai_router = ai_router

    async def validate_signal_from_stage3_data(
            self,
            signal: Dict,
            comprehensive_data: Dict
    ) -> Dict:
        """
        Валидация сигнала используя данные из Stage 3
        ВСЕГДА возвращает полный JSON с levels

        Args:
            signal: предварительный сигнал от AI
            comprehensive_data: все данные собранные в Stage 3

        Returns:
            Полный результат валидации с levels (даже если rejected)
        """
        symbol = signal['symbol']
        signal_direction = signal['signal']
        original_confidence = signal.get('confidence', 0)

        logger.info(f"Validating {symbol} {signal_direction} (confidence: {original_confidence})")

        # Извлекаем данные из Stage 3
        market_data = comprehensive_data.get('market_data', {})
        corr_data = comprehensive_data.get('correlation_data', {})
        vp_analysis = comprehensive_data.get('vp_analysis', {})
        orderflow_ai = comprehensive_data.get('orderflow_ai', {})
        smc_ai = comprehensive_data.get('smc_ai', {})

        adjustments = {
            'funding_rate': 0,
            'open_interest': 0,
            'spread': 0,
            'orderbook_pressure': 0,
            'taker_volume': 0,
            'btc_correlation': 0,
            'volume_profile': 0,
            'orderflow_ai': 0,
            'smc_patterns': 0
        }

        blocking_factors = []
        warnings = []

        try:
            # ========== КРИТИЧЕСКИЕ ПРОВЕРКИ ==========

            # 1. SPREAD (может заблокировать)
            if market_data.get('orderbook'):
                spread_pct = market_data['orderbook'].get('spread_pct', 0)

                if spread_pct > 0.15:  # >0.15% - illiquid
                    blocking_factors.append(f"SPREAD TOO WIDE: {spread_pct:.3f}%")
                elif spread_pct > 0.08:
                    adjustments['spread'] = -5
                    warnings.append(f"Wide spread {spread_pct:.3f}%")

            # 2. FUNDING RATE
            if market_data.get('funding_rate'):
                funding_rate = market_data['funding_rate'].get('funding_rate', 0)

                # Экстремальный funding против направления
                if signal_direction == 'LONG' and funding_rate > 0.001:  # >0.10%
                    adjustments['funding_rate'] = -15
                    warnings.append(f"High funding {funding_rate*100:.3f}% - longs overleveraged")
                elif signal_direction == 'SHORT' and funding_rate < -0.001:
                    adjustments['funding_rate'] = -15
                    warnings.append(f"Negative funding {funding_rate*100:.3f}% - shorts overleveraged")
                elif abs(funding_rate) < 0.0003:  # neutral
                    adjustments['funding_rate'] = 0
                else:  # умеренный в нашу пользу
                    adjustments['funding_rate'] = +5

            # 3. OPEN INTEREST
            if market_data.get('open_interest'):
                oi_trend = market_data['open_interest'].get('oi_trend', 'STABLE')
                oi_change = market_data['open_interest'].get('oi_change_24h', 0)

                # Определяем price direction из данных
                candles_1h = comprehensive_data.get('candles_1h', [])
                if len(candles_1h) >= 10:
                    from func_correlation import determine_trend, extract_prices_from_candles
                    prices = extract_prices_from_candles(candles_1h)
                    price_direction = determine_trend(prices, window=10)
                else:
                    price_direction = 'FLAT'

                # Price UP + OI UP = сильный тренд
                if price_direction == 'UP' and oi_trend == 'GROWING':
                    if signal_direction == 'LONG':
                        adjustments['open_interest'] = +12
                    else:
                        adjustments['open_interest'] = -10
                        warnings.append("OI growing on uptrend, SHORT risky")

                # Price DOWN + OI UP = сильный downtrend
                elif price_direction == 'DOWN' and oi_trend == 'GROWING':
                    if signal_direction == 'SHORT':
                        adjustments['open_interest'] = +12
                    else:
                        adjustments['open_interest'] = -10
                        warnings.append("OI growing on downtrend, LONG risky")

                # Price UP + OI DOWN = слабый rally (short covering)
                elif price_direction == 'UP' and oi_trend == 'DECLINING':
                    adjustments['open_interest'] = -8
                    warnings.append("OI declining on rally - weak move")

            # 4. ORDERBOOK PRESSURE
            if market_data.get('orderbook'):
                bids = market_data['orderbook'].get('bids', [])
                asks = market_data['orderbook'].get('asks', [])

                if bids and asks:
                    bid_volume = sum(size for price, size in bids[:10])
                    ask_volume = sum(size for price, size in asks[:10])

                    if ask_volume > 0:
                        ratio = bid_volume / ask_volume

                        if signal_direction == 'LONG':
                            if ratio > 1.5:  # strong bids
                                adjustments['orderbook_pressure'] = +10
                            elif ratio < 0.67:  # strong asks
                                adjustments['orderbook_pressure'] = -10
                                warnings.append(f"Orderbook against LONG (ratio {ratio:.2f})")

                        elif signal_direction == 'SHORT':
                            if ratio < 0.67:  # strong asks
                                adjustments['orderbook_pressure'] = +10
                            elif ratio > 1.5:  # strong bids
                                adjustments['orderbook_pressure'] = -10
                                warnings.append(f"Orderbook against SHORT (ratio {ratio:.2f})")

            # 5. TAKER VOLUME
            if market_data.get('taker_volume'):
                buy_pressure = market_data['taker_volume'].get('buy_pressure', 0.5)

                if signal_direction == 'LONG':
                    if buy_pressure > 0.60:  # strong buying
                        adjustments['taker_volume'] = +8
                    elif buy_pressure < 0.40:  # strong selling
                        adjustments['taker_volume'] = -12
                        warnings.append(f"Taker volume shows selling pressure: {buy_pressure*100:.1f}%")

                elif signal_direction == 'SHORT':
                    if buy_pressure < 0.40:  # strong selling
                        adjustments['taker_volume'] = +8
                    elif buy_pressure > 0.60:  # strong buying
                        adjustments['taker_volume'] = -12
                        warnings.append(f"Taker volume shows buying pressure: {buy_pressure*100:.1f}%")

            # 6. BTC CORRELATION (может заблокировать)
            if corr_data and corr_data.get('should_block_signal'):
                blocking_factors.append(corr_data['btc_alignment']['reasoning'])
            else:
                adjustments['btc_correlation'] = corr_data.get('total_confidence_adjustment', 0)

            # 7. VOLUME PROFILE
            if vp_analysis:
                adjustments['volume_profile'] = vp_analysis.get('total_confidence_adjustment', 0)

            # 8. AI ORDERFLOW
            if orderflow_ai:
                adjustments['orderflow_ai'] = orderflow_ai.get('confidence_adjustment', 0)

                # Проверка на spoofing
                if orderflow_ai.get('spoofing_risk') == 'HIGH':
                    warnings.append("High spoofing risk detected in orderbook")

            # 9. SMC PATTERNS
            if smc_ai:
                adjustments['smc_patterns'] = smc_ai.get('confidence_boost', 0)

            # ========== ФИНАЛЬНЫЙ РАСЧЕТ ==========

            total_adjustment = sum(adjustments.values())
            final_confidence = original_confidence + total_adjustment
            final_confidence = max(0, min(100, final_confidence))

            # Проверка порога
            MIN_CONFIDENCE = 70
            approved = final_confidence >= MIN_CONFIDENCE and len(blocking_factors) == 0

            # ========== ФОРМИРУЕМ LEVELS (ВСЕГДА) ==========

            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)

            # Take Profit Levels (3 уровня)
            take_profit_levels = self._calculate_take_profit_levels(
                signal,
                comprehensive_data,
                approved
            )

            # Hold Time
            hold_time = self._calculate_hold_time(
                signal,
                comprehensive_data,
                approved
            )

            # ========== РЕЗУЛЬТАТ ==========

            validation_summary = self._create_summary(
                symbol, signal_direction, original_confidence, final_confidence,
                adjustments, blocking_factors, warnings, approved
            )

            result = {
                'approved': approved,
                'final_confidence': final_confidence,
                'original_confidence': original_confidence,
                'total_adjustment': total_adjustment,
                'adjustments': adjustments,
                'blocking_factors': blocking_factors,
                'warnings': warnings,
                'validation_summary': validation_summary,

                # LEVELS (всегда присутствуют)
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit_levels': take_profit_levels,
                'hold_time_hours': hold_time,
                'risk_reward_ratio': self._calculate_best_rr(entry_price, stop_loss, take_profit_levels)
            }

            logger.info(f"{symbol}: {validation_summary}")

            return result

        except Exception as e:
            logger.error(f"Ошибка валидации {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Даже при ошибке возвращаем структуру с levels
            return {
                'approved': False,
                'final_confidence': 0,
                'original_confidence': original_confidence,
                'total_adjustment': 0,
                'adjustments': adjustments,
                'blocking_factors': [f"Validation error: {str(e)}"],
                'warnings': [],
                'validation_summary': f"ERROR: {str(e)}",
                'entry_price': signal.get('entry_price', 0),
                'stop_loss': signal.get('stop_loss', 0),
                'take_profit_levels': [signal.get('take_profit', 0)],
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
        Рассчитать 3 уровня Take Profit

        Логика:
        - TP1: консервативный (60% от полного движения)
        - TP2: базовый (100% - основной таргет)
        - TP3: расширенный (160% - при сильном импульсе)
        """
        signal_direction = signal.get('signal', 'LONG')
        entry = signal.get('entry_price', 0)
        stop = signal.get('stop_loss', 0)
        base_tp = signal.get('take_profit', 0)

        if entry == 0 or stop == 0:
            return [base_tp] if base_tp > 0 else []

        risk = abs(entry - stop)

        # Определяем силу сетапа
        vp_data = comprehensive_data.get('volume_profile', {})
        market_data = comprehensive_data.get('market_data', {})

        # Проверяем volume spike
        volume_spike = 1.0
        if market_data.get('taker_volume'):
            total_vol = market_data['taker_volume'].get('total_volume', 0)
            # Примерная оценка силы
            volume_spike = 1.5  # можно улучшить логику

        # Базовые множители
        if volume_spike > 2.0:
            multipliers = [1.5, 2.5, 4.0]  # сильный импульс
        elif volume_spike > 1.5:
            multipliers = [1.2, 2.0, 3.2]  # средний
        else:
            multipliers = [1.0, 1.5, 2.5]  # слабый

        # Рассчитываем levels
        if signal_direction == 'LONG':
            tp1 = entry + (risk * multipliers[0])
            tp2 = entry + (risk * multipliers[1])
            tp3 = entry + (risk * multipliers[2])
        else:  # SHORT
            tp1 = entry - (risk * multipliers[0])
            tp2 = entry - (risk * multipliers[1])
            tp3 = entry - (risk * multipliers[2])

        # Корректируем на ближайшие уровни сопротивления/поддержки
        if vp_data and vp_data.get('poc', 0) > 0:
            poc = vp_data['poc']
            va_high = vp_data.get('value_area_high', 0)
            va_low = vp_data.get('value_area_low', 0)

            # Проверяем не упираемся ли в POC
            for i, tp in enumerate([tp1, tp2, tp3]):
                if signal_direction == 'LONG':
                    # Если TP близко к POC сверху, сдвигаем чуть выше
                    if abs(tp - poc) / poc < 0.005 and tp > poc:
                        [tp1, tp2, tp3][i] = poc * 1.003
                else:  # SHORT
                    if abs(tp - poc) / poc < 0.005 and tp < poc:
                        [tp1, tp2, tp3][i] = poc * 0.997

        # Округляем до разумной точности
        tp_levels = [round(tp, 2) for tp in [tp1, tp2, tp3]]

        # Валидация: TP должны быть в правильном направлении
        if signal_direction == 'LONG':
            tp_levels = [tp for tp in tp_levels if tp > entry]
        else:
            tp_levels = [tp for tp in tp_levels if tp < entry]

        # Сортируем
        tp_levels.sort(reverse=(signal_direction == 'SHORT'))

        return tp_levels[:3]  # Максимум 3

    def _calculate_hold_time(
            self,
            signal: Dict,
            comprehensive_data: Dict,
            approved: bool
    ) -> Dict[str, int]:
        """
        Рассчитать время удержания позиции

        Returns:
            {'min': 4, 'max': 48}  # в часах
        """
        # Базовое время
        min_hold = 4
        max_hold = 48

        # Анализируем волатильность
        candles_1h = comprehensive_data.get('candles_1h', [])
        if len(candles_1h) >= 20:
            try:
                from func_trade import calculate_atr
                atr = calculate_atr(candles_1h, period=14)
                current_price = float(candles_1h[-1][4])

                if current_price > 0:
                    atr_pct = (atr / current_price) * 100

                    # Высокая волатильность = быстрее
                    if atr_pct > 2.0:
                        min_hold = 2
                        max_hold = 24
                    elif atr_pct > 1.5:
                        min_hold = 4
                        max_hold = 36
                    elif atr_pct < 0.5:
                        # Низкая волатильность = дольше
                        min_hold = 8
                        max_hold = 72
            except:
                pass

        # Проверяем силу сетапа
        smc_ai = comprehensive_data.get('smc_ai', {})
        if smc_ai:
            confidence_boost = smc_ai.get('confidence_boost', 0)
            if confidence_boost > 20:
                # Очень сильный сетап - может быть быстрое движение
                min_hold = max(2, min_hold - 2)
                max_hold = min(max_hold, 36)

        return {
            'min': min_hold,
            'max': max_hold
        }

    def _calculate_best_rr(
            self,
            entry: float,
            stop: float,
            tp_levels: List[float]
    ) -> float:
        """Рассчитать лучший R/R (обычно для TP2)"""
        if not tp_levels or entry == 0 or stop == 0:
            return 0

        risk = abs(entry - stop)
        if risk == 0:
            return 0

        # Используем TP2 (средний уровень) или TP1 если только один
        target_tp = tp_levels[1] if len(tp_levels) > 1 else tp_levels[0]
        reward = abs(target_tp - entry)

        return round(reward / risk, 2)

    def _create_summary(
            self,
            symbol: str,
            direction: str,
            orig_conf: int,
            final_conf: int,
            adjustments: Dict,
            blocking: List[str],
            warnings: List[str],
            approved: bool
    ) -> str:
        """Создать краткое резюме"""

        # Топ-3 adjustment
        top_adj = sorted(
            adjustments.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]

        adj_str = ", ".join([f"{k}: {v:+d}" for k, v in top_adj if v != 0])

        if approved:
            summary = f"✅ {symbol} {direction} APPROVED: {orig_conf}->{final_conf}%"
            if adj_str:
                summary += f" ({adj_str})"
        else:
            reason = blocking[0] if blocking else f"Low confidence {final_conf}%"
            summary = f"❌ {symbol} {direction} REJECTED: {reason}"
            if warnings:
                summary += f" | Warnings: {len(warnings)}"

        return summary


# ==================== BATCH VALIDATION (УЛУЧШЕННАЯ) ====================

async def validate_signals_batch_improved(
        validator: EnhancedSignalValidator,
        signals_with_data: List[Dict]
) -> Dict:
    """
    Пакетная валидация с данными из Stage 3

    Args:
        validator: экземпляр EnhancedSignalValidator
        signals_with_data: сигналы с comprehensive_data из Stage 3

    Returns:
        {'validated': [...], 'rejected': [...]}
    """
    validated_signals = []
    rejected_signals = []

    validation_tasks = []

    for signal in signals_with_data:
        comprehensive_data = signal.get('comprehensive_data', {})

        if not comprehensive_data:
            logger.warning(f"No comprehensive_data for {signal.get('symbol', 'UNKNOWN')}")
            rejected_signals.append({
                'symbol': signal.get('symbol'),
                'signal': signal.get('signal'),
                'rejection_reason': 'Missing comprehensive data from Stage 3'
            })
            continue

        task = validator.validate_signal_from_stage3_data(signal, comprehensive_data)
        validation_tasks.append((signal, task))

    # Выполняем все валидации
    results = []
    for signal, task in validation_tasks:
        try:
            result = await task
            results.append((signal, result))
        except Exception as e:
            logger.error(f"Validation failed for {signal.get('symbol')}: {e}")
            results.append((signal, {
                'approved': False,
                'final_confidence': 0,
                'blocking_factors': [f"Validation crashed: {str(e)}"],
                'entry_price': signal.get('entry_price', 0),
                'stop_loss': signal.get('stop_loss', 0),
                'take_profit_levels': [],
                'hold_time_hours': {'min': 4, 'max': 48}
            }))

    # Разделяем на approved/rejected
    for original_signal, validation_result in results:

        # Создаем финальный сигнал с validation данными
        final_signal = {
            'symbol': original_signal['symbol'],
            'signal': original_signal['signal'],
            'confidence': validation_result['final_confidence'],
            'original_confidence': validation_result['original_confidence'],
            'entry_price': validation_result['entry_price'],
            'stop_loss': validation_result['stop_loss'],
            'take_profit_levels': validation_result['take_profit_levels'],
            'hold_time_hours': validation_result['hold_time_hours'],
            'risk_reward_ratio': validation_result['risk_reward_ratio'],
            'analysis': original_signal.get('analysis', ''),
            'validation': validation_result,
            'timestamp': original_signal.get('timestamp')
        }

        if validation_result['approved']:
            validated_signals.append(final_signal)
        else:
            # Для rejected тоже возвращаем levels
            rejected_signals.append(final_signal)

    logger.info(f"Batch validation: {len(validated_signals)} approved, {len(rejected_signals)} rejected")

    return {
        'validated': validated_signals,
        'rejected': rejected_signals
    }


# ==================== QUICK MARKET CHECK (без изменений) ====================

async def quick_market_check(session, symbol: str) -> Dict:
    """Быстрая проверка критических параметров"""
    from func_market_data import MarketDataCollector, MarketDataAnalyzer

    collector = MarketDataCollector(session)
    analyzer = MarketDataAnalyzer()

    try:
        funding_data = await collector.get_funding_rate(symbol)
        orderbook_data = await collector.get_orderbook(symbol, depth=10)

        funding_ok = True
        spread_ok = True

        if funding_data:
            funding_analysis = analyzer.analyze_funding_rate(funding_data)
            if funding_analysis['risk_level'] == 'HIGH':
                funding_ok = False

        if orderbook_data:
            spread_analysis = analyzer.analyze_spread(orderbook_data)
            if not spread_analysis['tradeable']:
                spread_ok = False

        tradeable = funding_ok and spread_ok
        reason = "OK" if tradeable else "High risk detected"

        return {
            'tradeable': tradeable,
            'reason': reason,
            'spread_ok': spread_ok,
            'funding_ok': funding_ok
        }

    except Exception as e:
        return {
            'tradeable': False,
            'reason': f'Error: {str(e)}',
            'spread_ok': False,
            'funding_ok': False
        }


async def batch_quick_market_check(
        session,
        symbols: List[str],
        max_concurrent: int = 10
) -> Dict[str, Dict]:
    """Пакетная быстрая проверка"""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def check_with_semaphore(symbol):
        async with semaphore:
            return await quick_market_check(session, symbol)

    tasks = [check_with_semaphore(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        symbol: result if isinstance(result, dict) else {
            'tradeable': False,
            'reason': f'Error: {str(result)}'
        }
        for symbol, result in zip(symbols, results)
    }


def get_validation_statistics(validation_results: List[Dict]) -> Dict:
    """Собрать статистику по валидации"""
    if not validation_results:
        return {
            'total': 0,
            'approved': 0,
            'rejected': 0,
            'avg_confidence_change': 0
        }

    approved = [r for r in validation_results if r.get('approved', False)]

    confidence_changes = []
    for result in validation_results:
        if 'original_confidence' in result and 'final_confidence' in result:
            change = result['final_confidence'] - result['original_confidence']
            confidence_changes.append(change)

    avg_change = sum(confidence_changes) / len(confidence_changes) if confidence_changes else 0

    return {
        'total': len(validation_results),
        'approved': len(approved),
        'rejected': len(validation_results) - len(approved),
        'approval_rate': round(len(approved) / len(validation_results) * 100, 1),
        'avg_confidence_change': round(avg_change, 1)
    }