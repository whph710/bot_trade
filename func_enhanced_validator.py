"""
Расширенный валидатор сигналов
Интегрирует все новые проверки: funding, OI, correlations, orderflow, SMC, VP
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class EnhancedSignalValidator:
    """
    Комплексный валидатор сигналов со всеми проверками
    """

    def __init__(
            self,
            session,  # aiohttp session
            ai_router  # твой AIRouter
    ):
        self.session = session
        self.ai_router = ai_router

        # Импортируем модули (будут доступны после интеграции)
        from func_market_data import MarketDataCollector, MarketDataAnalyzer
        from func_correlation import CorrelationAnalyzer
        from func_volume_profile import VolumeProfileCalculator, VolumeProfileAnalyzer
        from ai_advanced_analysis import AIAdvancedAnalyzer

        self.market_collector = MarketDataCollector(session)
        self.market_analyzer = MarketDataAnalyzer()
        self.corr_analyzer = CorrelationAnalyzer()
        self.vp_calculator = VolumeProfileCalculator()
        self.vp_analyzer = VolumeProfileAnalyzer()
        self.ai_analyzer = AIAdvancedAnalyzer(ai_router)

    async def validate_signals_batch(
            validator: EnhancedSignalValidator,
            signals: List[Dict],
            candles_data: Dict[str, Dict],  # {symbol: {'1h': candles, '4h': candles}}
            btc_candles_1h: List[List],
            sector_candles: Optional[Dict[str, Dict]] = None
    ) -> Dict:
        """
        Пакетная валидация множества сигналов

        Args:
            validator: экземпляр EnhancedSignalValidator
            signals: список предварительных сигналов
            candles_data: данные свечей для всех символов
            btc_candles_1h: свечи BTC
            sector_candles: опционально - свечи секторов

        Returns:
            {
                'validated': [signals],
                'rejected': [signals],
                'validation_stats': {...}
            }
        """

    validated_signals = []
    rejected_signals = []

    validation_tasks = []

    for signal in signals:
        symbol = signal['symbol']

        # Проверяем наличие данных
        if symbol not in candles_data:
            logger.warning(f"Нет данных свечей для {symbol}, пропускаем")
            rejected_signals.append({
                **signal,
                'rejection_reason': 'No candle data available'
            })
            continue

        symbol_data = candles_data[symbol]

        # Подготавливаем данные сектора если есть
        symbol_sector_candles = None
        if sector_candles:
            from func_correlation import CorrelationAnalyzer
            analyzer = CorrelationAnalyzer()
            peers = analyzer.get_sector_peers(symbol)

            if peers:
                symbol_sector_candles = {}
                for peer in peers:
                    if peer in sector_candles:
                        symbol_sector_candles[peer] = sector_candles[peer].get('1h', [])

        # Создаем задачу валидации
        task = validator.validate_signal_comprehensive(
            signal,
            symbol_data.get('1h', []),
            symbol_data.get('4h', []),
            btc_candles_1h,
            symbol_sector_candles
        )

        validation_tasks.append((signal, task))

    # Выполняем все валидации параллельно
    results = []
    for signal, task in validation_tasks:
        try:
            result = await task
            results.append((signal, result))
        except Exception as e:
            logger.error(f"Ошибка валидации {signal['symbol']}: {e}")
            results.append((signal, {
                'approved': False,
                'final_confidence': 0,
                'blocking_factors': [f"Validation failed: {str(e)}"]
            }))

    # Разделяем на одобренные и отклоненные
    for original_signal, validation_result in results:
        if validation_result['approved']:
            # Обновляем сигнал с новыми данными
            updated_signal = {
                **original_signal,
                'confidence': validation_result['final_confidence'],
                'original_confidence': validation_result['original_confidence'],
                'validation': validation_result
            }
            validated_signals.append(updated_signal)
        else:
            # Создаем очищенную версию для rejected
            rejected_signal = {
                'symbol': original_signal['symbol'],
                'signal': original_signal['signal'],
                'confidence': original_signal.get('confidence', 0),
                'entry_price': original_signal.get('entry_price', 0),
                'rejection_reason': validation_result.get('blocking_factors', ['Unknown'])[0],
                'validation_summary': validation_result.get('validation_summary', '')
            }
            rejected_signals.append(rejected_signal)

    # Статистика
    validation_stats = {
        'total_signals': len(signals),
        'validated': len(validated_signals),
        'rejected': len(rejected_signals),
        'validation_rate': round(len(validated_signals) / len(signals) * 100, 1) if signals else 0
    }

    logger.info(f"Batch validation: {validation_stats['validated']}/{validation_stats['total_signals']} approved")

    return {
        'validated': validated_signals,
        'rejected': rejected_signals,
        'validation_stats': validation_stats
    }


# ==================== QUICK CHECKS ====================

async def quick_market_check(
        session,
        symbol: str
) -> Dict:
    """
    Быстрая проверка только критических параметров (для фильтрации на ранних этапах)

    Args:
        session: aiohttp session
        symbol: торговая пара

    Returns:
        {
            'tradeable': True / False,
            'reason': str,
            'spread_ok': bool,
            'funding_ok': bool
        }
    """
    from func_market_data import MarketDataCollector, MarketDataAnalyzer

    collector = MarketDataCollector(session)
    analyzer = MarketDataAnalyzer()

    try:
        # Собираем только критичные данные
        funding_data = await collector.get_funding_rate(symbol)
        orderbook_data = await collector.get_orderbook(symbol, depth=10)

        # Проверка funding
        funding_ok = True
        funding_reason = ""

        if funding_data:
            funding_analysis = analyzer.analyze_funding_rate(funding_data)
            if funding_analysis['risk_level'] == 'HIGH':
                funding_ok = False
                funding_reason = funding_analysis['reasoning']

        # Проверка spread
        spread_ok = True
        spread_reason = ""

        if orderbook_data:
            spread_analysis = analyzer.analyze_spread(orderbook_data)
            if not spread_analysis['tradeable']:
                spread_ok = False
                spread_reason = spread_analysis['reasoning']

        # Итоговое решение
        tradeable = funding_ok and spread_ok

        if not tradeable:
            reason = spread_reason if not spread_ok else funding_reason
        else:
            reason = "All quick checks passed"

        return {
            'tradeable': tradeable,
            'reason': reason,
            'spread_ok': spread_ok,
            'funding_ok': funding_ok
        }

    except Exception as e:
        logger.error(f"Ошибка quick check для {symbol}: {e}")
        return {
            'tradeable': False,
            'reason': f"Error: {str(e)}",
            'spread_ok': False,
            'funding_ok': False
        }


async def batch_quick_market_check(
        session,
        symbols: List[str],
        max_concurrent: int = 10
) -> Dict[str, Dict]:
    """
    Пакетная быстрая проверка множества пар

    Returns:
        {symbol: quick_check_result}
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def check_with_semaphore(symbol):
        async with semaphore:
            return await quick_market_check(session, symbol)

    tasks = [check_with_semaphore(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        symbol: result if isinstance(result, dict) else {
            'tradeable': False,
            'reason': f'Error: {str(result)}',
            'spread_ok': False,
            'funding_ok': False
        }
        for symbol, result in zip(symbols, results)
    }


# ==================== HELPER FUNCTIONS ====================

def calculate_confidence_components(adjustments: Dict) -> Dict:
    """
    Разбить adjustments на категории для анализа

    Returns:
        {
            'market_data_score': int,
            'correlation_score': int,
            'ai_analysis_score': int,
            'technical_score': int
        }
    """
    market_data_score = (
            adjustments.get('funding_rate', 0) +
            adjustments.get('open_interest', 0) +
            adjustments.get('spread', 0) +
            adjustments.get('orderbook_imbalance', 0) +
            adjustments.get('taker_volume', 0)
    )

    correlation_score = (
            adjustments.get('btc_correlation', 0) +
            adjustments.get('sector_analysis', 0)
    )

    ai_analysis_score = (
            adjustments.get('orderflow_ai', 0) +
            adjustments.get('smc_patterns', 0)
    )

    technical_score = (
            adjustments.get('volume_profile', 0) +
            adjustments.get('key_levels', 0)
    )

    return {
        'market_data_score': market_data_score,
        'correlation_score': correlation_score,
        'ai_analysis_score': ai_analysis_score,
        'technical_score': technical_score,
        'total_score': market_data_score + correlation_score + ai_analysis_score + technical_score
    }


def create_validation_report(validation_results: List[Dict]) -> str:
    """
    Создать текстовый отчет по валидации

    Args:
        validation_results: список результатов валидации

    Returns:
        Форматированный текстовый отчет
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("VALIDATION REPORT")
    report_lines.append("=" * 80)

    approved = [r for r in validation_results if r.get('approved', False)]
    rejected = [r for r in validation_results if not r.get('approved', False)]

    report_lines.append(f"\nTotal Signals: {len(validation_results)}")
    report_lines.append(f"✅ Approved: {len(approved)}")
    report_lines.append(f"❌ Rejected: {len(rejected)}")
    report_lines.append(f"Success Rate: {len(approved) / len(validation_results) * 100:.1f}%")

    if approved:
        report_lines.append("\n" + "-" * 80)
        report_lines.append("APPROVED SIGNALS:")
        report_lines.append("-" * 80)

        for result in approved:
            report_lines.append(f"\n{result.get('validation_summary', 'N/A')}")

            # Breakdown adjustments
            if 'adjustments' in result:
                components = calculate_confidence_components(result['adjustments'])
                report_lines.append(f"  Breakdown: Market={components['market_data_score']:+d}, "
                                    f"Corr={components['correlation_score']:+d}, "
                                    f"AI={components['ai_analysis_score']:+d}, "
                                    f"Tech={components['technical_score']:+d}")

    if rejected:
        report_lines.append("\n" + "-" * 80)
        report_lines.append("REJECTED SIGNALS:")
        report_lines.append("-" * 80)

        for result in rejected:
            report_lines.append(f"\n{result.get('validation_summary', 'N/A')}")

            if 'blocking_factors' in result and result['blocking_factors']:
                for factor in result['blocking_factors'][:3]:  # Top 3
                    report_lines.append(f"  - {factor}")

    report_lines.append("\n" + "=" * 80)

    return "\n".join(report_lines)


def get_validation_statistics(validation_results: List[Dict]) -> Dict:
    """
    Собрать статистику по валидации

    Returns:
        Подробная статистика
    """
    if not validation_results:
        return {
            'total': 0,
            'approved': 0,
            'rejected': 0,
            'avg_confidence_change': 0,
            'top_rejection_reasons': []
        }

    approved = [r for r in validation_results if r.get('approved', False)]
    rejected = [r for r in validation_results if not r.get('approved', False)]

    # Среднее изменение confidence
    confidence_changes = []
    for result in validation_results:
        if 'original_confidence' in result and 'final_confidence' in result:
            change = result['final_confidence'] - result['original_confidence']
            confidence_changes.append(change)

    avg_confidence_change = sum(confidence_changes) / len(confidence_changes) if confidence_changes else 0

    # Топ причин отклонения
    rejection_reasons = []
    for result in rejected:
        if 'blocking_factors' in result and result['blocking_factors']:
            rejection_reasons.extend(result['blocking_factors'])

    from collections import Counter
    top_rejections = Counter(rejection_reasons).most_common(5)

    return {
        'total': len(validation_results),
        'approved': len(approved),
        'rejected': len(rejected),
        'approval_rate': round(len(approved) / len(validation_results) * 100, 1),
        'avg_confidence_change': round(avg_confidence_change, 1),
        'top_rejection_reasons': [f"{reason} ({count}x)" for reason, count in top_rejections]
    }


signal_comprehensive(
    self,
    signal: Dict,
symbol_candles_1h: List[List],
symbol_candles_4h: List[List],
btc_candles_1h: List[List],
sector_candles: Optional[Dict[str, List[List]]] = None
) -> Dict:
"""
ПОЛНАЯ валидация сигнала со ВСЕМИ проверками

Args:
    signal: предварительный сигнал от AI
    symbol_candles_1h: свечи символа 1H
    symbol_candles_4h: свечи символа 4H
    btc_candles_1h: свечи BTC 1H
    sector_candles: опционально - свечи пар сектора

Returns:
    {
        'approved': True / False,
        'final_confidence': int,
        'original_confidence': int,
        'adjustments': {...},
        'blocking_factors': [...],
        'validation_summary': str
    }
"""
symbol = signal['symbol']
signal_direction = signal['signal']
original_confidence = signal.get('confidence', 0)

logger.info(f"Начало расширенной валидации {symbol} {signal_direction} (confidence: {original_confidence})")

# Инициализируем результаты
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
    # ========== ЭТАП 1: MARKET DATA (КРИТИЧНО) ==========
    logger.debug(f"{symbol}: Сбор market data...")

    # Определяем направление цены
    if len(symbol_candles_1h) >= 10:
        from func_correlation import determine_trend, extract_prices_from_candles

        prices_1h = extract_prices_from_candles(symbol_candles_1h)
        price_direction = determine_trend(prices_1h, window=10)
    else:
        price_direction = 'FLAT'

    # Собираем market snapshot
    market_snapshot = await self.market_collector.get_market_snapshot(symbol)

    # 1. Funding Rate
    if market_snapshot['funding_rate']:
        funding_analysis = self.market_analyzer.analyze_funding_rate(
            market_snapshot['funding_rate']
        )
        adjustments['funding_rate'] = funding_analysis['confidence_adjustment']
        validation_details.append(f"Funding: {funding_analysis['reasoning']}")

        if funding_analysis['risk_level'] == 'HIGH':
            blocking_factors.append(funding_analysis['reasoning'])

    # 2. Open Interest
    if market_snapshot['open_interest']:
        oi_analysis = self.market_analyzer.analyze_open_interest(
            market_snapshot['open_interest'],
            price_direction
        )
        adjustments['open_interest'] = oi_analysis['confidence_adjustment']
        validation_details.append(f"OI: {oi_analysis['reasoning']}")

    # 3. Spread (КРИТИЧНО - может заблокировать)
    if market_snapshot['orderbook']:
        spread_analysis = self.market_analyzer.analyze_spread(
            market_snapshot['orderbook']
        )
        adjustments['spread'] = spread_analysis['confidence_adjustment']
        validation_details.append(f"Spread: {spread_analysis['reasoning']}")

        if not spread_analysis['tradeable']:
            blocking_factors.append(f"BLOCK: {spread_analysis['reasoning']}")
            # Это критический блокер - сразу отклоняем
            return self._create_rejection_result(
                signal, original_confidence, adjustments, blocking_factors, validation_details
            )

    # 4. Orderbook Imbalance
    if market_snapshot['orderbook']:
        imbalance_analysis = self.market_analyzer.analyze_orderbook_imbalance(
            market_snapshot['orderbook']
        )
        adjustments['orderbook_imbalance'] = imbalance_analysis['confidence_adjustment']
        validation_details.append(f"Orderbook: {imbalance_analysis['reasoning']}")

    # 5. Taker Volume
    if market_snapshot['taker_volume']:
        taker_analysis = self.market_analyzer.analyze_taker_volume(
            market_snapshot['taker_volume']
        )
        adjustments['taker_volume'] = taker_analysis['confidence_adjustment']
        validation_details.append(f"Taker: {taker_analysis['reasoning']}")

    # ========== ЭТАП 2: CORRELATION ANALYSIS ==========
    logger.debug(f"{symbol}: Корреляционный анализ...")

    from func_correlation import get_comprehensive_correlation_analysis

    corr_analysis = await get_comprehensive_correlation_analysis(
        symbol,
        symbol_candles_1h,
        btc_candles_1h,
        signal_direction,
        sector_candles
    )

    # Проверка BTC alignment (может заблокировать)
    if corr_analysis.get('should_block_signal'):
        blocking_factors.append(corr_analysis['btc_alignment']['reasoning'])
        # Это критический блокер
        return self._create_rejection_result(
            signal, original_confidence, adjustments, blocking_factors, validation_details
        )

    adjustments['btc_correlation'] = corr_analysis.get('total_confidence_adjustment', 0)
    validation_details.append(f"BTC Corr: {corr_analysis['btc_correlation']['reasoning']}")

    if 'sector_analysis' in corr_analysis:
        adjustments['sector_analysis'] = corr_analysis['sector_analysis'].get('confidence_adjustment', 0)
        validation_details.append(f"Sector: {corr_analysis['sector_analysis']['reasoning']}")

    # ========== ЭТАП 3: VOLUME PROFILE ==========
    logger.debug(f"{symbol}: Расчет Volume Profile...")

    from func_volume_profile import (
        calculate_volume_profile_for_candles,
        analyze_volume_profile,
        analyze_key_levels
    )

    # Рассчитываем VP на 4H данных (больше истории)
    vp_data = calculate_volume_profile_for_candles(symbol_candles_4h, num_bins=50)

    current_price = float(symbol_candles_1h[-1][4]) if symbol_candles_1h else 0

    if vp_data and vp_data.get('poc', 0) > 0:
        vp_analysis = analyze_volume_profile(vp_data, current_price)
        adjustments['volume_profile'] = vp_analysis['total_confidence_adjustment']

        # Ключевые уровни (круглые числа, PDH/PDL, VP)
        key_levels_analysis = analyze_key_levels(current_price, vp_data, symbol_candles_1h)
        adjustments['key_levels'] = key_levels_analysis['total_confidence_adjustment']

        validation_details.append(
            f"VP: POC {vp_data['poc']:.2f}, VA [{vp_data['value_area_low']:.2f}-{vp_data['value_area_high']:.2f}]")
        validation_details.append(f"Key Levels: {key_levels_analysis['round_numbers']['reasoning']}")

    # ========== ЭТАП 4: AI ADVANCED ANALYSIS ==========
    logger.debug(f"{symbol}: AI продвинутый анализ...")

    # 4.1 Order Flow AI
    if market_snapshot['orderbook']:
        prices_recent = [float(c[4]) for c in symbol_candles_1h[-20:]]

        orderflow_ai = await self.ai_analyzer.analyze_orderbook_flow(
            symbol,
            market_snapshot['orderbook'],
            prices_recent
        )

        adjustments['orderflow_ai'] = orderflow_ai.get('confidence_adjustment', 0)
        validation_details.append(f"OrderFlow AI: {orderflow_ai.get('reasoning', 'N/A')}")

    # 4.2 Smart Money Concepts AI
    smc_ai = await self.ai_analyzer.detect_smart_money_concepts(
        symbol,
        symbol_candles_1h,
        current_price
    )

    adjustments['smc_patterns'] = smc_ai.get('confidence_boost', 0)
    validation_details.append(f"SMC AI: {smc_ai.get('reasoning', 'N/A')}")

    # ========== ЭТАП 5: ФИНАЛЬНЫЙ РАСЧЕТ ==========

    total_adjustment = sum(adjustments.values())
    final_confidence = original_confidence + total_adjustment

    # Ограничиваем диапазон 0-100
    final_confidence = max(0, min(100, final_confidence))

    # Проверка порога
    MIN_CONFIDENCE_THRESHOLD = 70
    approved = final_confidence >= MIN_CONFIDENCE_THRESHOLD and len(blocking_factors) == 0

    validation_summary = self._create_validation_summary(
        symbol, signal_direction, original_confidence, final_confidence,
        adjustments, blocking_factors, approved
    )

    logger.info(
        f"{symbol}: Валидация завершена. Confidence: {original_confidence}->{final_confidence}, Approved: {approved}")

    return {
        'approved': approved,
        'final_confidence': final_confidence,
        'original_confidence': original_confidence,
        'total_adjustment': total_adjustment,
        'adjustments': adjustments,
        'blocking_factors': blocking_factors,
        'validation_details': validation_details,
        'validation_summary': validation_summary,
        'market_data': market_snapshot,
        'correlation_data': corr_analysis,
        'volume_profile_data': vp_data if vp_data and vp_data.get('poc', 0) > 0 else None
    }

except Exception as e:
    logger.error(f"Ошибка валидации {symbol}: {e}")
    import traceback

    logger.error(f"Traceback: {traceback.format_exc()}")

    return {
        'approved': False,
        'final_confidence': 0,
        'original_confidence': original_confidence,
        'total_adjustment': 0,
        'adjustments': adjustments,
        'blocking_factors': [f"Validation error: {str(e)}"],
        'validation_details': validation_details,
        'validation_summary': f"ERROR during validation: {str(e)}"
    }


def _create_rejection_result(
        self,
        signal: Dict,
        original_confidence: int,
        adjustments: Dict,
        blocking_factors: List[str],
        validation_details: List[str]
) -> Dict:
    """Создать результат отклонения"""
    return {
        'approved': False,
        'final_confidence': 0,
        'original_confidence': original_confidence,
        'total_adjustment': sum(adjustments.values()),
        'adjustments': adjustments,
        'blocking_factors': blocking_factors,
        'validation_details': validation_details,
        'validation_summary': f"REJECTED: {blocking_factors[0] if blocking_factors else 'Unknown reason'}"
    }


def _create_validation_summary(
        self,
        symbol: str,
        signal_direction: str,
        original_conf: int,
        final_conf: int,
        adjustments: Dict,
        blocking_factors: List[str],
        approved: bool
) -> str:
    """Создать краткое резюме валидации"""

    # Топ-3 adjustment по модулю
    top_adjustments = sorted(
        adjustments.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    adj_summary = ", ".join([f"{k}: {v:+d}" for k, v in top_adjustments if v != 0])

    if approved:
        return f"✅ {symbol} {signal_direction} APPROVED: {original_conf}->{final_conf} ({adj_summary})"
    else:
        reason = blocking_factors[0] if blocking_factors else f"confidence {final_conf}<70"
        return f"❌ {symbol} {signal_direction} REJECTED: {reason} ({adj_summary})"


# ==================== BATCH VALIDATION ====================

async def validate_

    Не дописано , надо дописать 