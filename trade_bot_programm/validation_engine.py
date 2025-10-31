"""
Централизованная система валидации - БЕЗ 1D ДАННЫХ
Файл: trade_bot_programm/validation_engine.py
ИЗМЕНЕНИЯ:
- Убраны все проверки indicators_1d
- Убран параметр has_1d_data
- Используются только 1H и 4H данные
"""

from typing import Dict, Tuple, List
from logging_config import setup_module_logger

logger = setup_module_logger(__name__)


class ValidationEngine:
    """Централизованный движок валидации (БЕЗ 1D)"""

    @staticmethod
    def check_rsi_exhaustion(
            indicators_1h: Dict,
            indicators_4h: Dict,
            signal_type: str
    ) -> Tuple[bool, str]:
        """
        Проверка RSI exhaustion (ТОЛЬКО 1H и 4H)

        Args:
            indicators_1h: Индикаторы 1H
            indicators_4h: Индикаторы 4H
            signal_type: 'LONG' или 'SHORT'
        """
        rsi_1h = indicators_1h.get('current', {}).get('rsi', 50)
        rsi_4h = indicators_4h.get('current', {}).get('rsi', 50)

        if signal_type == 'LONG':
            # Extreme 1H RSI
            if rsi_1h > 78:
                return True, f"RSI 1H extreme overbought ({rsi_1h:.1f})"

            # Multi-TF exhaustion (1H+4H)
            if rsi_1h > 72 and rsi_4h > 68:
                return True, f"Multi-TF overbought (1H={rsi_1h:.1f}, 4H={rsi_4h:.1f})"

        elif signal_type == 'SHORT':
            # Extreme 1H RSI
            if rsi_1h < 22:
                return True, f"RSI 1H extreme oversold ({rsi_1h:.1f})"

            # Multi-TF exhaustion (1H+4H)
            if rsi_1h < 28 and rsi_4h < 32:
                return True, f"Multi-TF oversold (1H={rsi_1h:.1f}, 4H={rsi_4h:.1f})"

        return False, ""

    @staticmethod
    def check_correlation_blocking(corr_data: Dict) -> Tuple[bool, str]:
        """
        Проверка BTC correlation blocking (смягченная логика)
        Блокировка только при EXTREME correlation >0.85
        """
        if not corr_data.get('should_block_signal', False):
            return False, ""

        btc_corr = corr_data.get('btc_correlation', {})
        correlation = btc_corr.get('correlation', 0)

        # Блокируем ТОЛЬКО при EXTREME correlation >0.85
        if abs(correlation) > 0.85:
            return True, f"EXTREME BTC correlation {correlation:.2f} conflict"

        return False, f"BTC correlation {correlation:.2f} warning (not blocking)"

    @staticmethod
    def check_overextension(vp_analysis: Dict) -> Tuple[bool, str]:
        """Проверка overextension от Volume Profile POC"""
        if not vp_analysis:
            return False, ""

        va_analysis = vp_analysis.get('value_area_analysis', {})
        market_condition = va_analysis.get('market_condition', 'NORMAL')

        if market_condition == 'OVEREXTENDED':
            return True, "Price overextended from Value Area"

        poc_analysis = vp_analysis.get('poc_analysis', {})
        distance_pct = poc_analysis.get('distance_to_poc_pct', 0)

        if distance_pct > 15:
            return True, f"Price {distance_pct:.1f}% from POC (>15% overextended)"

        return False, ""

    @staticmethod
    def check_funding_rate_extreme(funding_data: Dict) -> Tuple[bool, str]:
        """Проверка экстремального funding rate"""
        if not funding_data:
            return False, ""

        funding_rate = funding_data.get('funding_rate', 0)

        if funding_rate > 0.001:
            return True, f"Extreme positive funding {funding_rate:.4f} (overleveraged longs)"

        if funding_rate < -0.001:
            return True, f"Extreme negative funding {funding_rate:.4f} (overleveraged shorts)"

        return False, ""

    @staticmethod
    def check_spread_illiquidity(orderbook_data: Dict) -> Tuple[bool, str]:
        """Проверка ликвидности через spread"""
        if not orderbook_data:
            return False, ""

        spread_pct = orderbook_data.get('spread_pct', 0)

        if spread_pct > 0.15:
            return True, f"Illiquid market (spread {spread_pct:.4f}% >0.15%)"

        return False, ""

    @classmethod
    def run_all_checks(
            cls,
            signal: Dict,
            comprehensive_data: Dict
    ) -> Tuple[bool, List[str]]:
        """
        Запустить все критические проверки (БЕЗ 1D)

        Returns:
            (passed: bool, reasons: List[str])
        """
        reasons = []

        # 1. Correlation - СМЯГЧЕНО
        blocked, reason = cls.check_correlation_blocking(
            comprehensive_data.get('correlation_data', {})
        )
        if blocked:
            reasons.append(reason)

        # 2. Overextension
        blocked, reason = cls.check_overextension(
            comprehensive_data.get('vp_analysis', {})
        )
        if blocked:
            reasons.append(reason)

        # 3. RSI exhaustion (ТОЛЬКО 1H+4H)
        blocked, reason = cls.check_rsi_exhaustion(
            comprehensive_data.get('indicators_1h', {}),
            comprehensive_data.get('indicators_4h', {}),
            signal.get('signal', 'NONE')
        )
        if blocked:
            reasons.append(reason)

        # 4. Funding rate
        blocked, reason = cls.check_funding_rate_extreme(
            comprehensive_data.get('market_data', {}).get('funding_rate', {})
        )
        if blocked:
            reasons.append(reason)

        # 5. Spread
        blocked, reason = cls.check_spread_illiquidity(
            comprehensive_data.get('market_data', {}).get('orderbook', {})
        )
        if blocked:
            reasons.append(reason)

        return len(reasons) == 0, reasons