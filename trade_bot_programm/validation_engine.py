"""
Централизованная система валидации торговых сигналов
Устраняет дублирование логики между simple_validator.py, shared_utils.py и AI промптами
Файл: trade_bot_programm/validation_engine.py
"""

from typing import Dict, Tuple, List
from logging_config import setup_module_logger

logger = setup_module_logger(__name__)


class ValidationEngine:
    """Централизованный движок валидации"""

    @staticmethod
    def check_correlation_blocking(corr_data: Dict) -> Tuple[bool, str]:
        """
        Проверка блокировки по корреляции с BTC

        Args:
            corr_data: Данные корреляционного анализа

        Returns:
            (should_block, reason)
        """
        if corr_data.get('should_block_signal', False):
            return True, "BTC correlation conflict"
        return False, ""

    @staticmethod
    def check_overextension(vp_analysis: Dict, threshold: float = 15.0) -> Tuple[bool, str]:
        """
        Проверка перерастяжения цены от POC

        Args:
            vp_analysis: Volume Profile анализ
            threshold: Максимальное отклонение от POC в %

        Returns:
            (should_block, reason)
        """
        poc_dist = vp_analysis.get('poc_analysis', {}).get('distance_to_poc_pct', 0)
        if abs(poc_dist) > threshold:
            return True, f"Overextended {poc_dist:.1f}% from POC"
        return False, ""

    @staticmethod
    def check_rsi_exhaustion(
            indicators_1h: Dict,
            indicators_4h: Dict,
            signal_type: str
    ) -> Tuple[bool, str]:
        """
        Проверка перекупленности/перепроданности RSI

        Args:
            indicators_1h: Индикаторы 1H таймфрейма
            indicators_4h: Индикаторы 4H таймфрейма
            signal_type: 'LONG' или 'SHORT'

        Returns:
            (should_block, reason)
        """
        rsi_1h = indicators_1h.get('current', {}).get('rsi', 50)
        rsi_4h = indicators_4h.get('current', {}).get('rsi', 50)

        if signal_type == 'LONG':
            if rsi_1h > 75:
                return True, f"RSI 1H overbought ({rsi_1h:.1f})"
            if rsi_1h > 70 and rsi_4h > 65:
                return True, f"Multi-TF overbought (1H={rsi_1h:.1f}, 4H={rsi_4h:.1f})"

        elif signal_type == 'SHORT':
            if rsi_1h < 25:
                return True, f"RSI 1H oversold ({rsi_1h:.1f})"
            if rsi_1h < 30 and rsi_4h < 35:
                return True, f"Multi-TF oversold (1H={rsi_1h:.1f}, 4H={rsi_4h:.1f})"

        return False, ""

    @staticmethod
    def check_funding_rate_extreme(funding_data: Dict, threshold: float = 0.1) -> Tuple[bool, str]:
        """
        Проверка экстремального funding rate

        Args:
            funding_data: Данные funding rate
            threshold: Пороговое значение (0.1 = 0.1%)

        Returns:
            (should_block, reason)
        """
        funding_rate = funding_data.get('funding_rate', 0)
        if abs(funding_rate) > threshold:
            return True, f"Extreme funding rate ({funding_rate:.4f}%)"
        return False, ""

    @staticmethod
    def check_spread_illiquidity(orderbook_data: Dict, max_spread: float = 0.15) -> Tuple[bool, str]:
        """
        Проверка спреда (ликвидность)

        Args:
            orderbook_data: Данные ордербука
            max_spread: Максимальный спред в %

        Returns:
            (should_block, reason)
        """
        spread_pct = orderbook_data.get('spread_pct', 0)
        if spread_pct > max_spread:
            return True, f"Wide spread ({spread_pct:.3f}%)"
        return False, ""

    @classmethod
    def run_all_checks(
            cls,
            signal: Dict,
            comprehensive_data: Dict
    ) -> Tuple[bool, List[str]]:
        """
        Запускает все критические проверки

        Args:
            signal: Данные сигнала (symbol, signal type, etc)
            comprehensive_data: Полные данные анализа

        Returns:
            (passed, reasons) - passed=True если все проверки пройдены
        """
        reasons = []

        # 1. Correlation
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

        # 3. RSI exhaustion
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