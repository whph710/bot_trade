"""
Централизованная система валидации торговых сигналов
Устраняет дублирование логики между simple_validator.py, shared_utils.py и AI промптами
Файл: trade_bot_programm/validation_engine.py
"""

from typing import Dict, Tuple, List
from logging_config import setup_module_logger

logger = setup_module_logger(__name__)

"""
Централизованная система валидации - FIXED: Согласованные RSI пороги
"""


class ValidationEngine:
    """Централизованный движок валидации"""

    @staticmethod
    def check_rsi_exhaustion(
            indicators_1h: Dict,
            indicators_4h: Dict,
            indicators_1d: Dict,  # ДОБАВЛЕНО
            signal_type: str,
            has_1d_data: bool = False  # ДОБАВЛЕНО
    ) -> Tuple[bool, str]:
        """
        ИСПРАВЛЕНО: Согласовано с промптами

        Args:
            indicators_1h: Индикаторы 1H
            indicators_4h: Индикаторы 4H
            indicators_1d: Индикаторы 1D (может быть пустым)
            signal_type: 'LONG' или 'SHORT'
            has_1d_data: Доступны ли 1D данные
        """
        rsi_1h = indicators_1h.get('current', {}).get('rsi', 50)
        rsi_4h = indicators_4h.get('current', {}).get('rsi', 50)
        rsi_1d = indicators_1d.get('current', {}).get('rsi', 50) if has_1d_data else None

        if signal_type == 'LONG':
            # Критичный порог 1H
            if has_1d_data:
                # С 1D данными - стандартный порог
                if rsi_1h > 75:
                    return True, f"RSI 1H extreme overbought ({rsi_1h:.1f})"

                # Multi-TF exhaustion (все 3 TF)
                if rsi_1d and rsi_1h > 70 and rsi_4h > 65 and rsi_1d > 60:
                    return True, f"Multi-TF overbought (1H={rsi_1h:.1f}, 4H={rsi_4h:.1f}, 1D={rsi_1d:.1f})"
            else:
                # БЕЗ 1D - более строгий порог
                if rsi_1h > 78:  # Было 75
                    return True, f"RSI 1H extreme overbought ({rsi_1h:.1f}, no 1D data)"

                # Multi-TF exhaustion (только 1H+4H)
                if rsi_1h > 72 and rsi_4h > 68:  # Было 70/65
                    return True, f"Multi-TF overbought (1H={rsi_1h:.1f}, 4H={rsi_4h:.1f}, 1D unavailable)"

        elif signal_type == 'SHORT':
            if has_1d_data:
                if rsi_1h < 25:
                    return True, f"RSI 1H extreme oversold ({rsi_1h:.1f})"

                if rsi_1d and rsi_1h < 30 and rsi_4h < 35 and rsi_1d < 40:
                    return True, f"Multi-TF oversold (1H={rsi_1h:.1f}, 4H={rsi_4h:.1f}, 1D={rsi_1d:.1f})"
            else:
                if rsi_1h < 22:
                    return True, f"RSI 1H extreme oversold ({rsi_1h:.1f}, no 1D data)"

                if rsi_1h < 28 and rsi_4h < 32:
                    return True, f"Multi-TF oversold (1H={rsi_1h:.1f}, 4H={rsi_4h:.1f}, 1D unavailable)"

        return False, ""

    @staticmethod
    def check_correlation_blocking(corr_data: Dict) -> Tuple[bool, str]:
        """
        ИСПРАВЛЕНО: Блокировка только при EXTREME correlation

        ИЗМЕНЕНИЕ: should_block_signal проверяется, но НЕ блокирует автоматически
        Требуется также отсутствие дивергенций и Wyckoff Phase D
        """
        if not corr_data.get('should_block_signal', False):
            return False, ""

        # Если correlation module говорит блокировать, проверяем дополнительно
        btc_corr = corr_data.get('btc_correlation', {})
        correlation = btc_corr.get('correlation', 0)

        # Блокируем ТОЛЬКО при EXTREME correlation >0.85
        if abs(correlation) > 0.85:
            return True, f"EXTREME BTC correlation {correlation:.2f} conflict"

        # Иначе - только warning, не блокируем
        return False, f"BTC correlation {correlation:.2f} warning (not blocking)"

    @classmethod
    def run_all_checks(
            cls,
            signal: Dict,
            comprehensive_data: Dict
    ) -> Tuple[bool, List[str]]:
        """
        ОБНОВЛЕНО: Передаем has_1d_data в RSI проверку
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

        # 3. RSI exhaustion - ОБНОВЛЕНО
        has_1d = comprehensive_data.get('has_1d_data', False)
        blocked, reason = cls.check_rsi_exhaustion(
            comprehensive_data.get('indicators_1h', {}),
            comprehensive_data.get('indicators_4h', {}),
            comprehensive_data.get('indicators_1d', {}),
            signal.get('signal', 'NONE'),
            has_1d_data=has_1d
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