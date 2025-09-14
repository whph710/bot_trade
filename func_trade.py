"""
Исправленный модуль индикаторов - устранены все ошибки расчетов
"""

import numpy as np
from typing import List, Dict, Any
from config import config
import logging

logger = logging.getLogger(__name__)


def safe_float(value) -> float:
    """Безопасное преобразование в float с логированием"""
    try:
        # Если это numpy array, берем последний элемент
        if isinstance(value, np.ndarray):
            if len(value) > 0:
                value = value[-1]
            else:
                return 0.0

        result = float(value)
        if np.isnan(result) or np.isinf(result):
            return 0.0
        return result
    except Exception as e:
        logger.debug(f"Ошибка конвертации в float: {value}, {e}")
        return 0.0


def validate_candles(candles: List[List[str]]) -> bool:
    """Валидация свечных данных"""
    if not candles or len(candles) < 10:
        return False

    # Проверим несколько свечей на корректность
    for candle in candles[:3]:
        try:
            if len(candle) < 6:
                return False
            # Проверяем что OHLCV числовые
            for i in [1, 2, 3, 4, 5]:  # open, high, low, close, volume
                float(candle[i])
        except (ValueError, IndexError):
            return False

    return True


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Расчет EMA с дополнительной защитой"""
    if len(prices) < period:
        return np.full_like(prices, prices[0] if len(prices) > 0 else 0)

    try:
        ema = np.zeros_like(prices, dtype=np.float64)
        alpha = 2.0 / (period + 1)
        ema[0] = prices[0]

        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

        return ema
    except Exception as e:
        logger.error(f"Ошибка расчета EMA: {e}")
        return np.full_like(prices, prices[0] if len(prices) > 0 else 0)


def calculate_rsi(prices: np.ndarray, period: int = 9) -> np.ndarray:
    """Расчет RSI с защитой от ошибок"""
    if len(prices) < period + 1:
        return np.full_like(prices, 50.0)

    try:
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Добавляем нулевой элемент для выравнивания размеров
        gains = np.concatenate([[0], gains])
        losses = np.concatenate([[0], losses])

        avg_gains = np.zeros_like(prices)
        avg_losses = np.zeros_like(prices)

        # Первый расчет
        if len(gains) >= period:
            avg_gains[period] = np.mean(gains[1:period+1])
            avg_losses[period] = np.mean(losses[1:period+1])

        # Скользящее среднее
        alpha = 1.0 / period
        for i in range(period + 1, len(prices)):
            avg_gains[i] = alpha * gains[i] + (1 - alpha) * avg_gains[i - 1]
            avg_losses[i] = alpha * losses[i] + (1 - alpha) * avg_losses[i - 1]

        # Расчет RSI
        rsi = np.full_like(prices, 50.0)
        for i in range(period, len(prices)):
            if avg_losses[i] != 0:
                rs = avg_gains[i] / avg_losses[i]
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100 if avg_gains[i] > 0 else 50

        return rsi
    except Exception as e:
        logger.error(f"Ошибка расчета RSI: {e}")
        return np.full_like(prices, 50.0)


def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, np.ndarray]:
    """Расчет MACD с защитой"""
    zero_array = np.zeros_like(prices)

    if len(prices) < max(fast, slow):
        return {'line': zero_array, 'signal': zero_array, 'histogram': zero_array}

    try:
        ema_fast = calculate_ema(prices, fast)
        ema_slow = calculate_ema(prices, slow)

        macd_line = ema_fast - ema_slow
        signal_line = calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line

        return {'line': macd_line, 'signal': signal_line, 'histogram': histogram}
    except Exception as e:
        logger.error(f"Ошибка расчета MACD: {e}")
        return {'line': zero_array, 'signal': zero_array, 'histogram': zero_array}


def calculate_atr(candles: List[List[str]], period: int = 14) -> float:
    """Расчет ATR с защитой"""
    if not validate_candles(candles) or len(candles) < period + 1:
        return 0.0

    try:
        highs = np.array([safe_float(c[2]) for c in candles])
        lows = np.array([safe_float(c[3]) for c in candles])
        closes = np.array([safe_float(c[4]) for c in candles])

        # Проверим что есть корректные данные
        if np.any(highs <= 0) or np.any(lows <= 0) or np.any(closes <= 0):
            return 0.0

        tr = np.zeros(len(candles))
        for i in range(1, len(candles)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )

        if len(tr) <= period:
            return safe_float(np.mean(tr[1:]))

        atr = np.mean(tr[1:period + 1])
        for i in range(period + 1, len(candles)):
            atr = (atr * (period - 1) + tr[i]) / period

        return safe_float(atr)
    except Exception as e:
        logger.error(f"Ошибка расчета ATR: {e}")
        return 0.0


def calculate_volume_ratios(volumes: np.ndarray, window: int = 20) -> np.ndarray:
    """Безопасный расчет отношения объемов"""
    try:
        if len(volumes) < window:
            return np.ones_like(volumes)

        ratios = np.ones_like(volumes)

        for i in range(window, len(volumes)):
            avg_volume = np.mean(volumes[max(0, i-window):i])
            if avg_volume > 0:
                ratios[i] = volumes[i] / avg_volume
            else:
                ratios[i] = 1.0

        return ratios
    except Exception as e:
        logger.error(f"Ошибка расчета volume ratios: {e}")
        return np.ones_like(volumes)


def calculate_basic_indicators(candles: List[List[str]]) -> Dict[str, Any]:
    """Базовые индикаторы с полной защитой от ошибок"""
    if not validate_candles(candles):
        logger.warning("Некорректные данные свечей для базовых индикаторов")
        return {}

    try:
        closes = np.array([safe_float(c[4]) for c in candles])
        volumes = np.array([safe_float(c[5]) for c in candles])

        # Проверим корректность данных
        if len(closes) < 20 or np.all(closes == 0) or np.all(volumes == 0):
            logger.warning("Недостаточно корректных данных для индикаторов")
            return {}

        # EMA система
        ema5 = calculate_ema(closes, config.EMA_FAST)
        ema8 = calculate_ema(closes, config.EMA_MEDIUM)
        ema20 = calculate_ema(closes, config.EMA_SLOW)

        # RSI
        rsi = calculate_rsi(closes, config.RSI_PERIOD)

        # MACD
        macd = calculate_macd(closes, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL)

        # ATR
        atr = calculate_atr(candles, config.ATR_PERIOD)

        # Объем
        volume_ratios = calculate_volume_ratios(volumes)

        # Возвращаем только скалярные значения
        return {
            'price': safe_float(closes[-1]),
            'ema5': safe_float(ema5[-1]),
            'ema8': safe_float(ema8[-1]),
            'ema20': safe_float(ema20[-1]),
            'rsi': safe_float(rsi[-1]),
            'macd_line': safe_float(macd['line'][-1]),
            'macd_signal': safe_float(macd['signal'][-1]),
            'macd_histogram': safe_float(macd['histogram'][-1]),
            'atr': safe_float(atr),
            'volume_ratio': safe_float(volume_ratios[-1])
        }

    except Exception as e:
        logger.error(f"Критическая ошибка расчета базовых индикаторов: {e}")
        return {}


def calculate_ai_indicators(candles: List[List[str]], history_length: int) -> Dict[str, Any]:
    """Индикаторы для ИИ с историей - исправлены все ошибки"""
    if not validate_candles(candles):
        logger.warning("Некорректные данные свечей для ИИ индикаторов")
        return {}

    try:
        closes = np.array([safe_float(c[4]) for c in candles])
        volumes = np.array([safe_float(c[5]) for c in candles])

        # Проверим корректность
        if len(closes) < history_length or np.all(closes == 0):
            logger.warning(f"Недостаточно данных для ИИ индикаторов: {len(closes)} свечей, нужно {history_length}")
            return {}

        # EMA с историей
        ema5 = calculate_ema(closes, config.EMA_FAST)
        ema8 = calculate_ema(closes, config.EMA_MEDIUM)
        ema20 = calculate_ema(closes, config.EMA_SLOW)

        # RSI с историей
        rsi = calculate_rsi(closes, config.RSI_PERIOD)

        # MACD с историей
        macd = calculate_macd(closes, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL)

        # Volume ratios с защитой
        volume_ratios = calculate_volume_ratios(volumes)

        # Проверим что массивы не пустые
        min_length = min(len(ema5), len(ema8), len(ema20), len(rsi), len(volume_ratios))
        if min_length < history_length:
            logger.warning(f"Массивы индикаторов слишком короткие: {min_length} < {history_length}")
            return {}

        # Безопасное извлечение истории
        def safe_history(arr, length):
            """Безопасное извлечение последних элементов"""
            try:
                if len(arr) >= length:
                    return [safe_float(x) for x in arr[-length:]]
                else:
                    # Дополняем нулями если не хватает данных
                    result = [0.0] * (length - len(arr))
                    result.extend([safe_float(x) for x in arr])
                    return result
            except Exception as e:
                logger.error(f"Ошибка извлечения истории: {e}")
                return [0.0] * length

        return {
            # История индикаторов
            'ema5_history': safe_history(ema5, history_length),
            'ema8_history': safe_history(ema8, history_length),
            'ema20_history': safe_history(ema20, history_length),
            'rsi_history': safe_history(rsi, history_length),
            'macd_line_history': safe_history(macd['line'], history_length),
            'macd_signal_history': safe_history(macd['signal'], history_length),
            'macd_histogram_history': safe_history(macd['histogram'], history_length),
            'volume_ratio_history': safe_history(volume_ratios, history_length),

            # Текущие значения - ТОЛЬКО СКАЛЯРЫ!
            'current': {
                'price': safe_float(closes[-1]),
                'ema5': safe_float(ema5[-1]),
                'ema8': safe_float(ema8[-1]),
                'ema20': safe_float(ema20[-1]),
                'rsi': safe_float(rsi[-1]),
                'macd_line': safe_float(macd['line'][-1]),
                'macd_histogram': safe_float(macd['histogram'][-1]),
                'volume_ratio': safe_float(volume_ratios[-1]),
                'atr': safe_float(calculate_atr(candles, config.ATR_PERIOD))
            }
        }

    except Exception as e:
        logger.error(f"Критическая ошибка расчета ИИ индикаторов: {e}")
        import traceback
        logger.error(f"Полная трассировка: {traceback.format_exc()}")
        return {}


def check_basic_signal(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """Проверка базового сигнала с защитой"""
    if not indicators:
        return {'signal': False, 'confidence': 0, 'direction': 'NONE'}

    try:
        # Извлекаем значения с дополнительной защитой
        price = safe_float(indicators.get('price', 0))
        ema5 = safe_float(indicators.get('ema5', 0))
        ema8 = safe_float(indicators.get('ema8', 0))
        ema20 = safe_float(indicators.get('ema20', 0))
        rsi = safe_float(indicators.get('rsi', 50))
        macd_hist = safe_float(indicators.get('macd_histogram', 0))
        volume_ratio = safe_float(indicators.get('volume_ratio', 1.0))
        atr = safe_float(indicators.get('atr', 0))

        # Валидация значений
        if price <= 0 or ema5 <= 0 or ema8 <= 0 or ema20 <= 0:
            logger.warning(f"Некорректные значения цены/EMA: {price}, {ema5}, {ema8}, {ema20}")
            return {'signal': False, 'confidence': 0, 'direction': 'NONE'}

        if not (0 <= rsi <= 100):
            logger.warning(f"Некорректное значение RSI: {rsi}")
            rsi = 50  # Используем нейтральное значение

        # Проверяем базовые условия
        conditions = []

        # EMA выравнивание для лонга
        if price > ema5 and ema5 > ema8 and ema8 > ema20:
            conditions.append(('LONG', 25))

        # EMA выравнивание для шорта
        if price < ema5 and ema5 < ema8 and ema8 < ema20:
            conditions.append(('SHORT', 25))

        # RSI в рабочем диапазоне
        if 30.0 < rsi < 70.0:
            conditions.append(('ANY', 15))

        # MACD активность
        if abs(macd_hist) > 0.001:
            conditions.append(('ANY', 15))

        # Объем подтверждение
        if volume_ratio >= config.MIN_VOLUME_RATIO:
            conditions.append(('ANY', 20))

        # ATR для торговли
        if atr > 0.001:
            conditions.append(('ANY', 10))

        if not conditions:
            return {'signal': False, 'confidence': 0, 'direction': 'NONE'}

        # Подсчет очков
        long_score = sum(score for direction, score in conditions if direction in ['LONG', 'ANY'])
        short_score = sum(score for direction, score in conditions if direction in ['SHORT', 'ANY'])

        # Результат
        if long_score > short_score and long_score >= config.MIN_CONFIDENCE:
            return {
                'signal': True,
                'confidence': int(long_score),
                'direction': 'LONG'
            }
        elif short_score > long_score and short_score >= config.MIN_CONFIDENCE:
            return {
                'signal': True,
                'confidence': int(short_score),
                'direction': 'SHORT'
            }
        else:
            return {
                'signal': False,
                'confidence': int(max(long_score, short_score)),
                'direction': 'NONE'
            }

    except Exception as e:
        logger.error(f"Ошибка проверки сигнала: {e}")
        return {'signal': False, 'confidence': 0, 'direction': 'NONE'}