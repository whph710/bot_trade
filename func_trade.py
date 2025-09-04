import numpy as np
from typing import List, Dict, Any, Tuple
import time
import math
import json

# УПРОЩЕННЫЕ СКАЛЬПИНГОВЫЕ ПАРАМЕТРЫ - ТОЛЬКО 3 ЛУЧШИХ ИНДИКАТОРА
SCALPING_PARAMS = {
    # 1. EMA для тренда (самый быстрый и точный)
    'ema_fast': 8,  # Быстрая EMA
    'ema_slow': 21,  # Медленная EMA

    # 2. RSI для momentum (классический)
    'rsi_period': 14,  # Стандартный RSI

    # 3. Объемы (самый важный для скальпинга)
    'volume_lookback': 10,  # Анализ объемов

    'min_confidence': 75  # Повышенная минимальная уверенность
}


def safe_float(value, default=0.0):
    """Безопасное преобразование в float"""
    try:
        if value is None:
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    """Безопасное преобразование в int"""
    try:
        if value is None:
            return default
        result = int(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average - основа системы"""
    if len(prices) < period:
        return np.array([])

    ema = np.zeros_like(prices, dtype=float)
    alpha = 2.0 / (period + 1)
    ema[0] = float(prices[0])

    for i in range(1, len(prices)):
        ema[i] = alpha * float(prices[i]) + (1 - alpha) * ema[i - 1]

    return ema


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI индикатор для momentum"""
    if len(prices) < period + 1:
        return np.array([])

    deltas = np.diff(prices.astype(float))
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gains = np.zeros_like(prices, dtype=float)
    avg_losses = np.zeros_like(prices, dtype=float)

    # Первое значение - простое среднее
    avg_gains[period] = np.mean(gains[:period])
    avg_losses[period] = np.mean(losses[:period])

    # Остальные - экспоненциальное сглаживание
    alpha = 1.0 / period
    for i in range(period + 1, len(prices)):
        avg_gains[i] = alpha * gains[i - 1] + (1 - alpha) * avg_gains[i - 1]
        avg_losses[i] = alpha * losses[i - 1] + (1 - alpha) * avg_losses[i - 1]

    # Избегаем деления на ноль
    rs = np.divide(avg_gains, avg_losses,
                   out=np.zeros_like(avg_gains),
                   where=avg_losses != 0)
    rsi = 100 - (100 / (1 + rs))

    # Заменяем NaN на 50
    rsi = np.nan_to_num(rsi, nan=50.0)

    return rsi


def analyze_volume_pattern(candles: List[List[str]], lookback: int = 10) -> Dict[str, Any]:
    """Анализ объемных паттернов - ключ к успешному скальпингу"""
    if len(candles) < lookback:
        return {
            'volume_spike': False,
            'volume_ratio': 1.0,
            'volume_trend': 'NEUTRAL',
            'volume_score': 0
        }

    try:
        volumes = np.array([safe_float(c[5]) for c in candles])
        if len(volumes) < lookback:
            return {'volume_spike': False, 'volume_ratio': 1.0, 'volume_trend': 'NEUTRAL', 'volume_score': 0}

        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-lookback:-1])  # Исключаем текущую свечу

        if avg_volume <= 0:
            return {'volume_spike': False, 'volume_ratio': 1.0, 'volume_trend': 'NEUTRAL', 'volume_score': 0}

        ratio = current_volume / avg_volume

        # Определяем тренд объемов
        recent_avg = np.mean(volumes[-5:]) if len(volumes) >= 5 else current_volume
        older_avg = np.mean(volumes[-lookback:-5]) if len(volumes) >= lookback else current_volume

        if older_avg > 0:
            volume_change = (recent_avg - older_avg) / older_avg
            if volume_change > 0.2:
                volume_trend = 'INCREASING'
            elif volume_change < -0.2:
                volume_trend = 'DECREASING'
            else:
                volume_trend = 'NEUTRAL'
        else:
            volume_trend = 'NEUTRAL'

        # Скоринг объемов (0-100)
        volume_score = 0
        if ratio > 2.0:
            volume_score = 100
        elif ratio > 1.5:
            volume_score = 75
        elif ratio > 1.2:
            volume_score = 50
        elif ratio > 1.0:
            volume_score = 25

        return {
            'volume_spike': bool(ratio > 1.5),
            'volume_ratio': safe_float(ratio),
            'volume_trend': volume_trend,
            'volume_score': safe_int(volume_score)
        }

    except Exception as e:
        return {'volume_spike': False, 'volume_ratio': 1.0, 'volume_trend': 'NEUTRAL', 'volume_score': 0}


def calculate_simplified_indicators(candles: List[List[str]]) -> Dict[str, Any]:
    """
    УПРОЩЕННАЯ СИСТЕМА: Только 3 самых важных индикатора для скальпинга
    1. EMA кроссовер (тренд)
    2. RSI (momentum)
    3. Volume pattern (подтверждение)
    """
    if len(candles) < 30:
        return {}

    try:
        # Извлекаем цены безопасно
        closes = np.array([safe_float(c[4]) for c in candles])
        highs = np.array([safe_float(c[2]) for c in candles])
        lows = np.array([safe_float(c[3]) for c in candles])

        # 1. EMA СИСТЕМА (главный тренд индикатор)
        ema_fast = calculate_ema(closes, SCALPING_PARAMS['ema_fast'])
        ema_slow = calculate_ema(closes, SCALPING_PARAMS['ema_slow'])

        # Определяем тренд и силу
        if len(ema_fast) > 0 and len(ema_slow) > 0:
            ema_diff = ema_fast[-1] - ema_slow[-1]
            ema_diff_percent = (ema_diff / closes[-1] * 100) if closes[-1] > 0 else 0

            # Проверяем кроссовер
            if len(ema_fast) >= 2 and len(ema_slow) >= 2:
                prev_diff = ema_fast[-2] - ema_slow[-2]
                current_diff = ema_fast[-1] - ema_slow[-1]

                if prev_diff <= 0 and current_diff > 0:
                    ema_signal = 'BULLISH_CROSS'
                elif prev_diff >= 0 and current_diff < 0:
                    ema_signal = 'BEARISH_CROSS'
                elif current_diff > 0:
                    ema_signal = 'BULLISH'
                elif current_diff < 0:
                    ema_signal = 'BEARISH'
                else:
                    ema_signal = 'NEUTRAL'
            else:
                ema_signal = 'NEUTRAL'
        else:
            ema_diff_percent = 0
            ema_signal = 'NEUTRAL'

        # 2. RSI СИСТЕМА (momentum индикатор)
        rsi = calculate_rsi(closes, SCALPING_PARAMS['rsi_period'])

        if len(rsi) > 0:
            current_rsi = safe_float(rsi[-1], 50.0)

            # Определяем RSI сигналы
            if current_rsi < 30:
                rsi_signal = 'OVERSOLD'
            elif current_rsi > 70:
                rsi_signal = 'OVERBOUGHT'
            elif 40 <= current_rsi <= 60:
                rsi_signal = 'NEUTRAL'
            elif current_rsi < 50:
                rsi_signal = 'BEARISH'
            else:
                rsi_signal = 'BULLISH'
        else:
            current_rsi = 50.0
            rsi_signal = 'NEUTRAL'

        # 3. ОБЪЕМЫ (подтверждающий индикатор)
        volume_data = analyze_volume_pattern(candles, SCALPING_PARAMS['volume_lookback'])

        # Расчет общего качества сигнала
        signal_quality = calculate_signal_quality(
            ema_signal, current_rsi, volume_data, closes[-1], highs[-5:], lows[-5:]
        )

        return {
            # EMA система
            'ema_fast_value': safe_float(ema_fast[-1] if len(ema_fast) > 0 else 0),
            'ema_slow_value': safe_float(ema_slow[-1] if len(ema_slow) > 0 else 0),
            'ema_diff_percent': safe_float(ema_diff_percent),
            'ema_signal': str(ema_signal),

            # RSI система
            'rsi_value': safe_float(current_rsi),
            'rsi_signal': str(rsi_signal),

            # Объемы
            'volume_spike': bool(volume_data['volume_spike']),
            'volume_ratio': safe_float(volume_data['volume_ratio']),
            'volume_trend': str(volume_data['volume_trend']),
            'volume_score': safe_int(volume_data['volume_score']),

            # Общие метрики
            'signal_quality': safe_int(signal_quality),
            'current_price': safe_float(closes[-1])
        }

    except Exception as e:
        print(f"Ошибка расчета индикаторов: {e}")
        return {}


def calculate_signal_quality(ema_signal: str, rsi_value: float, volume_data: Dict,
                             current_price: float, highs: np.ndarray, lows: np.ndarray) -> int:
    """Расчет качества сигнала на основе 3 индикаторов"""
    quality = 0

    # EMA качество (40 баллов максимум)
    if ema_signal in ['BULLISH_CROSS', 'BEARISH_CROSS']:
        quality += 40  # Сильный сигнал
    elif ema_signal in ['BULLISH', 'BEARISH']:
        quality += 25  # Умеренный сигнал
    elif ema_signal == 'NEUTRAL':
        quality += 10  # Слабый сигнал

    # RSI качество (30 баллов максимум)
    if rsi_value <= 30 or rsi_value >= 70:
        quality += 30  # Экстремальные значения отлично для скальпинга
    elif 35 <= rsi_value <= 65:
        quality += 15  # Нейтральная зона
    else:
        quality += 20  # Умеренные значения

    # Volume качество (30 баллов максимум)
    quality += min(30, volume_data['volume_score'] * 0.3)

    return min(100, max(0, quality))


def detect_scalping_entry(candles: List[List[str]]) -> Dict[str, Any]:
    """
    ОСНОВНАЯ ФУНКЦИЯ: Определение входа для скальпинга
    Использует только 3 проверенных индикатора
    """
    if len(candles) < 30:
        return {
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'reason': 'INSUFFICIENT_DATA',
            'indicators': {}
        }

    # Рассчитываем упрощенные индикаторы
    indicators = calculate_simplified_indicators(candles)

    if not indicators:
        return {
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'reason': 'CALCULATION_ERROR',
            'indicators': {}
        }

    # ОПРЕДЕЛЯЕМ СИГНАЛ НА ОСНОВЕ 3 ИНДИКАТОРОВ
    signal_type = 'NO_SIGNAL'
    confidence = 0
    entry_reasons = []

    ema_signal = indicators.get('ema_signal', 'NEUTRAL')
    rsi_signal = indicators.get('rsi_signal', 'NEUTRAL')
    volume_confirmed = indicators.get('volume_spike', False)

    # LONG условия
    long_score = 0
    if ema_signal in ['BULLISH', 'BULLISH_CROSS']:
        long_score += 3
        entry_reasons.append('EMA_BULLISH')

    if rsi_signal in ['OVERSOLD', 'BULLISH']:
        long_score += 2
        entry_reasons.append('RSI_SUPPORT')

    if volume_confirmed:
        long_score += 2
        entry_reasons.append('VOLUME_SPIKE')

    # SHORT условия
    short_score = 0
    if ema_signal in ['BEARISH', 'BEARISH_CROSS']:
        short_score += 3
        entry_reasons.append('EMA_BEARISH')

    if rsi_signal in ['OVERBOUGHT', 'BEARISH']:
        short_score += 2
        entry_reasons.append('RSI_RESISTANCE')

    if volume_confirmed:
        short_score += 2
        entry_reasons.append('VOLUME_SPIKE')

    # Определяем финальный сигнал
    if long_score >= 5 and long_score > short_score:
        signal_type = 'LONG'
        confidence = min(100, long_score * 15 + indicators.get('signal_quality', 0))
    elif short_score >= 5 and short_score > long_score:
        signal_type = 'SHORT'
        confidence = min(100, short_score * 15 + indicators.get('signal_quality', 0))

    # Проверяем минимальное качество
    if confidence < SCALPING_PARAMS['min_confidence']:
        signal_type = 'NO_SIGNAL'
        confidence = 0
        entry_reasons = []

    return {
        'signal': signal_type,
        'confidence': safe_int(confidence),
        'reason': 'QUALITY_SIGNAL' if signal_type != 'NO_SIGNAL' else 'LOW_QUALITY',
        'entry_reasons': [str(r) for r in entry_reasons],
        'indicators': indicators
    }


def prepare_ai_data(signals: List) -> Dict[str, Any]:
    """Подготовка данных для ИИ с гарантией JSON сериализации"""
    prepared_signals = []

    for signal in signals:
        try:
            # Проверяем, что у нас есть необходимые атрибуты
            if not hasattr(signal, 'pair') or not hasattr(signal, 'confidence'):
                continue

            # Безопасно извлекаем данные
            signal_data = {
                'pair': str(signal.pair),
                'signal_type': str(getattr(signal, 'signal_type', 'NO_SIGNAL')),
                'confidence': safe_int(getattr(signal, 'confidence', 0)),
                'entry_price': safe_float(getattr(signal, 'entry_price', 0)),

                # Упрощенные индикаторы
                'indicators': {
                    'ema_signal': str(getattr(signal, 'indicators_data', {}).get('ema_signal', 'NEUTRAL')),
                    'rsi_value': safe_float(getattr(signal, 'indicators_data', {}).get('rsi_value', 50)),
                    'rsi_signal': str(getattr(signal, 'indicators_data', {}).get('rsi_signal', 'NEUTRAL')),
                    'volume_spike': bool(getattr(signal, 'indicators_data', {}).get('volume_spike', False)),
                    'volume_ratio': safe_float(getattr(signal, 'indicators_data', {}).get('volume_ratio', 1.0)),
                    'signal_quality': safe_int(getattr(signal, 'indicators_data', {}).get('signal_quality', 0))
                },

                # Последние 5 свечей для контекста
                'recent_candles': []
            }

            # Добавляем свечи если есть
            if hasattr(signal, 'candles_data') and signal.candles_data:
                recent_candles = signal.candles_data[-5:]  # Только последние 5
                for candle in recent_candles:
                    if len(candle) >= 6:
                        signal_data['recent_candles'].append({
                            'open': safe_float(candle[1]),
                            'high': safe_float(candle[2]),
                            'low': safe_float(candle[3]),
                            'close': safe_float(candle[4]),
                            'volume': safe_float(candle[5])
                        })

            prepared_signals.append(signal_data)

        except Exception as e:
            print(f"Ошибка подготовки данных для {getattr(signal, 'pair', 'unknown')}: {e}")
            continue

    return {
        'signals_count': len(prepared_signals),
        'timeframe': '15m',
        'strategy': 'simplified_scalping',
        'timestamp': safe_int(time.time()),
        'signals': prepared_signals
    }


def test_json_serialization(data: Any) -> bool:
    """Тест JSON сериализации"""
    try:
        json.dumps(data, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"JSON ошибка: {e}")
        return False


# Тестирование системы
if __name__ == "__main__":
    # Пример тестовых данных
    test_candles = []
    base_price = 45000
    for i in range(50):
        price_change = np.random.uniform(-0.01, 0.01)
        price = base_price * (1 + price_change)
        volume = np.random.uniform(1000, 5000)

        test_candles.append([
            str(int(time.time() + i * 900)),  # timestamp
            str(price * 0.999),  # open
            str(price * 1.002),  # high
            str(price * 0.998),  # low
            str(price),  # close
            str(volume)  # volume
        ])

    print("Тестирование упрощенной системы...")

    # Тест индикаторов
    indicators = calculate_simplified_indicators(test_candles)
    print("Индикаторы:", indicators)

    # Тест сигнала
    signal = detect_scalping_entry(test_candles)
    print("Сигнал:", signal)

    # Тест JSON сериализации
    print("JSON тест:", test_json_serialization(indicators))
    print("Signal JSON тест:", test_json_serialization(signal))