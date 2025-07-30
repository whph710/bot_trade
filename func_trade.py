import numpy as np
from typing import List, Dict, Any, Tuple
import time


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Расчет Exponential Moving Average (точно как в Pine Script)

    Args:
        prices: Массив цен
        period: Период EMA

    Returns:
        Массив значений EMA
    """
    ema = np.zeros_like(prices)
    alpha = 2.0 / (period + 1)

    # Первое значение EMA равно первой цене
    ema[0] = prices[0]

    # Расчет последующих значений EMA
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema


def double_smooth(src: np.ndarray, long_period: int, short_period: int) -> np.ndarray:
    """
    Двойное сглаживание с помощью EMA (как в Pine Script)

    Args:
        src: Исходные данные
        long_period: Длинный период для первого сглаживания
        short_period: Короткий период для второго сглаживания

    Returns:
        Массив двойно сглаженных значений
    """
    # Первое сглаживание
    first_smooth = calculate_ema(src, long_period)

    # Второе сглаживание
    second_smooth = calculate_ema(first_smooth, short_period)

    return second_smooth


def calculate_tsi_with_momentum(candles: List[List[str]],
                                long_length: int = 8,
                                short_length: int = 4,
                                signal_length: int = 3) -> Dict[str, np.ndarray]:
    """
    Расчет True Strength Index (TSI) с momentum для скальпинга 15M

    Args:
        candles: Данные свечей в формате Bybit [timestamp, open, high, low, close, volume, turnover]
        long_length: Длинный период сглаживания (ускоренный для 15M)
        short_length: Короткий период сглаживания (ускоренный для 15M)
        signal_length: Период сигнальной линии (ускоренный для 15M)

    Returns:
        Словарь с массивами:
        - 'tsi': значения TSI
        - 'signal': сигнальная линия (EMA от TSI)
        - 'momentum': momentum изменения TSI
        - 'momentum_strength': сила momentum
    """
    if len(candles) < max(long_length, short_length, signal_length) + 10:
        return {
            'tsi': np.array([]),
            'signal': np.array([]),
            'momentum': np.array([]),
            'momentum_strength': 0.0
        }

    # Извлекаем цены закрытия
    prices = np.array([float(candle[4]) for candle in candles])

    # Вычисляем изменения цены (price change)
    pc = np.diff(prices, prepend=prices[0])  # Первое значение = 0

    # Двойное сглаживание изменений цены
    double_smoothed_pc = double_smooth(pc, long_length, short_length)

    # Двойное сглаживание абсолютных изменений цены
    abs_pc = np.abs(pc)
    double_smoothed_abs_pc = double_smooth(abs_pc, long_length, short_length)

    # Рассчитываем TSI (избегаем деление на ноль)
    tsi_values = np.zeros_like(double_smoothed_pc)
    non_zero_mask = double_smoothed_abs_pc != 0
    tsi_values[non_zero_mask] = 100 * (double_smoothed_pc[non_zero_mask] /
                                       double_smoothed_abs_pc[non_zero_mask])

    # Сигнальная линия - EMA от TSI
    signal_line = calculate_ema(tsi_values, signal_length)

    # Momentum = np.diff(tsi_values, prepend=tsi_values[0])
    momentum = np.diff(tsi_values, prepend=tsi_values[0])

    # Momentum strength = abs(momentum[-1])
    momentum_strength = abs(momentum[-1]) if len(momentum) > 0 else 0.0

    return {
        'tsi': tsi_values,
        'signal': signal_line,
        'momentum': momentum,
        'momentum_strength': momentum_strength
    }


def find_support_resistance_levels(candles: List[List[str]], window: int = 20) -> Dict[str, List[float]]:
    """
    Поиск уровней поддержки/сопротивления через локальные минимумы/максимумы

    Args:
        candles: Данные свечей
        window: Окно для поиска локальных экстремумов

    Returns:
        Словарь с уровнями поддержки и сопротивления
    """
    if len(candles) < window * 2:
        return {'support': [], 'resistance': []}

    highs = np.array([float(candle[2]) for candle in candles])
    lows = np.array([float(candle[3]) for candle in candles])

    support_levels = []
    resistance_levels = []

    # Поиск локальных минимумов (поддержка)
    for i in range(window, len(lows) - window):
        if lows[i] == min(lows[i - window:i + window + 1]):
            support_levels.append(float(lows[i]))

    # Поиск локальных максимумов (сопротивление)
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i - window:i + window + 1]):
            resistance_levels.append(float(highs[i]))

    # Убираем дубликаты и сортируем
    support_levels = sorted(list(set(support_levels)))
    resistance_levels = sorted(list(set(resistance_levels)))

    return {
        'support': support_levels,
        'resistance': resistance_levels
    }


def calculate_rsi_with_divergence(candles: List[List[str]], period: int = 9) -> Dict[str, Any]:
    """
    Расчет RSI с поиском дивергенций (оптимизированный период для 15M)

    Args:
        candles: Данные свечей
        period: Период RSI (ускоренный для скальпинга)

    Returns:
        Словарь с RSI и найденными дивергенциями
    """
    if len(candles) < period + 10:
        return {'rsi': [], 'divergences': []}

    prices = np.array([float(candle[4]) for candle in candles])

    # Расчет RSI
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gains = np.zeros_like(prices)
    avg_losses = np.zeros_like(prices)

    # Первое значение - простое среднее
    avg_gains[period] = np.mean(gains[:period])
    avg_losses[period] = np.mean(losses[:period])

    # Последующие значения - сглаженное среднее
    for i in range(period + 1, len(prices)):
        avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_losses[i] = (avg_losses[i - 1] * (period - 1) + losses[i - 1]) / period

    rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses != 0)
    rsi = 100 - (100 / (1 + rs))

    # Поиск дивергенций (улучшенный)
    divergences = []
    if len(rsi) > 50:
        # Ищем последние 20 свечей для дивергенций
        recent_prices = prices[-20:]
        recent_rsi = rsi[-20:]

        # Бычья дивергенция: цена падает, RSI растет
        if len(recent_prices) > 10:
            price_trend = recent_prices[-1] - recent_prices[-10]
            rsi_trend = recent_rsi[-1] - recent_rsi[-10]

            if price_trend < 0 and rsi_trend > 0 and recent_rsi[-1] < 30:
                divergences.append({'type': 'bullish', 'strength': abs(rsi_trend)})
            elif price_trend > 0 and rsi_trend < 0 and recent_rsi[-1] > 70:
                divergences.append({'type': 'bearish', 'strength': abs(rsi_trend)})

    return {
        'rsi': rsi.tolist(),
        'divergences': divergences
    }


def get_volume_anomalies(candles: List[List[str]],
                         threshold: float = 2.0,
                         lookback: int = 10) -> Dict[str, Any]:
    """
    Поиск объемных аномалий (повышенные требования для скальпинга)

    Args:
        candles: Данные свечей
        threshold: Порог для определения объемного всплеска (повышен до 2.0)
        lookback: Период анализа объемов

    Returns:
        Словарь с объемными данными
    """
    if len(candles) < 20:
        return {'spikes': [], 'profile': [], 'volume_ratio': 1.0}

    volumes = np.array([float(candle[5]) for candle in candles])
    prices = np.array([float(candle[4]) for candle in candles])

    # Поиск объемных всплесков
    if len(volumes) > lookback:
        avg_volume = np.mean(volumes[-lookback:])
        spikes = []

        for i in range(len(volumes)):
            if volumes[i] > avg_volume * threshold:
                spikes.append({
                    'index': i,
                    'volume': float(volumes[i]),
                    'price': float(prices[i]),
                    'ratio': float(volumes[i] / avg_volume)
                })

        # Текущий объемный коэффициент
        current_volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
    else:
        spikes = []
        current_volume_ratio = 1.0

    # Упрощенный объемный профиль (последние 100 свечей)
    profile = []
    if len(candles) > 50:
        recent_candles = candles[-100:] if len(candles) > 100 else candles
        price_min = min(float(c[3]) for c in recent_candles)
        price_max = max(float(c[2]) for c in recent_candles)

        # Разбиваем диапазон на 20 уровней
        price_levels = np.linspace(price_min, price_max, 20)

        for i in range(len(price_levels) - 1):
            level_volume = 0
            for candle in recent_candles:
                candle_low = float(candle[3])
                candle_high = float(candle[2])
                candle_volume = float(candle[5])

                # Если свеча пересекает уровень, добавляем объем
                if candle_low <= price_levels[i + 1] and candle_high >= price_levels[i]:
                    level_volume += candle_volume

            if level_volume > 0:
                profile.append({
                    'price_level': float((price_levels[i] + price_levels[i + 1]) / 2),
                    'volume': float(level_volume)
                })

    return {
        'spikes': spikes[-10:],  # Последние 10 всплесков
        'profile': sorted(profile, key=lambda x: x['volume'], reverse=True)[:10],
        'volume_ratio': current_volume_ratio  # Для фильтрации
    }


def simulate_higher_timeframes(candles: List[List[str]]) -> Dict[str, str]:
    """
    Симуляция анализа старших таймфреймов из 15-минутных данных

    Args:
        candles: 15-минутные свечи

    Returns:
        Словарь с направлениями трендов на старших ТФ
    """
    if len(candles) < 100:
        return {'1h_direction': 'UNKNOWN', '4h_direction': 'UNKNOWN'}

    prices = np.array([float(candle[4]) for candle in candles])

    # Симуляция 1H (4 свечи 15M = 1H)
    h1_direction = 'UNKNOWN'
    if len(prices) >= 16:  # Минимум 4 часа данных
        h1_prices = []
        for i in range(0, len(prices), 4):
            if i + 3 < len(prices):
                h1_prices.append(prices[i + 3])  # Цена закрытия часовой свечи

        if len(h1_prices) >= 4:
            recent_h1 = h1_prices[-4:]
            if recent_h1[-1] > recent_h1[0]:
                h1_direction = 'BULLISH'
            elif recent_h1[-1] < recent_h1[0]:
                h1_direction = 'BEARISH'
            else:
                h1_direction = 'SIDEWAYS'

    # Симуляция 4H (16 свечей 15M = 4H)
    h4_direction = 'UNKNOWN'
    if len(prices) >= 64:  # Минимум 16 часов данных
        h4_prices = []
        for i in range(0, len(prices), 16):
            if i + 15 < len(prices):
                h4_prices.append(prices[i + 15])  # Цена закрытия 4-часовой свечи

        if len(h4_prices) >= 4:
            recent_h4 = h4_prices[-4:]
            if recent_h4[-1] > recent_h4[0]:
                h4_direction = 'BULLISH'
            elif recent_h4[-1] < recent_h4[0]:
                h4_direction = 'BEARISH'
            else:
                h4_direction = 'SIDEWAYS'

    return {
        '1h_direction': h1_direction,
        '4h_direction': h4_direction
    }


def calculate_atr_stops(candles: List[List[str]], period: int = 10) -> Dict[str, Any]:
    """
    Расчет ATR и динамических стоп-лоссов (ускоренный для скальпинга)

    Args:
        candles: Данные свечей
        period: Период ATR (ускорен до 10 для скальпинга)

    Returns:
        Словарь с ATR и уровнями стопов
    """
    if len(candles) < period + 1:
        return {'atr': [], 'stop_levels': {}}

    highs = np.array([float(candle[2]) for candle in candles])
    lows = np.array([float(candle[3]) for candle in candles])
    closes = np.array([float(candle[4]) for candle in candles])

    # Расчет True Range
    tr = np.zeros(len(candles))
    for i in range(1, len(candles)):
        tr[i] = max(
            highs[i] - lows[i],  # High - Low
            abs(highs[i] - closes[i - 1]),  # High - Previous Close
            abs(lows[i] - closes[i - 1])  # Low - Previous Close
        )

    # Расчет ATR
    atr = np.zeros(len(candles))
    atr[period] = np.mean(tr[1:period + 1])

    for i in range(period + 1, len(candles)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    # Динамические стоп-лоссы (скальпинговые множители)
    stop_levels = {}
    if len(atr) > 0:
        current_price = closes[-1]
        current_atr = atr[-1]

        stop_levels = {
            'long_stop': current_price - (current_atr * 1.2),  # Скальпинговый множитель 1.2
            'short_stop': current_price + (current_atr * 1.2),  # Скальпинговый множитель 1.2
            'trailing_multiplier': 1.2,
            'current_volatility': current_atr / current_price * 100 if current_price > 0 else 0
        }

    return {
        'atr': atr.tolist(),
        'stop_levels': stop_levels
    }


# СКАЛЬПИНГОВЫЕ ПАРАМЕТРЫ ДЛЯ 15M (ОБНОВЛЕННЫЕ)
SCALPING_15M_PARAMS = {
    'ema_fast': 5,  # Ускорено с 7 для скальпинга
    'ema_medium': 8,  # Ускорено с 14 для скальпинга
    'ema_slow': 13,  # Ускорено с 28 для скальпинга
    'tsi_long': 8,  # Ускорено с 12 для скальпинга
    'tsi_short': 4,  # Ускорено с 6 для скальпинга
    'tsi_signal': 3,  # Ускорено с 6 для скальпинга
    'rsi_period': 9,  # Оставляем как есть
    'volume_lookback': 10,  # Для анализа объемных всплесков
    'atr_period': 10,  # Ускорено с 14 для скальпинга
    'atr_multiplier': 1.2,  # Снижено с 2.0 для скальпинга
    'min_tsi_momentum': 7.0,  # Повышено с 3.0 для качества
    'volume_threshold': 2.0,  # Повышено с 1.8 для качества
    'min_quality_score': 75  # Повышено с 50 для качества
}

# АДАПТИВНЫЕ ПАРАМЕТРЫ ПО ТИПАМ АКТИВОВ
ASSET_SPECIFIC_PARAMS = {
    'BTC_majors': {  # BTC, ETH - стабильные
        'ema_periods': [5, 8, 13],
        'tsi_params': [8, 4, 3],
        'min_momentum': 7.0,  # Повышено
        'volume_threshold': 2.0  # Повышено
    },
    'volatile_alts': {  # DOGE, SHIB, мем-коины
        'ema_periods': [4, 6, 10],
        'tsi_params': [6, 3, 2],
        'min_momentum': 10.0,  # Повышено значительно
        'volume_threshold': 2.5
    },
    'stable_alts': {  # Крупные альты
        'ema_periods': [5, 8, 13],
        'tsi_params': [8, 4, 3],
        'min_momentum': 8.0,  # Повышено
        'volume_threshold': 2.0  # Повышено
    }
}


def get_optimal_params_for_asset(symbol: str) -> Dict[str, Any]:
    """
    Получение оптимальных параметров для данного актива

    Args:
        symbol: Символ торговой пары

    Returns:
        Оптимальные параметры для данного актива
    """
    # Определяем тип актива
    if symbol.startswith(('BTC', 'ETH')):
        return ASSET_SPECIFIC_PARAMS['BTC_majors']
    elif symbol.startswith(('DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF')):
        return ASSET_SPECIFIC_PARAMS['volatile_alts']
    else:
        return ASSET_SPECIFIC_PARAMS['stable_alts']


def improved_crossover_detection(tsi_vals: List[float],
                                 signal_vals: List[float],
                                 tolerance: float = 0.3) -> Tuple[bool, bool]:
    """
    Улучшенное определение пересечений TSI с толерантностью

    Args:
        tsi_vals: Значения TSI
        signal_vals: Значения сигнальной линии
        tolerance: Толерантность для пересечения

    Returns:
        Tuple[cross_up, cross_down]
    """
    if len(tsi_vals) < 2 or len(signal_vals) < 2:
        return False, False

    prev_tsi, curr_tsi = tsi_vals[-2], tsi_vals[-1]
    prev_sig, curr_sig = signal_vals[-2], signal_vals[-1]

    cross_up = (prev_tsi <= prev_sig + tolerance) and (curr_tsi > curr_sig + tolerance)
    cross_down = (prev_tsi >= prev_sig - tolerance) and (curr_tsi < curr_sig - tolerance)

    return cross_up, cross_down


def check_ema_micro_consistency(indicators: Dict[str, Any], last_n: int = 3) -> bool:
    """
    Проверка микро-консистентности EMA на последних N свечах

    Args:
        indicators: Словарь индикаторов
        last_n: Количество последних свечей для анализа

    Returns:
        True если EMA выровнены консистентно
    """
    try:
        ema1_vals = indicators.get('ema1_values', [])
        ema2_vals = indicators.get('ema2_values', [])
        ema3_vals = indicators.get('ema3_values', [])

        if len(ema1_vals) < last_n or len(ema2_vals) < last_n or len(ema3_vals) < last_n:
            return False

        # Проверяем консистентность на последних N свечах
        for i in range(-last_n, 0):
            if not (ema1_vals[i] > ema2_vals[i] > ema3_vals[i] or
                    ema1_vals[i] < ema2_vals[i] < ema3_vals[i]):
                return False

        return True
    except Exception:
        return False


def check_higher_tf_conflict(indicators: Dict[str, Any]) -> bool:
    """
    Проверка конфликтов с старшими таймфреймами

    Args:
        indicators: Словарь индикаторов

    Returns:
        True если есть конфликт
    """
    h1_trend = indicators.get('h1_trend', 'UNKNOWN')
    h4_trend = indicators.get('h4_trend', 'UNKNOWN')

    # Если тренды противоположные - конфликт
    if ((h1_trend == 'BULLISH' and h4_trend == 'BEARISH') or
            (h1_trend == 'BEARISH' and h4_trend == 'BULLISH')):
        return True

    return False


def apply_ai_optimized_filters(candles: List[List[str]], indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Система качественных фильтров для ИИ (ОБНОВЛЕННАЯ)

    Args:
        candles: Данные свечей
        indicators: Рассчитанные индикаторы

    Returns:
        Результат фильтрации
    """
    filters_passed = []
    filters_failed = []

    # 1. TSI Momentum фильтр (КРИТИЧНЫЙ) - повышен порог
    momentum_strength = indicators.get('momentum_strength', 0)
    if momentum_strength > 7.0:  # Было 3.0 - повышено в 2.3 раза
        filters_passed.append('STRONG_TSI_MOMENTUM')
    else:
        filters_failed.append('WEAK_TSI_MOMENTUM')

    # 2. EMA выравнивание с микро-трендом
    ema_micro_trend = check_ema_micro_consistency(indicators, last_n=3)
    if ema_micro_trend:
        filters_passed.append('EMA_MICRO_CONSISTENT')
    else:
        filters_failed.append('EMA_INCONSISTENT')

    # 3. Объемное подтверждение (повышенные требования)
    volume_spike = indicators.get('volume_ratio', 1.0)
    if volume_spike > 2.0:  # Было 1.8 - повышено для скальпинга
        filters_passed.append('VOLUME_CONFIRMED')
    else:
        filters_failed.append('INSUFFICIENT_VOLUME')

    # 4. Отсутствие конфликтов с старшими ТФ
    mtf_conflict = check_higher_tf_conflict(indicators)
    if not mtf_conflict:
        filters_passed.append('NO_MTF_CONFLICT')
    else:
        filters_failed.append('MTF_CONFLICT')

    # 5. НОВЫЙ ФИЛЬТР: Проверка недавних движений
    if len(candles) >= 8:
        recent_moves = []
        for i in range(-8, 0):
            try:
                current_price = float(candles[i][4])
                prev_price = float(candles[i - 1][4])
                move = abs(current_price - prev_price) / prev_price * 100
                recent_moves.append(move)
            except (ValueError, ZeroDivisionError, IndexError):
                continue

        max_recent_move = max(recent_moves) if recent_moves else 0
        if max_recent_move <= 1.5:  # Нет сильных движений >1.5% за 8 свечей
            filters_passed.append('NO_STRONG_RECENT_MOVES')
        else:
            filters_failed.append('STRONG_RECENT_MOVES')

    # 6. НОВЫЙ ФИЛЬТР: Минимальный спред EMA
    ema1_vals = indicators.get('ema1_values', [0])
    ema3_vals = indicators.get('ema3_values', [0])
    if len(ema1_vals) > 0 and len(ema3_vals) > 0 and ema3_vals[-1] > 0:
        ema_spread = abs(ema1_vals[-1] - ema3_vals[-1]) / ema3_vals[-1] * 100
        if ema_spread >= 0.4:  # Минимальный спред 0.4%
            filters_passed.append('ADEQUATE_EMA_SPREAD')
        else:
            filters_failed.append('INSUFFICIENT_EMA_SPREAD')

    return {
        'filters_passed': filters_passed,
        'filters_failed': filters_failed,
        'quality_score': len(filters_passed) * 16.67  # Макс 100 (6 фильтров)
    }


def analyze_recent_price_action(candles: List[List[str]], lookback: int = 10) -> Dict[str, Any]:
    """
    Анализ последних свечей (детальный)

    Args:
        candles: Данные свечей
        lookback: Количество свечей для анализа

    Returns:
        Результат анализа микроструктуры
    """
    if len(candles) < lookback:
        return {
            'micro_trend_strength': 0,
            'momentum_building': False,
            'volume_pattern': 'UNKNOWN',
            'price_structure': 'UNKNOWN',
            'breakout_potential': False
        }

    recent = candles[-lookback:]
    closes = np.array([float(candle[4]) for candle in recent])
    highs = np.array([float(candle[2]) for candle in recent])
    lows = np.array([float(candle[3]) for candle in recent])
    volumes = np.array([float(candle[5]) for candle in recent])

    # Сила микротренда
    trend_strength = abs(closes[-1] - closes[0]) / closes[0] * 100 if closes[0] > 0 else 0

    # Нарастание импульса
    momentum_building = np.mean(closes[-3:]) > np.mean(closes[:3]) if len(closes) >= 6 else False

    # Паттерн объемов
    avg_volume = np.mean(volumes)
    volume_pattern = 'INCREASING' if volumes[-1] > avg_volume else 'DECREASING'

    # Структура цены
    higher_highs = highs[-1] > highs[-2] > highs[-3] if len(highs) >= 3 else False
    higher_lows = lows[-1] > lows[-2] > lows[-3] if len(lows) >= 3 else False

    if higher_highs and higher_lows:
        price_structure = 'UPTREND'
    elif not higher_highs and not higher_lows:
        price_structure = 'DOWNTREND'
    else:
        price_structure = 'CONSOLIDATION'

    return {
        'micro_trend_strength': trend_strength,
        'momentum_building': momentum_building,
        'volume_pattern': volume_pattern,
        'price_structure': price_structure,
        'breakout_potential': trend_strength > 1.0 and momentum_building
    }


def calculate_predictive_signals(candles: List[List[str]], indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Предиктивные индикаторы

    Args:
        candles: Данные свечей
        indicators: Рассчитанные индикаторы

    Returns:
        Предиктивные сигналы
    """
    if len(candles) < 10:
        return {
            'momentum_acceleration': 0,
            'volume_leading': False,
            'ema_compression': False,
            'price_velocity_change': False
        }

    # Ускорение TSI momentum
    momentum_vals = indicators.get('momentum', [])
    if len(momentum_vals) >= 3:
        momentum_acceleration = momentum_vals[-1] - momentum_vals[-3]
    else:
        momentum_acceleration = 0

    # Опережающие объемные сигналы
    volume_spikes = indicators.get('volume_spikes', [])
    volume_leading = len([s for s in volume_spikes if s.get('ratio', 1) > 2.0]) > 0

    # Сжатие EMA перед движением
    ema1_vals = indicators.get('ema1_values', [])
    ema3_vals = indicators.get('ema3_values', [])
    if len(ema1_vals) >= 5 and len(ema3_vals) >= 5:
        current_spread = abs(ema1_vals[-1] - ema3_vals[-1])
        prev_spread = abs(ema1_vals[-5] - ema3_vals[-5])
        ema_compression = current_spread < prev_spread * 0.8
    else:
        ema_compression = False

    # Изменение скорости цены
    closes = np.array([float(candle[4]) for candle in candles[-10:]])
    if len(closes) >= 5:
        recent_velocity = abs(closes[-1] - closes[-3])
        prev_velocity = abs(closes[-3] - closes[-5])
        price_velocity_change = recent_velocity > prev_velocity * 1.2
    else:
        price_velocity_change = False

    return {
        'momentum_acceleration': momentum_acceleration,
        'volume_leading': volume_leading,
        'ema_compression': ema_compression,
        'price_velocity_change': price_velocity_change
    }


def calculate_signal_strength_for_ai(candles: List[List[str]],
                                     indicators: Dict[str, Any],
                                     signal_type: str) -> Dict[str, Any]:
    """
    Расчет силы сигнала специально для ИИ

    Args:
        candles: Данные свечей
        indicators: Рассчитанные индикаторы
        signal_type: Тип сигнала ('LONG' или 'SHORT')

    Returns:
        Метрики силы для ИИ
    """
    if not candles or not indicators:
        return {
            'momentum_acceleration': 0,
            'trend_consistency': 0,
            'volume_confirmation': False,
            'price_velocity': 0,
            'market_microstructure': 'UNKNOWN',
            'confluence_score': 0,
            'signal_freshness': 1,
            'volatility_regime': 'UNKNOWN',
            'confluence_factors': []
        }

    # Скорость изменения TSI momentum
    momentum_vals = indicators.get('momentum', [])
    momentum_acceleration = 0
    if len(momentum_vals) >= 3:
        momentum_acceleration = abs(momentum_vals[-1] - momentum_vals[-3])

    # Консистентность EMA направлений
    ema1_vals = indicators.get('ema1_values', [])
    ema2_vals = indicators.get('ema2_values', [])
    ema3_vals = indicators.get('ema3_values', [])

    trend_consistency = 0
    if len(ema1_vals) >= 5:
        if signal_type == 'LONG':
            trend_consistency = sum(1 for i in range(-5, 0)
                                    if ema1_vals[i] > ema2_vals[i] > ema3_vals[i]) / 5
        else:
            trend_consistency = sum(1 for i in range(-5, 0)
                                    if ema1_vals[i] < ema2_vals[i] < ema3_vals[i]) / 5

    # Подтверждение объемом
    volume_confirmation = indicators.get('volume_ratio', 1.0) > 2.0  # Повышено с 1.8

    # Скорость движения цены
    if len(candles) >= 5:
        price_start = float(candles[-5][4])
        price_end = float(candles[-1][4])
        price_velocity = abs(price_end - price_start) / price_start * 100 if price_start > 0 else 0
    else:
        price_velocity = 0

    # Состояние микроструктуры
    price_action = analyze_recent_price_action(candles, 10)
    market_microstructure = price_action.get('price_structure', 'UNKNOWN')

    # Общий скор совпадения индикаторов
    confluence_factors = []
    if indicators.get('momentum_strength', 0) > 7.0:  # Повышено с 3.0
        confluence_factors.append('STRONG_TSI')
    if volume_confirmation:
        confluence_factors.append('VOLUME_SPIKE')
    if indicators.get('mtf_confluence', False):
        confluence_factors.append('MTF_ALIGNED')
    if len(indicators.get('nearby_support', [])) > 0 or len(indicators.get('nearby_resistance', [])) > 0:
        confluence_factors.append('NEAR_LEVELS')

    confluence_score = len(confluence_factors) * 25  # Макс 100

    # Возраст сигнала в свечах (всегда 1 для текущего сигнала)
    signal_freshness = 1

    # Текущий режим волатильности
    atr_current = indicators.get('current_atr', 0)
    current_price = float(candles[-1][4]) if candles else 1
    volatility_percent = (atr_current / current_price * 100) if current_price > 0 else 0

    if volatility_percent > 2.0:
        volatility_regime = 'HIGH'
    elif volatility_percent > 1.0:
        volatility_regime = 'MEDIUM'
    else:
        volatility_regime = 'LOW'

    return {
        'momentum_acceleration': momentum_acceleration,
        'trend_consistency': trend_consistency * 100,
        'volume_confirmation': volume_confirmation,
        'price_velocity': price_velocity,
        'market_microstructure': market_microstructure,
        'confluence_score': confluence_score,
        'signal_freshness': signal_freshness,
        'volatility_regime': volatility_regime,
        'confluence_factors': confluence_factors
    }


def enhanced_signal_detection(candles: List[List[str]]) -> Dict[str, Any]:
    """
    Улучшенная логика определения сигналов (ГЛАВНАЯ ФУНКЦИЯ для main.py)

    Args:
        candles: Данные свечей

    Returns:
        Результат улучшенного определения сигнала
    """
    if len(candles) < 50:
        return {
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'strength_metrics': {},
            'market_context': {},
            'ai_input_data': {}
        }

    # Используем оптимизированные параметры для 15M
    params = SCALPING_15M_PARAMS

    # Рассчитываем все индикаторы
    indicators = calculate_all_indicators_extended(
        candles,
        params['ema_fast'],
        params['ema_medium'],
        params['ema_slow'],
        params['tsi_long'],
        params['tsi_short'],
        params['tsi_signal']
    )

    if not indicators:
        return {
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'strength_metrics': {},
            'market_context': {},
            'ai_input_data': {}
        }

    # Проверяем базовые условия с более мягкими пересечениями
    tsi_vals = indicators.get('tsi_values', [])
    signal_vals = indicators.get('tsi_signal_values', [])
    ema1_vals = indicators.get('ema1_values', [])
    ema2_vals = indicators.get('ema2_values', [])
    ema3_vals = indicators.get('ema3_values', [])

    if len(tsi_vals) < 2 or len(ema1_vals) < 1:
        return {
            'signal': 'NO_SIGNAL',
            'confidence': 0,
            'strength_metrics': {},
            'market_context': {},
            'ai_input_data': {}
        }

    # Улучшенное определение пересечений
    cross_up, cross_down = improved_crossover_detection(tsi_vals, signal_vals, tolerance=0.3)

    # EMA условия
    current_ema1 = ema1_vals[-1]
    current_ema2 = ema2_vals[-1]
    current_ema3 = ema3_vals[-1]

    ema_bullish = current_ema1 > current_ema2 > current_ema3
    ema_bearish = current_ema1 < current_ema2 < current_ema3

    # Определяем сигнал
    signal = 'NO_SIGNAL'
    if ema_bullish and cross_up:
        signal = 'LONG'
    elif ema_bearish and cross_down:
        signal = 'SHORT'

    # Применяем фильтры качества
    filter_results = apply_ai_optimized_filters(candles, indicators)

    # Рассчитываем уверенность (0-100)
    confidence = filter_results['quality_score']

    # Дополнительные проверки для повышения уверенности
    if signal != 'NO_SIGNAL':
        # Проверяем силу TSI momentum
        momentum_strength = indicators.get('momentum_strength', 0)
        if momentum_strength > 10.0:  # Очень сильный momentum
            confidence += 15
        elif momentum_strength > 7.0:  # Сильный momentum
            confidence += 10

        # Проверяем объемное подтверждение
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 2.5:  # Очень сильный объемный всплеск
            confidence += 15
        elif volume_ratio > 2.0:  # Сильный объемный всплеск
            confidence += 10

        # Проверяем MTF confluence
        if indicators.get('mtf_confluence', False):
            confidence += 10

    # Ограничиваем confidence в пределах 0-100
    confidence = min(100, max(0, confidence))

    # Если confidence слишком низкий, отменяем сигнал
    if confidence < SCALPING_15M_PARAMS['min_quality_score']:  # 75
        signal = 'NO_SIGNAL'

    # Рассчитываем метрики силы для ИИ
    strength_metrics = calculate_signal_strength_for_ai(candles, indicators, signal)

    # Контекст рынка
    market_context = {
        'h1_trend': indicators.get('h1_trend', 'UNKNOWN'),
        'h4_trend': indicators.get('h4_trend', 'UNKNOWN'),
        'rsi_current': indicators.get('rsi_current', 50),
        'near_support': len(indicators.get('nearby_support', [])) > 0,
        'near_resistance': len(indicators.get('nearby_resistance', [])) > 0,
        'volume_spikes_recent': len(indicators.get('volume_spikes', [])) > 0
    }

    # Данные для ИИ
    ai_input_data = prepare_ai_context_data(candles, indicators, signal)

    return {
        'signal': signal,
        'confidence': confidence,
        'strength_metrics': strength_metrics,
        'market_context': market_context,
        'ai_input_data': ai_input_data,
        'filter_results': filter_results
    }


def prepare_ai_context_data(candles: List[List[str]],
                            indicators: Dict[str, Any],
                            signal_type: str) -> Dict[str, Any]:
    """
    Обогащение данных для ИИ

    Args:
        candles: Данные свечей
        indicators: Рассчитанные индикаторы
        signal_type: Тип сигнала

    Returns:
        Контекстные данные для ИИ
    """
    if not candles or not indicators:
        return {
            'market_regime': 'UNKNOWN',
            'volatility_percentile': 50,
            'volume_profile_analysis': {'high_volume_levels': 0, 'price_at_high_volume': False},
            'signal_context': {'time_since_last_signal': 1, 'recent_signal_success_rate': 0.75,
                               'confluence_factors': [], 'risk_factors': []},
            'technical_context': {'trend_alignment': False, 'momentum_phase': 'NEUTRAL',
                                  'mean_reversion_risk': 'UNKNOWN', 'breakout_continuation': False}
        }

    current_price = float(candles[-1][4])

    # Определение режима рынка
    atr_current = indicators.get('current_atr', 0)
    rsi_current = indicators.get('rsi_current', 50)
    volume_ratio = indicators.get('volume_ratio', 1.0)

    if volume_ratio > 2.5 and atr_current / current_price > 0.02:  # Повышены пороги
        market_regime = 'HIGH_VOLATILITY_BREAKOUT'
    elif rsi_current > 70 or rsi_current < 30:
        market_regime = 'OVERBOUGHT_OVERSOLD'
    elif volume_ratio < 0.8:
        market_regime = 'LOW_VOLUME_CONSOLIDATION'
    else:
        market_regime = 'NORMAL_TRADING'

    # Перцентиль волатильности
    atr_values = indicators.get('atr_values', [])
    if len(atr_values) >= 20:
        current_atr_percentile = sum(1 for atr in atr_values[-20:] if atr < atr_current) / 20 * 100
    else:
        current_atr_percentile = 50

    # Анализ объемного профиля
    volume_profile = indicators.get('volume_profile', [])
    volume_profile_analysis = {
        'high_volume_levels': len([level for level in volume_profile if level.get('volume', 0) > 0]),
        'price_at_high_volume': len(volume_profile) > 0
    }

    # Контекст сигнала
    signal_context = {
        'time_since_last_signal': 1,  # Всегда свежий сигнал
        'recent_signal_success_rate': 0.75,  # Условное значение
        'confluence_factors': indicators.get('confluence_factors', []),
        'risk_factors': []
    }

    # Добавляем факторы риска
    if indicators.get('h1_trend') != indicators.get('h4_trend'):
        signal_context['risk_factors'].append('MTF_DIVERGENCE')
    if rsi_current > 80 or rsi_current < 20:
        signal_context['risk_factors'].append('EXTREME_RSI')
    if volume_ratio < 0.6:  # Повышен порог
        signal_context['risk_factors'].append('LOW_VOLUME')

    # Техническая картина
    h1_trend = indicators.get('h1_trend', 'UNKNOWN')
    h4_trend = indicators.get('h4_trend', 'UNKNOWN')

    if signal_type == 'LONG':
        trend_alignment = h1_trend == 'BULLISH' and h4_trend == 'BULLISH'
        momentum_phase = 'BUILDING' if indicators.get('momentum_strength', 0) > 7 else 'WEAK'
    elif signal_type == 'SHORT':
        trend_alignment = h1_trend == 'BEARISH' and h4_trend == 'BEARISH'
        momentum_phase = 'BUILDING' if indicators.get('momentum_strength', 0) > 7 else 'WEAK'
    else:
        trend_alignment = False
        momentum_phase = 'NEUTRAL'

    # Риск возврата к среднему
    ema1_current = indicators.get('ema1_values', [0])[-1]
    ema3_current = indicators.get('ema3_values', [0])[-1]
    if ema3_current > 0:
        ema_divergence = abs(ema1_current - ema3_current) / ema3_current * 100
        mean_reversion_risk = 'HIGH' if ema_divergence > 3 else 'LOW'
    else:
        mean_reversion_risk = 'UNKNOWN'

    # Потенциал продолжения
    price_action = analyze_recent_price_action(candles, 10)
    breakout_continuation = price_action.get('breakout_potential', False)

    technical_context = {
        'trend_alignment': trend_alignment,
        'momentum_phase': momentum_phase,
        'mean_reversion_risk': mean_reversion_risk,
        'breakout_continuation': breakout_continuation
    }

    return {
        # Текущее состояние рынка
        'market_regime': market_regime,
        'volatility_percentile': current_atr_percentile,
        'volume_profile_analysis': volume_profile_analysis,

        # Контекст сигнала
        'signal_context': signal_context,

        # Техническая картина
        'technical_context': technical_context
    }


def signal_quality_validator(signal_data: Dict[str, Any], market_context: Dict[str, Any]) -> bool:
    """
    Система подтверждения качества (ИСПОЛЬЗУЕТСЯ в main.py)

    Args:
        signal_data: Данные сигнала
        market_context: Контекст рынка

    Returns:
        True если сигнал прошел валидацию
    """
    quality_checks = {}

    # momentum_sufficient: momentum >= пороговых значений
    momentum_strength = signal_data.get('strength_metrics', {}).get('momentum_acceleration', 0)
    quality_checks['momentum_sufficient'] = momentum_strength >= 7.0  # Повышено с 3.0

    # volume_adequate: объем выше среднего
    volume_confirmation = signal_data.get('strength_metrics', {}).get('volume_confirmation', False)
    quality_checks['volume_adequate'] = volume_confirmation

    # trend_aligned: тренды не конфликтуют
    technical_context = signal_data.get('ai_input_data', {}).get('technical_context', {})
    quality_checks['trend_aligned'] = technical_context.get('trend_alignment', False)

    # timing_optimal: время входа оптимально
    signal_freshness = signal_data.get('strength_metrics', {}).get('signal_freshness', 0)
    quality_checks['timing_optimal'] = signal_freshness <= 2

    # risk_acceptable: риск в допустимых пределах
    risk_factors = signal_data.get('ai_input_data', {}).get('signal_context', {}).get('risk_factors', [])
    quality_checks['risk_acceptable'] = len(risk_factors) <= 1

    # Подсчитываем качество
    quality_score = sum(quality_checks.values()) / len(quality_checks) * 100

    return quality_score >= 80  # Повышен порог с 75 до 80


def format_ai_input_data(signal_result: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """
    Стандартизация выходных данных (ИСПОЛЬЗУЕТСЯ в main.py)

    Args:
        signal_result: Результат анализа сигнала
        symbol: Символ торговой пары

    Returns:
        Стандартизированные данные для ИИ
    """
    strength_metrics = signal_result.get('strength_metrics', {})
    market_context = signal_result.get('market_context', {})
    ai_data = signal_result.get('ai_input_data', {})

    # Определяем качество сигнала
    signal_quality = signal_result.get('confidence', 0)

    # Определяем условия рынка
    risk_factors = ai_data.get('signal_context', {}).get('risk_factors', [])
    if len(risk_factors) == 0:
        market_conditions = 'FAVORABLE'
    elif len(risk_factors) <= 1:
        market_conditions = 'NEUTRAL'
    else:
        market_conditions = 'ADVERSE'

    # Confluence факторы
    confluence_factors = strength_metrics.get('confluence_factors', [])

    # Определяем режим волатильности
    volatility_regime = strength_metrics.get('volatility_regime', 'MEDIUM')

    return {
        'entry_signal': {
            'pair': symbol,
            'direction': signal_result.get('signal', 'NO_SIGNAL').lower(),
            'confidence': signal_result.get('confidence', 0),
            'entry_price': 0,  # Будет заполнено при получении текущей цены
            'timestamp': int(time.time())
        },
        'technical_analysis': {
            'trend_strength': min(100, strength_metrics.get('trend_consistency', 0)),
            'momentum_score': min(100, strength_metrics.get('momentum_acceleration', 0) * 10),
            'volume_confirmation': strength_metrics.get('volume_confirmation', False),
            'support_resistance': {
                'near_support': market_context.get('near_support', False),
                'near_resistance': market_context.get('near_resistance', False)
            },
            'volatility_regime': volatility_regime.lower()
        },
        'risk_assessment': {
            'signal_quality': signal_quality,
            'market_conditions': market_conditions.lower(),
            'confluence_factors': confluence_factors,
            'warning_signals': risk_factors
        },
        'context_data': {
            'time_of_day': int(time.strftime('%H')),
            'market_session': 'unknown',  # Можно улучшить определение сессии
            'recent_performance': 75,  # Условное значение
            'similar_setups': 70  # Условное значение
        }
    }


def calculate_all_indicators_extended(candles: List[List[str]],
                                      ema1: int = 5,
                                      ema2: int = 8,
                                      ema3: int = 13,
                                      tsi_long: int = 8,
                                      tsi_short: int = 4,
                                      tsi_signal: int = 3) -> Dict[str, Any]:
    """
    Главная функция расчета всех индикаторов (ИСПОЛЬЗУЕТСЯ в main.py)

    Args:
        candles: Данные свечей
        ema1, ema2, ema3: Периоды EMA (ускоренные для 15M)
        tsi_long, tsi_short, tsi_signal: Параметры TSI (ускоренные для 15M)

    Returns:
        Словарь со всеми рассчитанными индикаторами
    """
    if not candles:
        return {}

    # Базовые индикаторы
    base_indicators = calculate_indicators_for_candles(
        candles, ema1, ema2, ema3, tsi_long, tsi_short, tsi_signal
    )

    # Дополнительные индикаторы
    levels = find_support_resistance_levels(candles)
    rsi_data = calculate_rsi_with_divergence(candles)
    volume_data = get_volume_anomalies(candles)
    mtf_data = simulate_higher_timeframes(candles)
    atr_data = calculate_atr_stops(candles, period=SCALPING_15M_PARAMS['atr_period'])

    # Извлекаем цены для дополнительного анализа
    prices = [float(candle[4]) for candle in candles]
    current_price = prices[-1] if prices else 0

    # Анализ близости к уровням
    nearby_support = []
    nearby_resistance = []

    if current_price > 0:
        # Ищем ближайшие уровни поддержки (ниже текущей цены)
        for level in levels['support']:
            if level < current_price:
                distance = abs(current_price - level) / current_price * 100
                if distance <= 2.0:  # В пределах 2%
                    nearby_support.append({'level': level, 'distance_percent': distance})

        # Ищем ближайшие уровни сопротивления (выше текущей цены)
        for level in levels['resistance']:
            if level > current_price:
                distance = abs(level - current_price) / current_price * 100
                if distance <= 2.0:  # В пределах 2%
                    nearby_resistance.append({'level': level, 'distance_percent': distance})

    # Объединяем все данные
    extended_indicators = {
        # Базовые индикаторы (EMA + TSI)
        **base_indicators,

        # Уровневый анализ
        'support_levels': levels['support'][-5:],  # Последние 5 уровней
        'resistance_levels': levels['resistance'][-5:],  # Последние 5 уровней
        'nearby_support': nearby_support,
        'nearby_resistance': nearby_resistance,

        # RSI + дивергенции
        'rsi_values': rsi_data['rsi'],
        'rsi_divergences': rsi_data['divergences'],
        'rsi_current': rsi_data['rsi'][-1] if rsi_data['rsi'] else 50,

        # Объемный анализ
        'volume_spikes': volume_data['spikes'],
        'volume_profile': volume_data['profile'],
        'volume_ratio': volume_data['volume_ratio'],  # Для фильтров

        # Мультитаймфрейм
        'h1_trend': mtf_data['1h_direction'],
        'h4_trend': mtf_data['4h_direction'],
        'mtf_confluence': mtf_data['1h_direction'] == mtf_data['4h_direction'],

        # ATR и стопы
        'atr_values': atr_data['atr'],
        'dynamic_stops': atr_data['stop_levels'],
        'current_atr': atr_data['atr'][-1] if atr_data['atr'] else 0,

        # Дополнительные поля для фильтров
        'momentum_strength': base_indicators.get('momentum_strength', 0)
    }

    return extended_indicators


def calculate_indicators_for_candles(candles: List[List[str]],
                                     ema1_period: int = 5,
                                     ema2_period: int = 8,
                                     ema3_period: int = 13,
                                     tsi_long: int = 8,
                                     tsi_short: int = 4,
                                     tsi_signal: int = 3) -> Dict[str, Any]:
    """
    Функция расчета базовых индикаторов для свечей

    Args:
        candles: Данные свечей
        ema1_period: Период первой EMA (ускорен для 15M)
        ema2_period: Период второй EMA (ускорен для 15M)
        ema3_period: Период третьей EMA (ускорен для 15M)
        tsi_long: Длинный период TSI (ускорен для 15M)
        tsi_short: Короткий период TSI (ускорен для 15M)
        tsi_signal: Период сигнальной линии TSI (ускорен для 15M)

    Returns:
        Словарь с массивами значений индикаторов
    """
    if not candles:
        return {}

    # Извлекаем цены закрытия
    prices = np.array([float(candle[4]) for candle in candles])

    # Рассчитываем EMA
    ema1_values = calculate_ema(prices, ema1_period)
    ema2_values = calculate_ema(prices, ema2_period)
    ema3_values = calculate_ema(prices, ema3_period)

    # Рассчитываем TSI с momentum
    tsi_data = calculate_tsi_with_momentum(candles, tsi_long, tsi_short, tsi_signal)

    return {
        'ema1_values': ema1_values.tolist(),
        'ema2_values': ema2_values.tolist(),
        'ema3_values': ema3_values.tolist(),
        'tsi_values': tsi_data['tsi'].tolist(),
        'tsi_signal_values': tsi_data['signal'].tolist(),
        'momentum': tsi_data['momentum'].tolist(),
        'momentum_strength': tsi_data['momentum_strength']
    }