import pandas as pd
import numpy as np


def calculate_atr(candles: list) -> float:
    """
    Вычисляет Average True Range (ATR) на основе свечных данных.

    Args:
        candles: Список данных свечей формата
               [['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'], ...]

    Returns:
        float: Значение ATR или 0, если недостаточно данных
    """
    if len(candles) < 2:
        return 0.0  # Возвращаем 0, если недостаточно данных

    tr_values = []
    for i in range(1, len(candles)):
        try:
            high = float(candles[i][2])
            low = float(candles[i][3])
            prev_close = float(candles[i - 1][4])
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)
        except (IndexError, ValueError):
            # Пропускаем итерацию, если данные некорректные
            continue

    if not tr_values:
        return 0.0  # Возвращаем 0, если не удалось вычислить TR

    atr = sum(tr_values) / len(tr_values)
    return atr


def detect_candlestick_signals(data: dict) -> dict:
    longs, shorts = set(), set()

    for symbol, content in data.items():
        candles = content['candles']
        if len(candles) != 3:
            continue

        o = []; h = []; l = []; c = []
        for ts, op, hi, lo, cl, *_ in candles:
            op, hi, lo, cl = map(float, (op, hi, lo, cl))
            o.append(op); h.append(hi); l.append(lo); c.append(cl)

        bodies = [abs(c[i] - o[i]) for i in range(3)]
        upper = [h[i] - max(o[i], c[i]) for i in range(3)]
        lower = [min(o[i], c[i]) - l[i] for i in range(3)]
        is_bull = [c[i] > o[i] for i in range(3)]
        is_bear = [not b for b in is_bull]
        mid1 = (o[0] + c[0]) / 2

        # Hammer → long
        if lower[2] >= 2*bodies[2] and upper[2] <= bodies[2]:
            longs.add(symbol)
        # Shooting Star → short
        if upper[2] >= 2*bodies[2] and lower[2] <= bodies[2]:
            shorts.add(symbol)
        # Bullish Engulfing → long
        if is_bear[0] and is_bull[1] and o[1] < c[0] and c[1] > o[0]:
            longs.add(symbol)
        # Bearish Engulfing → short
        if is_bull[0] and is_bear[1] and o[1] > c[0] and c[1] < o[0]:
            shorts.add(symbol)
        # Piercing Line → long
        if is_bear[0] and is_bull[1] and o[1] < l[0] and c[1] > mid1:
            longs.add(symbol)
        # Dark Cloud Cover → short
        if is_bull[0] and is_bear[1] and o[1] > h[0] and c[1] < mid1:
            shorts.add(symbol)
        # Morning Star → long
        if is_bear[0] and bodies[1] < (h[1]-l[1])*0.3 and is_bull[2] and c[2] > mid1:
            longs.add(symbol)
        # Evening Star → short
        if is_bull[0] and bodies[1] < (h[1]-l[1])*0.3 and is_bear[2] and c[2] < mid1:
            shorts.add(symbol)
        # Three White Soldiers → long
        if all(is_bull) and c[0] < c[1] < c[2]:
            longs.add(symbol)
        # Three Black Crows → short
        if all(is_bear) and c[0] > c[1] > c[2]:
            shorts.add(symbol)

    return {
        'long': sorted(longs),
        'short': sorted(shorts)
    }


def compute_cvd_signals(data, period_ma=100):
    """
    Реализация Cumulative Volume Delta индикатора по алгоритму в точности как в TradingView.

    Параметры:
    data: список строк формата [timestamp, open, high, low, close, volume, ...]
    period_ma: длина периода скользящей средней, по умолчанию 100

    Возвращает:
    список сигналов 'long'/'short' для каждой свечи
    """
    # Подготовка данных
    n = len(data)
    opens = [float(row[1]) for row in data]
    highs = [float(row[2]) for row in data]
    lows = [float(row[3]) for row in data]
    closes = [float(row[4]) for row in data]
    volumes = [float(row[5]) for row in data]

    # Инициализация массивов
    bull_power = [0.0] * n
    bear_power = [0.0] * n
    bull_volume = [0.0] * n
    bear_volume = [0.0] * n
    delta = [0.0] * n
    cvd = [0.0] * n
    cvd_ma = [None] * n
    signals = [None] * n

    # Pine Script использует NaN для первых свечей, где не хватает истории
    # Для простоты мы будем рассчитывать все со второй свечи
    for i in range(n):
        if i == 0:
            # Для первой свечи нет предыдущих данных, нуль - безопасное значение
            continue

        c = closes[i]
        o = opens[i]
        h = highs[i]
        l = lows[i]
        c1 = closes[i - 1]  # close[1] в Pine Script
        v = volumes[i]

        # Точное вычисление bullPower согласно Pine Script логике
        # Первый уровень: проверка close < open (медвежья свеча)
        if c < o:
            # Второй уровень: проверка close[1] < open
            if c1 < o:
                bull = max(h - c1, c - l)
            else:
                bull = max(h - o, c - l)
        # Первый уровень: проверка close > open (бычья свеча)
        elif c > o:
            # Второй уровень: проверка close[1] > open
            if c1 > o:
                bull = h - l
            else:
                bull = max(o - c1, h - l)
        # Первый уровень: close == open (доджи)
        else:
            # Второй уровень: проверка high - close > close - low
            if h - c > c - l:
                # Третий уровень: проверка close[1] < open
                if c1 < o:
                    bull = max(h - c1, c - l)
                else:
                    bull = h - o
            # Второй уровень: проверка high - close < close - low
            elif h - c < c - l:
                # Третий уровень: проверка close[1] > open
                if c1 > o:
                    bull = h - l
                else:
                    bull = max(o - c1, h - l)
            # Второй уровень: high - close == close - low
            else:
                # Третий уровень: проверка close[1] > open
                if c1 > o:
                    bull = max(h - o, c - l)
                # Третий уровень: проверка close[1] < open
                elif c1 < o:
                    bull = max(o - c1, h - l)
                # Третий уровень: close[1] == open
                else:
                    bull = h - l

        # Точное вычисление bearPower согласно Pine Script логике
        # Первый уровень: проверка close < open (медвежья свеча)
        if c < o:
            # Второй уровень: проверка close[1] > open
            if c1 > o:
                bear = max(c1 - o, h - l)
            else:
                bear = h - l
        # Первый уровень: проверка close > open (бычья свеча)
        elif c > o:
            # Второй уровень: проверка close[1] > open
            if c1 > o:
                bear = max(c1 - l, h - c)
            else:
                bear = max(o - l, h - c)
        # Первый уровень: close == open (доджи)
        else:
            # Второй уровень: проверка high - close > close - low
            if h - c > c - l:
                # Третий уровень: проверка close[1] > open
                if c1 > o:
                    bear = max(c1 - o, h - l)
                else:
                    bear = h - l
            # Второй уровень: проверка high - close < close - low
            elif h - c < c - l:
                # Третий уровень: проверка close[1] > open
                if c1 > o:
                    bear = max(c1 - l, h - c)
                else:
                    bear = o - l
            # Второй уровень: high - close == close - low
            else:
                # Третий уровень: проверка close[1] > open
                if c1 > o:
                    bear = max(c1 - o, h - l)
                # Третий уровень: проверка close[1] < open
                elif c1 < o:
                    bear = max(o - l, h - c)
                # Третий уровень: close[1] == open
                else:
                    bear = h - l

        # Сохраняем значения
        bull_power[i] = bull
        bear_power[i] = bear

        # Расчет Bull & Bear Volume
        divisor = bull + bear
        if divisor > 0:
            bull_volume[i] = (bull / divisor) * v
            bear_volume[i] = (bear / divisor) * v
        else:
            # Избегаем деления на ноль
            bull_volume[i] = bear_volume[i] = 0.0

        # Расчет Delta
        delta[i] = bull_volume[i] - bear_volume[i]

    # Расчет Cumulative Volume Delta (CVD)
    # cum() в Pine Script вычисляет кумулятивную сумму
    for i in range(n):
        if i == 0:
            cvd[i] = delta[i]
        else:
            cvd[i] = cvd[i - 1] + delta[i]

    # Расчет Simple Moving Average (SMA) для CVD
    for i in range(n):
        if i >= period_ma - 1:  # Нужно иметь хотя бы period_ma значений
            cvd_ma[i] = sum(cvd[i - (period_ma - 1):i + 1]) / period_ma

            # Генерация сигналов
            # В Pine Script: customColor = cvd > cvdMa ? color.teal : color.red
            # Teal (бирюзовый) = bullish/long, Red = bearish/short
            signals[i] = 'long' if cvd[i] > cvd_ma[i] else 'short'

    return signals


def compute_trend_signals(data, period_ma=100, ema_short=9, ema_medium=21, ema_long=50,
                          rsi_period=14, rsi_overbought=70, rsi_oversold=30,
                          macd_fast=12, macd_slow=26, macd_signal=9):
    """
    Комбинированный индикатор тренда, который заменяет CVD индикатор.

    Параметры:
    data: список строк формата [timestamp, open, high, low, close, volume, ...],
          отсортированных от более старых к более новым свечам
    period_ma: длина периода скользящей средней (не используется в этой реализации, но сохранен для совместимости)
    ema_short, ema_medium, ema_long: периоды для экспоненциальных скользящих средних
    rsi_period: период для расчета RSI
    rsi_overbought, rsi_oversold: пороговые значения для RSI
    macd_fast, macd_slow, macd_signal: периоды для расчета MACD

    Возвращает:
    список сигналов 'long'/'short'/None для каждой свечи в том же порядке (от старых к новым)
    """
    n = len(data)
    opens = [float(row[1]) for row in data]
    highs = [float(row[2]) for row in data]
    lows = [float(row[3]) for row in data]
    closes = [float(row[4]) for row in data]
    volumes = [float(row[5]) for row in data]

    # Инициализируем результаты
    signals = [None] * n

    # 1. Расчет EMA (экспоненциальная скользящая средняя)
    ema_short_values = calculate_ema(closes, ema_short)
    ema_medium_values = calculate_ema(closes, ema_medium)
    ema_long_values = calculate_ema(closes, ema_long)

    # 2. Расчет RSI (индекс относительной силы)
    rsi_values = calculate_rsi(closes, rsi_period)

    # 3. Расчет MACD
    macd_line, signal_line, histogram = calculate_macd(closes, macd_fast, macd_slow, macd_signal)

    # 4. Определение тренда по объему
    volume_trend = analyze_volume_trend(closes, volumes, 5)

    # 5. Определяем сигналы на основе комбинации индикаторов
    for i in range(n):
        # Пропускаем первые свечи, где недостаточно истории для расчета индикаторов
        if i < max(ema_long, rsi_period, macd_slow + macd_signal):
            continue

        # Счетчики сигналов
        bullish_signals = 0
        bearish_signals = 0

        # Проверяем EMA (тренд)
        if ema_short_values[i] > ema_medium_values[i] > ema_long_values[i]:
            bullish_signals += 2  # Сильный восходящий тренд
        elif ema_short_values[i] > ema_medium_values[i]:
            bullish_signals += 1  # Умеренный восходящий тренд
        elif ema_short_values[i] < ema_medium_values[i] < ema_long_values[i]:
            bearish_signals += 2  # Сильный нисходящий тренд
        elif ema_short_values[i] < ema_medium_values[i]:
            bearish_signals += 1  # Умеренный нисходящий тренд

        # Проверяем RSI (перекупленность/перепроданность)
        if rsi_values[i] < rsi_oversold:
            bullish_signals += 1  # Потенциальный разворот вверх
        elif rsi_values[i] > rsi_overbought:
            bearish_signals += 1  # Потенциальный разворот вниз

        # Проверяем MACD (импульс и развороты)
        if macd_line[i] > signal_line[i] and macd_line[i] > 0:
            bullish_signals += 1  # Положительный импульс
        elif macd_line[i] < signal_line[i] and macd_line[i] < 0:
            bearish_signals += 1  # Отрицательный импульс

        # Проверяем пересечение MACD и сигнальной линии (более сильный сигнал)
        if i > 0:
            if macd_line[i] > signal_line[i] and macd_line[i - 1] <= signal_line[i - 1]:
                bullish_signals += 2  # Бычье пересечение
            elif macd_line[i] < signal_line[i] and macd_line[i - 1] >= signal_line[i - 1]:
                bearish_signals += 2  # Медвежье пересечение

        # Проверяем объемный тренд
        if volume_trend[i] > 0:
            bullish_signals += 1  # Повышение объема на росте
        elif volume_trend[i] < 0:
            bearish_signals += 1  # Повышение объема на падении

        # Формируем итоговый сигнал на основе баланса бычьих и медвежьих сигналов
        if bullish_signals >= 3 and bullish_signals > bearish_signals + 1:
            signals[i] = 'long'
        elif bearish_signals >= 3 and bearish_signals > bullish_signals + 1:
            signals[i] = 'short'
        else:
            signals[i] = None

    return signals


def calculate_ema(prices, period):
    """Рассчитывает экспоненциальную скользящую среднюю"""
    ema = [None] * len(prices)
    # Инициализация первого значения как SMA
    ema[period - 1] = sum(prices[:period]) / period

    # Множитель сглаживания
    multiplier = 2 / (period + 1)

    # Расчет EMA для остальных точек
    for i in range(period, len(prices)):
        ema[i] = prices[i] * multiplier + ema[i - 1] * (1 - multiplier)

    return ema


def calculate_rsi(prices, period):
    """Рассчитывает индекс относительной силы (RSI)"""
    rsi = [None] * len(prices)
    gains = [0] * len(prices)
    losses = [0] * len(prices)

    # Вычисляем изменения цены
    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains[i] = change
        else:
            losses[i] = abs(change)

    # Первый средний выигрыш и проигрыш
    avg_gain = sum(gains[1:period + 1]) / period
    avg_loss = sum(losses[1:period + 1]) / period

    # Рассчитываем RSI для каждой точки
    for i in range(period, len(prices)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(prices, fast_period, slow_period, signal_period):
    """Рассчитывает MACD (схождение-расхождение скользящих средних)"""
    n = len(prices)
    macd_line = [None] * n
    signal_line = [None] * n
    histogram = [None] * n

    # Рассчитываем EMA для быстрого и медленного периодов
    fast_ema = calculate_ema(prices, fast_period)
    slow_ema = calculate_ema(prices, slow_period)

    # Рассчитываем MACD линию (разница между быстрой и медленной EMA)
    for i in range(slow_period - 1, n):
        if fast_ema[i] is not None and slow_ema[i] is not None:
            macd_line[i] = fast_ema[i] - slow_ema[i]

    # Рассчитываем сигнальную линию (EMA от MACD линии)
    # Берем только ненулевые значения MACD для расчета
    macd_values = [v for v in macd_line if v is not None]

    if len(macd_values) >= signal_period:
        # Инициализация первого значения как SMA
        start_idx = slow_period - 1 + signal_period - 1
        signal_line[start_idx] = sum(macd_line[slow_period - 1:start_idx + 1]) / signal_period

        # Множитель сглаживания
        multiplier = 2 / (signal_period + 1)

        # Расчет сигнальной линии
        for i in range(start_idx + 1, n):
            signal_line[i] = macd_line[i] * multiplier + signal_line[i - 1] * (1 - multiplier)

    # Рассчитываем гистограмму (разница между MACD и сигнальной линией)
    for i in range(n):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram[i] = macd_line[i] - signal_line[i]

    return macd_line, signal_line, histogram


def analyze_volume_trend(prices, volumes, period=5):
    """Анализирует тренд объема относительно движения цены"""
    n = len(prices)
    volume_trend = [None] * n

    for i in range(period, n):
        price_change = prices[i] - prices[i - period]
        avg_volume_current = sum(volumes[i - period + 1:i + 1]) / period
        avg_volume_previous = sum(
            volumes[i - 2 * period + 1:i - period + 1]) / period if i >= 2 * period else avg_volume_current

        # Если цена растет и объем увеличивается = сильный восходящий тренд
        if price_change > 0 and avg_volume_current > avg_volume_previous:
            volume_trend[i] = 1
        # Если цена падает и объем увеличивается = сильный нисходящий тренд
        elif price_change < 0 and avg_volume_current > avg_volume_previous:
            volume_trend[i] = -1
        # Если цена растет, но объем падает = слабый восходящий тренд
        elif price_change > 0 and avg_volume_current < avg_volume_previous:
            volume_trend[i] = 0.5
        # Если цена падает, но объем падает = слабый нисходящий тренд
        elif price_change < 0 and avg_volume_current < avg_volume_previous:
            volume_trend[i] = -0.5
        else:
            volume_trend[i] = 0

    return volume_trend