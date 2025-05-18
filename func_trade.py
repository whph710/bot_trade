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


def compute_indicators(candles: list, params: dict) -> dict:
    """
    Принимает данные свечного графика в формате
    [['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'], ...]
    и словарь params с параметрами индикаторов.
    Возвращает словарь: ключ — название индикатора, значение — список данных.
    """
    # Проверка на пустые данные
    if not candles or len(candles) < 2:
        return {}  # Возвращаем пустой словарь, если данных недостаточно

    # Создаем DataFrame и приводим типы
    try:
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Заменяем NaN на предыдущие значения или 0 (избегая устаревших методов)
        df.ffill(inplace=True)  # Заменено с df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)
    except Exception as e:
        print(f"Ошибка при создании DataFrame: {e}")
        return {}

    result = {}

    # EMA
    for period in params.get('EMA', []):
        if period > 0 and period < len(df):
            ema = df['close'].ewm(span=period, adjust=False).mean()
            result[f'EMA{period}'] = ema.tolist()

    # RSI
    rsi_period = params.get('RSI')
    if rsi_period and rsi_period > 0 and rsi_period < len(df):
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()

        # Безопасное деление - избегаем деления на ноль
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100.0)
        rsi = 100 - (100 / (1 + rs))
        result[f'RSI{rsi_period}'] = pd.Series(rsi).fillna(50).tolist()  # Заполняем NaN значением 50 (нейтральный RSI)

    # MACD
    macd_p = params.get('MACD', {})
    fast = macd_p.get('fast', 12)
    slow = macd_p.get('slow', 26)
    signal = macd_p.get('signal', 9)

    if fast > 0 and slow > 0 and signal > 0 and slow > fast and len(df) > slow:
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        result[f'MACD_{fast}_{slow}'] = macd_line.tolist()
        result[f'MACD_signal_{signal}'] = macd_signal.tolist()

    # Bollinger Bands
    bb_p = params.get('BBANDS', {})
    period = bb_p.get('period', 20)
    dev = bb_p.get('dev', 2)

    if period > 0 and period < len(df):
        ma = df['close'].rolling(window=period, min_periods=1).mean()
        std = df['close'].rolling(window=period, min_periods=1).std()
        upper = ma + dev * std
        lower = ma - dev * std
        result[f'BB_upper_{period}_{dev}'] = upper.tolist()
        result[f'BB_middle_{period}'] = ma.tolist()
        result[f'BB_lower_{period}_{dev}'] = lower.tolist()

    # CVD (Cumulative Volume Delta) - исправленная логика
    if params.get('CVD', False):
        # Направление свечи: вверх (1), вниз (-1), или без изменений (0)
        df['direction'] = np.where(df['close'] > df['open'], 1,
                                   np.where(df['close'] < df['open'], -1, 0))

        # CVD = сумма (объем * направление свечи)
        df['cvd_change'] = df['volume'] * df['direction']
        df['cvd'] = df['cvd_change'].cumsum()
        result['CVD'] = df['cvd'].tolist()

    # EMA для объемов
    for period in params.get('EMA_volume', []):
        if period > 0 and period < len(df):
            ema_vol = df['volume'].ewm(span=period, adjust=False).mean()
            result[f'EMA_volume_{period}'] = ema_vol.tolist()

    # OBV (On-Balance Volume) - исправленная логика
    if params.get('OBV', False):
        # Настоящий OBV должен использовать направление изменения цены закрытия
        df['price_change'] = df['close'].diff()
        df['obv_change'] = np.where(df['price_change'] > 0, df['volume'],
                                    np.where(df['price_change'] < 0, -df['volume'], 0))
        df['obv'] = df['obv_change'].cumsum()
        result['OBV'] = df['obv'].tolist()

    return result