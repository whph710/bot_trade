import pandas as pd

def calculate_atr(candles: list) -> float:
    tr_values = []
    for i in range(1, len(candles)):
        high = float(candles[i][2])
        low = float(candles[i][3])
        prev_close = float(candles[i - 1][4])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)

    atr = sum(tr_values) / len(tr_values)
    return atr




def compute_indicators(candles: list, params: dict) -> dict:
    """
    Принимает данные свечного графика в формате
    [['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'], ...]
    и словарь params с параметрами индикаторов.
    Возвращает словарь: ключ — название индикатора, значение — список данных.
    """
    # Создаем DataFrame и приводим типы
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)

    result = {}

    # EMA
    for period in params.get('EMA', []):
        ema = df['close'].ewm(span=period, adjust=False).mean()
        result[f'EMA{period}'] = ema.tolist()

    # RSI
    rsi_period = params.get('RSI')
    if rsi_period:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        result[f'RSI{rsi_period}'] = rsi.tolist()

    # MACD
    macd_p = params.get('MACD', {})
    fast = macd_p.get('fast', 12)
    slow = macd_p.get('slow', 26)
    signal = macd_p.get('signal', 9)
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
    ma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper = ma + dev * std
    lower = ma - dev * std
    result[f'BB_upper_{period}_{dev}'] = upper.tolist()
    result[f'BB_middle_{period}'] = ma.tolist()
    result[f'BB_lower_{period}_{dev}'] = lower.tolist()

    # CVD (Cumulative Volume Delta)
    if params.get('CVD', False):
        df['delta'] = df['close'].diff()
        df['cvd'] = (df['delta'] > 0).astype(int) * df['volume'] - (df['delta'] < 0).astype(int) * df['volume']
        df['cvd'] = df['cvd'].cumsum()
        result['CVD'] = df['cvd'].tolist()

    # EMA для объемов
    for period in params.get('EMA_volume', []):
        ema_vol = df['volume'].ewm(span=period, adjust=False).mean()
        result[f'EMA_volume_{period}'] = ema_vol.tolist()

    # OBV (On-Balance Volume)
    if params.get('OBV', False):
        df['obv'] = (df['close'].diff() > 0).astype(int) * df['volume'] - (df['close'].diff() < 0).astype(int) * df['volume']
        df['obv'] = df['obv'].cumsum()
        result['OBV'] = df['obv'].tolist()

    return result