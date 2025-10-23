# ============================================================================
# TRADING BOT SPECIFIC PARAMETERS
# ============================================================================
TIMEFRAME_SHORT = "60"   # 1H
TIMEFRAME_LONG = "240"   # 4H
TIMEFRAME_HTF = "D"      # 1D

TIMEFRAME_SHORT_NAME = "1H"
TIMEFRAME_LONG_NAME = "4H"
TIMEFRAME_HTF_NAME = "1D"

QUICK_SCAN_CANDLES = 48

# Stage 2 (Compact data для DeepSeek)
STAGE2_CANDLES_1H = 30
STAGE2_CANDLES_4H = 30
STAGE2_CANDLES_1D = 10

# Stage 3 (Full data для Sonnet)
STAGE3_CANDLES_1H = 100
STAGE3_CANDLES_4H = 60
STAGE3_CANDLES_1D = 20

AI_INDICATORS_HISTORY = 30
FINAL_INDICATORS_HISTORY = 30

# ВАЖНО: MAX_BULK_PAIRS убран - больше НЕ используется для ограничения Stage 2!
# Все пары из Stage 1 идут в Stage 2
# MAX_BULK_PAIRS = 50  # <-- УДАЛИТЬ или закомментировать эту строку

# Финальный лимит: сколько пар DeepSeek должен ВЫБРАТЬ из всех переданных
MAX_FINAL_PAIRS = 3  # DeepSeek вернет максимум 3 лучшие пары