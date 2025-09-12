import asyncio
import json
import logging
import time
import math
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import re
import datetime

# ИМПОРТЫ
from func_async import get_klines_async, get_usdt_trading_pairs
from deepseek import deep_seek_selection, deep_seek_analysis, cleanup_http_client

# Импорт с индикаторами по инструкции
from func_trade import detect_instruction_based_signals, calculate_indicators_by_instruction

# Импорт конфигурации
from config import config

# Настройка логирования без эмоджи
logging.basicConfig(
    level=getattr(logging, config.system.LOG_LEVEL),
    format=config.system.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.system.LOG_FILE, encoding=config.system.ENCODING),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def clean_value(value):
    """Очистка значений от NaN и Infinity"""
    if isinstance(value, (np.integer, np.floating)):
        value = float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, np.ndarray):
        return [clean_value(x) for x in value.tolist()]

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value
    elif isinstance(value, dict):
        return {k: clean_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [clean_value(item) for item in value]
    else:
        return value


def safe_json_serialize(obj: Any) -> Any:
    """Безопасная JSON сериализация"""
    return clean_value(obj)


@dataclass
class InstructionBasedSignal:
    """Сигнал согласно инструкции с мультитаймфреймным анализом"""
    pair: str
    signal_type: str  # 'LONG', 'SHORT', 'NO_SIGNAL'
    confidence: int
    entry_price: float
    timestamp: int

    # Данные по инструкции
    pattern_type: str  # 'MOMENTUM_BREAKOUT', 'PULLBACK_ENTRY', etc.
    higher_tf_trend: str  # Тренд 15m
    validation_score: str  # "5/5" чек-пунктов
    atr_current: float  # Текущий ATR для волатильности
    volume_ratio: float  # Соотношение объемов

    # Для ИИ (только краткие данные для отбора)
    candles_5m: List = None  # 5m свечи для входа
    candles_15m: List = None  # 15m свечи для контекста
    indicators_data: Dict = None


class InstructionBasedAnalyzer:
    """Анализатор согласно инструкции: 15m контекст + 5m вход"""

    def __init__(self):
        self.session_start = time.time()
        logger.info("Анализатор по инструкции запущен (15m+5m)")

    def passes_liquidity_filter(self, symbol: str, candles: List) -> bool:
        """Фильтр ликвидности согласно инструкции"""
        if not candles:
            return False

        # Примерная оценка объема (последние 24 свечи 5m = 2 часа)
        recent_volumes = [float(c[5]) * float(c[4]) for c in candles[-24:]]  # Объем в USD
        avg_hourly_volume = sum(recent_volumes) * 12  # Приблизительно за 24ч

        return avg_hourly_volume > config.trading.MIN_LIQUIDITY_VOLUME

    def check_spread_quality(self, candles: List) -> bool:
        """Проверка стабильности спреда (упрощенно через ATR)"""
        if len(candles) < 5:
            return False

        # Используем ATR как прокси для спреда
        highs = [float(c[2]) for c in candles[-5:]]
        lows = [float(c[3]) for c in candles[-5:]]
        closes = [float(c[4]) for c in candles[-5:]]

        avg_price = sum(closes) / len(closes)
        avg_range = sum(h - l for h, l in zip(highs, lows)) / len(highs)

        spread_estimate = (avg_range / avg_price) * 100
        return spread_estimate < config.trading.MAX_SPREAD_PERCENT

    async def quick_scan_pair(self, symbol: str) -> Optional[InstructionBasedSignal]:
        """Быстрое сканирование пары согласно инструкции"""
        try:
            # Получаем данные для быстрого сканирования (меньше данных)
            candles_5m = await get_klines_async(symbol, config.timeframe.ENTRY_TF,
                                                limit=config.timeframe.CANDLES_5M_QUICK)
            candles_15m = await get_klines_async(symbol, config.timeframe.CONTEXT_TF,
                                                 limit=config.timeframe.CANDLES_15M_QUICK)

            if not candles_5m or not candles_15m:
                return None

            # Фильтры согласно инструкции
            if not self.passes_liquidity_filter(symbol, candles_5m):
                return None

            if not self.check_spread_quality(candles_5m):
                return None

            # Определяем сигнал согласно инструкции (мультитаймфрейм)
            signal_result = detect_instruction_based_signals(candles_5m, candles_15m)

            if signal_result['signal'] == 'NO_SIGNAL':
                return None

            # Создаем сигнал
            entry_price = float(candles_5m[-1][4])
            confidence = int(signal_result['confidence'])

            if math.isnan(entry_price) or confidence < config.trading.MIN_CONFIDENCE:
                return None

            return InstructionBasedSignal(
                pair=symbol,
                signal_type=signal_result['signal'],
                confidence=confidence,
                entry_price=entry_price,
                timestamp=int(time.time()),

                # Данные согласно инструкции
                pattern_type=signal_result.get('pattern_type', 'UNKNOWN'),
                higher_tf_trend=signal_result.get('higher_tf_trend', 'UNKNOWN'),
                validation_score=signal_result.get('validation_score', '0/5'),
                atr_current=signal_result.get('atr_current', 0.0),
                volume_ratio=signal_result.get('volume_ratio', 1.0),

                # Данные для ИИ отбора (краткие)
                candles_5m=candles_5m[-config.timeframe.CANDLES_FOR_AI_SELECTION:],
                candles_15m=candles_15m[-config.timeframe.CANDLES_FOR_CONTEXT:],
                indicators_data=clean_value(signal_result.get('indicators', {}))
            )

        except Exception as e:
            logger.error(f"Ошибка сканирования {symbol}: {e}")
            return None

    async def mass_scan_markets(self) -> List[InstructionBasedSignal]:
        """Массовое сканирование с фильтрацией согласно инструкции"""
        start_time = time.time()
        logger.info("ЭТАП 1: Сканирование с фильтрами по инструкции")

        try:
            pairs = await get_usdt_trading_pairs()
            if not pairs:
                return []

            logger.info(f"Сканируем {len(pairs)} пар (15m контекст + 5m вход)")

            promising_signals = []

            # Обрабатываем батчами
            for i in range(0, len(pairs), config.processing.BATCH_SIZE):
                batch = pairs[i:i + config.processing.BATCH_SIZE]
                tasks = [self.quick_scan_pair(pair) for pair in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, InstructionBasedSignal):
                        promising_signals.append(result)

                processed = min(i + config.processing.BATCH_SIZE, len(pairs))
                logger.info(f"Обработано: {processed}/{len(pairs)}")

                if i + config.processing.BATCH_SIZE < len(pairs):
                    await asyncio.sleep(config.processing.BATCH_DELAY)

            # Сортируем по уверенности
            promising_signals.sort(key=lambda x: x.confidence, reverse=True)

            execution_time = time.time() - start_time
            logger.info(f"ЭТАП 1: {len(promising_signals)} сигналов за {execution_time:.2f}сек")

            return promising_signals

        except Exception as e:
            logger.error(f"Ошибка сканирования: {e}")
            return []


class InstructionBasedAISelector:
    """ИИ селектор согласно инструкции"""

    def __init__(self):
        self.selection_prompt = self._load_prompt(config.ai.SELECTION_PROMPT_FILE)
        self.analysis_prompt = self._load_prompt(config.ai.ANALYSIS_PROMPT_FILE)

    def _load_prompt(self, filename: str) -> str:
        """Загрузка промпта"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error(f"Файл {filename} не найден")
            return ""

    def _prepare_signals_for_ai(self, signals: List[InstructionBasedSignal]) -> Dict[str, Any]:
        """Подготовка данных для ИИ отбора (краткие данные)"""
        prepared_data = []

        for signal in signals:
            # Свечи 5m для анализа входа (краткие)
            recent_5m = signal.candles_5m[-30:] if signal.candles_5m else []
            # Свечи 15m для контекста (краткие)
            recent_15m = signal.candles_15m[-20:] if signal.candles_15m else []

            signal_data = {
                'pair': signal.pair,
                'signal_type': signal.signal_type,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,

                # Данные согласно инструкции
                'pattern_type': signal.pattern_type,
                'higher_tf_trend': signal.higher_tf_trend,
                'validation_score': signal.validation_score,
                'atr_current': signal.atr_current,
                'volume_ratio': signal.volume_ratio,

                # Краткие мультитаймфреймные данные для отбора
                'timeframes': {
                    '5m_candles': [
                        {
                            'timestamp': int(c[0]),
                            'open': float(c[1]),
                            'high': float(c[2]),
                            'low': float(c[3]),
                            'close': float(c[4]),
                            'volume': float(c[5])
                        } for c in recent_5m
                    ],
                    '15m_context': [
                        {
                            'timestamp': int(c[0]),
                            'open': float(c[1]),
                            'high': float(c[2]),
                            'low': float(c[3]),
                            'close': float(c[4]),
                            'volume': float(c[5])
                        } for c in recent_15m
                    ]
                },

                # Краткие индикаторы для отбора
                'technical_indicators': safe_json_serialize({
                    'ema_system': {
                        'ema5_current': signal.indicators_data.get('ema5', [])[-1] if signal.indicators_data.get('ema5') else 0,
                        'ema8_current': signal.indicators_data.get('ema8', [])[-1] if signal.indicators_data.get