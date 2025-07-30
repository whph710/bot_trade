import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import re
import datetime

from func_trade import (
    enhanced_signal_detection,
    format_ai_input_data,
    signal_quality_validator,
    SCALPING_15M_PARAMS,
    get_optimal_params_for_asset
)
from func_async import get_klines_async, get_usdt_trading_pairs
from deepseek import deep_seek

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def convert_to_json_serializable(obj):
    """Конвертирует объект в JSON-сериализуемый формат"""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, bool):
        return obj  # bool является подклассом int в Python, поэтому должен сериализоваться
    elif isinstance(obj, (int, float, str, type(None))):
        return obj
    else:
        return str(obj)  # Преобразуем любые другие типы в строку


@dataclass
class TradingSignal:
    """Стандартизированный торговый сигнал"""
    pair: str
    signal_type: str  # 'LONG', 'SHORT', 'NO_SIGNAL'
    confidence: int  # 0-100
    entry_price: float
    timestamp: int

    # Технический анализ
    trend_strength: int
    momentum_score: int
    volume_confirmation: bool
    volatility_regime: str

    # Контекст рынка
    market_conditions: str
    confluence_factors: List[str]
    warning_signals: List[str]

    # Метрики качества
    signal_quality: int
    filters_passed: List[str]
    filters_failed: List[str]


@dataclass
class PairAnalysisResult:
    """Результат анализа торговой пары"""
    pair: str
    signal: Optional[TradingSignal] = None
    error: Optional[str] = None
    execution_time: float = 0.0

    def is_valid_signal(self) -> bool:
        """Проверка валидности сигнала"""
        return (self.signal is not None and
                self.signal.signal_type in ['LONG', 'SHORT'] and
                self.signal.confidence >= 50)


class OptimizedTradingAnalyzer:
    """Оптимизированный анализатор торговых сигналов для скальпинга 15M"""

    def __init__(self, batch_size: int = 100, min_confidence: int = 75):
        """
        Инициализация анализатора с оптимизированными параметрами

        Args:
            batch_size: Размер батча для параллельной обработки
            min_confidence: Минимальная уверенность для сигнала
        """
        self.batch_size = batch_size
        self.min_confidence = min_confidence
        self.required_candles = 200  # Достаточно для всех расчетов

        # Кэш для оптимизации
        self._params_cache = {}
        self._last_candles = []  # Для хранения последних свечей

        logger.info(f"🚀 Анализатор инициализирован:")
        logger.info(f"   • Размер батча: {batch_size}")
        logger.info(f"   • Мин. уверенность: {min_confidence}%")
        logger.info(f"   • Параметры: {SCALPING_15M_PARAMS}")

    def _get_asset_params(self, symbol: str) -> Dict[str, Any]:
        """Получение оптимальных параметров для актива с кэшированием"""
        if symbol not in self._params_cache:
            self._params_cache[symbol] = get_optimal_params_for_asset(symbol)
        return self._params_cache[symbol]

    def _analyze_recent_moves(self, candles):
        """Анализ недавних движений цены"""
        if len(candles) < 2:
            return {'max_move': 0, 'avg_move': 0, 'strong_moves_count': 0}

        moves = []
        for i in range(1, len(candles)):
            try:
                current_price = float(candles[i][4])
                prev_price = float(candles[i - 1][4])
                move = abs(current_price - prev_price) / prev_price * 100
                moves.append(move)
            except (ValueError, ZeroDivisionError):
                continue

        if not moves:
            return {'max_move': 0, 'avg_move': 0, 'strong_moves_count': 0}

        return {
            'max_move': max(moves),
            'avg_move': sum(moves) / len(moves),
            'strong_moves_count': len([m for m in moves if m > 1.5])
        }

    def _calculate_avg_volatility(self, candles):
        """Расчет средней волатильности"""
        if len(candles) < 1:
            return 0.0

        volatilities = []
        for candle in candles:
            try:
                high = float(candle[2])
                low = float(candle[3])
                if low > 0:
                    volatility = (high - low) / low * 100
                    volatilities.append(volatility)
            except (ValueError, ZeroDivisionError):
                continue

        return sum(volatilities) / len(volatilities) if volatilities else 0.0

    def _detect_consolidation(self, candles):
        """Определение периода консолидации"""
        if len(candles) < 2:
            return 0

        small_moves = 0
        for i in range(1, len(candles)):
            try:
                current_price = float(candles[i][4])
                prev_price = float(candles[i - 1][4])
                move = abs(current_price - prev_price) / prev_price * 100
                if move < 0.5:
                    small_moves += 1
            except (ValueError, ZeroDivisionError):
                continue

        return small_moves

    def _distance_to_recent_extremes(self, candles):
        """Расстояние до недавних экстремумов"""
        if len(candles) < 1:
            return {'to_high': 0, 'to_low': 0, 'position_in_range': 50}

        try:
            current_price = float(candles[-1][4])
            recent_high = max(float(c[2]) for c in candles)
            recent_low = min(float(c[3]) for c in candles)

            if current_price == 0 or recent_high == recent_low:
                return {'to_high': 0, 'to_low': 0, 'position_in_range': 50}

            distance_to_high = (recent_high - current_price) / current_price * 100
            distance_to_low = (current_price - recent_low) / current_price * 100
            position_in_range = (current_price - recent_low) / (recent_high - recent_low) * 100

            return {
                'to_high': distance_to_high,
                'to_low': distance_to_low,
                'position_in_range': position_in_range
            }
        except (ValueError, ZeroDivisionError):
            return {'to_high': 0, 'to_low': 0, 'position_in_range': 50}

    def _analyze_spread(self, symbol):
        """Анализ спредов для скальпинга"""

        # Пока возвращаем примерные данные на основе популярности пары
        major_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'SOLUSDT']

        if symbol in major_pairs:
            return {
                'current_spread_pct': 0.05,  # Низкий спред для мажорных пар
                'avg_spread_5min': 0.06,
                'spread_stability': 'stable'
            }
        else:
            return {
                'current_spread_pct': 0.12,  # Выше спред для минорных пар
                'avg_spread_5min': 0.15,
                'spread_stability': 'volatile'
            }

    def _analyze_liquidity(self, symbol):
        """Анализ ликвидности в стакане"""

        # Пока классификация по известным парам
        high_liquidity = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        medium_liquidity = ['ADAUSDT', 'XRPUSDT', 'SOLUSDT', 'DOGEUSDT', 'DOTUSDT']

        if symbol in high_liquidity:
            return {
                'depth_5_levels': 'excellent',
                'bid_ask_imbalance': 0.02,
                'large_walls_nearby': False
            }
        elif symbol in medium_liquidity:
            return {
                'depth_5_levels': 'good',
                'bid_ask_imbalance': 0.05,
                'large_walls_nearby': False
            }
        else:
            return {
                'depth_5_levels': 'poor',
                'bid_ask_imbalance': 0.15,
                'large_walls_nearby': True
            }

    def _get_session_info(self):
        """Информация о текущей торговой сессии"""
        utc_hour = datetime.datetime.utcnow().hour

        if 0 <= utc_hour < 8:
            return {
                'session': 'asian',
                'liquidity_level': 'medium',
                'optimal_for_scalping': False
            }
        elif 8 <= utc_hour < 16:
            return {
                'session': 'european',
                'liquidity_level': 'high',
                'optimal_for_scalping': True
            }
        elif 16 <= utc_hour < 24:
            return {
                'session': 'american',
                'liquidity_level': 'high',
                'optimal_for_scalping': True
            }
        else:
            return {
                'session': 'unknown',
                'liquidity_level': 'low',
                'optimal_for_scalping': False
            }

    def _check_news_calendar(self):
        """Проверка близости важных новостей"""

        # Пока простая проверка времени (избегаем часы выхода важных новостей)
        utc_hour = datetime.datetime.utcnow().hour

        # Часы выхода важных новостей США (13:30, 15:00 UTC)
        high_risk_hours = [13, 14, 15]

        return {
            'major_news_30min': utc_hour in high_risk_hours,
            'risk_level': 'high' if utc_hour in high_risk_hours else 'low'
        }

    def _is_weekend(self):
        """Проверка выходных дней"""
        weekday = datetime.datetime.utcnow().weekday()
        return weekday >= 5  # 5=суббота, 6=воскресенье

    def _get_tick_size(self, symbol):
        """Получение минимального шага цены"""
        # Стандартные tick size для популярных пар
        tick_sizes = {
            'BTCUSDT': 0.01,
            'ETHUSDT': 0.01,
            'BNBUSDT': 0.001,
            'ADAUSDT': 0.0001,
            'XRPUSDT': 0.0001,
            'SOLUSDT': 0.001,
            'DOGEUSDT': 0.00001,
            'DOTUSDT': 0.001,
            'MATICUSDT': 0.0001,
            'LINKUSDT': 0.001
        }
        return tick_sizes.get(symbol, 0.0001)  # Дефолтный минимальный шаг

    async def analyze_single_pair(self, symbol: str) -> PairAnalysisResult:
        """
        Анализ одной торговой пары с оптимизациями

        Args:
            symbol: Символ торговой пары

        Returns:
            Результат анализа пары
        """
        start_time = time.time()

        try:
            # Получаем свечные данные
            candles = await get_klines_async(
                symbol,
                interval="15",
                limit=self.required_candles
            )

            if not candles or len(candles) < 100:
                return PairAnalysisResult(
                    pair=symbol,
                    error="INSUFFICIENT_DATA",
                    execution_time=time.time() - start_time
                )

            # Используем улучшенную систему определения сигналов
            signal_result = enhanced_signal_detection(candles)

            if signal_result['signal'] == 'NO_SIGNAL':
                return PairAnalysisResult(
                    pair=symbol,
                    error="NO_SIGNAL_DETECTED",
                    execution_time=time.time() - start_time
                )

            # Проверяем качество сигнала
            if not signal_quality_validator(signal_result, signal_result.get('market_context', {})):
                return PairAnalysisResult(
                    pair=symbol,
                    error="SIGNAL_QUALITY_TOO_LOW",
                    execution_time=time.time() - start_time
                )

            # Форматируем данные для ИИ
            ai_data = format_ai_input_data(signal_result, symbol)

            # Создаем стандартизированный сигнал
            trading_signal = self._create_trading_signal(ai_data, candles, candles)

            return PairAnalysisResult(
                pair=symbol,
                signal=trading_signal,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"❌ Ошибка анализа {symbol}: {e}")
            return PairAnalysisResult(
                pair=symbol,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def _create_trading_signal(self, ai_data: Dict[str, Any], candles: List, raw_candles: List = None) -> TradingSignal:
        """Создание стандартизированного торгового сигнала"""
        entry_signal = ai_data['entry_signal']
        technical_analysis = ai_data['technical_analysis']
        risk_assessment = ai_data['risk_assessment']

        # Сохраняем свечи для использования в _prepare_selection_data
        if raw_candles:
            self._last_candles = raw_candles

        return TradingSignal(
            pair=entry_signal['pair'],
            signal_type=entry_signal['direction'].upper(),
            confidence=entry_signal['confidence'],
            entry_price=float(candles[-1][4]),  # Текущая цена
            timestamp=entry_signal['timestamp'],

            # Технический анализ
            trend_strength=technical_analysis['trend_strength'],
            momentum_score=technical_analysis['momentum_score'],
            volume_confirmation=technical_analysis['volume_confirmation'],
            volatility_regime=technical_analysis['volatility_regime'],

            # Контекст рынка
            market_conditions=risk_assessment['market_conditions'],
            confluence_factors=risk_assessment['confluence_factors'],
            warning_signals=risk_assessment['warning_signals'],

            # Метрики качества
            signal_quality=risk_assessment['signal_quality'],
            filters_passed=[],  # Будет заполнено из signal_result
            filters_failed=[]  # Будет заполнено из signal_result
        )

    async def scan_all_markets(self) -> Dict[str, Any]:
        """
        Массовое сканирование всех рынков с оптимизацией

        Returns:
            Результат сканирования всех рынков
        """
        start_time = time.time()
        logger.info("🔍 НАЧАЛО: Массовое сканирование рынков")

        try:
            # Получаем список торговых пар
            pairs = await get_usdt_trading_pairs()
            if not pairs:
                raise Exception("Не удалось получить список торговых пар")

            logger.info(f"📊 Анализ {len(pairs)} торговых пар")

            # Параллельная обработка батчами
            all_results = await self._process_pairs_in_batches(pairs)

            # Фильтруем валидные сигналы
            valid_signals = [
                result for result in all_results
                if result.is_valid_signal()
            ]

            # Сортируем по уверенности
            valid_signals.sort(
                key=lambda x: x.signal.confidence if x.signal else 0,
                reverse=True
            )

            execution_time = time.time() - start_time

            # Статистика
            stats = self._calculate_statistics(all_results)

            logger.info(f"✅ ЗАВЕРШЕНО: Найдено {len(valid_signals)} качественных сигналов")
            logger.info(f"⏱️  Время выполнения: {execution_time:.2f}сек")
            logger.info(f"📈 Статистика: {stats}")

            return {
                'success': True,
                'valid_signals': [asdict(result) for result in valid_signals],
                'all_results': [asdict(result) for result in all_results],
                'statistics': stats,
                'execution_time': execution_time,
                'total_pairs': len(pairs)
            }

        except Exception as e:
            logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА сканирования: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }

    async def _process_pairs_in_batches(self, pairs: List[str]) -> List[PairAnalysisResult]:
        """Обработка пар батчами с контролем нагрузки"""
        all_results = []

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]

            # Создаем задачи для батча
            tasks = [self.analyze_single_pair(pair) for pair in batch]

            # Выполняем батч с обработкой исключений
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Обрабатываем результаты
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"❌ Исключение в батче: {result}")
                    # Создаем результат с ошибкой
                    all_results.append(PairAnalysisResult(
                        pair="UNKNOWN",
                        error=str(result)
                    ))
                else:
                    all_results.append(result)

            # Логируем прогресс
            processed = min(i + self.batch_size, len(pairs))
            logger.info(f"⏳ Обработано: {processed}/{len(pairs)} пар")

            # Пауза между батчами для снижения нагрузки
            if i + self.batch_size < len(pairs):
                await asyncio.sleep(0.1)

        return all_results

    def _calculate_statistics(self, results: List[PairAnalysisResult]) -> Dict[str, Any]:
        """Расчет статистики анализа"""
        total = len(results)
        valid_signals = sum(1 for r in results if r.is_valid_signal())
        errors = sum(1 for r in results if r.error is not None)

        signal_types = {'LONG': 0, 'SHORT': 0}
        confidence_levels = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}

        for result in results:
            if result.signal:
                signal_types[result.signal.signal_type] = signal_types.get(result.signal.signal_type, 0) + 1

                if result.signal.confidence >= 80:
                    confidence_levels['HIGH'] += 1
                elif result.signal.confidence >= 60:
                    confidence_levels['MEDIUM'] += 1
                else:
                    confidence_levels['LOW'] += 1

        avg_execution_time = sum(r.execution_time for r in results) / total if total > 0 else 0

        return {
            'total_pairs': total,
            'valid_signals': valid_signals,
            'errors': errors,
            'success_rate': f"{(valid_signals / total * 100):.1f}%" if total > 0 else "0%",
            'signal_types': signal_types,
            'confidence_levels': confidence_levels,
            'avg_execution_time': f"{avg_execution_time:.3f}s"
        }


class AITradingOrchestrator:
    """Оркестратор для работы с ИИ"""

    def __init__(self):
        self.selection_prompt = self._load_prompt('prompt2.txt')
        self.analysis_prompt = self._load_prompt('prompt.txt')

    def _load_prompt(self, filename: str) -> str:
        """Загрузка промпта из файла"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error(f"❌ Файл {filename} не найден")
            return ""
        except Exception as e:
            logger.error(f"❌ Ошибка чтения {filename}: {e}")
            return ""

    async def select_best_pairs(self, valid_signals: List[Dict[str, Any]]) -> List[str]:
        """
        Отбор лучших пар через ИИ

        Args:
            valid_signals: Список валидных сигналов

        Returns:
            Список выбранных пар
        """
        if not self.selection_prompt:
            logger.error("❌ Промпт для отбора не загружен")
            return []

        if not valid_signals:
            logger.info("ℹ️  Нет сигналов для отбора")
            return []

        logger.info(f"🤖 ИИ отбор из {len(valid_signals)} сигналов")

        try:
            # Подготавливаем данные для ИИ
            selection_data = self._prepare_selection_data(valid_signals)

            # Конвертируем данные в JSON-сериализуемый формат
            serializable_data = convert_to_json_serializable(selection_data)

            # Формируем запрос
            message = f"""{self.selection_prompt}

=== ДАННЫЕ ДЛЯ ОТБОРА ===
КОЛИЧЕСТВО СИГНАЛОВ: {len(valid_signals)}
ТАЙМФРЕЙМ: 15 минут (скальпинг)
СТРАТЕГИЯ: Оптимизированная EMA+TSI

{json.dumps(serializable_data, indent=2, ensure_ascii=False)}

Верни JSON в формате: {{"pairs": ["BTCUSDT", "ETHUSDT"]}} или {{"pairs": []}}
"""

            # Отправляем запрос к ИИ
            ai_response = await deep_seek(message)

            if not ai_response:
                logger.error("❌ ИИ не вернул ответ")
                return []

            # Парсим ответ
            selected_pairs = self._parse_ai_selection(ai_response)

            logger.info(f"✅ ИИ выбрал {len(selected_pairs)} пар: {selected_pairs}")
            return selected_pairs

        except Exception as e:
            logger.error(f"❌ Ошибка работы с ИИ: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _prepare_selection_data(self, valid_signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Подготовка данных для отбора ИИ"""
        selection_data = []

        for signal_data in valid_signals:
            signal = signal_data.get('signal')
            if not signal:
                continue

            # Создаем временный объект анализатора для доступа к методам
            temp_analyzer = OptimizedTradingAnalyzer()

            # Mock данные для свечей (в реальности должны передаваться из analyze_single_pair)
            mock_candles = []

            # Безопасное извлечение данных с проверкой типов
            selection_item = {
                'pair': str(signal.get('pair', '')),
                'signal_type': str(signal.get('signal_type', '')),
                'confidence': int(signal.get('confidence', 0)),
                'entry_price': float(signal.get('entry_price', 0.0)),

                # Ключевые метрики
                'trend_strength': int(signal.get('trend_strength', 0)),
                'momentum_score': int(signal.get('momentum_score', 0)),
                'volume_confirmation': bool(signal.get('volume_confirmation', False)),
                'volatility_regime': str(signal.get('volatility_regime', '')),

                # Оценка рисков
                'market_conditions': str(signal.get('market_conditions', '')),
                'confluence_factors': list(signal.get('confluence_factors', [])),
                'warning_signals': list(signal.get('warning_signals', [])),
                'signal_quality': int(signal.get('signal_quality', 0)),

                # НОВЫЕ СКАЛЬПИНГОВЫЕ МЕТРИКИ
                'recent_price_moves': temp_analyzer._analyze_recent_moves(
                    mock_candles[-20:] if len(mock_candles) >= 20 else []),
                'avg_volatility': temp_analyzer._calculate_avg_volatility(
                    mock_candles[-10:] if len(mock_candles) >= 10 else []),
                'consolidation_period': temp_analyzer._detect_consolidation(
                    mock_candles[-15:] if len(mock_candles) >= 15 else []),
                'distance_to_extremes': temp_analyzer._distance_to_recent_extremes(
                    mock_candles[-20:] if len(mock_candles) >= 20 else []),

                # Скальпинговые условия
                'spread_analysis': temp_analyzer._analyze_spread(signal.get('pair', '')),
                'liquidity_depth': temp_analyzer._analyze_liquidity(signal.get('pair', '')),
                'session_timing': temp_analyzer._get_session_info(),
                'news_proximity': temp_analyzer._check_news_calendar(),
                'weekend_check': temp_analyzer._is_weekend(),
                'tick_size': temp_analyzer._get_tick_size(signal.get('pair', ''))
            }

            selection_data.append(selection_item)

        return selection_data

    def _parse_ai_selection(self, ai_response: str) -> List[str]:
        """Парсинг ответа ИИ для отбора"""
        try:
            # Ищем JSON в ответе
            json_match = re.search(r'\{[^}]*"pairs"[^}]*\}', ai_response)
            if json_match:
                response_data = json.loads(json_match.group())
                return response_data.get('pairs', [])

            # Если не найден JSON в простом формате, ищем более сложные структуры
            json_matches = re.findall(r'\{.*?"pairs".*?\}', ai_response, re.DOTALL)
            for match in json_matches:
                try:
                    response_data = json.loads(match)
                    if 'pairs' in response_data:
                        return response_data.get('pairs', [])
                except json.JSONDecodeError:
                    continue

            return []
        except json.JSONDecodeError as e:
            logger.error(f"❌ Ошибка парсинга JSON: {e}")
            logger.error(f"Ответ ИИ: {ai_response[:500]}...")
            return []

    async def analyze_selected_pair(self, pair_name: str, signal_data: Dict[str, Any]) -> Optional[str]:
        """
        Детальный анализ выбранной пары

        Args:
            pair_name: Название пары
            signal_data: Данные сигнала

        Returns:
            Результат детального анализа или None при ошибке
        """
        if not self.analysis_prompt:
            logger.error("❌ Промпт для анализа не загружен")
            return None

        logger.info(f"🔬 Детальный анализ: {pair_name}")

        signal = signal_data.get('signal')
        if not signal:
            logger.error(f"❌ Нет данных сигнала для {pair_name}")
            return None

        try:
            # Формируем детальный запрос с безопасным извлечением данных
            message = f"""{self.analysis_prompt}

=== ДЕТАЛЬНЫЙ АНАЛИЗ ПАРЫ ===
ПАРА: {signal.get('pair', pair_name)}
СИГНАЛ: {signal.get('signal_type', 'UNKNOWN')}
УВЕРЕННОСТЬ: {signal.get('confidence', 0)}%
ЦЕНА ВХОДА: {signal.get('entry_price', 0.0)}
ВРЕМЯ: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(signal.get('timestamp', time.time())))}

=== ТЕХНИЧЕСКИЙ АНАЛИЗ ===
Сила тренда: {signal.get('trend_strength', 0)}/100
Моментум: {signal.get('momentum_score', 0)}/100
Подтверждение объемом: {'Да' if signal.get('volume_confirmation', False) else 'Нет'}
Режим волатильности: {signal.get('volatility_regime', 'Неизвестно')}

=== ОЦЕНКА РИСКОВ ===
Условия рынка: {signal.get('market_conditions', 'Неизвестно')}
Факторы подтверждения: {signal.get('confluence_factors', [])}
Предупреждения: {signal.get('warning_signals', [])}
Качество сигнала: {signal.get('signal_quality', 0)}/100

Проанализируй сигнал и дай рекомендации по торговле.
"""

            # Отправляем на анализ
            analysis_result = await deep_seek(message)

            if analysis_result:
                logger.info(f"✅ Получен анализ для {pair_name}")

                # Сохраняем результат
                self._save_analysis_result(pair_name, signal, analysis_result)

                return analysis_result
            else:
                logger.error(f"❌ Пустой ответ для {pair_name}")
                return None

        except Exception as e:
            logger.error(f"❌ Ошибка анализа {pair_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_analysis_result(self, pair_name: str, signal: Dict[str, Any], analysis: str):
        """Сохранение результата анализа в файл"""
        try:
            with open('ai_trading_analysis.log', 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"ПАРА: {pair_name}\n")
                f.write(f"ВРЕМЯ: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(
                    f"СИГНАЛ: {signal.get('signal_type', 'UNKNOWN')} | УВЕРЕННОСТЬ: {signal.get('confidence', 0)}%\n")
                f.write(f"ЦЕНА: {signal.get('entry_price', 0.0)}\n")
                f.write(f"КАЧЕСТВО: {signal.get('signal_quality', 0)}/100\n")
                f.write(f"\nАНАЛИЗ ИИ:\n{analysis}\n")
                f.write("=" * 80 + "\n")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения анализа: {e}")


async def main():
    """
    Главная функция торгового бота
    Использует двухэтапный подход:
    1. Массовое сканирование всех пар
    2. ИИ отбор и детальный анализ лучших
    """
    logger.info("🚀 ЗАПУСК ТОРГОВОГО БОТА")
    logger.info("📈 Стратегия: Оптимизированная EMA+TSI для скальпинга 15M")
    logger.info("🎯 Цель: Поиск высококачественных торговых сигналов")

    # Инициализация компонентов
    analyzer = OptimizedTradingAnalyzer(
        batch_size=100,
        min_confidence=60  # Минимальная уверенность для отбора
    )

    ai_orchestrator = AITradingOrchestrator()

    try:
        # ===== ЭТАП 1: МАССОВОЕ СКАНИРОВАНИЕ =====
        logger.info("\n🔍 ЭТАП 1: Массовое сканирование рынков")

        scan_result = await analyzer.scan_all_markets()

        if not scan_result['success']:
            logger.error(f"❌ ЭТАП 1 ПРОВАЛЕН: {scan_result.get('error')}")
            return

        valid_signals = scan_result['valid_signals']

        if not valid_signals:
            logger.info("ℹ️  Качественных сигналов не найдено")
            logger.info(f"📊 Статистика: {scan_result['statistics']}")
            return

        logger.info(f"✅ ЭТАП 1 ЗАВЕРШЕН: Найдено {len(valid_signals)} качественных сигналов")

        # ===== ЭТАП 2: ИИ ОТБОР =====
        logger.info("\n🤖 ЭТАП 2: ИИ отбор лучших сигналов")

        selected_pairs = await ai_orchestrator.select_best_pairs(valid_signals)

        if not selected_pairs:
            logger.info("ℹ️  ИИ не выбрал ни одной пары")
            return

        logger.info(f"✅ ЭТАП 2 ЗАВЕРШЕН: Выбрано {len(selected_pairs)} пар")

        # ===== ЭТАП 3: ДЕТАЛЬНЫЙ АНАЛИЗ =====
        logger.info("\n📊 ЭТАП 3: Детальный анализ выбранных пар")

        successful_analyses = 0

        for pair_name in selected_pairs:
            # Находим данные сигнала
            signal_data = None
            for signal_info in valid_signals:
                if signal_info.get('signal', {}).get('pair') == pair_name:
                    signal_data = signal_info
                    break

            if not signal_data:
                logger.error(f"❌ Данные для {pair_name} не найдены")
                continue

            # Анализируем пару
            analysis_result = await ai_orchestrator.analyze_selected_pair(pair_name, signal_data)

            if analysis_result:
                successful_analyses += 1
                logger.info(f"✅ Анализ {pair_name} завершен")
            else:
                logger.error(f"❌ Ошибка анализа {pair_name}")

            # Пауза между запросами к ИИ
            await asyncio.sleep(1)

        # ===== ИТОГИ =====
        logger.info(f"\n🎉 ВСЕ ЭТАПЫ ЗАВЕРШЕНЫ!")
        logger.info(f"📈 Проанализировано пар: {scan_result['total_pairs']}")
        logger.info(f"🎯 Найдено качественных сигналов: {len(valid_signals)}")
        logger.info(f"🤖 ИИ выбрал пар: {len(selected_pairs)}")
        logger.info(f"📊 Успешных анализов: {successful_analyses}")
        logger.info(f"📁 Результаты сохранены в: ai_trading_analysis.log")
        logger.info(f"⏱️  Общее время: {scan_result['execution_time']:.2f}сек")

    except KeyboardInterrupt:
        logger.info("⏹️  Остановка по запросу пользователя")
    except Exception as e:
        logger.error(f"💥 КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    """Точка входа в программу"""
    logger.info("=" * 80)
    logger.info("🎯 ТОРГОВЫЙ БОТ - СКАЛЬПИНГ EMA+TSI")
    logger.info("📊 Версия: 3.1 (Скальпинговая оптимизация)")
    logger.info("⏰ Таймфрейм: 15 минут")
    logger.info("🚀 Режим: Продакшн (реальные деньги)")
    logger.info("=" * 80)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Остановка программы")
    except Exception as e:
        logger.error(f"💥 ФАТАЛЬНАЯ ОШИБКА: {e}")
    finally:
        logger.info("🔚 Работа завершена")