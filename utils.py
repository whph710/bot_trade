"""
Утилиты для работы с файлами и результатами бота
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


def save_bot_result(result: Dict[str, Any], output_dir: str = "bot_results") -> str:
    """
    Сохранить результат работы бота в JSON файл

    Args:
        result: Словарь с результатами работы бота
        output_dir: Директория для сохранения (относительно текущей)

    Returns:
        Полный путь к сохранённому файлу
    """
    try:
        # Создать директорию если не существует
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Генерация имени файла
        timestamp = result.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
        filename = f"bot_result_{timestamp}.json"

        # Полный путь к файлу
        file_path = output_path / filename

        # Сохранить JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Result saved: {file_path}")
        return str(file_path.absolute())

    except Exception as e:
        logger.error(f"❌ Failed to save result: {e}")
        return ""


def print_bot_result(result: Dict[str, Any]) -> None:
    """
    Красиво вывести результат работы бота в консоль

    Args:
        result: Словарь с результатами работы бота
    """
    print("\n" + "=" * 60)
    print(f"RESULT: {result['result']}")
    print("=" * 60)

    # Время выполнения
    total_time = result.get('total_time', 0)
    print(f"⏱️  Time: {total_time:.1f}s")

    # Статистика
    stats = result.get('stats', {})
    print(f"\n📊 STATISTICS:")
    print(f"  Pairs scanned: {stats.get('pairs_scanned', 0)}")
    print(f"  Signal pairs found: {stats.get('signal_pairs_found', 0)}")
    print(f"  AI selected: {stats.get('ai_selected', 0)}")
    print(f"  Analyzed: {stats.get('analyzed', 0)}")
    print(f"  Validated: {stats.get('validated_signals', 0)}")
    print(f"  Rejected: {stats.get('rejected_signals', 0)}")
    print(f"  Speed: {stats.get('processing_speed', 0):.1f} pairs/sec")

    # Если валидация пропущена
    if result.get('result') == 'VALIDATION_SKIPPED':
        print(f"\n⏰ {result.get('reason', 'Validation skipped')}")
        return

    # Одобренные сигналы
    validated = result.get('validated_signals', [])
    if validated:
        print(f"\n✅ VALIDATED SIGNALS ({len(validated)}):")
        for sig in validated:
            tp_levels = sig.get('take_profit_levels', [0, 0, 0])
            print(f"\n  {sig['symbol']}: {sig['signal']} (Confidence: {sig['confidence']}%)")
            print(f"    Entry: ${sig['entry_price']:.2f}")
            print(f"    Stop:  ${sig['stop_loss']:.2f}")
            print(f"    TP1:   ${tp_levels[0]:.2f}")
            print(f"    TP2:   ${tp_levels[1]:.2f}")
            print(f"    TP3:   ${tp_levels[2]:.2f}")
            print(f"    R/R:   1:{sig.get('risk_reward_ratio', 0):.1f}")

    # Отклонённые сигналы (первые 3)
    rejected = result.get('rejected_signals', [])
    if rejected:
        print(f"\n❌ REJECTED SIGNALS ({len(rejected)}):")
        for rej in rejected[:3]:
            reason = rej.get('rejection_reason', 'Unknown')
            print(f"  {rej['symbol']}: {reason}")

        if len(rejected) > 3:
            print(f"  ... and {len(rejected) - 3} more")

    # Статистика валидации
    vstats = result.get('validation_stats', {})
    if vstats:
        print(f"\n📈 VALIDATION STATS:")
        print(f"  Approval rate: {vstats.get('approval_rate', 0)}%")
        print(f"  Avg R/R: 1:{vstats.get('avg_risk_reward', 0):.1f}")

        top_reasons = vstats.get('top_rejection_reasons', [])
        if top_reasons:
            print(f"  Top rejection reasons: {', '.join(top_reasons[:2])}")

    print()


def get_latest_result(results_dir: str = "bot_results") -> Dict[str, Any] | None:
    """
    Получить последний сохранённый результат

    Args:
        results_dir: Директория с результатами

    Returns:
        Словарь с последним результатом или None
    """
    try:
        results_path = Path(results_dir)

        if not results_path.exists():
            return None

        # Найти все JSON файлы
        json_files = list(results_path.glob("bot_result_*.json"))

        if not json_files:
            return None

        # Отсортировать по времени модификации (последний)
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)

        # Прочитать JSON
        with open(latest_file, 'r', encoding='utf-8') as f:
            result = json.load(f)

        logger.info(f"📁 Loaded latest result: {latest_file.name}")
        return result

    except Exception as e:
        logger.error(f"❌ Failed to load latest result: {e}")
        return None


def cleanup_old_results(results_dir: str = "bot_results", keep_last: int = 10) -> int:
    """
    Удалить старые результаты, оставив только последние N

    Args:
        results_dir: Директория с результатами
        keep_last: Сколько последних файлов оставить

    Returns:
        Количество удалённых файлов
    """
    try:
        results_path = Path(results_dir)

        if not results_path.exists():
            return 0

        # Найти все JSON файлы
        json_files = list(results_path.glob("bot_result_*.json"))

        if len(json_files) <= keep_last:
            return 0

        # Отсортировать по времени модификации
        json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Удалить старые файлы
        files_to_delete = json_files[keep_last:]
        deleted_count = 0

        for file_path in files_to_delete:
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {file_path.name}: {e}")

        if deleted_count > 0:
            logger.info(f"🗑️  Cleaned up {deleted_count} old result files")

        return deleted_count

    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return 0