"""
Утилиты для работы с файлами и результатами бота
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from logging_config import setup_module_logger

logger = setup_module_logger(__name__)


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
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = result.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
        filename = f"bot_result_{timestamp}.json"
        file_path = output_path / filename

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return str(file_path.absolute())

    except Exception as e:
        logger.error(f"Failed to save result: {e}")
        return ""


def print_bot_result(result: Dict[str, Any]) -> None:
    """
    Красиво вывести результат работы бота в консоль

    Args:
        result: Словарь с результатами работы бота
    """
    print("\n" + "=" * 80)
    print(f"{'RESULT: ' + result['result']:^80}")
    print("=" * 80)

    # Время выполнения
    total_time = result.get('total_time', 0)
    print(f"\n⏱️  Total time: {total_time:.1f}s")

    # Статистика
    stats = result.get('stats', {})
    print(f"\n📊 STATISTICS:")
    print(f"  • Pairs scanned:       {stats.get('pairs_scanned', 0)}")
    print(f"  • Signal pairs found:  {stats.get('signal_pairs_found', 0)}")
    print(f"  • AI selected:         {stats.get('ai_selected', 0)}")
    print(f"  • Analyzed:            {stats.get('analyzed', 0)}")
    print(f"  • Validated signals:   {stats.get('validated_signals', 0)}")
    print(f"  • Rejected signals:    {stats.get('rejected_signals', 0)}")
    print(f"  • Processing speed:    {stats.get('processing_speed', 0):.1f} pairs/sec")

    # Если валидация пропущена
    if result.get('result') == 'VALIDATION_SKIPPED':
        print(f"\n⏰ Validation skipped: {result.get('reason', 'Unknown reason')}")
        print("=" * 80 + "\n")
        return

    # Одобренные сигналы
    validated = result.get('validated_signals', [])
    if validated:
        print(f"\n✅ VALIDATED SIGNALS ({len(validated)}):")
        print("-" * 80)
        for sig in validated:
            tp_levels = sig.get('take_profit_levels', [0, 0, 0])
            print(f"\n  {sig['symbol']:>10} | {sig['signal']:<5} | Confidence: {sig['confidence']}%")
            print(f"  {'':>10} | Entry:  ${sig['entry_price']:.2f}")
            print(f"  {'':>10} | Stop:   ${sig['stop_loss']:.2f}")
            print(f"  {'':>10} | TP 1/2/3: ${tp_levels[0]:.2f} / ${tp_levels[1]:.2f} / ${tp_levels[2]:.2f}")
            print(f"  {'':>10} | R/R:    1:{sig.get('risk_reward_ratio', 0):.2f}")
            print(f"  {'':>10} | Hold:   {sig.get('hold_duration_minutes', 0)} min")

    # Отклонённые сигналы (первые 5)
    rejected = result.get('rejected_signals', [])
    if rejected:
        print(f"\n❌ REJECTED SIGNALS ({len(rejected)}):")
        print("-" * 80)
        for rej in rejected[:5]:
            reason = rej.get('rejection_reason', 'Unknown')
            print(f"  {rej['symbol']:>10} | {reason}")

        if len(rejected) > 5:
            print(f"  {'...':>10} | and {len(rejected) - 5} more")

    # Статистика валидации
    vstats = result.get('validation_stats', {})
    if vstats and vstats.get('total', 0) > 0:
        print(f"\n📈 VALIDATION STATS:")
        print(f"  • Approval rate:      {vstats.get('approval_rate', 0):.1f}%")
        print(f"  • Avg R/R:            1:{vstats.get('avg_risk_reward', 0):.2f}")

        rr_stats = vstats.get('rr_stats', {})
        if rr_stats.get('samples_counted', 0) > 0:
            print(f"  • R/R range:          1:{rr_stats.get('min_rr', 0):.2f} - 1:{rr_stats.get('max_rr', 0):.2f}")

        top_reasons = vstats.get('top_rejection_reasons', [])
        if top_reasons:
            print(f"  • Top rejection:")
            for reason in top_reasons[:2]:
                print(f"    - {reason}")

    print("\n" + "=" * 80 + "\n")


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
            logger.warning(f"Results directory not found: {results_dir}")
            return None

        json_files = list(results_path.glob("bot_result_*.json"))

        if not json_files:
            logger.warning("No result files found")
            return None

        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)

        with open(latest_file, 'r', encoding='utf-8') as f:
            result = json.load(f)

        logger.info(f"Latest result loaded: {latest_file.name}")
        return result

    except Exception as e:
        logger.error(f"Failed to load latest result: {e}")
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

        json_files = list(results_path.glob("bot_result_*.json"))

        if len(json_files) <= keep_last:
            return 0

        json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        files_to_delete = json_files[keep_last:]
        deleted_count = 0

        for file_path in files_to_delete:
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {file_path.name}: {e}")

        if deleted_count > 0:
            logger.info(f"Cleanup: Removed {deleted_count} old result file(s)")

        return deleted_count

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return 0