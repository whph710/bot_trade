"""
Checkpoint manager для recovery после сбоев
Файл: trade_bot_programm/checkpoint_manager.py
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from logging_config import setup_module_logger

logger = setup_module_logger(__name__)


class CheckpointManager:
    """Менеджер чекпоинтов для восстановления выполнения"""

    def __init__(self, checkpoint_dir: str = "bot_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.current_checkpoint_file = None

    def start_checkpoint(self, cycle_id: str) -> str:
        """
        Начинает новый чекпоинт для цикла

        Args:
            cycle_id: ID цикла

        Returns:
            Путь к файлу чекпоинта
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_checkpoint_file = self.checkpoint_dir / f"checkpoint_{timestamp}_{cycle_id}.json"

        checkpoint_data = {
            'cycle_id': cycle_id,
            'started_at': datetime.now().isoformat(),
            'stage': 0,
            'stage1_complete': False,
            'stage2_complete': False,
            'stage3_complete': False,
            'stage4_complete': False,
            'data': {}
        }

        self._save_checkpoint(checkpoint_data)
        logger.info(f"Checkpoint started: {self.current_checkpoint_file.name}")
        return str(self.current_checkpoint_file)

    def save_stage(self, stage: int, data: Dict):
        """
        Сохраняет завершение стадии

        Args:
            stage: Номер стадии (1-4)
            data: Данные стадии
        """
        if not self.current_checkpoint_file or not self.current_checkpoint_file.exists():
            logger.warning("No active checkpoint file")
            return

        checkpoint_data = self._load_checkpoint()

        checkpoint_data['stage'] = stage
        checkpoint_data[f'stage{stage}_complete'] = True
        checkpoint_data['data'][f'stage{stage}'] = data
        checkpoint_data['last_updated'] = datetime.now().isoformat()

        self._save_checkpoint(checkpoint_data)
        logger.debug(f"Stage {stage} saved to checkpoint")

    def get_last_checkpoint(self) -> Optional[Dict]:
        """
        Получает последний чекпоинт если существует

        Returns:
            Данные чекпоинта или None
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.json"))

        if not checkpoints:
            return None

        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)

        try:
            checkpoint_data = self._load_checkpoint(latest_checkpoint)

            if checkpoint_data.get('stage4_complete', False):
                logger.info("Last checkpoint is complete, no recovery needed")
                return None

            logger.info(f"Found incomplete checkpoint: {latest_checkpoint.name}")
            logger.info(f"  Last stage: {checkpoint_data.get('stage', 0)}")

            return checkpoint_data

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None

    def clear_checkpoint(self):
        """Очищает текущий чекпоинт после успешного завершения"""
        if self.current_checkpoint_file and self.current_checkpoint_file.exists():
            try:
                self.current_checkpoint_file.unlink()
                logger.info("Checkpoint cleared (cycle completed)")
            except Exception as e:
                logger.warning(f"Failed to clear checkpoint: {e}")

    def _save_checkpoint(self, data: Dict):
        """Сохраняет данные чекпоинта"""
        if not self.current_checkpoint_file:
            return

        try:
            with open(self.current_checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self, filepath: Path = None) -> Dict:
        """Загружает данные чекпоинта"""
        if filepath is None:
            filepath = self.current_checkpoint_file

        if not filepath or not filepath.exists():
            return {}

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return {}