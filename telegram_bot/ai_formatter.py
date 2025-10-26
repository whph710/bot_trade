"""
AI форматтер для сигналов - FIXED: Unified prompt loader
Файл: telegram_bot/ai_formatter.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any
from openai import AsyncOpenAI

# Добавляем родительскую директорию в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from logging_config import setup_module_logger

logger = setup_module_logger(__name__)

_prompt_cache = None


# Загружаем переменные окружения напрямую
def load_env():
    """Загрузить .env файл"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


# Загружаем переменные при импорте модуля
load_env()


def load_formatter_prompt() -> str:
    """
    Загрузить промпт для форматирования
    FIXED: Unified search strategy как в других модулях
    """
    global _prompt_cache

    if _prompt_cache:
        return _prompt_cache

    # ИСПРАВЛЕНО: Ищем в нескольких местах
    search_paths = [
        # 1. В папке telegram_bot/prompts (текущий поиск)
        Path(__file__).parent / 'prompts' / 'signal_formatter_prompt.txt',

        # 2. В корневой папке prompts
        Path(__file__).parent.parent / 'prompts' / 'signal_formatter_prompt.txt',

        # 3. В trade_bot_programm/prompts (где реально лежит)
        Path(__file__).parent.parent / 'trade_bot_programm' / 'prompts' / 'signal_formatter_prompt.txt',
    ]

    prompt_path = None
    for path in search_paths:
        if path.exists() and path.is_file():
            prompt_path = path
            logger.info(f"[Formatter] Prompt found at: {path}")
            break

    if not prompt_path:
        error_msg = f"Signal formatter prompt not found. Searched in:\n"
        for path in search_paths:
            error_msg += f"  - {path.absolute()}\n"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                raise ValueError(f"Prompt file is empty: {prompt_path}")
            _prompt_cache = content
            logger.info(f"[Formatter] Prompt cached: {prompt_path.name} ({len(content)} chars)")
            return content
    except Exception as e:
        logger.error(f"[Formatter] Error loading prompt: {e}")
        raise


class AISignalFormatter:
    """Форматирует сигналы через DeepSeek AI"""

    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        self.model = os.getenv('FORMATTER_MODEL', 'deepseek-chat')
        self.base_url = os.getenv('DEEPSEEK_URL', 'https://api.deepseek.com')
        self.temperature = float(os.getenv('FORMATTER_TEMPERATURE', '0.3'))
        self.max_tokens = int(os.getenv('FORMATTER_MAX_TOKENS', '2000'))

        logger.info(f"[Formatter] ╔{'═'*60}╗")
        logger.info(f"[Formatter] ║ {'AI FORMATTER ИНИЦИАЛИЗАЦИЯ':^60} ║")
        logger.info(f"[Formatter] ╠{'═'*60}╣")
        logger.info(f"[Formatter] ║ Провайдер: DeepSeek{'':<44} ║")
        logger.info(f"[Formatter] ║ Модель: {self.model:<49} ║")
        logger.info(f"[Formatter] ║ Temperature: {self.temperature:<46} ║")
        logger.info(f"[Formatter] ║ Max tokens: {self.max_tokens:<47} ║")
        logger.info(f"[Formatter] ╚{'═'*60}╝")

    async def format_signal(self, signal_data: Dict[str, Any]) -> str:
        """
        Форматировать один сигнал через DeepSeek AI

        Args:
            signal_data: JSON данные сигнала

        Returns:
            HTML-форматированный текст для Telegram
        """
        try:
            if not self.api_key:
                logger.error("[Formatter] DeepSeek API key not configured")
                raise ValueError("DeepSeek API key not found")

            # Загружаем промпт (теперь с unified search)
            prompt_template = load_formatter_prompt()

            # Конвертируем данные сигнала в JSON
            signal_json = json.dumps(signal_data, ensure_ascii=False, indent=2)

            # Формируем финальный промпт
            full_prompt = f"{prompt_template}\n\nSignal Data:\n{signal_json}"

            logger.debug(f"[Formatter] Formatting signal {signal_data.get('symbol', 'UNKNOWN')} via DeepSeek")

            # Вызов DeepSeek API
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                ),
                timeout=30
            )

            formatted_text = response.choices[0].message.content.strip()

            logger.info(
                f"[Formatter] ✅ Signal {signal_data.get('symbol', 'UNKNOWN')} formatted successfully ({len(formatted_text)} chars)")

            return formatted_text

        except asyncio.TimeoutError:
            logger.error("[Formatter] ❌ AI formatting timeout (30s)")
            raise
        except Exception as e:
            logger.error(f"[Formatter] ❌ AI formatting error: {e}")
            raise

    async def format_multiple_signals(self, signals: list[Dict[str, Any]]) -> list[str]:
        """
        Форматировать несколько сигналов параллельно через DeepSeek

        Args:
            signals: Список JSON данных сигналов

        Returns:
            Список HTML-форматированных текстов
        """
        if not signals:
            return []

        try:
            logger.info(f"[Formatter] {'='*60}")
            logger.info(f"[Formatter] 📝 FORMATTING {len(signals)} SIGNAL(S) VIA DEEPSEEK")
            logger.info(f"[Formatter] {'='*60}")

            tasks = [self.format_signal(signal) for signal in signals]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            formatted_signals = []
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"[Formatter] ❌ Failed to format signal {idx}: {result}")
                    continue
                formatted_signals.append(result)

            logger.info(f"[Formatter] ✅ Formatted {len(formatted_signals)}/{len(signals)} signals successfully")
            logger.info(f"[Formatter] {'='*60}")
            return formatted_signals

        except Exception as e:
            logger.error(f"[Formatter] ❌ Error in batch formatting: {e}")
            return []