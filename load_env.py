"""
Загрузчик переменных окружения из .env файла
"""

import os

def load_env():
    """Загрузка переменных из .env файла"""
    try:
        with open('.env', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("Переменные окружения загружены из .env")
        return True
    except FileNotFoundError:
        print("Файл .env не найден")
        return False
    except Exception as e:
        print(f"Ошибка загрузки .env: {e}")
        return False

# Загружаем при импорте
load_env()