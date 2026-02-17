import os
from pathlib import Path


def create_project_structure():
    """Создает полную структуру папок и файлов проекта"""

    # Основные директории
    directories = ["logs"]

    # Лог-файлы
    log_files = ["logs/debug.log", "logs/logger.log", "logs/warnings.log"]

    # Создаем директории
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Создана директория: {directory}")

    # Создаем лог-файлы
    for log_file in log_files:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(log_file).touch()
        print(f"✅ Создан лог-файл: {log_file}")


if __name__ == "__main__":
    create_project_structure()
