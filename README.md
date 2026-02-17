# 🚀 RAG Project Template

Современный шаблон для проектов Retrieval-Augmented Generation (RAG) с использованием uv.

## 📦 Быстрый старт

### 1. Установка и настройка
```bash
# Создание виртуального окружения
uv venv .venv

# Активация окружения
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Установка зависимостей
uv sync
```
### 2. Запуск проекта
Если это первый запуск создайте директорию под логи 
```bash
python .\setup_directories.py
```
Затем\последующие запуски
```bash
python main.py
```
### 3. Пример curl запроса (лучше используйте postman)
```bash
curl --location 'http://localhost:8000/api/hello_world' \
--header 'Authorization: Bearer SECRET' \
--header 'Content-Type: text/plain' \
--data '{"text": "Hello world!!!"}'
```

### PS. 
Посмотрите пожалуйста src\api\handlers\hello_world.py строка 12