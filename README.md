# RAG соцказначейство 

Проект реализует **Vanilla Retrieval-Augmented Generation (RAG)** для интеллектуального поиска по нормативно-правовым документам  
(постановления, законы, приказы) с использованием:
- FAISS (векторный поиск)
- SentenceTransformers (эмбеддинги)
- локальной LLM **qwen2.5:7b-instruct** через **Ollama**
- Telegram-бота как пользовательского интерфейса

---

## 📁 Структура проекта
```text
rag_project/
├── data/
│   ├── raw_docx/          # Исходные .docx документы
│   ├── chunks_json/       # Распарсенные чанки (JSON)
│   └── faiss_index/       # Сохранённый FAISS индекс
│
├── src/
│   ├── utils/
│   │   ├── process.py         # Парсинг DOCX → чанки
│   │   ├── embedder.py        # Эмбеддер (multilingual-e5)
│   │   ├── vector_store.py    # FAISS-хранилище
│   │   ├── retriever.py       # Retriever
│   │   ├── generator.py       # LLM (Qwen через Ollama)
│   │   └── rag_pipeline.py    # Vanilla RAG пайплайн
│   │
│   └── integrations/
│       └── telegram/
│           └── bot.py         # Telegram-бот
│
├── scripts/               # Тестовые и отладочные скрипты
├── setup_directories.py   # Создание директорий (логи и т.п.)
├── pyproject.toml
└── README.md
```
---

## 🔁 Общая схема работы

1. `.docx` документы помещаются в `data/raw_docx/`
2. Парсер извлекает главы / статьи / пункты → JSON-чанки
3. Чанки преобразуются в эмбеддинги и сохраняются в FAISS
4. Пользователь задаёт вопрос через Telegram
5. RAG:
   - ищет релевантные чанки
   - собирает контекст
   - отправляет запрос в Qwen
6. Ответ + источники возвращаются пользователю

---

## 📦 Быстрый старт

### 1. Установка и настройка окружения
```bash
# Клонировать проект
git clone <repo_url>
cd rag_project

# Создать виртуальное окружение
uv venv .venv

# Активировать окружение
# Windows:
.venv\Scripts\activate
# Linux / Mac:
source .venv/bin/activate

# Установить зависимости
uv sync
```
### 2. Запуск проекта
Если это первый запуск создайте директорию под логи 
```bash
python .\setup_directories.py
```
### 3. Установка qwen локально
```bash
# 1. Установить Ollama с официального сайта:
# https://ollama.com

# 2. Скачать модель Qwen:
ollama pull qwen2.5:7b-instruct

# 3. Проверить работу модели:
ollama run qwen2.5:7b-instruct
```
### 4. Telegram-бот
```bash
1. В Telegram открой @BotFather
2. /newbot
3. Скопируй токен
4. в терминал: setx TELEGRAM_BOT_TOKEN "ВАШ_ТОКЕН"
(проверка (в терминал): echo $env:TELEGRAM_BOT_TOKEN)
5. Запуска uv run python -m src.integrations.telegram.bot
```

---

## ⚙️ Требования
* Python 3.12+
* uv
* Ollama (~15 gb)
* Telegram Bot Token

## 🔧 Конфигурация
