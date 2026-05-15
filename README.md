# RAG соцказначейство 

Проект реализует **Vanilla Retrieval-Augmented Generation (RAG)** для интеллектуального поиска по нормативно-правовым документам  
(постановления, законы, приказы) с использованием:
- FAISS (векторный поиск)
- SentenceTransformers (эмбеддинги)
- локальной LLM **qwen2.5:7b-instruct** через **Ollama**
- Telegram-бота как пользовательского интерфейса
- Gradio Web UI для поиска по векторной БД и RAG-запросов

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

### 5. Gradio Web UI
```bash
uv run python -m src.integrations.gradio.app
```

По умолчанию интерфейс поднимется на `http://127.0.0.1:7860`.
Во вкладках доступны:
- `Search` — поиск фрагментов по векторной базе знаний
- `Ask` — RAG-ответ с указанием источников
- `Documents` — заготовка под будущую drag-and-drop загрузку документов

---

## ⚙️ Требования
* Python 3.12+
* uv
* Ollama (~15 gb)
* Telegram Bot Token

## 🔧 Конфигурация

## RAG API для поставки заказчику

Сервис можно использовать как самостоятельный RAG backend. Основной API:
- `GET /api/v1/health` — процесс жив
- `GET /api/v1/ready` — индекс загружен
- `GET /api/v1/config` — текущий runtime config
- `POST /api/v1/search` — поиск по векторной базе
- `POST /api/v1/ask` — RAG-ответ с источниками

Для подключения LLM заказчика используйте OpenAI-compatible endpoint, например vLLM:
```env
LLM_PROVIDER=openai_compatible
OPENAI_BASE_URL=http://customer-llm:8000/v1
OPENAI_API_KEY=dummy
OPENAI_MODEL=customer-model
```

Пример запроса к API:
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change-me" \
  -d '{"query":"единовременная денежная выплата","top_k":5}'
```

## Docker Compose

1) Создайте `.env` из примера:
```bash
cp .env.example .env
```

2) Настройте `.env`:
- `API_SECRET` — Bearer token для API
- `OPENAI_BASE_URL`, `OPENAI_MODEL` — LLM заказчика
- `INDEX_NAME`, `TOP_K`, `USE_RERANKER` — параметры RAG

3) Запуск RAG API + Gradio UI:
```bash
docker compose up --build
```

4) Локальное демо с Ollama:
```bash
docker compose --profile ollama up --build
```

При Ollama-демо в `.env` укажите:
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
```

5) Опциональный Telegram bot:
```bash
docker compose --profile telegram up --build telegram-bot
```
