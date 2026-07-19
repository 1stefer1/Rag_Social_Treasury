# Contributing

1. Создайте небольшую ветку и не добавляйте документы, индексы, модели, `.env` или customer data.
2. Установите окружение: `uv sync --group dev` и `uv run pre-commit install`.
3. Добавьте узкий тест для изменяемого поведения. LLM-тесты должны использовать fake/mock transport и не обращаться к платным API.
4. Запустите Ruff, mypy, unit/integration tests и Compose validation командами из README.
5. В PR опишите изменение pipeline, prompt/model/retrieval impact, latency/cost risk, тесты и rollback.

Изменения prompt, output schema, provider routing, tenant scope и side-effectful integrations требуют отдельного review и regression evaluation. Не ослабляйте validation и не логируйте raw private content.
