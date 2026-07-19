# Engineering decisions

## uv + pyproject + lockfile

Runtime, dev и evaluation зависимости разделены. `uv.lock` является единственным lockfile; PyTorch явно берётся из CPU index. Это устраняет дублирование установки в Docker и случайное скачивание CUDA wheels.

## Lazy ML runtime

FastAPI import и `/health` не должны загружать SentenceTransformers, reranker и FAISS. Runtime создаётся только для readiness/search/ask и кешируется в процессе. Ошибка индекса отражается через `/ready`, а не ломает liveness.

## Stdout JSON logging

Контейнер пишет структурированные логи в stdout. Ротация и хранение принадлежат платформе. Raw prompts, chunks, tokens и customer payloads по умолчанию не логируются.

## Local FAISS plus optional Elasticsearch

FAISS сохраняет простой и быстрый semantic baseline. Elasticsearch остаётся отдельным модулем: объединение retrieval policies без evaluation могло бы молча изменить качество. Для production multi-tenant deployment требуется namespace/filter invariant и отдельные isolation tests.

## No deployment-shaped CD

CI собирает образ, но не публикует и не развёртывает его. CD должен появиться вместе с registry, immutable tags, environment approvals, secret manager, SBOM/signing и rollback policy.
