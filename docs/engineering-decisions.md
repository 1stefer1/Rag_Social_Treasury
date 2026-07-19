# Engineering decisions

## uv + pyproject + lockfile

Runtime, dev и evaluation зависимости разделены. `uv.lock` является единственным lockfile; PyTorch явно берётся из CPU index. Это устраняет дублирование установки в Docker и случайное скачивание CUDA wheels.

## Lazy ML runtime

FastAPI import и `/health` не должны загружать SentenceTransformers и reranker. Runtime создаётся только для readiness/search/ask и кешируется в процессе. Недоступность Qdrant или Elasticsearch отражается через `/ready`, а не ломает liveness.

## Stdout JSON logging

Контейнер пишет структурированные логи в stdout. Ротация и хранение принадлежат платформе. Raw prompts, chunks, tokens и customer payloads по умолчанию не логируются.

## Qdrant and Elasticsearch hybrid retrieval

Qdrant хранит dense-вектора и payload чанков, Elasticsearch выполняет BM25-поиск. Результаты объединяются Reciprocal Rank Fusion, потому что cosine similarity и BM25 score имеют разные шкалы. Cross-encoder получает объединённый пул после fusion. При изменении candidate limits, RRF constant или модели reranker требуется повторный offline evaluation.

## No deployment-shaped CD

CI собирает образ, но не публикует и не развёртывает его. CD должен появиться вместе с registry, immutable tags, environment approvals, secret manager, SBOM/signing и rollback policy.
