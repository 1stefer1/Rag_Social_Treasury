# Evaluation strategy

Проект содержит offline-каркас на базе RAGAS и MLflow (`scripts/eval_ragas_stage1.py`, `src/utils/eval_tracker.py`). Evaluation вынесен в optional dependency profile, чтобы production-образ не включал pandas/MLflow/RAGAS.

## Что измерять

- retrieval: Recall@k, MRR/nDCG, доля вопросов без релевантного чанка;
- generation: faithfulness, answer relevance, context precision/recall;
- safety: abstention для out-of-scope, prompt-injection cases, отсутствие утечки скрытых данных;
- operations: p50/p95 latency, input/output tokens, retry rate и cost на вопрос.

## Воспроизводимый протокол

1. Зафиксировать hash обезличенного gold dataset и corpus snapshot.
2. Зафиксировать embedding/reranker/LLM identifiers, prompt version и retrieval settings.
3. Разделить development и holdout-наборы.
4. Запустить baseline и candidate на одинаковых данных.
5. Сохранить per-example outputs, агрегаты и failure categories в MLflow.
6. Применять quality gate только после согласования допустимых порогов и статистической устойчивости.

В репозитории нет подтверждённых публично воспроизводимых значений метрик, поэтому численные claims намеренно отсутствуют. Для CI следующим шагом нужен небольшой лицензированный и обезличенный JSON fixture; реальные пользовательские вопросы и документы публиковать нельзя.
