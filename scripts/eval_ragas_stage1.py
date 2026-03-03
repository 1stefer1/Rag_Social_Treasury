# uv run python scripts/eval_ragas_stage1.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional
import math

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets import Dataset
from openpyxl import Workbook, load_workbook
from ragas import evaluate
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._nv_metrics import ContextRelevance
from ragas.run_config import RunConfig

from src.utils.embedder import Embedder
from src.utils.generator import LLM
from src.utils.retriever import Retriever
from src.utils.rag_pipeline import VanillaRAG
from src.utils.ragas_support import (
    E5RagasEmbeddings,
    OllamaRagasLLM,
    build_retrieved_contexts,
    warmup_embedder,
)


@dataclass
class Row:
    id: str
    question: str
    gold_answer: str
    rag_answer: str
    gold_source: str
    matched_doc: str


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def _load_rows(xlsx_path: Path) -> List[Row]:
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active
    header = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
    col = {str(name).strip(): i for i, name in enumerate(header) if name is not None}
    required = [
        "id",
        "question",
        "gold_answer",
        "rag_answer",
        "gold_source",
        "matched_doc",
    ]
    for r in required:
        if r not in col:
            raise KeyError(f"Missing column '{r}' in {xlsx_path}")

    rows: List[Row] = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        row = Row(
            id=_safe_str(r[col["id"]]).strip(),
            question=_safe_str(r[col["question"]]).strip(),
            gold_answer=_safe_str(r[col["gold_answer"]]).strip(),
            rag_answer=_safe_str(r[col["rag_answer"]]).strip(),
            gold_source=_safe_str(r[col["gold_source"]]).strip(),
            matched_doc=_safe_str(r[col["matched_doc"]]).strip(),
        )
        if not row.question:
            continue
        rows.append(row)
    return rows


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def _mean(values: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in values if v is not None]
    if not xs:
        return None
    return mean(xs)


def _write_output(
    out_path: Path,
    input_rows: List[Row],
    answers: List[str],
    contexts: List[List[str]],
    scores: List[Dict[str, Any]],
) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "scored"

    metric_cols = list(scores[0].keys()) if scores else []
    header = [
        "id",
        "question",
        "gold_answer",
        "rag_answer",
        "gold_source",
        "matched_doc",
        "used_top_k",
        "context_chars",
        "retrieved_contexts",
    ] + metric_cols
    ws.append(header)

    for row, ans, ctxs, sc in zip(input_rows, answers, contexts, scores):
        used_top_k = len(ctxs)
        ctx_join = "\n\n---\n\n".join(ctxs)
        ctx_chars = len(ctx_join)
        ws.append(
            [
                row.id,
                row.question,
                row.gold_answer,
                ans,
                row.gold_source,
                row.matched_doc,
                used_top_k,
                ctx_chars,
                ctx_join,
            ]
            + [sc.get(c) for c in metric_cols]
        )

    # summary sheet
    ws2 = wb.create_sheet("summary")
    ws2.append(["metric", "mean"])
    for c in metric_cols:
        vals = [_to_float(s.get(c)) for s in scores]
        ws2.append([c, _mean(vals)])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


async def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage-1 RAGAS evaluation using local Ollama Qwen as judge (no external API)."
    )
    ap.add_argument(
        "--input-xlsx",
        default="data/gold/gold_in_scope.xlsx",
        help="Input gold xlsx with filled rag_answer (default: %(default)s)",
    )
    ap.add_argument(
        "--generate-answers",
        action="store_true",
        help="Generate rag answers via current RAG pipeline (ignores rag_answer column)",
    )
    ap.add_argument(
        "--out-xlsx",
        default="data/gold/gold_in_scope_scored_qwen.xlsx",
        help="Output xlsx with metrics (default: %(default)s)",
    )
    ap.add_argument(
        "--index-dir",
        default="data/faiss_index",
        help="FAISS index directory (default: %(default)s)",
    )
    ap.add_argument(
        "--index-name",
        default="moscow_kb",
        help="FAISS index name (default: %(default)s)",
    )
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-context-chars", type=int, default=14000)
    ap.add_argument(
        "--use-bm25", action="store_true", help="Enable hybrid retrieval (FAISS + BM25)"
    )
    ap.add_argument("--bm25-weight", type=float, default=0.25)
    ap.add_argument("--bm25-candidates-k", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only first N rows")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--ollama-model", default="qwen2.5:7b-instruct")
    args = ap.parse_args()

    in_path = Path(args.input_xlsx)
    out_path = Path(args.out_xlsx)

    rows = _load_rows(in_path)
    if not args.generate_answers:
        rows = [r for r in rows if r.rag_answer]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    if not rows:
        raise RuntimeError("No rows to evaluate (check rag_answer column)")

    embedder = Embedder()
    await warmup_embedder(embedder)

    retriever = Retriever(
        embedder,
        top_k=args.top_k,
        use_bm25=bool(args.use_bm25),
        bm25_weight=float(args.bm25_weight),
        bm25_candidates_k=int(args.bm25_candidates_k),
    )
    retriever.load(Path(args.index_dir), name=args.index_name)

    rag: VanillaRAG | None = None
    if args.generate_answers:
        llm = LLM()
        rag = VanillaRAG(
            retriever,
            llm,
            default_top_k=args.top_k,
            max_context_chars=args.max_context_chars,
        )

    # Build contexts + (optional) generate answers
    contexts: List[List[str]] = []
    answers: List[str] = []
    for r in rows:
        # Retrieve once to keep contexts and generation aligned
        chunks = await retriever.aretrieve(r.question, top_k=args.top_k)
        # Build contexts (same locator style as pipeline)
        ctxs: List[str] = []
        total = 0
        for i, c in enumerate(chunks, 1):
            from src.utils.ragas_support import format_locator

            loc = format_locator(i, c.meta or {})
            block = f"{loc}\n{c.text}".strip()
            if total + len(block) > args.max_context_chars:
                break
            ctxs.append(block)
            total += len(block)
        contexts.append(ctxs)

        if args.generate_answers:
            assert rag is not None
            prompt, used_chunks = rag._build_prompt(r.question, chunks)
            ans = await rag.llm.arun(prompt, temperature=0.0, max_tokens=700)
            answers.append((ans or "").strip())
        else:
            answers.append(r.rag_answer)

    dataset = Dataset.from_dict(
        {
            "user_input": [r.question for r in rows],
            "response": answers,
            "retrieved_contexts": contexts,
        }
    )

    judge_llm = OllamaRagasLLM(model=args.ollama_model, base_url=args.ollama_url)
    judge_emb = E5RagasEmbeddings(embedder)

    run_config = RunConfig(timeout=600, max_workers=4)

    # Use fresh metric instances (avoid global singletons being mutated between runs)
    metrics = [AnswerRelevancy(), Faithfulness(), ContextRelevance()]

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=True,
        batch_size=4,
    )

    scores = result.scores
    _write_output(out_path, rows, answers, contexts, scores)

    # Print quick summary
    print(f"Rows evaluated: {len(rows)}")
    for k in scores[0].keys():
        vals = [_to_float(s.get(k)) for s in scores]
        m = _mean(vals)
        if m is not None:
            print(f"{k}: mean={m:.4f}")
        else:
            print(f"{k}: mean=N/A")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main()))
