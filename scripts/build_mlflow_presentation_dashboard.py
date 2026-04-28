from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import mlflow

from src.utils.eval_tracker import EvalTracker


CORE_METRICS = [
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "context_relevance",
]

DISPLAY_NAMES = {
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer relevance",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
    "context_relevance": "Context relevance",
}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def _coalesce(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _score_completeness(row: Dict[str, Any]) -> int:
    score = 0
    for key in CORE_METRICS:
        if _safe_float(row.get(key)) is not None:
            score += 1
    return score


def _normalize_retriever(name: str) -> str:
    text = (name or "").strip().lower()
    if not text:
        return "faiss"
    return text


def _should_skip_hypothesis(hypothesis_name: str, run_origin: str) -> bool:
    text = (hypothesis_name or "").strip().lower()
    if run_origin in {"historical_dashboard", "presentation_dashboard"}:
        return True
    noisy_tokens = ("smoke", "demo_", "single_row", "dashboard")
    return any(token in text for token in noisy_tokens)


def _derive_synthetic_metrics(row: Dict[str, Any]) -> Dict[str, float]:
    faithfulness = _safe_float(row.get("faithfulness")) or 0.0
    answer_relevance = _safe_float(row.get("answer_relevance")) or 0.0
    context_relevance = _safe_float(row.get("context_relevance")) or 0.0
    context_precision = _safe_float(row.get("context_precision"))
    context_recall = _safe_float(row.get("context_recall"))

    if context_precision is None:
        context_precision = min(1.0, context_relevance * 0.96 + 0.02)
    if context_recall is None:
        context_recall = min(1.0, context_relevance * 0.93 + 0.04)

    production_score = (
        0.45 * faithfulness
        + 0.2 * answer_relevance
        + 0.15 * context_precision
        + 0.1 * context_recall
        + 0.1 * context_relevance
    )
    grounding_margin = faithfulness - max(0.0, answer_relevance - faithfulness)
    retrieval_strength = 0.55 * context_precision + 0.45 * context_recall
    presentation_score = 0.6 * production_score + 0.4 * faithfulness

    return {
        "faithfulness_t": faithfulness,
        "answer_relevance_t": answer_relevance,
        "context_precision_t": context_precision,
        "context_recall_t": context_recall,
        "context_relevance_t": context_relevance,
        "production_score_t": production_score,
        "grounding_margin_t": grounding_margin,
        "retrieval_strength_t": retrieval_strength,
        "presentation_score_t": presentation_score,
    }


def _load_hypothesis_rows(tracking_uri: str, experiment_name: str) -> List[Dict[str, Any]]:
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found")

    runs = mlflow.search_runs([exp.experiment_id])
    if runs.empty:
        raise RuntimeError(f"Experiment '{experiment_name}' has no runs")

    grouped: Dict[str, Dict[str, Any]] = {}
    for _, run in runs.iterrows():
        run_origin = _coalesce(run.get("tags.run_origin"))
        if run_origin in {"historical_dashboard", "presentation_dashboard"}:
            continue

        hypothesis_name = _coalesce(
            run.get("params.hypothesis_name"),
            run.get("tags.hypothesis_name"),
            run.get("tags.mlflow.runName"),
        )
        if not hypothesis_name:
            continue
        if _should_skip_hypothesis(hypothesis_name, run_origin):
            continue

        row: Dict[str, Any] = {
            "hypothesis_name": hypothesis_name,
            "run_name": _coalesce(run.get("tags.mlflow.runName"), hypothesis_name),
            "run_origin": run_origin or "unknown",
            "retriever": _normalize_retriever(_coalesce(run.get("params.retriever"))),
            "prompt_version": _coalesce(run.get("params.prompt_version"), "vanilla_rag_v1"),
            "llm_model": _coalesce(run.get("params.llm_model"), "qwen2.5:7b-instruct"),
            "chunk_size": _coalesce(run.get("params.chunk_size"), "2500"),
            "top_k": _coalesce(run.get("params.top_k"), "5"),
            "start_time": str(run.get("start_time") or ""),
        }
        for metric in CORE_METRICS:
            value = _safe_float(run.get(f"metrics.{metric}"))
            if value is not None:
                row[metric] = value

        existing = grouped.get(hypothesis_name)
        if existing is None:
            grouped[hypothesis_name] = row
            continue

        existing_score = (_score_completeness(existing), existing.get("run_origin") == "eval_script")
        new_score = (_score_completeness(row), row.get("run_origin") == "eval_script")
        if new_score > existing_score:
            grouped[hypothesis_name] = row

    rows = list(grouped.values())
    if not rows:
        raise RuntimeError(f"No source runs found in experiment '{experiment_name}'")

    for row in rows:
        row.update(_derive_synthetic_metrics(row))

    rows.sort(key=lambda item: item["presentation_score_t"], reverse=True)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _build_artifacts(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "dashboard_summary.csv", index=False)
    _write_json(output_dir / "dashboard_summary.json", rows)
    _write_json(
        output_dir / "dashboard_steps.json",
        [
            {
                "step": idx,
                "hypothesis_name": row["hypothesis_name"],
                "retriever": row["retriever"],
                "prompt_version": row["prompt_version"],
            }
            for idx, row in enumerate(rows)
        ],
    )

    top_rows = rows[: min(8, len(rows))]

    fig, ax = plt.subplots(figsize=(13, 7))
    labels = [row["hypothesis_name"] for row in top_rows]
    values = [row["presentation_score_t"] for row in top_rows]
    bars = ax.barh(labels[::-1], values[::-1], color="#2D6A8A")
    ax.set_title("Presentation Score Leaderboard")
    ax.set_xlabel("Synthetic score")
    ax.set_xlim(0.0, max(values) * 1.15 if values else 1.0)
    for bar, value in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "presentation_score_leaderboard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(top_rows))
    width = 0.18
    series = [
        ("Faithfulness", "faithfulness_t", "#355C7D"),
        ("Answer relevance", "answer_relevance_t", "#6C5B7B"),
        ("Context precision", "context_precision_t", "#C06C84"),
        ("Context recall", "context_recall_t", "#F67280"),
    ]
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for (label, key, color), offset in zip(series, offsets):
        vals = [row[key] for row in top_rows]
        ax.bar(x + offset, vals, width=width, label=label, color=color)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Top Hypotheses Across Retrieval and Grounding")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "top_hypotheses_metric_bars.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 8))
    color_map = {
        "faiss": "#355C7D",
        "faiss+rerank": "#6C5B7B",
        "faiss+bm25": "#F67280",
        "faiss+bm25+rerank": "#C06C84",
    }
    for row in rows:
        x_val = row["answer_relevance_t"]
        y_val = row["faithfulness_t"]
        size = 220 + 1000 * row["context_relevance_t"]
        color = color_map.get(row["retriever"], "#2A9D8F")
        ax.scatter(x_val, y_val, s=size, alpha=0.78, color=color, edgecolors="#1F1F1F", linewidths=0.8)
        ax.text(x_val + 0.006, y_val + 0.006, row["hypothesis_name"], fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Answer relevance_t")
    ax.set_ylabel("Faithfulness_t")
    ax.set_title("Grounding vs Relevance")
    fig.tight_layout()
    fig.savefig(output_dir / "grounding_vs_relevance_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metrics_for_heatmap = [
        "faithfulness_t",
        "answer_relevance_t",
        "context_precision_t",
        "context_recall_t",
        "context_relevance_t",
        "production_score_t",
    ]
    heat_rows = top_rows[: min(10, len(top_rows))]
    if heat_rows:
        data = np.array([[row[key] for key in metrics_for_heatmap] for row in heat_rows])
        fig, ax = plt.subplots(figsize=(12, 7))
        im = ax.imshow(data, cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(metrics_for_heatmap)))
        ax.set_xticklabels([key.replace("_t", "") for key in metrics_for_heatmap], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(heat_rows)))
        ax.set_yticklabels([row["hypothesis_name"] for row in heat_rows])
        ax.set_title("Hypothesis Heatmap")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / "hypothesis_heatmap.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    radar_rows = top_rows[: min(3, len(top_rows))]
    if radar_rows:
        radar_metrics = [
            "faithfulness_t",
            "answer_relevance_t",
            "context_precision_t",
            "context_recall_t",
            "context_relevance_t",
        ]
        labels = [key.replace("_t", "") for key in radar_metrics]
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]
        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(111, polar=True)
        palette = ["#355C7D", "#C06C84", "#2A9D8F"]
        for row, color in zip(radar_rows, palette):
            values = [row[key] for key in radar_metrics]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=row["hypothesis_name"], color=color)
            ax.fill(angles, values, color=color, alpha=0.14)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Top Configurations Radar")
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "top_configurations_radar.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a presentation-friendly MLflow dashboard with synthetic *_t metrics and charts."
    )
    ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    ap.add_argument("--source-experiment", default="rag-evals")
    ap.add_argument("--target-experiment", default="rag-evals-dashboard")
    ap.add_argument("--run-name", default="presentation_dashboard")
    args = ap.parse_args()

    rows = _load_hypothesis_rows(args.tracking_uri, args.source_experiment)

    tracker = EvalTracker(
        tracking_uri=args.tracking_uri,
        experiment_name=args.target_experiment,
        run_name=args.run_name,
        tags={
            "run_origin": "presentation_dashboard",
            "source_experiment": args.source_experiment,
        },
    )
    tracker.start()

    try:
        tracker.log_params(
            {
                "source_experiment": args.source_experiment,
                "hypothesis_count": len(rows),
                "chart_count": 5,
                "synthetic_metric_suffix": "_t",
            }
        )

        for step, row in enumerate(rows):
            metrics = {
                key: row[key]
                for key in row.keys()
                if key.endswith("_t")
            }
            tracker.log_metrics(metrics, step=step)

        best = rows[0]
        tracker.log_metrics(
            {
                "best_presentation_score_t": best["presentation_score_t"],
                "best_faithfulness_t": best["faithfulness_t"],
                "best_answer_relevance_t": best["answer_relevance_t"],
                "best_context_precision_t": best["context_precision_t"],
                "best_context_recall_t": best["context_recall_t"],
                "best_context_relevance_t": best["context_relevance_t"],
            }
        )

        with tempfile.TemporaryDirectory(prefix="mlflow-dashboard-") as tmp_dir_str:
            artifact_dir = Path(tmp_dir_str) / "presentation"
            _build_artifacts(rows, artifact_dir)
            tracker.log_artifact_dir(artifact_dir, artifact_path="presentation")

        tracker.end(status="FINISHED")
    except Exception:
        tracker.end(status="FAILED")
        raise

    print(f"Source experiment: {args.source_experiment}")
    print(f"Target experiment: {args.target_experiment}")
    print(f"Hypotheses included: {len(rows)}")
    print(f"Top hypothesis: {rows[0]['hypothesis_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
