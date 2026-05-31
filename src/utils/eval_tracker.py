# uv run mlflow ui
from __future__ import annotations

import csv
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt


_PRIMARY_METRICS = (
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "context_relevance",
)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_metric_name(name: str) -> str:
    mapping = {
        "answer_relevancy": "answer_relevance",
        "nv_context_relevance": "context_relevance",
    }
    return mapping.get(name, name)


def _safe_filename(text: str) -> str:
    cleaned = [ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text]
    return "".join(cleaned).strip("_") or "artifact"


def _sanitize_param_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _top_rows_by_metric(
    rows: Sequence[Mapping[str, Any]], metric_name: str, *, limit: int, reverse: bool
) -> List[Dict[str, Any]]:
    normalized = _normalize_metric_name(metric_name)
    decorated: List[tuple[float, Dict[str, Any]]] = []
    for row in rows:
        value = _safe_float(row.get(normalized))
        if value is None:
            continue
        decorated.append((value, dict(row)))
    decorated.sort(key=lambda item: item[0], reverse=reverse)
    return [row for _, row in decorated[:limit]]


def create_history_report(
    run_summaries: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    top_n: int = 12,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for summary in run_summaries:
        row = dict(summary)
        hypothesis = str(
            row.get("hypothesis_name") or row.get("run_name") or row.get("source_file") or "run"
        )
        row["hypothesis_name"] = hypothesis
        rows.append(row)

    if not rows:
        return []

    csv_path = output_dir / "historical_runs.csv"
    json_path = output_dir / "historical_runs.json"
    _write_csv(csv_path, rows)
    _write_json(json_path, rows)

    plt.style.use("seaborn-v0_8-whitegrid")

    faith_sorted = sorted(
        rows,
        key=lambda row: _safe_float(row.get("faithfulness")) or -1.0,
        reverse=True,
    )
    leaders = faith_sorted[:top_n]
    if leaders:
        fig, ax = plt.subplots(figsize=(12, 7))
        labels = [row["hypothesis_name"] for row in leaders]
        values = [max((_safe_float(row.get("faithfulness")) or 0.0), 0.0) for row in leaders]
        bars = ax.barh(labels[::-1], values[::-1], color="#355C7D")
        ax.set_title("Faithfulness Leaderboard")
        ax.set_xlabel("Mean score")
        ax.set_xlim(0.0, max(values) * 1.15 if values else 1.0)
        for bar, value in zip(bars, values[::-1]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
        fig.tight_layout()
        fig.savefig(output_dir / "faithfulness_leaderboard.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    main_chart_rows = faith_sorted[:top_n]
    if main_chart_rows:
        labels = [row["hypothesis_name"] for row in main_chart_rows]
        x = list(range(len(labels)))
        width = 0.24
        fig, ax = plt.subplots(figsize=(14, 7))
        series = [
            ("Faithfulness", "faithfulness", "#355C7D"),
            ("Answer relevance", "answer_relevance", "#6C5B7B"),
            ("Context relevance", "context_relevance", "#C06C84"),
        ]
        offsets = (-width, 0.0, width)
        for (label, key, color), offset in zip(series, offsets):
            vals = [max((_safe_float(row.get(key)) or 0.0), 0.0) for row in main_chart_rows]
            ax.bar([i + offset for i in x], vals, width=width, label=label, color=color)
        ax.set_ylim(0.0, 1.05)
        ax.set_title("RAG Hypothesis Comparison")
        ax.set_ylabel("Mean score")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "hypothesis_metric_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8))
        for row in main_chart_rows:
            x_val = _safe_float(row.get("answer_relevance")) or 0.0
            y_val = _safe_float(row.get("faithfulness")) or 0.0
            size = 250 + 1200 * ((_safe_float(row.get("context_relevance")) or 0.0))
            ax.scatter(x_val, y_val, s=size, alpha=0.75, color="#F67280", edgecolors="#2A363B", linewidths=0.8)
            ax.text(x_val + 0.005, y_val + 0.005, row["hypothesis_name"], fontsize=9)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Answer relevance")
        ax.set_ylabel("Faithfulness")
        ax.set_title("Faithfulness vs Answer Relevance")
        fig.tight_layout()
        fig.savefig(output_dir / "faithfulness_vs_relevance.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    return list(output_dir.iterdir())


class EvalTracker:
    def __init__(
        self,
        *,
        tracking_uri: str = "sqlite:///mlflow.db",
        experiment_name: str = "rag-evals",
        run_name: Optional[str] = None,
        enabled: bool = True,
        tags: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.enabled = enabled
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = dict(tags or {})
        self._mlflow = None
        self._active_run = None

    def start(self) -> None:
        if not self.enabled:
            return

        try:
            import mlflow
        except ImportError as e:
            raise RuntimeError(
                "MLflow не установлен. Выполни `uv sync`, чтобы использовать eval tracking."
            ) from e

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._active_run = mlflow.start_run(run_name=self.run_name)
        self._mlflow = mlflow
        all_tags = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **self.tags,
        }
        for key, value in all_tags.items():
            mlflow.set_tag(key, _sanitize_param_value(value))

    def end(self, status: str = "FINISHED") -> None:
        if not self.enabled or self._mlflow is None:
            return
        self._mlflow.end_run(status=status)
        self._active_run = None

    def log_params(self, params: Mapping[str, Any]) -> None:
        if not self.enabled or self._mlflow is None:
            return
        sanitized = {key: _sanitize_param_value(value) for key, value in params.items()}
        self._mlflow.log_params(sanitized)

    def log_metrics(self, metrics: Mapping[str, Any], *, step: int = 0) -> None:
        if not self.enabled or self._mlflow is None:
            return
        sanitized: Dict[str, float] = {}
        for key, value in metrics.items():
            cast = _safe_float(value)
            if cast is None:
                continue
            sanitized[key] = cast
        if sanitized:
            self._mlflow.log_metrics(sanitized, step=step)

    def log_artifact_file(self, path: Path, artifact_path: Optional[str] = None) -> None:
        if not self.enabled or self._mlflow is None:
            return
        self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_artifact_dir(self, path: Path, artifact_path: Optional[str] = None) -> None:
        if not self.enabled or self._mlflow is None:
            return
        self._mlflow.log_artifacts(str(path), artifact_path=artifact_path)

    def log_eval_payload(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        params: Mapping[str, Any],
        summary_metrics: Mapping[str, Any],
        out_xlsx: Optional[Path] = None,
        worst_case_limit: int = 10,
        history_runs: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        if not self.enabled or self._mlflow is None:
            return

        with tempfile.TemporaryDirectory(prefix="eval-tracker-") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            eval_dir = tmp_dir / "eval"
            plots_dir = eval_dir / "plots"
            worst_dir = eval_dir / "worst_cases"
            eval_dir.mkdir(parents=True, exist_ok=True)
            plots_dir.mkdir(parents=True, exist_ok=True)
            worst_dir.mkdir(parents=True, exist_ok=True)

            _write_csv(eval_dir / "eval_results.csv", rows)
            _write_json(eval_dir / "eval_results.json", list(rows))
            _write_json(eval_dir / "run_config.json", dict(params))
            _write_json(eval_dir / "summary_metrics.json", dict(summary_metrics))

            plt.style.use("seaborn-v0_8-whitegrid")
            primary = [
                (
                    name,
                    _safe_float(summary_metrics.get(name))
                    or _safe_float(summary_metrics.get(f"{name}_mean")),
                )
                for name in _PRIMARY_METRICS
                if (_safe_float(summary_metrics.get(name)) or _safe_float(summary_metrics.get(f"{name}_mean"))) is not None
            ]
            if primary:
                fig, ax = plt.subplots(figsize=(10, 6))
                labels = [name.replace("_", " ").title() for name, _ in primary]
                values = [float(value) for _, value in primary if value is not None]
                bars = ax.bar(labels, values, color=["#355C7D", "#6C5B7B", "#C06C84", "#F67280", "#99B898"][: len(values)])
                ax.set_ylim(0.0, 1.05)
                ax.set_title("Eval Metrics")
                ax.set_ylabel("Mean score")
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
                fig.tight_layout()
                fig.savefig(plots_dir / "eval_metrics.png", dpi=180, bbox_inches="tight")
                plt.close(fig)

            latency_keys = [
                "retrieval_latency_mean_ms",
                "generation_latency_mean_ms",
                "end_to_end_latency_mean_ms",
            ]
            latency_points = [
                (key, _safe_float(summary_metrics.get(key)))
                for key in latency_keys
                if _safe_float(summary_metrics.get(key)) is not None
            ]
            if latency_points:
                fig, ax = plt.subplots(figsize=(9, 5))
                labels = [key.replace("_latency_mean_ms", "").replace("_", " ").title() for key, _ in latency_points]
                values = [float(value) for _, value in latency_points if value is not None]
                bars = ax.bar(labels, values, color="#355C7D")
                ax.set_title("Pipeline Latency")
                ax.set_ylabel("Mean latency, ms")
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.03, f"{value:.1f}", ha="center")
                fig.tight_layout()
                fig.savefig(plots_dir / "latency.png", dpi=180, bbox_inches="tight")
                plt.close(fig)

            for metric_name in ("faithfulness", "answer_relevance", "context_relevance"):
                worst_rows = _top_rows_by_metric(rows, metric_name, limit=worst_case_limit, reverse=False)
                if not worst_rows:
                    continue
                stem = _safe_filename(metric_name)
                _write_csv(worst_dir / f"worst_{stem}.csv", worst_rows)
                _write_json(worst_dir / f"worst_{stem}.json", worst_rows)

            if history_runs:
                create_history_report(history_runs, plots_dir / "history")

            if out_xlsx is not None and out_xlsx.exists():
                self._mlflow.log_artifact(str(out_xlsx), artifact_path="eval")

            self._mlflow.log_artifacts(str(eval_dir), artifact_path="eval")
