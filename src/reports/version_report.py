from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelArtifacts:
    model_name: str
    metadata_path: Optional[Path]
    evaluation_path: Optional[Path]
    threshold_comparison_path: Optional[Path]
    plots_dir: Optional[Path]
    hyperparams_dir: Optional[Path]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_float(x: Any, ndigits: int = 4) -> str:
    try:
        if x is None:
            return "-"
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not headers:
        return ""
    out: List[str] = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _discover_models_for_version(version: str) -> List[str]:
    models_dir = PROJECT_ROOT / "models" / version
    if not models_dir.exists():
        return []
    return sorted([p.name for p in models_dir.iterdir() if p.is_dir()])


def _model_artifacts(version: str, model_name: str) -> ModelArtifacts:
    metadata_path = PROJECT_ROOT / "models" / version / model_name / "metadata.json"
    evaluation_path = PROJECT_ROOT / "data" / "output" / version / "evaluation" / model_name / "evaluation.json"
    threshold_path = (
        PROJECT_ROOT
        / "data"
        / "output"
        / version
        / "threshold_analysis"
        / model_name
        / "threshold_comparison_test.json"
    )
    plots_dir = PROJECT_ROOT / "data" / "output" / version / "model_plots" / model_name
    hyperparams_dir = PROJECT_ROOT / "data" / "output" / version / "hyperparameter_search" / model_name

    return ModelArtifacts(
        model_name=model_name,
        metadata_path=metadata_path if metadata_path.exists() else None,
        evaluation_path=evaluation_path if evaluation_path.exists() else None,
        threshold_comparison_path=threshold_path if threshold_path.exists() else None,
        plots_dir=plots_dir if plots_dir.exists() else None,
        hyperparams_dir=hyperparams_dir if hyperparams_dir.exists() else None,
    )


def _list_plot_images(plots_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    images = [p for p in plots_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    # Stable ordering: prioritize key plots, then alphabetical
    priority = [
        "confusion_matrix.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "calibration_curve.png",
        "threshold_analysis.png",
        "feature_importance.png",
    ]
    by_name = {p.name: p for p in images}
    ordered: List[Path] = [by_name[n] for n in priority if n in by_name]
    ordered += sorted([p for p in images if p.name not in set(priority)], key=lambda p: p.name)
    return ordered


def _relpath_from_root(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except Exception:
        return str(path)


def _render_model_section(version: str, artifacts: ModelArtifacts) -> str:
    lines: List[str] = []
    lines.append(f"## Modelo: {artifacts.model_name}")

    metadata: Dict[str, Any] = {}
    if artifacts.metadata_path:
        metadata = _read_json(artifacts.metadata_path)

    evaluation: Dict[str, Any] = {}
    if artifacts.evaluation_path:
        evaluation = _read_json(artifacts.evaluation_path)

    threshold_cmp: Dict[str, Any] = {}
    if artifacts.threshold_comparison_path:
        threshold_cmp = _read_json(artifacts.threshold_comparison_path)

    # Quick links
    links: List[str] = []
    if artifacts.metadata_path:
        links.append(f"metadata: `{_relpath_from_root(artifacts.metadata_path)}`")
    if artifacts.evaluation_path:
        links.append(f"evaluation: `{_relpath_from_root(artifacts.evaluation_path)}`")
    if artifacts.threshold_comparison_path:
        links.append(f"thresholds: `{_relpath_from_root(artifacts.threshold_comparison_path)}`")
    if links:
        lines.append("\n" + " | ".join(links) + "\n")

    # Dataset summary
    n_train = metadata.get("n_samples")
    class_dist = metadata.get("class_distribution")
    if n_train is not None or class_dist:
        lines.append("### Dataset (train)")
        if n_train is not None:
            lines.append(f"- n_samples: {n_train}")
        if isinstance(class_dist, dict) and class_dist:
            # keys can be '0.0'/'1.0'
            pos = class_dist.get("1") or class_dist.get("1.0")
            neg = class_dist.get("0") or class_dist.get("0.0")
            if pos is not None and neg is not None:
                try:
                    total = float(pos) + float(neg)
                    lines.append(f"- positives: {pos} ({float(pos)/total:.1%})")
                    lines.append(f"- negatives: {neg} ({float(neg)/total:.1%})")
                except Exception:
                    lines.append(f"- class_distribution: {class_dist}")
            else:
                lines.append(f"- class_distribution: {class_dist}")

    # Feature pruning
    pruning = metadata.get("feature_pruning")
    if isinstance(pruning, dict) and pruning.get("enabled"):
        lines.append("### Feature pruning")
        lines.append(f"- min_total_ones: {pruning.get('min_total_ones')}")
        lines.append(f"- n_dropped: {pruning.get('n_dropped')}")
        dropped_cols = pruning.get("dropped_columns")
        if isinstance(dropped_cols, list) and dropped_cols:
            lines.append("- dropped_columns: " + ", ".join(map(str, dropped_cols)))

    # Hyperparameters
    best_params = metadata.get("optimized_params") or (metadata.get("hyperparameter_search") or {}).get("best_params")
    if isinstance(best_params, dict) and best_params:
        lines.append("### Hiperparámetros (congelados / best)")
        # stable ordering for readability
        key_order = [
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "max_depth",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
            "min_split_gain",
            "reg_alpha",
            "reg_lambda",
            "scale_pos_weight",
            "random_state",
            "n_jobs",
        ]
        ordered_items: List[Tuple[str, Any]] = []
        for k in key_order:
            if k in best_params:
                ordered_items.append((k, best_params[k]))
        for k in sorted(best_params.keys()):
            if k not in set(key_order):
                ordered_items.append((k, best_params[k]))
        rows = [[k, str(v)] for k, v in ordered_items]
        lines.append(_md_table(["param", "value"], rows))

    # CV summary (from metadata)
    cv_results = metadata.get("cv_results")
    if isinstance(cv_results, dict) and isinstance(cv_results.get("metrics"), dict):
        metrics = cv_results["metrics"]
        wanted = ["precision", "recall", "f1", "f2", "roc_auc", "average_precision"]
        rows = []
        for m in wanted:
            if m in metrics and isinstance(metrics[m], dict):
                rows.append([
                    m,
                    _fmt_float(metrics[m].get("test_mean")),
                    _fmt_float(metrics[m].get("test_std")),
                    _fmt_float(metrics[m].get("train_mean")),
                    _fmt_float(metrics[m].get("train_std")),
                ])
        if rows:
            lines.append("### Cross-validation")
            lines.append(_md_table(["metric", "test_mean", "test_std", "train_mean", "train_std"], rows))

    # Test evaluation
    if evaluation:
        lines.append("### Evaluación (test)")
        rows = [
            ["accuracy", _fmt_float(evaluation.get("accuracy"))],
            ["precision", _fmt_float(evaluation.get("precision"))],
            ["recall", _fmt_float(evaluation.get("recall"))],
            ["specificity", _fmt_float(evaluation.get("specificity"))],
            ["f1_score", _fmt_float(evaluation.get("f1_score"))],
            ["roc_auc", _fmt_float(evaluation.get("roc_auc"))],
            ["average_precision", _fmt_float(evaluation.get("average_precision"))],
        ]
        lines.append(_md_table(["metric", "value"], rows))
        cm = evaluation.get("confusion_matrix")
        if isinstance(cm, dict):
            lines.append(
                "- confusion_matrix (tn, fp, fn, tp): "
                f"{cm.get('tn')}, {cm.get('fp')}, {cm.get('fn')}, {cm.get('tp')}"
            )

        opt = evaluation.get("optimal_threshold_results")
        if isinstance(opt, dict):
            lines.append(
                f"- optimal_threshold (train-optimized): {_fmt_float(opt.get('threshold'), ndigits=3)} "
                f"| recall={_fmt_float(opt.get('recall'))} | precision={_fmt_float(opt.get('precision'))}"
            )

    # Threshold comparison table (test)
    if isinstance(threshold_cmp, dict) and isinstance(threshold_cmp.get("results"), list):
        rows = []
        for r in threshold_cmp["results"]:
            if not isinstance(r, dict):
                continue
            rows.append([
                _fmt_float(r.get("threshold"), ndigits=3),
                _fmt_float(r.get("recall")),
                _fmt_float(r.get("precision")),
                _fmt_float(r.get("specificity")),
                _fmt_float(r.get("f1_score")),
                _fmt_float(r.get("f2_score")),
                "OPTIMAL" if str(r.get("is_optimal", "")).strip().upper() == "OPTIMAL" else "",
            ])
        if rows:
            lines.append("### Umbrales (test)")
            lines.append(_md_table(["thr", "recall", "precision", "spec", "f1", "f2", "note"], rows))

    # Plots
    if artifacts.plots_dir:
        images = _list_plot_images(artifacts.plots_dir)
        if images:
            lines.append("### Plots")
            for img in images:
                rel = _relpath_from_root(img)
                lines.append(f"- {img.name}: ![]({rel})")

    return "\n".join(lines).strip() + "\n"


def generate_report_for_version(
    version: str,
    models: Optional[Sequence[str]] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    version = str(version)
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "output" / version / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(models) if models else _discover_models_for_version(version)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append(f"# Reporte de resultados — {version}")
    lines.append(f"Generado: {generated_at}")

    # Add quick summary table per model (test metrics)
    summary_rows: List[List[str]] = []
    for model_name in model_names:
        artifacts = _model_artifacts(version, model_name)
        if artifacts.evaluation_path:
            ev = _read_json(artifacts.evaluation_path)
            summary_rows.append(
                [
                    model_name,
                    _fmt_float(ev.get("roc_auc")),
                    _fmt_float(ev.get("average_precision")),
                    _fmt_float(ev.get("recall")),
                    _fmt_float(ev.get("precision")),
                    _fmt_float(ev.get("f2_score")) if "f2_score" in ev else "-",
                ]
            )
    if summary_rows:
        lines.append("\n## Resumen (test)")
        lines.append(_md_table(["model", "roc_auc", "avg_precision", "recall", "precision", "f2"], summary_rows))

    for model_name in model_names:
        artifacts = _model_artifacts(version, model_name)
        lines.append("\n" + _render_model_section(version, artifacts))

    out_path = output_dir / f"report_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return out_path


def generate_reports_for_versions(
    versions: Sequence[str],
    models: Optional[Sequence[str]] = None,
    output_base: Optional[Path] = None,
) -> List[Path]:
    paths: List[Path] = []
    for v in versions:
        out_dir = output_base if output_base is not None else None
        paths.append(generate_report_for_version(v, models=models, output_dir=out_dir))
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Markdown reports per version.")
    parser.add_argument(
        "--versions",
        nargs="+",
        required=False,
        help="Versions to include (e.g., v10 v11). Defaults to all under models/.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=False,
        help="Optional model names to include (e.g., lightgbm). Defaults to all under models/<version>/.",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        help="Optional output directory. If omitted, writes to data/output/<version>/reports/.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    versions = args.versions
    if not versions:
        versions = sorted([p.name for p in (PROJECT_ROOT / "models").iterdir() if p.is_dir()])

    output_dir = Path(args.output_dir) if args.output_dir else None

    out_paths = generate_reports_for_versions(versions, models=args.models, output_base=output_dir)
    for p in out_paths:
        print(str(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
