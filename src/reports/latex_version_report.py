from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.utils import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelArtifacts:
    model_name: str
    metadata_path: Optional[Path]
    evaluation_path: Optional[Path]
    threshold_comparison_path: Optional[Path]
    plots_dir: Optional[Path]
    hyperparams_path: Optional[Path]


def _read_json(path: Path) -> Dict[str, Any]:
    return DataLoader.load(path)


def _fmt_float(x: Any, ndigits: int = 4) -> str:
    try:
        if x is None:
            return "-"
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)


def _latex_escape(text: str) -> str:
    # Minimal LaTeX escaping for text content (not paths).
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = "".join(replacements.get(ch, ch) for ch in str(text))
    return out


def _detok(path_str: str) -> str:
    # Safe for LaTeX file paths.
    return r"\detokenize{" + path_str + "}"


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
    hyperparams_path = PROJECT_ROOT / "data" / "output" / version / "hyperparameter_search" / model_name / "best_params.json"

    return ModelArtifacts(
        model_name=model_name,
        metadata_path=metadata_path if metadata_path.exists() else None,
        evaluation_path=evaluation_path if evaluation_path.exists() else None,
        threshold_comparison_path=threshold_path if threshold_path.exists() else None,
        plots_dir=plots_dir if plots_dir.exists() else None,
        hyperparams_path=hyperparams_path if hyperparams_path.exists() else None,
    )


def _list_plot_images(plots_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    images = [p for p in plots_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]

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


def _relpath(from_dir: Path, target: Path) -> str:
    return str(target.relative_to(from_dir) if target.is_relative_to(from_dir) else target)


def _relpath_any(from_dir: Path, target: Path) -> str:
    try:
        return str(target.relative_to(from_dir)).replace("\\", "/")
    except Exception:
        return str(Path(".") / target).replace("\\", "/")


def _render_table(rows: List[Tuple[str, str]]) -> str:
    lines: List[str] = []
    lines.append(r"\begin{tabular}{ll}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{M\'etrica} & \textbf{Valor} \\")
    lines.append(r"\midrule")
    for k, v in rows:
        lines.append(f"{_latex_escape(k)} & {_latex_escape(v)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _render_hparams_table(hparams: Dict[str, Any]) -> str:
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
        "objective",
        "random_state",
        "n_jobs",
    ]

    items: List[Tuple[str, Any]] = []
    for k in key_order:
        if k in hparams:
            items.append((k, hparams[k]))
    for k in sorted(hparams.keys()):
        if k not in set(key_order):
            items.append((k, hparams[k]))

    lines: List[str] = []
    lines.append(r"\begin{tabular}{ll}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Par\'ametro} & \textbf{Valor} \\")
    lines.append(r"\midrule")
    for k, v in items:
        lines.append(f"{_latex_escape(k)} & {_latex_escape(str(v))} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _render_threshold_longtable(results: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(r"\begin{longtable}{rrrrrrl}")
    lines.append(r"\toprule")
    lines.append(r"thr & recall & precision & spec & f1 & f2 & note \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"thr & recall & precision & spec & f1 & f2 & note \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    for r in results:
        note = "OPTIMAL" if str(r.get("is_optimal", "")).strip().upper() == "OPTIMAL" else ""
        lines.append(
            f"{_fmt_float(r.get('threshold'), 3)} & "
            f"{_fmt_float(r.get('recall'))} & "
            f"{_fmt_float(r.get('precision'))} & "
            f"{_fmt_float(r.get('specificity'))} & "
            f"{_fmt_float(r.get('f1_score'))} & "
            f"{_fmt_float(r.get('f2_score'))} & "
            f"{_latex_escape(note)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def generate_latex_report_for_version(
    version: str,
    models: Optional[Sequence[str]] = None,
    output_dir: Optional[Path] = None,
    compile_pdf: bool = True,
) -> Tuple[Path, Optional[Path]]:
    version = str(version)

    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "output" / version / "reports_latex"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(models) if models else _discover_models_for_version(version)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    tex_path = output_dir / f"report_{version}_{now}.tex"
    pdf_path = output_dir / f"report_{version}_{now}.pdf"

    created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parts: List[str] = []
    parts.append(r"\documentclass[11pt,letterpaper]{article}")
    parts.append(r"\usepackage[utf8]{inputenc}")
    parts.append(r"\usepackage[T1]{fontenc}")
    parts.append(r"\usepackage[spanish,es-tabla]{babel}")
    parts.append(r"\usepackage{geometry}")
    parts.append(r"\geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}")
    parts.append(r"\usepackage{graphicx}")
    parts.append(r"\usepackage{grffile}")
    parts.append(r"\usepackage{float}")
    parts.append(r"\usepackage{booktabs}")
    parts.append(r"\usepackage{longtable}")
    parts.append(r"\usepackage{hyperref}")
    parts.append(r"\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue}")
    parts.append(r"\setlength{\parskip}{0.35em}")
    parts.append(r"\setlength{\parindent}{0pt}")
    parts.append(r"\begin{document}")

    parts.append(r"\begin{titlepage}")
    parts.append(r"\centering")
    parts.append(r"{\Large\bfseries Reporte de resultados\\[0.5em]}")
    parts.append(rf"{{\large Versi\'on: { _latex_escape(version) }\\[0.5em]}}")
    parts.append(rf"{{\small Generado: { _latex_escape(created) }\\}}")
    parts.append(r"\vfill")
    parts.append(r"\end{titlepage}")

    parts.append(r"\tableofcontents")
    parts.append(r"\newpage")

    # Summary table (test metrics) using evaluation.json only
    summary_rows: List[Tuple[str, str]] = []
    summary_table_lines: List[str] = []
    summary_table_lines.append(r"\section{Resumen (validaci\'on final / test)}")

    # We render a small tabular per model
    tab_lines: List[str] = []
    tab_lines.append(r"\begin{tabular}{lrrrrrr}")
    tab_lines.append(r"\toprule")
    tab_lines.append(r"Modelo & ROC-AUC & AP & Recall & Precision & F1 & Accuracy \\")
    tab_lines.append(r"\midrule")

    any_eval = False
    for model_name in model_names:
        artifacts = _model_artifacts(version, model_name)
        if not artifacts.evaluation_path:
            continue
        ev = _read_json(artifacts.evaluation_path)
        any_eval = True
        tab_lines.append(
            f"{_latex_escape(model_name)} & "
            f"{_fmt_float(ev.get('roc_auc'))} & "
            f"{_fmt_float(ev.get('average_precision'))} & "
            f"{_fmt_float(ev.get('recall'))} & "
            f"{_fmt_float(ev.get('precision'))} & "
            f"{_fmt_float(ev.get('f1_score'))} & "
            f"{_fmt_float(ev.get('accuracy'))} \\\\"
        )

    tab_lines.append(r"\bottomrule")
    tab_lines.append(r"\end{tabular}")
    if any_eval:
        summary_table_lines.extend(tab_lines)
    else:
        summary_table_lines.append(r"No se encontr\'o ning\'un archivo de validaci\'on final (evaluation.json) para esta versi\'on.")

    parts.extend(summary_table_lines)

    # Per-model sections
    for model_name in model_names:
        artifacts = _model_artifacts(version, model_name)
        parts.append(r"\newpage")
        parts.append(rf"\section{{Modelo: {_latex_escape(model_name)}}}")

        # Load JSONs
        metadata: Dict[str, Any] = _read_json(artifacts.metadata_path) if artifacts.metadata_path else {}
        evaluation: Dict[str, Any] = _read_json(artifacts.evaluation_path) if artifacts.evaluation_path else {}
        threshold_cmp: Dict[str, Any] = (
            _read_json(artifacts.threshold_comparison_path) if artifacts.threshold_comparison_path else {}
        )
        best_params: Dict[str, Any] = _read_json(artifacts.hyperparams_path) if artifacts.hyperparams_path else {}

        # Artifact pointers
        parts.append(r"\subsection{Artefactos}")
        bullet_lines: List[str] = []
        bullet_lines.append(r"\begin{itemize}")
        if artifacts.evaluation_path:
            rel = _relpath_any(tex_path.parent, artifacts.evaluation_path)
            bullet_lines.append(rf"\item Validaci\'on final (test): \texttt{{{_latex_escape(rel)}}}")
        if artifacts.metadata_path:
            rel = _relpath_any(tex_path.parent, artifacts.metadata_path)
            bullet_lines.append(rf"\item Metadata de entrenamiento: \texttt{{{_latex_escape(rel)}}}")
        if artifacts.threshold_comparison_path:
            rel = _relpath_any(tex_path.parent, artifacts.threshold_comparison_path)
            bullet_lines.append(rf"\item Comparaci\'on de umbrales (test): \texttt{{{_latex_escape(rel)}}}")
        if artifacts.plots_dir:
            rel = _relpath_any(tex_path.parent, artifacts.plots_dir)
            bullet_lines.append(rf"\item Plots: \texttt{{{_latex_escape(rel)}}}")
        bullet_lines.append(r"\end{itemize}")
        parts.extend(bullet_lines)

        # Final evaluation table (uses evaluation.json)
        parts.append(r"\subsection{Validaci\'on final (test)}")
        if evaluation:
            rows = [
                ("accuracy", _fmt_float(evaluation.get("accuracy"))),
                ("precision", _fmt_float(evaluation.get("precision"))),
                ("recall", _fmt_float(evaluation.get("recall"))),
                ("specificity", _fmt_float(evaluation.get("specificity"))),
                ("f1_score", _fmt_float(evaluation.get("f1_score"))),
                ("roc_auc", _fmt_float(evaluation.get("roc_auc"))),
                ("average_precision", _fmt_float(evaluation.get("average_precision"))),
                ("n_samples", str(evaluation.get("n_samples", "-"))),
                ("n_positive", str(evaluation.get("n_positive", "-"))),
                ("n_negative", str(evaluation.get("n_negative", "-"))),
            ]
            parts.append(_render_table([(k, v) for k, v in rows]))

            cm = evaluation.get("confusion_matrix")
            if isinstance(cm, dict):
                parts.append(r"\\[0.75em]")
                parts.append(r"\textbf{Matriz de confusi\'on (test)}\\")
                parts.append(
                    _render_table(
                        [
                            ("tn", str(cm.get("tn"))),
                            ("fp", str(cm.get("fp"))),
                            ("fn", str(cm.get("fn"))),
                            ("tp", str(cm.get("tp"))),
                        ]
                    )
                )

            opt = evaluation.get("optimal_threshold_results")
            if isinstance(opt, dict):
                parts.append(r"\\[0.75em]")
                parts.append(r"\textbf{Umbral operativo reportado}\\")
                parts.append(
                    _render_table(
                        [
                            ("threshold", _fmt_float(opt.get("threshold"), 3)),
                            ("recall", _fmt_float(opt.get("recall"))),
                            ("precision", _fmt_float(opt.get("precision"))),
                            ("specificity", _fmt_float(opt.get("specificity"))),
                            ("f2_score", _fmt_float(opt.get("f2_score"))),
                        ]
                    )
                )
        else:
            parts.append(r"No se encontr\'o el archivo de validaci\'on final (evaluation.json).")

        # Hyperparameters
        if best_params:
            parts.append(r"\subsection{Hiperpar\'ametros (best")
            parts.append(r" / congelados)}")
            parts.append(_render_hparams_table(best_params))
        else:
            # Fallback to metadata optimized_params if present
            meta_best = metadata.get("optimized_params")
            if isinstance(meta_best, dict) and meta_best:
                parts.append(r"\subsection{Hiperpar\'ametros (best / congelados)}")
                parts.append(_render_hparams_table(meta_best))

        # Feature pruning
        pruning = metadata.get("feature_pruning")
        if isinstance(pruning, dict) and pruning.get("enabled"):
            parts.append(r"\subsection{Feature pruning}")
            parts.append(r"\begin{itemize}")
            parts.append(rf"\item min\_total\_ones: {_latex_escape(str(pruning.get('min_total_ones')))}")
            parts.append(rf"\item n\_dropped: {_latex_escape(str(pruning.get('n_dropped')))}")
            dropped = pruning.get("dropped_columns")
            if isinstance(dropped, list) and dropped:
                parts.append(r"\item dropped\_columns: " + _latex_escape(", ".join(map(str, dropped))))
            parts.append(r"\end{itemize}")

        # Threshold comparison
        if isinstance(threshold_cmp, dict) and isinstance(threshold_cmp.get("results"), list):
            parts.append(r"\subsection{An\'alisis de umbrales (test)}")
            parts.append(_render_threshold_longtable(threshold_cmp["results"]))

        # Plots (graphics)
        if artifacts.plots_dir:
            images = _list_plot_images(artifacts.plots_dir)
            if images:
                parts.append(r"\subsection{Gr\'aficas}")
                for img in images:
                    rel_img = _relpath_any(tex_path.parent, img)
                    parts.append(r"\begin{figure}[H]")
                    parts.append(r"\centering")
                    parts.append(rf"\includegraphics[width=0.95\linewidth]{{{_detok(rel_img)}}}")
                    parts.append(rf"\caption{{{_latex_escape(img.name)}}}")
                    parts.append(r"\end{figure}")

    parts.append(r"\end{document}")

    tex_path.write_text("\n".join(parts) + "\n", encoding="utf-8")

    compiled_pdf: Optional[Path] = None
    if compile_pdf:
        if shutil.which("pdflatex") is None:
            raise RuntimeError("pdflatex no est\'a disponible en el entorno.")

        # Run pdflatex twice to settle TOC.
        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={str(output_dir)}",
            str(tex_path),
        ]
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        if pdf_path.exists():
            compiled_pdf = pdf_path

    return tex_path, compiled_pdf


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate LaTeX/PDF reports per version.")
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
        help="Optional output directory. If omitted, writes to data/output/<version>/reports_latex/.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Generate .tex only (skip pdflatex).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    versions = args.versions
    if not versions:
        versions = sorted([p.name for p in (PROJECT_ROOT / "models").iterdir() if p.is_dir()])

    output_dir = Path(args.output_dir) if args.output_dir else None

    for v in versions:
        tex, pdf = generate_latex_report_for_version(
            v,
            models=args.models,
            output_dir=output_dir,
            compile_pdf=not args.no_compile,
        )
        print(str(tex))
        if pdf is not None:
            print(str(pdf))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
