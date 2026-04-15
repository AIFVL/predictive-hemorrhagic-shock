"""
Visualization module for model evaluation.

Generates evaluation plots for shock prediction model assessment.
All plotting logic is delegated to src.utils.plotting functions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)

from src.utils import (
    DataLoader,
    logger,
    log_section,
    get_config,
    setup_plot_style,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance,
    plot_calibration_curve,
    plot_threshold_analysis,
    plot_performance_radar,
    plot_normalized_confusion_matrix,
    plot_cv_fold_boxplot,
    plot_prediction_bias,
    plot_model_comparison_bar,
    plot_model_comparison_grouped,
    save_figure
)

# Metric display config: (cv_key, label, color)
_CV_METRICS_CONFIG = [
    ("accuracy",         "Accuracy",          "#2196F3"),
    ("precision",        "Precision",          "#4CAF50"),
    ("recall",           "Recall (Sensitivity)","#F44336"),
    ("f1",               "F1-Score",           "#FF9800"),
    ("f2",               "F2-Score (β=2)",     "#FF5722"),
    ("roc_auc",          "ROC-AUC",            "#9C27B0"),
    ("kappa",            "Cohen's Kappa (κ)",  "#00BCD4"),
]


def generate_all_plots(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "Model"
) -> Dict[str, str]:
    """
    Generate all per-model evaluation plots.

    Outputs (inside data/output/{version}/{model_name}/model_plots/):
      - roc_curve.png
      - precision_recall_curve.png
      - confusion_matrix.png           (raw counts, default threshold)
      - confusion_matrix_normalized.png (row-%, optimal threshold)
      - feature_importance.png
      - threshold_analysis.png
      - calibration_curve.png
      - performance_radar.png
      - prediction_bias.png
      - cv_fold_{metric}.png           (one per CV metric)

    Args:
        model:      Trained sklearn Pipeline or estimator.
        X_test:     Test features.
        y_test:     Test labels.
        model_name: Name of the model.

    Returns:
        Dict mapping plot key → file path string.
    """
    config = get_config()
    base_plots_dir = Path(config.get_path('eval_plots'))
    output_dir = base_plots_dir / model_name.lower().replace(' ', '_')
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_plot_style()
    log_section(f"GENERATING EVALUATION PLOTS FOR {model_name.upper()}")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    plots = {}

    # ── 1. ROC Curve ─────────────────────────────────────────────────────────
    logger.info("Creating CV ROC curve (Mean over folds)...")
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    n_folds = config.get('cross_validation.n_folds')
    random_seed = config.get('general_config.random_seed')
    shuffle = config.get('data_split.shuffle')

    cv = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_seed)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        fold_model = clone(model)
        fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        y_prob_fold = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]
        
        fpr_fold, tpr_fold, _ = roc_curve(y_train.iloc[val_idx], y_prob_fold)
        fold_auc = auc(fpr_fold, tpr_fold)
        
        interp_tpr = np.interp(mean_fpr, fpr_fold, tpr_fold)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(fold_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)

    fig = plot_roc_curve(
        mean_fpr, mean_tpr, mean_auc, 
        title=f"CV Mean ROC Curve - {model_name}"
    )
        
    plots['roc_curve'] = output_dir / 'roc_curve.png'
    save_figure(fig, plots['roc_curve'])

    # ── 2. Precision-Recall Curve ────────────────────────────────────────────
    logger.info("Creating Precision-Recall curve...")
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    fig = plot_precision_recall_curve(recall_vals, precision_vals, avg_precision,
                                      title=f"Precision-Recall Curve - {model_name}")
    plots['pr_curve'] = output_dir / 'precision_recall_curve.png'
    save_figure(fig, plots['pr_curve'])

    # ── 3. Confusion Matrix (raw, default threshold) ─────────────────────────
    logger.info("Creating confusion matrix (raw)...")
    cm = confusion_matrix(y_test, y_pred)
    fig = plot_confusion_matrix(cm, class_names=['No Shock', 'Shock'],
                                title=f"Confusion Matrix - {model_name}")
    plots['confusion_matrix'] = output_dir / 'confusion_matrix.png'
    save_figure(fig, plots['confusion_matrix'])

    # ── 4. Feature Importance ────────────────────────────────────────────────
    logger.info("Creating feature importance plot...")

    def _build_importance_df(estimator, X_in: pd.DataFrame) -> pd.DataFrame:
        importances = getattr(estimator, 'feature_importances_', None)
        if importances is None:
            raise ValueError("Estimator has no feature_importances_")
        est_names = getattr(estimator, 'feature_name_', None)
        feature_names = (
            list(est_names)
            if isinstance(est_names, (list, tuple)) and len(est_names) == len(importances)
            else list(X_in.columns)
        )
        if len(feature_names) != len(importances):
            m = min(len(feature_names), len(importances))
            logger.warning(f"Feature importance length mismatch – truncating to {m}.")
            feature_names = feature_names[:m]
            importances = importances[:m]
        return (
            pd.DataFrame({'feature': feature_names, 'importance': importances})
            .sort_values('importance', ascending=False)
        )

    try:
        if hasattr(model, 'feature_importances_'):
            importance_df = _build_importance_df(model, X_test)
        elif hasattr(model, 'named_steps') and hasattr(
                model.named_steps.get('classifier'), 'feature_importances_'):
            classifier = model.named_steps['classifier']
            X_for_names = X_test
            if 'prune_features' in model.named_steps:
                X_for_names = model.named_steps['prune_features'].transform(X_for_names)
            importance_df = _build_importance_df(classifier, X_for_names)
        else:
            importance_df = None

        if importance_df is not None:
            fig = plot_feature_importance(importance_df, top_n=20,
                                          title=f"Feature Importance - {model_name}")
            plots['feature_importance'] = output_dir / 'feature_importance.png'
            save_figure(fig, plots['feature_importance'])
        else:
            logger.warning("Model does not support feature importances")
    except Exception as e:
        logger.warning(f"Skipping feature importance plot due to error: {e}")

    # ── 5. Threshold Analysis ────────────────────────────────────────────────
    logger.info("Creating threshold analysis...")
    fig = plot_threshold_analysis(y_test, y_prob, title="Metrics vs Decision Threshold")
    plots['threshold_analysis'] = output_dir / 'threshold_analysis.png'
    save_figure(fig, plots['threshold_analysis'])

    # ── 6. Calibration Curve ─────────────────────────────────────────────────
    logger.info("Creating calibration curve...")
    fig = plot_calibration_curve(y_test, y_prob, n_bins=10,
                                 title=f"Calibration Curve - {model_name}")
    plots['calibration_curve'] = output_dir / 'calibration_curve.png'
    save_figure(fig, plots['calibration_curve'])

    # ── 7-onward: metadata-dependent plots ───────────────────────────────────
    metadata_path = (
        Path(config.get_path('model_output', model_name=model_name)).parent / 'metadata.json'
    )
    if not metadata_path.exists():
        logger.warning(
            f"metadata.json not found at {metadata_path} — "
            "skipping metadata-dependent plots"
        )
        logger.success(f"Generated {len(plots)} evaluation plots in {output_dir}")
        return {k: str(v) for k, v in plots.items()}

    metadata = DataLoader.load(metadata_path)

    operating_point = metadata.get('operating_point', {})
    cv_results      = metadata.get('cv_results', {})
    cv_metrics      = cv_results.get('metrics', {})
    class_dist      = metadata.get('class_distribution', {})

    # ── 7. Normalized Confusion Matrix (optimal threshold) ───────────────────
    if operating_point:
        logger.info("Creating normalized confusion matrix (optimal threshold)...")
        cm_data = operating_point.get('confusion_matrix', {})
        fig = plot_normalized_confusion_matrix(
            confusion_matrix_data=cm_data,
            operating_point=operating_point,
            model_name=model_name,
        )
        plots['confusion_matrix_normalized'] = output_dir / 'confusion_matrix_normalized.png'
        save_figure(fig, plots['confusion_matrix_normalized'])

    # ── 8. Performance Radar ─────────────────────────────────────────────────
    if operating_point:
        logger.info("Creating performance radar chart...")
        op_metrics = operating_point.get('metrics', {})
        radar_entry = {
            'label':       model_name,
            'recall':      op_metrics.get('recall',      0.0),
            'specificity': op_metrics.get('specificity', 0.0),
            'precision':   op_metrics.get('precision',   0.0),
            'f1_score':    op_metrics.get('f1_score',    0.0),
            'accuracy':    op_metrics.get('accuracy',    0.0),
            'kappa':       op_metrics.get('kappa',       0.0),
        }
        fig = plot_performance_radar(
            [radar_entry],
            title=f"Performance Radar — {model_name}"
        )
        plots['performance_radar'] = output_dir / 'performance_radar.png'
        save_figure(fig, plots['performance_radar'])

    # ── 9. Prediction Bias ───────────────────────────────────────────────────
    if operating_point and class_dist:
        logger.info("Creating prediction bias chart...")
        fig = plot_prediction_bias(
            confusion_matrix_data=operating_point.get('confusion_matrix', {}),
            class_distribution=class_dist,
            operating_point=operating_point,
            model_name=model_name,
        )
        plots['prediction_bias'] = output_dir / 'prediction_bias.png'
        save_figure(fig, plots['prediction_bias'])

    # ── 10. CV fold boxplot per metric ───────────────────────────────────────
    if cv_metrics:
        cv_plots_dir = output_dir / 'cv_fold_plots'
        cv_plots_dir.mkdir(parents=True, exist_ok=True)
        for cv_key, metric_label, color in _CV_METRICS_CONFIG:
            metric_data = cv_metrics.get(cv_key, {})
            scores = metric_data.get('test_scores', [])
            if not scores:
                logger.warning(f"No CV scores for {cv_key}, skipping boxplot.")
                continue
            logger.info(f"Creating CV fold boxplot: {metric_label}...")
            fig = plot_cv_fold_boxplot(
                scores=scores,
                metric_name=metric_label,
                model_name=model_name,
                color=color,
            )
            plot_key = f'cv_fold_{cv_key}'
            plots[plot_key] = cv_plots_dir / f'cv_fold_{cv_key}.png'
            save_figure(fig, plots[plot_key])

    logger.success(f"Generated {len(plots)} evaluation plots in {output_dir}")
    return {k: str(v) for k, v in plots.items()}


def generate_comparison_plots() -> Dict[str, str]:
    """
    Generate cross-model comparison charts.

    Reads all metadata.json files for every trained model under models/{version}/
    and produces grouped vertical bar charts saved to:
        data/output/{version}/comparison_plots/op/   (operating-point metrics)
        data/output/{version}/comparison_plots/cv/   (CV mean ± std metrics)

    Group 1 — classification_group.png : Accuracy, Precision, Recall, Specificity
    Group 2 — scores_group.png         : F1, F2, ROC-AUC, Kappa

    Also generates a combined multi-model Performance Radar.

    Returns:
        Dict mapping plot key → file path string.
    """
    config = get_config()
    version = config.get('general_config.version')

    # Locate all model metadata files for the current version
    models_base = Path(config.get_path('output_base')) / 'models'
    metadata_files = sorted(models_base.glob('*/metadata.json'))

    if not metadata_files:
        logger.warning(f"No metadata.json found under {models_base} — skipping comparison plots")
        return {}

    log_section("GENERATING MODEL COMPARISON PLOTS")

    # Output sub-directories
    base_comparison = Path(config.get_path('output_base')) / 'comparison_plots'
    op_dir = base_comparison / 'op'
    cv_dir = base_comparison / 'cv'
    op_dir.mkdir(parents=True, exist_ok=True)
    cv_dir.mkdir(parents=True, exist_ok=True)

    setup_plot_style()

    # ── Collect per-model data ────────────────────────────────────────────────
    op_rows: List[dict]       = []
    cv_rows: List[dict]       = []
    radar_entries: List[dict] = []

    for mf in metadata_files:
        model_name = mf.parent.name
        try:
            md = DataLoader.load(mf)
        except Exception as e:
            logger.warning(f"Could not read {mf}: {e}")
            continue

        op   = md.get('operating_point', {})
        op_m = op.get('metrics', {})
        cv_m = md.get('cv_results', {}).get('metrics', {})

        op_rows.append({
            'label':       model_name,
            'recall':      op_m.get('recall',      0.0),
            'specificity': op_m.get('specificity', 0.0),
            'precision':   op_m.get('precision',   0.0),
            'f1_score':    op_m.get('f1_score',    0.0),
            'f2_score':    op_m.get('f2_score',    0.0),
            'accuracy':    op_m.get('accuracy',    0.0),
            'kappa':       op_m.get('kappa',       0.0),
        })

        cv_entry = {'label': model_name}
        for cv_key, _, _ in _CV_METRICS_CONFIG:
            m_data = cv_m.get(cv_key, {})
            cv_entry[cv_key]              = m_data.get('test_mean', 0.0)
            cv_entry[f'{cv_key}_std']     = m_data.get('test_std',  0.0)
        cv_rows.append(cv_entry)

        radar_entries.append({
            'label':       model_name,
            'recall':      op_m.get('recall',      0.0),
            'specificity': op_m.get('specificity', 0.0),
            'precision':   op_m.get('precision',   0.0),
            'f1_score':    op_m.get('f1_score',    0.0),
            'accuracy':    op_m.get('accuracy',    0.0),
            'kappa':       op_m.get('kappa',       0.0),
        })

    if not op_rows:
        logger.warning("No valid model metadata found — aborting comparison plots")
        return {}

    plots = {}

    # ── Metric definitions ────────────────────────────────────────────────────
    # Each list has exactly 4 items → one per 2×2 cell
    OP_CLASSIFICATION = [
        ("accuracy",    "Accuracy"),
        ("precision",   "Precision"),
        ("recall",      "Recall"),
        ("specificity", "Specificity"),
    ]
    OP_SCORES = [
        ("f1_score", "F1-Score"),
        ("f2_score", "F2-Score (β=2)"),
        ("kappa",    "Cohen's κ"),
        ("accuracy", "Accuracy"),   # reuse accuracy as 4th cell for symmetry
    ]

    # CV keys follow _CV_METRICS_CONFIG
    CV_CLASSIFICATION = [
        ("accuracy",  "Accuracy"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
        ("roc_auc",   "ROC-AUC"),
    ]
    CV_SCORES = [
        ("f1",      "F1-Score"),
        ("f2",      "F2-Score (β=2)"),
        ("kappa",   "Cohen's κ"),
        ("roc_auc", "ROC-AUC"),
    ]

    # ── Operating-point charts ────────────────────────────────────────────────
    logger.info("Creating OP classification metrics chart (2×2)...")
    fig = plot_model_comparison_grouped(
        op_rows,
        metrics=OP_CLASSIFICATION,
        title="Model Comparison — Classification Metrics (Operating Point)",
    )
    plots['op_classification'] = op_dir / 'classification_group.png'
    save_figure(fig, plots['op_classification'])

    logger.info("Creating OP score metrics chart (2×2)...")
    fig = plot_model_comparison_grouped(
        op_rows,
        metrics=OP_SCORES,
        title="Model Comparison — Score Metrics (Operating Point)",
    )
    plots['op_scores'] = op_dir / 'scores_group.png'
    save_figure(fig, plots['op_scores'])

    # ── CV charts ─────────────────────────────────────────────────────────────
    logger.info("Creating CV classification metrics chart (2×2)...")
    fig = plot_model_comparison_grouped(
        cv_rows,
        metrics=CV_CLASSIFICATION,
        title="Model Comparison — Classification Metrics (CV Mean)",
        std_suffix="_std",
    )
    plots['cv_classification'] = cv_dir / 'classification_group.png'
    save_figure(fig, plots['cv_classification'])

    logger.info("Creating CV score metrics chart (2×2)...")
    fig = plot_model_comparison_grouped(
        cv_rows,
        metrics=CV_SCORES,
        title="Model Comparison — Score Metrics (CV Mean)",
        std_suffix="_std",
    )
    plots['cv_scores'] = cv_dir / 'scores_group.png'
    save_figure(fig, plots['cv_scores'])

    # ── Multi-model Performance Radar ─────────────────────────────────────────
    if len(radar_entries) >= 1:
        logger.info("Creating multi-model performance radar...")
        fig = plot_performance_radar(
            radar_entries,
            title="Multi-Model Performance Radar (Operating Point)"
        )
        plots['multi_model_radar'] = base_comparison / 'multi_model_radar.png'
        save_figure(fig, plots['multi_model_radar'])

    logger.success(f"Generated {len(plots)} comparison plots in {base_comparison}")
    return {k: str(v) for k, v in plots.items()}


