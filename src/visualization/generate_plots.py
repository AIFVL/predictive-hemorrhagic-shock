"""
Visualization module for model evaluation.
Generates evaluation plots and creates localized plot metadata JSONs.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    brier_score_loss
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
    plot_decision_tree_structure,
    plot_calibration_curve,
    plot_threshold_analysis,
    plot_normalized_confusion_matrix,
    plot_cv_fold_boxplot,
    plot_metric_pair_bar,
    plot_single_metrics_bars,
    plot_multi_roc_curve,
    plot_multi_calibration_curve,
    save_figure
)

def generate_all_plots(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "Model"
) -> Dict[str, str]:
    config = get_config()
    version = config.get('general_config.version')
    output_dir = Path(config.get_path('eval_plots', model_name=model_name))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = Path(config.get_path('model_output', model_name=model_name)).parent / f'{model_name}_metadata.json'
    metadata = DataLoader.load(metadata_path) if metadata_path.exists() else {}
    
    setup_plot_style()
    log_section(f"GENERATING EVALUATION PLOTS FOR {model_name.upper()}")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    plots = {}
    plot_metadata = {
        "experiment": version,
        "model_name": model_name,
        "plots": {}
    }

    # 1. ROC Curve CV and Calibration CV (OOF data)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    n_folds = config.get('cross_validation.n_folds')
    random_seed = config.get('general_config.random_seed')
    shuffle = config.get('data_split.shuffle')

    cv = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_seed)
    
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    oof_y_true, oof_y_prob = [], []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        fold_model = clone(model)
        fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        y_prob_fold = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]
        
        fpr_fold, tpr_fold, _ = roc_curve(y_train.iloc[val_idx], y_prob_fold)
        aucs.append(auc(fpr_fold, tpr_fold))
        
        interp_tpr = np.interp(mean_fpr, fpr_fold, tpr_fold)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        oof_y_true.extend(y_train.iloc[val_idx])
        oof_y_prob.extend(y_prob_fold)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = float(np.mean(aucs))
    cv_brier = float(brier_score_loss(oof_y_true, oof_y_prob))

    # Temporarily save raw data to a json strictly for multi_comparison mapping later, but NOT in plot_metadata!
    cv_curves_data = {
        'mean_fpr': mean_fpr.tolist(),
        'mean_tpr': mean_tpr.tolist(),
        'mean_auc': float(mean_auc),
        'y_true': np.array(oof_y_true).tolist(),
        'y_prob': np.array(oof_y_prob).tolist(),
        'brier_score': cv_brier
    }
    DataLoader.save(cv_curves_data, output_dir / '.cv_curves_internal.json') # Hidden internal file
    
    fig = plot_roc_curve(mean_fpr, mean_tpr, mean_auc, title=f"CV Mean ROC Curve - {model_name}")
    roc_cv_path = output_dir / 'roc_curve_cv.png'
    save_figure(fig, roc_cv_path)
    plots['roc_curve_cv'] = str(roc_cv_path)
    plot_metadata["plots"]["roc_curve_cv"] = {
        "description": "ROC Curve based on Train CV validation probabilities",
        "path": "roc_curve_cv.png",
        "metrics": {"roc_auc": mean_auc}
    }

    fig = plot_calibration_curve(np.array(oof_y_true), np.array(oof_y_prob), n_bins=10, title=f"CV Calibration Curve (Brier: {cv_brier:.3f}) - {model_name}")
    cal_cv_path = output_dir / 'calibration_curve_cv.png'
    save_figure(fig, cal_cv_path)
    plots['calibration_curve_cv'] = str(cal_cv_path)
    plot_metadata["plots"]["calibration_curve_cv"] = {
        "description": "Calibration curve based on Train CV validation probabilities",
        "path": "calibration_curve_cv.png",
        "metrics": {"brier_score": cv_brier}
    }

    # 2. Precision-Recall Curve (Test set)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    fig = plot_precision_recall_curve(recall_vals, precision_vals, avg_precision, title=f"Precision-Recall Curve (Test) - {model_name}")
    pr_path = output_dir / 'precision_recall_curve.png'
    save_figure(fig, pr_path)
    plots['precision_recall'] = str(pr_path)
    plot_metadata["plots"]["precision_recall"] = {
        "description": "Precision-Recall curve on Test set",
        "path": "precision_recall_curve.png",
        "metrics": {"average_precision": float(avg_precision)}
    }

    # 3. Decision Threshold Analysis (Test set)
    metric_thresh = metadata.get('operating_point', {}).get('threshold', 0.5)
    fig = plot_threshold_analysis(y_test, y_prob, optimal_threshold=metric_thresh, title="Metrics vs Decision Threshold")
    thresh_path = output_dir / 'threshold_analysis.png'
    save_figure(fig, thresh_path)
    plots['decision_threshold'] = str(thresh_path)
    plot_metadata["plots"]["decision_threshold"] = {
        "description": "Decision threshold curve on Test set highlighting optimal choice",
        "path": "threshold_analysis.png",
        "metrics": {"optimal_threshold": metric_thresh}
    }

    # 4. Feature Importance
    def _build_importance_df(estimator, X_in):
        importances = getattr(estimator, 'feature_importances_', None)
        if importances is None: return None
        # FORCE use of actual dataframe columns to avoid LightGBM 'Column_0' masking
        fnames = list(X_in.columns)
        if len(fnames) != len(importances):
            m = min(len(fnames), len(importances))
            fnames, importances = fnames[:m], importances[:m]
        return pd.DataFrame({'feature': fnames, 'importance': importances}).sort_values('importance', ascending=False)

    if hasattr(model, 'feature_importances_'):
        imp_df = _build_importance_df(model, X_test)
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('classifier'), 'feature_importances_'):
        cls = model.named_steps['classifier']
        Xin = X_test if 'prune_features' not in model.named_steps else model.named_steps['prune_features'].transform(X_test)
        imp_df = _build_importance_df(cls, Xin)
    else:
        imp_df = None

    if imp_df is not None:
        fig = plot_feature_importance(imp_df, top_n=20, title=f"Feature Importance - {model_name}")
        feat_path = output_dir / 'feature_importance.png'
        save_figure(fig, feat_path)
        plots['feature_importance'] = str(feat_path)
        
        # Add pct to feature_data before exporting
        df_imp_top = imp_df.head(20).copy()
        total_imp = df_imp_top['importance'].sum()
        df_imp_top['pct'] = (df_imp_top['importance'] / total_imp * 100) if total_imp > 0 else 0

        # Replace numpy float64 with native float for clean JSON structure
        feature_data = []
        for _, row in df_imp_top.iterrows():
            feature_data.append({
                "feature": row["feature"],
                "importance": float(row["importance"]),
                "pct": round(float(row["pct"]), 2)
            })

        plot_metadata["plots"]["feature_importance"] = {
            "description": "Top feature importances from the trained model",
            "path": "feature_importance.png",
            "metrics": {},
            "top_features": feature_data 
        }

    # 4B. Full Decision Tree Structure (only for decision tree models)
    tree_estimator = None
    if hasattr(model, 'tree_'):
        tree_estimator = model
    elif hasattr(model, 'named_steps'):
        cls = model.named_steps.get('classifier')
        if cls is not None and hasattr(cls, 'tree_'):
            tree_estimator = cls

    if tree_estimator is not None and 'decision_tree' in str(model_name).lower():
        # Resolve feature names robustly against possible pruning/transform steps.
        n_features_tree = int(getattr(tree_estimator, 'n_features_in_', 0))
        feature_names = list(X_test.columns)

        if len(feature_names) != n_features_tree and hasattr(model, 'named_steps'):
            prune_step = model.named_steps.get('prune_features')
            if prune_step is not None:
                try:
                    X_pruned = prune_step.transform(X_test)
                    if hasattr(X_pruned, 'columns'):
                        feature_names = list(X_pruned.columns)
                    elif hasattr(prune_step, 'get_feature_names_out'):
                        feature_names = list(prune_step.get_feature_names_out())
                except Exception:
                    pass

        if len(feature_names) != n_features_tree:
            feature_names = [f'feature_{i}' for i in range(n_features_tree)]

        class_names = [str(c) for c in sorted(pd.Series(y_test).unique().tolist())]

        fig = plot_decision_tree_structure(
            tree_estimator=tree_estimator,
            feature_names=feature_names,
            class_names=class_names,
            title=f"Decision Tree Structure - {model_name}",
            max_depth=None,
        )
        tree_path = output_dir / 'decision_tree_full.png'
        save_figure(fig, tree_path)
        plots['decision_tree_full'] = str(tree_path)
        plot_metadata["plots"]["decision_tree_full"] = {
            "description": "Complete decision tree structure",
            "path": "decision_tree_full.png",
            "metrics": {
                "depth": int(tree_estimator.get_depth()),
                "n_leaves": int(tree_estimator.get_n_leaves()),
                "node_count": int(tree_estimator.tree_.node_count)
            }
        }
        
    # 5. Extract Operating Point Metrics & Confusion Matrix (Optimal Threshold)
    operating_point = metadata.get('operating_point', {})
    if operating_point:
        op_metrics = operating_point.get('metrics', {})
        cm_data = operating_point.get('confusion_matrix', {})
        tn = cm_data.get('tn', 0); fp = cm_data.get('fp', 0); fn = cm_data.get('fn', 0); tp = cm_data.get('tp', 0)
        total = tn + fp + fn + tp
        
        actual_neg = tn + fp
        actual_pos = fn + tp
        
        # Adjust pct to row-based (true distribution) like in plot_normalized_confusion_matrix
        cm_pct = {
            'tn_pct': round((tn/actual_neg)*100, 2) if actual_neg > 0 else 0,
            'fp_pct': round((fp/actual_neg)*100, 2) if actual_neg > 0 else 0,
            'fn_pct': round((fn/actual_pos)*100, 2) if actual_pos > 0 else 0,
            'tp_pct': round((tp/actual_pos)*100, 2) if actual_pos > 0 else 0,
        }
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # Performance Pair Group: Precision vs NPV
        p_npv_fig = plot_metric_pair_bar("Precision", op_metrics.get("precision", 0.0), "NPV", float(npv), title=f"Precision vs NPV - {model_name}")
        p_npv_path = output_dir / 'perf_precision_npv.png'
        save_figure(p_npv_fig, p_npv_path)
        plots['perf_precision_npv'] = str(p_npv_path)
        plot_metadata["plots"]["perf_precision_npv"] = {
            "description": "Grouped bar plot without space comparing Precision and NPV from Operating Point",
            "path": "perf_precision_npv.png",
            "metrics": {"precision": op_metrics.get("precision", 0.0), "npv": float(npv)}
        }
        
        # Performance Pair Group: Recall vs Specificity
        r_sp_fig = plot_metric_pair_bar("Recall", op_metrics.get("recall", 0.0), "Specificity", op_metrics.get("specificity", 0.0), title=f"Recall vs Specificity - {model_name}")
        r_sp_path = output_dir / 'perf_recall_specificity.png'
        save_figure(r_sp_fig, r_sp_path)
        plots['perf_recall_specificity'] = str(r_sp_path)
        plot_metadata["plots"]["perf_recall_specificity"] = {
            "description": "Grouped bar plot without space comparing Recall and Specificity from Operating Point",
            "path": "perf_recall_specificity.png",
            "metrics": {"recall": op_metrics.get("recall", 0.0), "specificity": op_metrics.get("specificity", 0.0)}
        }
        
        # Performance Pair Group: F1 vs F2
        f_fig = plot_metric_pair_bar("F1", op_metrics.get("f1_score", 0.0), "F2", op_metrics.get("f2_score", 0.0), title=f"F1 vs F2 Score - {model_name}")
        f_path = output_dir / 'perf_f1_f2.png'
        save_figure(f_fig, f_path)
        plots['perf_f1_f2'] = str(f_path)
        plot_metadata["plots"]["perf_f1_f2"] = {
            "description": "Grouped bar plot without space comparing F1 and F2 from Operating Point",
            "path": "perf_f1_f2.png",
            "metrics": {"f1_score": op_metrics.get("f1_score", 0.0), "f2_score": op_metrics.get("f2_score", 0.0)}
        }
        
        # Accuracy
        acc_metric = {"accuracy": op_metrics.get("accuracy", 0.0)}
        fig = plot_single_metrics_bars(acc_metric, title=f"Accuracy (OP) - {model_name}")
        acc_path = output_dir / 'accuracy.png'
        save_figure(fig, acc_path)
        plots['accuracy'] = str(acc_path)
        plot_metadata["plots"]["accuracy"] = {
            "description": "Single bar plot for isolated accuracy from operating point",
            "path": "accuracy.png",
            "metrics": acc_metric
        }

        # Kappa
        kappa_metric = {"kappa": op_metrics.get("kappa", 0.0)}
        fig = plot_single_metrics_bars(kappa_metric, title=f"Kappa (OP) - {model_name}")
        kappa_path = output_dir / 'kappa.png'
        save_figure(fig, kappa_path)
        plots['kappa'] = str(kappa_path)
        plot_metadata["plots"]["kappa"] = {
            "description": "Single bar plot for isolated kappa from operating point",
            "path": "kappa.png",
            "metrics": kappa_metric
        }

        # Confusion Matrix (Normalized)
        fig = plot_normalized_confusion_matrix(cm_data, operating_point, model_name=model_name)
        conf_path = output_dir / 'confusion_matrix_normalized.png'
        save_figure(fig, conf_path)
        plots['confusion_matrix'] = str(conf_path)
        plot_metadata["plots"]["confusion_matrix"] = {
            "description": "Confusion matrix with percentages using optimal threshold",
            "path": "confusion_matrix_normalized.png",
            "metrics": {"counts": cm_data, "percentages": cm_pct}
        }
        
    # 6. CV Fold Boxplot (ROC AUC)
    cv_metrics = metadata.get('cv_results', {}).get('metrics', {})
    if cv_metrics and 'roc_auc' in cv_metrics:
        roc_data = cv_metrics['roc_auc']
        scores = roc_data.get('test_scores', [])
        fig = plot_cv_fold_boxplot(scores, "ROC-AUC", model_name, "#9C27B0")
        box_path = output_dir / 'cv_fold_plots' / 'cv_fold_roc_auc.png'
        box_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, box_path)
        plots['roc_boxplot_cv'] = str(box_path)
        plot_metadata["plots"]["roc_boxplot_cv"] = {
            "description": "Boxplot showing ROC-AUC distribution across CV folds",
            "path": "cv_fold_plots/cv_fold_roc_auc.png",
            "metrics": {
                "mean": float(roc_data.get("test_mean", 0.0)),
                "std": float(roc_data.get("test_std", 0.0)),
                "folds": [float(x) for x in scores]
            }
        }

    # Save Individual JSON
    plot_meta_path = output_dir / 'plot_metadata.json'
    DataLoader.save(plot_metadata, plot_meta_path)
    logger.success(f"Generated {len(plots)} plots and plot_metadata.json for {model_name}")
    
    return {k: str(v) for k, v in plots.items()}

def generate_comparison_plots() -> Dict[str, str]:
    config = get_config()
    version = config.get('general_config.version')
    models_base = Path(config.get_path('output_base')) / 'models'
    metadata_files = sorted(models_base.glob('*/*_metadata.json'))

    if not metadata_files: return {}

    base_comparison = Path(config.get_path('output_base')) / 'comparison_plots'
    base_comparison.mkdir(parents=True, exist_ok=True)
    setup_plot_style()
    plots = {}
    
    global_meta = {
        "experiment": version,
        "plots": {}
    }

    roc_data, cal_data, brier_models, cv_auc_models, cv_folds = [], [], {}, {}, {}
    
    for mf in metadata_files:
        model_name = mf.parent.name
        
        # Load internal hidden curve data
        curves_file = mf.parent / 'plots' / '.cv_curves_internal.json'
        
        # Also need metrics from plot_metadata
        plot_meta_file = mf.parent / 'plots' / 'plot_metadata.json'
        
        if plot_meta_file.exists():
            pm = DataLoader.load(plot_meta_file)
            bx = pm.get("plots", {}).get("roc_boxplot_cv", {})
            if "metrics" in bx:
                cv_folds[model_name] = bx["metrics"]
                cv_auc_models[model_name] = bx["metrics"]["mean"]
                
            cal = pm.get("plots", {}).get("calibration_curve_cv", {})
            if "metrics" in cal:
                brier_models[model_name] = cal["metrics"]["brier_score"]
                
        if curves_file.exists():
            cdata = DataLoader.load(curves_file)
            roc_data.append({'label': model_name, 'fpr': cdata['mean_fpr'], 'tpr': cdata['mean_tpr'], 'auc': cdata['mean_auc']})
            # Format label to include brier 
            brier_label = f"{model_name} (Brier: {cdata['brier_score']:.3f})"
            cal_data.append({'label': brier_label, 'y_true': cdata['y_true'], 'y_prob': cdata['y_prob']})

    if roc_data:
        fig = plot_multi_roc_curve(roc_data, title="Model Comparison - ROC Curve (CV)")
        mroc_path = base_comparison / 'roc_comparison.png'
        save_figure(fig, mroc_path)
        plots['roc_comparison'] = str(mroc_path)
        global_meta["plots"]["roc_comparison"] = {
            "description": "ROC Curves comparing all evaluated models",
            "path": "roc_comparison.png",
            "models": {m: {"roc_auc": a} for m, a in cv_auc_models.items()}
        }

    if cal_data:
        fig = plot_multi_calibration_curve(cal_data, title="Model Comparison - Calibration Curve (CV)")
        mcal_path = base_comparison / 'calibration_comparison.png'
        save_figure(fig, mcal_path)
        plots['calibration_comparison'] = str(mcal_path)
        global_meta["plots"]["calibration_comparison"] = {
            "description": "Calibration curves comparing models with Brier score",
            "path": "calibration_comparison.png",
            "models": {m: {"brier_score": bs} for m, bs in brier_models.items()}
        }

    if cv_folds:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = list(cv_folds.keys())
        data = [cv_folds[k]["folds"] for k in labels]
        ax.boxplot(data, labels=labels, patch_artist=True)
        ax.set_title("CV Fold Distribution - ROC AUC")
        ax.set_ylabel("ROC AUC")
        # Adding rotation so labels do not overlap
        ax.set_xticklabels(labels, rotation=45, ha='right')
        fig.tight_layout()
        bx_path = base_comparison / 'roc_boxplot_comparison.png'
        save_figure(fig, bx_path)
        plots['roc_boxplot'] = str(bx_path)
        global_meta["plots"]["roc_boxplot"] = {
            "description": "Boxplot comparing CV ROC AUC distribution across models",
            "path": "roc_boxplot_comparison.png",
            "models": cv_folds
        }

    comp_meta_path = base_comparison / 'comparison_metadata.json'
    DataLoader.save(global_meta, comp_meta_path)
    
    return plots
