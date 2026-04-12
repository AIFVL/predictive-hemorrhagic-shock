"""
Model evaluation module.

Provides functionality for evaluating trained models
with clinical-appropriate metrics for hemorrhagic shock prediction.

Reference: docs/project_specification.md Section 8
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from datetime import datetime
import json

from src.utils import logger, log_section, log_subsection

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    cohen_kappa_score
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    
    Metrics are selected for clinical relevance in shock prediction:
    - Sensitivity (Recall): Critical to avoid missing shock cases
    - Specificity: Important to avoid unnecessary interventions
    - F1-Score: Balance between precision and recall
    - AUC-ROC: Overall discriminative ability
    - AUC-PR: Better for imbalanced classes
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for positive class (optional)
    
    Returns:
        Dict with all metrics
    """
    # Basic metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),  # Sensitivity
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'n_samples': len(y_true),
        'n_positive': int(y_true.sum()),
        'n_negative': int(len(y_true) - y_true.sum())
    }
    
    # Probability-based metrics (if available)
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        metrics['average_precision'] = average_precision_score(y_true, y_prob)
        
        # Optimal threshold OPTIMIZED FOR RECALL
        # Instead of Youden's J (sensitivity + specificity - 1), 
        # we find threshold that maximizes recall while maintaining minimum specificity
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Find threshold that gives at least 40% specificity with maximum recall
        valid_indices = np.where((1 - fpr) >= 0.40)[0]
        if len(valid_indices) > 0:
            # Among valid thresholds, pick the one with highest recall
            optimal_idx = valid_indices[np.argmax(tpr[valid_indices])]
        else:
            # Fallback: just maximize recall (lower threshold)
            optimal_idx = np.argmax(tpr)
        
        metrics['optimal_threshold'] = float(thresholds[optimal_idx])
        metrics['optimal_sensitivity'] = float(tpr[optimal_idx])
        metrics['optimal_specificity'] = float(1 - fpr[optimal_idx])
    
    return metrics


def analyze_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float] = None
) -> pd.DataFrame:
    """
    Analyze model performance across multiple classification thresholds.
    
    For each threshold, calculates:
    - Recall (Sensitivity): True Positive Rate
    - Precision: Positive Predictive Value
    - F2 Score: Weighted harmonic mean (recall 2x more important)
    - Specificity: True Negative Rate
    - F1 Score: Traditional F-measure
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        thresholds: List of thresholds to test (default: [0.2, 0.25, 0.3, 0.35, 0.4, 0.5])
    
    Returns:
        DataFrame with metrics for each threshold
    """
    from sklearn.metrics import fbeta_score
    
    if thresholds is None:
        raise ValueError("Thresholds list must be provided.")
    
    results = []
    
    for threshold in thresholds:
        # Apply threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        kappa = cohen_kappa_score(y_true, y_pred)
        
        results.append({
            'threshold': threshold,
            'recall': recall,
            'precision': precision,
            'specificity': specificity,
            'f1_score': f1,
            'f2_score': f2,
            'kappa': kappa,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'total_positive_predictions': tp + fp,
            'total_positive_actual': tp + fn
        })
    
    df = pd.DataFrame(results)
    return df


def find_optimal_threshold_for_target_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.90,
    thresholds: List[float] = None,
    test_thresholds: List[float] = None
) -> Dict:
    """
    Find the threshold that achieves closest to target recall while maximizing precision.
    
    Strategy:
    1. Test all candidate thresholds
    2. Filter those that achieve at least target_recall
    3. Among valid thresholds, pick the one with highest precision (fewer false positives)
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        target_recall: Desired recall level (e.g., 0.90 for 90% sensitivity)
        thresholds: List of thresholds to test for optimization (search range)
        test_thresholds: List of specific thresholds that were tested (for metadata)
    
    Returns:
        Dict with threshold_optimization and operating_point structured metadata
    """
    if thresholds is None:
        # Test finer-grained thresholds
        thresholds = np.arange(0.05, 0.60, 0.05).tolist()
    
    # Analyze all thresholds
    df_results = analyze_thresholds(y_true, y_prob, thresholds)
    
    # Filter thresholds that meet recall target
    valid_thresholds = df_results[df_results['recall'] >= target_recall]
    
    if len(valid_thresholds) == 0:
        # If no threshold meets target, return the one with highest recall
        logger.warning(f"No threshold achieves target recall of {target_recall:.2f}")
        best_row = df_results.loc[df_results['recall'].idxmax()]
        logger.info(f"Using threshold with maximum recall: {best_row['recall']:.3f}")
    else:
        # Among valid thresholds, pick the one with highest precision
        best_row = valid_thresholds.loc[valid_thresholds['precision'].idxmax()]
        logger.success(f"Found threshold {best_row['threshold']:.3f} with recall={best_row['recall']:.3f}, precision={best_row['precision']:.3f}")
    
    # Calculate accuracy
    y_pred = (y_prob >= best_row['threshold']).astype(int)
    accuracy = (y_pred == y_true).mean()
    
    # Return structured metadata
    return {
        'threshold_optimization': {
            'criterion': 'maximize_precision',
            'constraint': f'recall >= {target_recall:.2f}',
            'threshold_candidates': test_thresholds if test_thresholds else thresholds,
            'optimal_threshold': float(best_row['threshold'])
        },
        'operating_point': {
            'threshold': float(best_row['threshold']),
            'confusion_matrix': {
                'tp': int(best_row['tp']),
                'fp': int(best_row['fp']),
                'tn': int(best_row['tn']),
                'fn': int(best_row['fn'])
            },
            'metrics': {
                'recall': float(best_row['recall']),
                'precision': float(best_row['precision']),
                'specificity': float(best_row['specificity']),
                'f1_score': float(best_row['f1_score']),
                'f2_score': float(best_row['f2_score']),
                'kappa': float(best_row['kappa']),
                'accuracy': float(accuracy)
            }
        }
    }


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "test",
    use_optimal_threshold: bool = True
) -> Dict:
    """
    Evaluate a trained model on a dataset.
    
    Args:
        model: Trained sklearn model or Pipeline
        X: Features DataFrame
        y: True labels
        dataset_name: Name for reporting
        use_optimal_threshold: If True, recalculate metrics using optimized threshold
    
    Returns:
        Dict with evaluation results
    """
    log_section(f"EVALUATING MODEL ON {dataset_name.upper()} SET")
    
    # Get predictions with default 0.5 threshold
    y_pred_default = model.predict(X)
    
    # Get probabilities if available
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]
    
    # Calculate metrics with default threshold
    metrics = evaluate_predictions(y, y_pred_default, y_prob)
    metrics['dataset'] = dataset_name
    metrics['timestamp'] = datetime.now().isoformat()
    
    # If we have probabilities and want to use optimal threshold
    if y_prob is not None and use_optimal_threshold:
        optimal_th = metrics['optimal_threshold']
        y_pred_optimal = (y_prob >= optimal_th).astype(int)
        
        # Recalculate metrics with optimal threshold
        tn, fp, fn, tp = confusion_matrix(y, y_pred_optimal).ravel()
        
        metrics['accuracy_optimal'] = accuracy_score(y, y_pred_optimal)
        metrics['precision_optimal'] = precision_score(y, y_pred_optimal, zero_division=0)
        metrics['recall_optimal'] = recall_score(y, y_pred_optimal, zero_division=0)
        metrics['specificity_optimal'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['f1_score_optimal'] = f1_score(y, y_pred_optimal, zero_division=0)
        metrics['kappa_optimal'] = cohen_kappa_score(y, y_pred_optimal)
        metrics['confusion_matrix_optimal'] = {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    
    # Log results
    log_subsection(f"Results on {dataset_name} set (n={len(y)}) - DEFAULT threshold (0.5)")
    logger.info({
        "metrics": {
            "accuracy": f"{metrics['accuracy']:.4f}",
            "precision": f"{metrics['precision']:.4f}",
            "recall": f"{metrics['recall']:.4f}",
            "specificity": f"{metrics['specificity']:.4f}",
            "f1_score": f"{metrics['f1_score']:.4f}",
            "kappa": f"{metrics['kappa']:.4f}"
        }
    })
    
    if y_prob is not None:
        logger.info({
            "probability_metrics": {
                "roc_auc": f"{metrics['roc_auc']:.4f}",
                "average_precision": f"{metrics['average_precision']:.4f}",
                "optimal_threshold": f"{metrics['optimal_threshold']:.3f} (min_spec=0.40)"
            }
        })
        
        if use_optimal_threshold:
            log_subsection(f"Results with OPTIMAL threshold ({metrics['optimal_threshold']:.3f})")
            logger.info({
                "optimal_metrics": {
                    "accuracy": f"{metrics['accuracy_optimal']:.4f}",
                    "precision": f"{metrics['precision_optimal']:.4f}",
                    "recall": f"{metrics['recall_optimal']:.4f} ⭐",
                    "specificity": f"{metrics['specificity_optimal']:.4f}",
                    "f1_score": f"{metrics['f1_score_optimal']:.4f}",
                    "kappa": f"{metrics['kappa_optimal']:.4f}"
                }
            })
    
    # Log confusion matrices
    cm = metrics['confusion_matrix']
    logger.info({
        "confusion_matrix_default": {
            "TN": cm['tn'],
            "FP": cm['fp'],
            "FN": cm['fn'],
            "TP": cm['tp']
        }
    })
    
    if use_optimal_threshold and y_prob is not None:
        cm_opt = metrics['confusion_matrix_optimal']
        logger.info({
            "confusion_matrix_optimal": {
                "TN": cm_opt['tn'],
                "FP": cm_opt['fp'],
                "FN": cm_opt['fn'],
                "TP": cm_opt['tp']
            }
        })
    
    return metrics


def compare_models(
    results: List[Dict],
    primary_metric: str = 'f1_score'
) -> pd.DataFrame:
    """
    Compare multiple model evaluation results.
    
    Args:
        results: List of evaluation result dicts
        primary_metric: Metric to sort by
    
    Returns:
        DataFrame with comparison
    """
    comparison_metrics = [
        'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'kappa'
    ]
    
    # Add probability metrics if available
    if 'roc_auc' in results[0]:
        comparison_metrics.extend(['roc_auc', 'average_precision'])
    
    rows = []
    for result in results:
        row = {'model': result.get('model_name', 'unknown')}
        for metric in comparison_metrics:
            row[metric] = result.get(metric, np.nan)
        rows.append(row)
    
    df = pd.DataFrame(rows).sort_values(primary_metric, ascending=False)
    return df


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = None
) -> str:
    """
    Get sklearn classification report as string.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional class names
    
    Returns:
        Classification report string
    """
    if target_names is None:
        target_names = ['No Shock', 'Shock']
    
    return classification_report(y_true, y_pred, target_names=target_names)


def get_roc_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict:
    """
    Get ROC curve data for plotting.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
    
    Returns:
        Dict with fpr, tpr, thresholds
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
        'auc': roc_auc_score(y_true, y_prob)
    }


def get_pr_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict:
    """
    Get Precision-Recall curve data for plotting.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
    
    Returns:
        Dict with precision, recall, thresholds
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'thresholds': thresholds.tolist(),
        'average_precision': average_precision_score(y_true, y_prob)
    }


def save_evaluation_results(
    results: Dict,
    model_name: str
) -> str:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Evaluation results dict
        model_name: Name of the model (used to construct path)
    
    Returns:
        Path to saved file
    """
    from src.utils import get_config, DataLoader
    
    config = get_config()
    output_path = Path(config.get_path('evaluation_output', model_name=model_name))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert any non-serializable objects to strings
    import json as json_module
    serializable_results = json_module.loads(json_module.dumps(results, default=str))
    DataLoader.save(serializable_results, output_path)
    
    logger.success(f"Evaluation results saved to: {output_path}")
    return str(output_path)
