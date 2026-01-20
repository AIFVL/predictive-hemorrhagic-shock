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
    precision_recall_curve
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
    print("\n" + "="*60)
    print(f"EVALUATING MODEL ON {dataset_name.upper()} SET")
    print("="*60)
    
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
        metrics['confusion_matrix_optimal'] = {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    
    # Print results (default threshold)
    print(f"\nResults on {dataset_name} set (n={len(y)}) - DEFAULT threshold (0.5):")
    print("-" * 40)
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall (Sens.):    {metrics['recall']:.4f}")
    print(f"  Specificity:       {metrics['specificity']:.4f}")
    print(f"  F1-Score:          {metrics['f1_score']:.4f}")
    
    if y_prob is not None:
        print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
        print(f"  Average Precision: {metrics['average_precision']:.4f}")
        print(f"\n  Optimal Threshold: {metrics['optimal_threshold']:.3f} (min_spec=0.40)")
        
        if use_optimal_threshold:
            print(f"\n Results with OPTIMAL threshold ({metrics['optimal_threshold']:.3f}):")
            print("-" * 40)
            print(f"  Accuracy:          {metrics['accuracy_optimal']:.4f}")
            print(f"  Precision:         {metrics['precision_optimal']:.4f}")
            print(f"  Recall (Sens.):    {metrics['recall_optimal']:.4f} ⭐")
            print(f"  Specificity:       {metrics['specificity_optimal']:.4f}")
            print(f"  F1-Score:          {metrics['f1_score_optimal']:.4f}")
    
    print(f"\nConfusion Matrix (default 0.5):")
    cm = metrics['confusion_matrix']
    print(f"  TN: {cm['tn']:4d}  |  FP: {cm['fp']:4d}")
    print(f"  FN: {cm['fn']:4d}  |  TP: {cm['tp']:4d}")
    
    if use_optimal_threshold and y_prob is not None:
        print(f"\nConfusion Matrix (optimal {metrics['optimal_threshold']:.3f}):")
        cm_opt = metrics['confusion_matrix_optimal']
        print(f"  TN: {cm_opt['tn']:4d}  |  FP: {cm_opt['fp']:4d}")
        print(f"  FN: {cm_opt['fn']:4d}  |  TP: {cm_opt['tp']:4d}")
    
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
        'accuracy', 'precision', 'recall', 'specificity', 'f1_score'
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
    output_path: Union[str, Path]
) -> str:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Evaluation results dict
        output_path: Path to save results
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Evaluation results saved to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    """CLI interface for model evaluation."""
    import argparse
    import joblib
    
    parser = argparse.ArgumentParser(description="Evaluate shock prediction model")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to trained model file"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to test data file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="SHOCK",
        help="Target variable name (default: SHOCK)"
    )
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = joblib.load(args.model)
    
    # Load data
    print("Loading test data...")
    if args.data.endswith('.parquet'):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)
    
    # Prepare features and target
    y = df[args.target]
    X = df.drop(columns=[args.target])
    
    print(f"Loaded {len(df)} samples with {X.shape[1]} features")
    
    # Evaluate
    results = evaluate_model(model, X, y, dataset_name="test")
    
    # Get classification report
    y_pred = model.predict(X)
    print("\nClassification Report:")
    print(get_classification_report(y, y_pred))
    
    # Get curve data if probabilities available
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]
        results['roc_curve'] = get_roc_curve_data(y, y_prob)
        results['pr_curve'] = get_pr_curve_data(y, y_prob)
    
    # Save results
    save_evaluation_results(results, args.output)
    
    print("\n✓ Evaluation complete")
