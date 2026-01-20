"""
Visualization module for model evaluation.

Generates clinical-appropriate plots for shock prediction model assessment.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Union
import json

from sklearn.metrics import (
    roc_curve, 
    auc, 
    precision_recall_curve,
    confusion_matrix
)


# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Union[str, Path],
    title: str = "ROC Curve - Shock Prediction"
) -> str:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        output_path: Path to save plot
        title: Plot title
    
    Returns:
        Path to saved plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ ROC curve saved to: {output_path}")
    return str(output_path)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Precision-Recall Curve - Shock Prediction"
) -> str:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        output_path: Path to save plot
        title: Plot title
    
    Returns:
        Path to saved plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    # Calculate average precision using sklearn
    from sklearn.metrics import average_precision_score
    avg_precision = average_precision_score(y_true, y_prob)
    
    # Baseline (random classifier)
    baseline = y_true.sum() / len(y_true)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.3f})')
    plt.axhline(y=baseline, color='red', linestyle='--', lw=2,
                label=f'Baseline ({baseline:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Precision-Recall curve saved to: {output_path}")
    return str(output_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Confusion Matrix - Shock Prediction"
) -> str:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save plot
        title: Plot title
    
    Returns:
        Path to saved plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Count'},
        xticklabels=['No Shock (0)', 'Shock (1)'],
        yticklabels=['No Shock (0)', 'Shock (1)'],
        annot_kws={'fontsize': 14, 'fontweight': 'bold'}
    )
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add text annotations with percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({pct:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to: {output_path}")
    return str(output_path)


def plot_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    output_path: Union[str, Path],
    top_n: int = 20,
    title: str = "Feature Importance - Shock Prediction"
) -> str:
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importances: Feature importance values
        output_path: Path to save plot
        top_n: Number of top features to show
        title: Plot title
    
    Returns:
        Path to saved plot
    """
    # Create DataFrame and sort
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    plt.barh(range(len(df)), df['importance'].values, color=colors)
    plt.yticks(range(len(df)), df['feature'].values)
    plt.xlabel('Importance (Mean Decrease in Impurity)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Feature importance plot saved to: {output_path}")
    return str(output_path)


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Metrics vs Decision Threshold"
) -> str:
    """
    Plot metrics at different decision thresholds.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        output_path: Path to save plot
        title: Plot title
    
    Returns:
        Path to saved plot
    """
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    specificities = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        f1_scores.append(f1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall (Sensitivity)', linewidth=2)
    plt.plot(thresholds, specificities, label='Specificity', linewidth=2)
    plt.plot(thresholds, f1_scores, label='F1 Score', linewidth=2, linestyle='--')
    
    plt.xlabel('Decision Threshold', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Threshold analysis plot saved to: {output_path}")
    return str(output_path)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Union[str, Path],
    n_bins: int = 10,
    title: str = "Calibration Curve - Shock Prediction"
) -> str:
    """
    Plot calibration curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        output_path: Path to save plot
        n_bins: Number of bins for calibration
        title: Plot title
    
    Returns:
        Path to saved plot
    """
    from sklearn.calibration import calibration_curve
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, 's-',
             label='Model', linewidth=2, markersize=8)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Calibration curve saved to: {output_path}")
    return str(output_path)


def generate_all_plots(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Union[str, Path],
    model_name: str = "RandomForest"
) -> Dict[str, str]:
    """
    Generate all evaluation plots.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        output_dir: Directory to save plots
        model_name: Name of the model
    
    Returns:
        Dict with paths to all generated plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING EVALUATION PLOTS")
    print("="*60 + "\n")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    plots = {}
    
    # 1. ROC Curve
    plots['roc_curve'] = plot_roc_curve(
        y_test, y_prob,
        output_dir / 'roc_curve.png',
        f"ROC Curve - {model_name}"
    )
    
    # 2. Precision-Recall Curve
    plots['pr_curve'] = plot_precision_recall_curve(
        y_test, y_prob,
        output_dir / 'precision_recall_curve.png',
        f"Precision-Recall Curve - {model_name}"
    )
    
    # 3. Confusion Matrix
    plots['confusion_matrix'] = plot_confusion_matrix(
        y_test, y_pred,
        output_dir / 'confusion_matrix.png',
        f"Confusion Matrix - {model_name}"
    )
    
    # 4. Feature Importance
    if hasattr(model, 'feature_importances_'):
        plots['feature_importance'] = plot_feature_importance(
            X_test.columns.tolist(),
            model.feature_importances_,
            output_dir / 'feature_importance.png',
            top_n=20,
            title=f"Feature Importance - {model_name}"
        )
    elif hasattr(model.named_steps['classifier'], 'feature_importances_'):
        plots['feature_importance'] = plot_feature_importance(
            X_test.columns.tolist(),
            model.named_steps['classifier'].feature_importances_,
            output_dir / 'feature_importance.png',
            top_n=20,
            title=f"Feature Importance - {model_name}"
        )
    
    # 5. Threshold Analysis
    plots['threshold_analysis'] = plot_threshold_analysis(
        y_test, y_prob,
        output_dir / 'threshold_analysis.png',
        "Metrics vs Decision Threshold"
    )
    
    # 6. Calibration Curve
    plots['calibration_curve'] = plot_calibration_curve(
        y_test, y_prob,
        output_dir / 'calibration_curve.png',
        n_bins=10,
        title=f"Calibration Curve - {model_name}"
    )
    
    print("\n✓ All plots generated successfully")
    print(f"  Output directory: {output_dir}")
    
    return plots


if __name__ == "__main__":
    """CLI interface for generating evaluation plots."""
    import argparse
    import joblib
    
    parser = argparse.ArgumentParser(description="Generate evaluation plots for shock prediction model")
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
        help="Output directory for plots"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="SHOCK",
        help="Target variable name (default: SHOCK)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="RandomForest",
        help="Model name for plot titles (default: RandomForest)"
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
    y_test = df[args.target]
    X_test = df.drop(columns=[args.target])
    
    print(f"Loaded {len(df)} samples with {X_test.shape[1]} features")
    
    # Generate plots
    plots = generate_all_plots(
        model, X_test, y_test,
        args.output,
        args.model_name
    )
    
    # Save plot paths to JSON
    plots_manifest = Path(args.output) / 'plots_manifest.json'
    with open(plots_manifest, 'w') as f:
        json.dump(plots, f, indent=2)
    
    print(f"\n✓ Plot manifest saved to: {plots_manifest}")
