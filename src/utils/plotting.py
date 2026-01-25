"""
Plotting utilities for data visualization.

Contains individual plotting functions that can be reused across
EDA and evaluation visualization modules.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List

from .logger import logger


def setup_plot_style() -> None:
    """Configure matplotlib and seaborn plot styling."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 300


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    cmap: str = 'Blues',
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: Names for classes
        normalize: Whether to normalize values
        title: Plot title
        cmap: Colormap name
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm, annot=True, fmt='.2f' if normalize else 'd',
        cmap=cmap, square=True, ax=ax,
        xticklabels=class_names or ['0', '1'],
        yticklabels=class_names or ['0', '1'],
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title)
    
    plt.tight_layout()
    logger.debug(f"Created confusion matrix plot: {title}")
    
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    title: str = 'ROC Curve',
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: Area under curve
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    logger.debug(f"Created ROC curve plot: {title}")
    
    return fig


def plot_precision_recall_curve(
    recall: np.ndarray,
    precision: np.ndarray,
    avg_precision: float,
    title: str = 'Precision-Recall Curve',
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        recall: Recall values
        precision: Precision values
        avg_precision: Average precision score
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})', linewidth=2)
    ax.axhline(y=precision[0], color='k', linestyle='--', 
               label='Baseline', linewidth=1)
    
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    logger.debug(f"Created precision-recall curve plot: {title}")
    
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = 'Feature Importance',
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Create horizontal bar plot
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features['importance'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    logger.debug(f"Created feature importance plot: {title}")
    
    return fig


def plot_distribution(
    data: pd.Series,
    title: str = 'Distribution',
    xlabel: str = None,
    bins: int = 30,
    kde: bool = True,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot distribution of a variable.
    
    Args:
        data: Series with data
        title: Plot title
        xlabel: X-axis label
        bins: Number of bins for histogram
        kde: Whether to show KDE overlay
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.histplot(data, bins=bins, kde=kde, ax=ax, stat='density')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel or data.name)
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    logger.debug(f"Created distribution plot: {title}")
    
    return fig


def plot_boxplot_by_target(
    df: pd.DataFrame,
    feature: str,
    target: str,
    title: str = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot boxplot of a feature grouped by target.
    
    Args:
        df: DataFrame with data
        feature: Feature column name
        target: Target column name
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.boxplot(data=df, x=target, y=feature, ax=ax)
    
    ax.set_title(title or f'{feature} by {target}')
    ax.set_xlabel(target)
    ax.set_ylabel(feature)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    logger.debug(f"Created boxplot: {feature} by {target}")
    
    return fig


def plot_count_by_target(
    df: pd.DataFrame,
    feature: str,
    target: str,
    title: str = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot count plot of a categorical feature grouped by target.
    
    Args:
        df: DataFrame with data
        feature: Feature column name
        target: Target column name
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.countplot(data=df, x=feature, hue=target, ax=ax)
    
    ax.set_title(title or f'{feature} by {target}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    ax.legend(title=target)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    logger.debug(f"Created count plot: {feature} by {target}")
    
    return fig


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    title: str = 'Correlation Matrix',
    cmap: str = 'coolwarm',
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot correlation matrix heatmap.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        title: Plot title
        cmap: Colormap name
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        center=0,
        square=True,
        ax=ax,
        cbar_kws={'label': 'Correlation'}
    )
    
    ax.set_title(title)
    plt.tight_layout()
    logger.debug(f"Created correlation matrix plot: {title}")
    
    return fig


def plot_calibration_curve(
    prob_true: np.ndarray,
    prob_pred: np.ndarray,
    title: str = 'Calibration Curve',
    n_bins: int = 10,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot calibration curve (reliability diagram).
    
    Args:
        prob_true: True probabilities
        prob_pred: Predicted probabilities
        title: Plot title
        n_bins: Number of bins for calibration
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    from sklearn.calibration import calibration_curve
    
    fig, ax = plt.subplots(figsize=figsize)
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        prob_true, prob_pred, n_bins=n_bins
    )
    
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    logger.debug(f"Created calibration curve plot: {title}")
    
    return fig


def save_figure(
    fig: plt.Figure,
    output_path: Path,
    dpi: int = 300,
    bbox_inches: str = 'tight'
) -> None:
    """
    Save a matplotlib figure to disk.
    
    Args:
        fig: Figure to save
        output_path: Path to save the figure
        dpi: Dots per inch
        bbox_inches: Bounding box option
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    
    logger.debug(f"Saved figure to: {output_path}")


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot target variable distribution with bar chart and pie chart.
    
    Args:
        df: DataFrame with data
        target_col: Target column name
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    counts = df[target_col].value_counts()
    
    # Bar chart
    axes[0].bar(counts.index.astype(str), counts.values, color=['#3498db', '#e74c3c'])
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Distribution of {target_col}')
    axes[0].grid(True, alpha=0.3)
    
    total = len(df)
    for i, (idx, val) in enumerate(counts.items()):
        pct = (val / total) * 100
        axes[0].text(i, val, f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    axes[1].pie(counts.values, labels=[f'Class {i}' for i in counts.index], 
               autopct='%1.1f%%', startangle=90, colors=['#3498db', '#e74c3c'])
    axes[1].set_title(f'Proportion of {target_col}')
    
    plt.tight_layout()
    logger.debug(f"Created target distribution plot: {target_col}")
    
    return fig


def plot_missing_values(
    df: pd.DataFrame,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8)
) -> Optional[plt.Figure]:
    """
    Plot missing values percentage.
    
    Args:
        df: DataFrame with data
        top_n: Number of top features with missing values to show
        figsize: Figure size
    
    Returns:
        matplotlib Figure object or None if no missing values
    """
    missing = df.isnull().sum()
    if missing.sum() == 0:
        return None
    
    missing_pct = (missing / len(df) * 100).sort_values(ascending=True)
    missing_pct = missing_pct[missing_pct > 0].tail(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    missing_pct.plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel('Missing Percentage (%)')
    ax.set_title(f'Top {top_n} Features with Missing Values')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    logger.debug(f"Created missing values plot")
    
    return fig


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Metrics vs Decision Threshold",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot metrics at different decision thresholds.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    specificities = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        f1_scores.append(f1)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(thresholds, precisions, label='Precision', linewidth=2)
    ax.plot(thresholds, recalls, label='Recall (Sensitivity)', linewidth=2)
    ax.plot(thresholds, specificities, label='Specificity', linewidth=2)
    ax.plot(thresholds, f1_scores, label='F1 Score', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Decision Threshold', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    logger.debug(f"Created threshold analysis plot: {title}")
    
    return fig
    """
    Save a matplotlib figure to disk.
    
    Args:
        fig: Figure to save
        output_path: Path to save the figure
        dpi: Dots per inch
        bbox_inches: Bounding box option
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    
    logger.debug(f"Saved figure to: {output_path}")
