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
from typing import Optional, Tuple, List, Dict, Any

from .logger import logger


def setup_plot_style() -> None:
    """Configure matplotlib and seaborn plot styling."""
    from .config_manager import get_config

    viz = get_config().get('visualization')
    w, h = viz['figure_size']
    dpi = viz['dpi']
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (w, h)
    plt.rcParams['figure.dpi'] = dpi


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
    top_features = importance_df.head(top_n).copy()

    # Normalize to percentages (sum of displayed features = 100%)
    total = top_features['importance'].sum()
    top_features['pct'] = (
        top_features['importance'] / total * 100
        if total > 0 else top_features['importance'] * 0
    )

    # Create horizontal bar plot
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_features['importance'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')

    # Annotate each bar with its percentage
    x_max = top_features['importance'].max()
    for bar, pct in zip(bars, top_features['pct']):
        width = bar.get_width()
        offset = x_max * 0.01  # small gap from bar end
        ax.text(
            width + offset,
            bar.get_y() + bar.get_height() / 2,
            f'{pct:.1f}%',
            va='center',
            ha='left',
            fontsize=9
        )

    # Extend x-axis slightly so labels don't get clipped
    ax.set_xlim(right=x_max * 1.15)

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
    dpi: int = None,
    bbox_inches: str = 'tight'
) -> None:
    """
    Save a matplotlib figure to disk.
    
    Args:
        fig: Figure to save
        output_path: Path to save the figure
        dpi: Dots per inch (defaults to visualization.dpi from config)
        bbox_inches: Bounding box option
    """
    if dpi is None:
        from .config_manager import get_config

        dpi = get_config().get('visualization.dpi')
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


def plot_performance_radar(
    models_metrics: List[dict],
    title: str = "Performance Radar",
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Spider / radar chart comparing operating-point metrics across one or more models.

    Each entry in ``models_metrics`` must be a dict with keys:
        - ``label``       : str – display name of the model
        - ``recall``      : float  (Sensitivity)
        - ``specificity`` : float
        - ``precision``   : float
        - ``f1_score``    : float
        - ``accuracy``    : float
        - ``kappa``       : float  (Cohen's Kappa, range -1..1; clipped to 0..1 for display)

    Args:
        models_metrics: List of dicts, one per model.
        title: Plot title.
        figsize: Figure size (square recommended).

    Returns:
        matplotlib Figure object
    """
    import math

    AXES = [
        ("Recall\n(Sensitivity)", "recall"),
        ("Specificity", "specificity"),
        ("Precision", "precision"),
        ("F1-Score", "f1_score"),
        ("Accuracy", "accuracy"),
        ("Cohen's\nKappa", "kappa"),
    ]
    n_axes = len(AXES)
    angles = [i * 2 * math.pi / n_axes for i in range(n_axes)]
    angles += angles[:1]  # close polygon

    COLORS = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)

    # Draw outer grid rings at 0.2 intervals with value labels
    for ring in np.arange(0.2, 1.2, 0.2):
        ax.plot(angles, [ring] * (n_axes + 1), color="grey", linewidth=0.5,
                linestyle="--", alpha=0.5)
        ax.text(angles[0], ring + 0.02, f"{ring:.1f}",
                ha="center", va="bottom", fontsize=7, color="grey")

    # Set axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [label for label, _ in AXES],
        fontsize=10, fontweight="bold"
    )
    ax.set_yticks([])          # hide radial tick marks
    ax.set_ylim(0, 1)

    # Plot each model
    for idx, model in enumerate(models_metrics):
        color = COLORS[idx % len(COLORS)]
        values = []
        for _, key in AXES:
            raw = float(model.get(key, 0.0))
            # Kappa can be negative; clip to [0, 1] for radar display
            values.append(max(0.0, min(1.0, raw)))
        values += values[:1]  # close polygon

        ax.plot(angles, values, color=color, linewidth=2,
                linestyle="solid", label=model.get("label", f"Model {idx+1}"))
        ax.fill(angles, values, color=color, alpha=0.15)

        # Annotate exact metric value at each vertex
        for angle, val, (axis_label, key) in zip(angles[:-1], values[:-1], AXES):
            raw_val = model.get(key, 0.0)
            ax.annotate(
                f"{raw_val:.3f}",
                xy=(angle, val),
                xytext=(angle, val + 0.07),
                ha="center", va="center",
                fontsize=8,
                color=color,
                fontweight="bold",
            )

    ax.set_title(title, size=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    plt.tight_layout()
    logger.debug(f"Created performance radar plot: {title}")
    return fig


def plot_normalized_confusion_matrix(
    confusion_matrix_data: dict,
    operating_point: dict,
    model_name: str = "Model",
    title: str = None,
    figsize: Tuple[int, int] = (7, 6)
) -> plt.Figure:
    """
    Plot a row-normalized confusion matrix showing both % and raw count per cell.

    Args:
        confusion_matrix_data: dict with keys tp, fp, tn, fn.
        operating_point:       full operating_point dict (used to read threshold).
        model_name:            display name for the subtitle.
        title:                 override title (optional).
        figsize:               figure size.

    Returns:
        matplotlib Figure object
    """
    tn = confusion_matrix_data.get("tn", 0)
    fp = confusion_matrix_data.get("fp", 0)
    fn = confusion_matrix_data.get("fn", 0)
    tp = confusion_matrix_data.get("tp", 0)
    threshold = operating_point.get("threshold", "?")

    cm_arr  = np.array([[tn, fp], [fn, tp]], dtype=float)
    cm_norm = cm_arr / cm_arr.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row proportion")

    class_names = ["No Shock", "Shock"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)

    threshold_str = f"{threshold:.3f}" if isinstance(threshold, float) else str(threshold)
    ax.set_title(
        title or f"Normalized Confusion Matrix — {model_name}\n(threshold = {threshold_str})",
        fontsize=12, fontweight="bold"
    )

    raw_vals = [[tn, fp], [fn, tp]]
    thresh_color = cm_norm.max() / 2
    for i in range(2):
        for j in range(2):
            pct = cm_norm[i, j] * 100
            raw = int(raw_vals[i][j])
            color = "white" if cm_norm[i, j] > thresh_color else "black"
            ax.text(j, i, f"{pct:.1f}%\n(n={raw})",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color=color)

    plt.tight_layout()
    logger.debug(f"Created normalized confusion matrix: {model_name}")
    return fig


def plot_cv_fold_boxplot(
    scores: List[float],
    metric_name: str,
    model_name: str = "Model",
    color: str = "#2196F3",
    title: str = None,
    figsize: Tuple[int, int] = (7, 6)
) -> plt.Figure:
    """
    Boxplot of k-fold cross-validation scores for a single metric.
    Each individual fold score is shown as a labelled dot.

    Args:
        scores:      List of per-fold test scores.
        metric_name: Display name of the metric (e.g. "F2-Score").
        model_name:  Model display name.
        color:       Color for the box and annotations.
        title:       Override title (optional).
        figsize:     Figure size.

    Returns:
        matplotlib Figure object
    """
    scores = list(scores)
    mean_v = float(np.mean(scores))
    std_v  = float(np.std(scores))

    fig, ax = plt.subplots(figsize=figsize)

    ax.boxplot(
        scores, positions=[1], widths=0.45,
        patch_artist=True, notch=False,
        boxprops=dict(facecolor=color, alpha=0.30),
        medianprops=dict(color=color, linewidth=2.5),
        whiskerprops=dict(linestyle="--", linewidth=1.2),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor=color, markersize=6),
    )

    # Mean dashed line
    ax.axhline(mean_v, color=color, linewidth=1.5, linestyle="--",
               label=f"Mean = {mean_v:.3f}")

    # Jittered individual fold dots + labels
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.06, 0.06, len(scores))
    for i, (s, j) in enumerate(zip(scores, jitter)):
        ax.scatter(1 + j, s, color=color, zorder=6, s=55, alpha=0.9)
        ax.annotate(
            f"fold {i+1}: {s:.3f}",
            xy=(1 + j, s),
            xytext=(1.28, s),
            fontsize=8, color="dimgrey",
            arrowprops=dict(arrowstyle="-", color="lightgrey", lw=0.7),
        )

    # Stats box (top-right)
    ax.text(
        0.97, 0.97,
        f"μ = {mean_v:.3f}\nσ = {std_v:.3f}\nn_folds = {len(scores)}",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=color, alpha=0.8),
    )

    ax.set_title(
        title or f"{metric_name} — {len(scores)}-Fold CV\n{model_name}",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_xticks([1])
    ax.set_xticklabels(["CV Folds"])
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xlim(0.5, 1.9)

    plt.tight_layout()
    logger.debug(f"Created CV fold boxplot: {metric_name} — {model_name}")
    return fig


def plot_prediction_bias(
    confusion_matrix_data: dict,
    class_distribution: dict,
    operating_point: dict,
    model_name: str = "Model",
    title: str = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Bar chart comparing actual class distribution (from training set)
    against the model's predicted distribution at the operating threshold.

    Args:
        confusion_matrix_data: dict with tp, fp, tn, fn.
        class_distribution:    dict from metadata (keys '0'/'0.0' and '1'/'1.0').
        operating_point:       full operating_point dict (used for threshold label).
        model_name:            display name.
        title:                 override title (optional).
        figsize:               figure size.

    Returns:
        matplotlib Figure object
    """
    def _get_class(d, target):
        for k in d:
            if float(k) == float(target):
                return d[k]
        return 0

    tn = confusion_matrix_data.get("tn", 0)
    fp = confusion_matrix_data.get("fp", 0)
    fn = confusion_matrix_data.get("fn", 0)
    tp = confusion_matrix_data.get("tp", 0)
    threshold = operating_point.get("threshold", "?")

    actual_neg = _get_class(class_distribution, 0)
    actual_pos = _get_class(class_distribution, 1)
    actual_total = actual_neg + actual_pos

    pred_neg = tn + fn
    pred_pos = tp + fp
    pred_total = pred_neg + pred_pos

    actual_pcts = [
        actual_neg / actual_total * 100 if actual_total else 0,
        actual_pos / actual_total * 100 if actual_total else 0,
    ]
    pred_pcts = [
        pred_neg / pred_total * 100 if pred_total else 0,
        pred_pos / pred_total * 100 if pred_total else 0,
    ]

    categories = ["No Shock (0)", "Shock (1)"]
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    threshold_str = f"{threshold:.3f}" if isinstance(threshold, float) else str(threshold)

    bars_a = ax.bar(x - width / 2, actual_pcts, width,
                    label="Actual (train set)", color=["#42A5F5", "#EF5350"], alpha=0.85)
    bars_p = ax.bar(x + width / 2, pred_pcts, width,
                    label=f"Predicted (thr={threshold_str})",
                    color=["#1565C0", "#B71C1C"], alpha=0.85)

    actual_counts = [actual_neg, actual_pos]
    pred_counts   = [pred_neg,   pred_pos]
    for bar, pct, cnt in zip(bars_a, actual_pcts, actual_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%\n(n={cnt})", ha="center", va="bottom", fontsize=9)
    for bar, pct, cnt in zip(bars_p, pred_pcts, pred_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%\n(n={cnt})", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Proportion (%)", fontsize=11)
    ax.set_title(
        title or f"Prediction Bias: Actual vs Predicted — {model_name}",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(max(actual_pcts), max(pred_pcts)) * 1.30)

    plt.tight_layout()
    logger.debug(f"Created prediction bias chart: {model_name}")
    return fig


def plot_model_comparison_bar(
    models_data: List[dict],
    metric_key: str,
    metric_label: str,
    title: str = None,
    color: str = "#2196F3",
    figsize: Tuple[int, int] = (9, 6)
) -> plt.Figure:
    """
    Horizontal bar chart comparing a single metric across all trained models.

    Each entry in ``models_data`` must contain:
        - ``label``         : str  – model display name
        - ``<metric_key>``  : float – the metric value to plot
        - ``std`` (optional): float – std dev to draw as error bar

    Args:
        models_data:  List of dicts, one per model.
        metric_key:   Key to read from each dict (e.g. "recall").
        metric_label: Human-readable metric name for axis label.
        title:        Override title (optional).
        color:        Bar color.
        figsize:      Figure size.

    Returns:
        matplotlib Figure object
    """
    labels = [m["label"] for m in models_data]
    values = [float(m.get(metric_key, 0.0)) for m in models_data]
    errors = [float(m.get("std", 0.0)) for m in models_data]

    # Sort by value descending
    order = sorted(range(len(values)), key=lambda i: values[i])
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    errors = [errors[i] for i in order]

    COLORS = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0",
              "#00BCD4", "#FF5722", "#607D8B"]
    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, xerr=errors if any(e > 0 for e in errors) else None,
                   color=bar_colors, alpha=0.85, capsize=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel(metric_label, fontsize=11)
    ax.set_title(title or f"Model Comparison — {metric_label}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Annotate exact value on each bar
    x_max = max(values) if values else 1.0
    for bar, val, err in zip(bars, values, errors):
        txt = f"{val:.3f}"
        if err > 0:
            txt += f" ±{err:.3f}"
        ax.text(
            bar.get_width() + x_max * 0.01,
            bar.get_y() + bar.get_height() / 2,
            txt, va="center", ha="left", fontsize=9, fontweight="bold"
        )

    ax.set_xlim(right=x_max * 1.20)
    plt.tight_layout()
    logger.debug(f"Created model comparison bar: {metric_label}")
    return fig


def plot_model_comparison_grouped(
    models_data: List[dict],
    metrics: List[Tuple[str, str]],
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (14, 10),
    std_suffix: str = "_std",
) -> plt.Figure:
    """
    2×2 grid where each subplot shows a vertical bar chart for ONE metric,
    with one bar per model.

    Args:
        models_data: List of dicts, one per model.  Each dict must have keys
                     for every metric in ``metrics`` plus optional
                     ``<key>_std`` entries.
        metrics:     Exactly 4 (metric_key, metric_label) pairs — one per
                     subplot cell in reading order (top-left → top-right →
                     bottom-left → bottom-right).  Fewer than 4 are allowed;
                     extra cells are hidden.
        title:       Overall figure suptitle.
        figsize:     Figure size.
        std_suffix:  Suffix used to look up std-dev values in each row.

    Returns:
        matplotlib Figure object.
    """
    COLORS = [
        "#2196F3", "#E91E63", "#4CAF50", "#FF9800",
        "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
    ]
    model_labels = [m["label"] for m in models_data]
    n_models     = len(models_data)
    n_metrics    = len(metrics)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes_flat = axes.flatten().tolist()

    bar_width = 0.8 / max(n_models, 1)
    offsets   = (np.arange(n_models) - (n_models - 1) / 2.0) * bar_width

    for m_idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes_flat[m_idx]

        for i, (model_row, model_label) in enumerate(zip(models_data, model_labels)):
            color = COLORS[i % len(COLORS)]
            val   = float(model_row.get(metric_key, 0.0))

            ax.bar(
                offsets[i],
                val,
                width=bar_width * 0.92,
                label=model_label,
                color=color,
                alpha=0.85,
            )

            # White dot at the top of each bar
            ax.plot(offsets[i], val, marker="o", color="white", markersize=5, zorder=5)

            # Horizontal value label above the bar
            ax.text(
                offsets[i],
                val + 0.012,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold",
            )

        ax.set_xticks(offsets)
        ax.set_xticklabels(model_labels, fontsize=9)
        ax.set_xlim(offsets[0] - bar_width, offsets[-1] + bar_width)
        ax.set_ylim(0, 1.22)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    # Hide unused cells if fewer than 4 metrics provided
    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    logger.debug(f"Created 2×2 per-metric comparison chart: {title}")
    return fig


def plot_multi_roc_curve(
    roc_data: List[Dict[str, Any]],
    title: str = 'Model Comparison - ROC Curve (CV Mean)',
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot overlaid ROC curves for multiple models.

    Args:
        roc_data: List of dicts, each with 'label' (model name), 'fpr' (mean FPR), 
                  'tpr' (mean TPR), and 'auc' (mean ROC AUC)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for item in roc_data:
        label = f"{item['label']} (AUC = {item['auc']:.4f})"
        ax.plot(item['fpr'], item['tpr'], lw=2, label=label)
        
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 0.05))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    return fig


def plot_multi_calibration_curve(
    calibration_data: List[Dict[str, Any]],
    title: str = 'Model Comparison - Calibration Curve (CV)',
    n_bins: int = 10,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot overlaid calibration curves for multiple models.

    Args:
        calibration_data: List of dicts, each with 'label', 'y_true', and 'y_prob'
    """
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', lw=2)
    
    for item in calibration_data:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            item['y_true'], item['y_prob'], n_bins=n_bins
        )
        ax.plot(mean_predicted_value, fraction_of_positives, 's-', lw=2, label=item['label'])
        
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    return fig
