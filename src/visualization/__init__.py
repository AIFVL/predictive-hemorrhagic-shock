"""
Visualization module for shock prediction model evaluation.

Provides clinical-appropriate plots and analysis visualizations.
"""

from .generate_plots import (
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_threshold_analysis,
    plot_calibration_curve,
    generate_all_plots
)

from .generate_eda_plots import generate_all_eda_plots

__all__ = [
    # Model evaluation plots
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_confusion_matrix',
    'plot_feature_importance',
    'plot_threshold_analysis',
    'plot_calibration_curve',
    'generate_all_plots',
    # EDA plots
    'generate_all_eda_plots'
]
