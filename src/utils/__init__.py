"""
Utilities module.

Provides common utilities for configuration management, logging,
statistical computations, and plotting.
"""

from .config_manager import ConfigurationManager, get_config
from .logger import setup_logger, logger, log_section, log_subsection
from .stats import cramers_v, cramers_v_matrix, compute_correlation_matrix
from .data_loader import DataLoader, FileFormatSpec
from .plotting import (
    setup_plot_style,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance,
    plot_decision_tree_structure,
    plot_distribution,
    plot_boxplot_by_target,
    plot_count_by_target,
    plot_correlation_matrix,
    plot_calibration_curve,
    plot_target_distribution,
    plot_missing_values,
    plot_threshold_analysis,
    plot_performance_radar,
    plot_normalized_confusion_matrix,
    plot_cv_fold_boxplot,
    plot_prediction_bias,
    plot_model_comparison_bar,
    plot_model_comparison_grouped,
    plot_multi_roc_curve,
    plot_multi_calibration_curve,
    plot_performance_bars_single,
    plot_single_metrics_bars,
    plot_metric_pair_bar,
    save_figure
)

__all__ = [
    # Configuration
    'ConfigurationManager',
    'get_config',
    # Logging
    'setup_logger',
    'logger',
    'log_section',
    'log_subsection',
    # Statistics
    'cramers_v',
    'cramers_v_matrix',
    'compute_correlation_matrix',
    # Data I/O
    'DataLoader',
    'FileFormatSpec',
    # Plotting
    'setup_plot_style',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_feature_importance',
    'plot_decision_tree_structure',
    'plot_distribution',
    'plot_boxplot_by_target',
    'plot_count_by_target',
    'plot_correlation_matrix',
    'plot_calibration_curve',
    'plot_target_distribution',
    'plot_missing_values',
    'plot_threshold_analysis',
    'plot_performance_radar',
    'plot_normalized_confusion_matrix',
    'plot_cv_fold_boxplot',
    'plot_prediction_bias',
    'plot_model_comparison_bar',
    'plot_model_comparison_grouped',
    'plot_multi_roc_curve',
    'plot_multi_calibration_curve',
    'plot_performance_bars_single',
    'plot_single_metrics_bars',
    'plot_metric_pair_bar',
    'save_figure',
]
