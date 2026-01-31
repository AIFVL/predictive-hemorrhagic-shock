"""
Machine learning models for hemorrhagic shock prediction.

Provides training and evaluation modules with clinical-appropriate
metrics and strategies.
"""

from .train import (
    train_model,
    cross_validate_model,
    optimize_hyperparameters,
    save_model,
    load_model,
    update_model_metadata,
    get_model_from_config,
    create_pipeline,
)

from .evaluate import (
    evaluate_model,
    evaluate_predictions,
    analyze_thresholds,
    find_optimal_threshold_for_target_recall,
    compare_models,
    get_classification_report,
    get_roc_curve_data,
    get_pr_curve_data,
    save_evaluation_results
)

__all__ = [
    # Training
    'train_model',
    'cross_validate_model',
    'optimize_hyperparameters',
    'save_model',
    'load_model',
    'update_model_metadata',
    'get_model_from_config',
    'create_pipeline',
    # Evaluation
    'evaluate_model',
    'evaluate_predictions',
    'analyze_thresholds',
    'find_optimal_threshold_for_target_recall',
    'compare_models',
    'get_classification_report',
    'get_roc_curve_data',
    'get_pr_curve_data',
    'save_evaluation_results'
]
