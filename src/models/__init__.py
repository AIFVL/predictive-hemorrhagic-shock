"""
Machine learning models for hemorrhagic shock prediction.

Provides training and evaluation modules with clinical-appropriate
metrics and strategies.
"""

from .train import (
    train_model,
    cross_validate_model,
    save_model,
    load_model,
    get_model_from_config,
    create_pipeline,
    train_all_models
)

from .evaluate import (
    evaluate_model,
    evaluate_predictions,
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
    'save_model',
    'load_model',
    'get_model_from_config',
    'create_pipeline',
    'train_all_models',
    # Evaluation
    'evaluate_model',
    'evaluate_predictions',
    'compare_models',
    'get_classification_report',
    'get_roc_curve_data',
    'get_pr_curve_data',
    'save_evaluation_results'
]
