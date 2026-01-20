"""
Machine learning models for hemorrhagic shock prediction.

Provides training and evaluation modules with clinical-appropriate
metrics and strategies.
"""

from .train import (
    train_model,
    cross_validate_model,
    get_feature_importance,
    save_model,
    load_model,
    get_model,
    create_pipeline,
    MODEL_CONFIGS
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
    'get_feature_importance',
    'save_model',
    'load_model',
    'get_model',
    'create_pipeline',
    'MODEL_CONFIGS',
    # Evaluation
    'evaluate_model',
    'evaluate_predictions',
    'compare_models',
    'get_classification_report',
    'get_roc_curve_data',
    'get_pr_curve_data',
    'save_evaluation_results'
]
