"""
Visualization module for model evaluation.

Generates evaluation plots for shock prediction model assessment.
All plotting logic is delegated to src.utils.plotting functions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union

from sklearn.metrics import (
    roc_curve, 
    auc, 
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)

from src.utils import (
    logger,
    log_section,
    get_config,
    setup_plot_style,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance,
    plot_calibration_curve,
    plot_threshold_analysis,
    save_figure
)


def generate_all_plots(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, str]:
    """
    Generate all evaluation plots.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
    
    Returns:
        Dict with paths to all generated plots
    """
    config = get_config()
    # Use model_name in path
    base_plots_dir = Path(config.get_path('eval_plots'))
    output_dir = base_plots_dir / model_name.lower().replace(' ', '_')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_plot_style()
    log_section(f"GENERATING EVALUATION PLOTS FOR {model_name.upper()}")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    plots = {}
    
    # 1. ROC Curve
    logger.info("Creating ROC curve...")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig = plot_roc_curve(fpr, tpr, roc_auc, title=f"ROC Curve - {model_name}")
    plots['roc_curve'] = output_dir / 'roc_curve.png'
    save_figure(fig, plots['roc_curve'])
    
    # 2. Precision-Recall Curve
    logger.info("Creating Precision-Recall curve...")
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    fig = plot_precision_recall_curve(recall, precision, avg_precision, 
                                      title=f"Precision-Recall Curve - {model_name}")
    plots['pr_curve'] = output_dir / 'precision_recall_curve.png'
    save_figure(fig, plots['pr_curve'])
    
    # 3. Confusion Matrix
    logger.info("Creating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    fig = plot_confusion_matrix(cm, class_names=['No Shock', 'Shock'], 
                                title=f"Confusion Matrix - {model_name}")
    plots['confusion_matrix'] = output_dir / 'confusion_matrix.png'
    save_figure(fig, plots['confusion_matrix'])
    
    # 4. Feature Importance
    logger.info("Creating feature importance plot...")
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig = plot_feature_importance(importance_df, top_n=20, title=f"Feature Importance - {model_name}")
        plots['feature_importance'] = output_dir / 'feature_importance.png'
        save_figure(fig, plots['feature_importance'])
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('classifier'), 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.named_steps['classifier'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig = plot_feature_importance(importance_df, top_n=20, title=f"Feature Importance - {model_name}")
        plots['feature_importance'] = output_dir / 'feature_importance.png'
        save_figure(fig, plots['feature_importance'])
    else:
        logger.warning("Model does not support feature importances")
    
    # 5. Threshold Analysis
    logger.info("Creating threshold analysis...")
    fig = plot_threshold_analysis(y_test, y_prob, title="Metrics vs Decision Threshold")
    plots['threshold_analysis'] = output_dir / 'threshold_analysis.png'
    save_figure(fig, plots['threshold_analysis'])
    
    # 6. Calibration Curve
    logger.info("Creating calibration curve...")
    fig = plot_calibration_curve(y_test, y_prob, n_bins=10,
                                 title=f"Calibration Curve - {model_name}")
    plots['calibration_curve'] = output_dir / 'calibration_curve.png'
    save_figure(fig, plots['calibration_curve'])
    
    logger.success(f"Generated {len(plots)} evaluation plots in {output_dir}")
    
    return {k: str(v) for k, v in plots.items()}
