"""
Model training module.

Provides functionality for training classification models
for hemorrhagic shock prediction with clinical-appropriate
evaluation strategies.

Reference: docs/project_specification.md Section 8
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union
from datetime import datetime
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, fbeta_score


# Default model configurations
# OPTIMIZED FOR MAXIMUM RECALL (Sensitivity)
# Using aggressive class_weight to penalize false negatives heavily
MODEL_CONFIGS = {
    'random_forest': {
        'class': RandomForestClassifier,
        'params': {
            'n_estimators': 200,  # Increased for better ensemble
            'max_depth': 12,  # Slightly deeper to capture patterns
            'min_samples_split': 4,  # Allow more granular splits
            'min_samples_leaf': 1,  # Allow leaf nodes with few samples
            'class_weight': {0: 1, 1: 4},  # Aggressive: 4x penalty for FN vs FP
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'gradient_boosting': {
        'class': GradientBoostingClassifier,
        'params': {
            'n_estimators': 150,  # More estimators for recall
            'max_depth': 6,  # Deeper trees
            'learning_rate': 0.08,  # Slightly slower for better learning
            'min_samples_split': 4,
            'min_samples_leaf': 1,
            'random_state': 42
        }
    },
    'logistic_regression': {
        'class': LogisticRegression,
        'params': {
            'max_iter': 2000,
            'class_weight': {0: 1, 1: 4},  # Aggressive penalty
            'random_state': 42,
            'solver': 'lbfgs',
            'C': 0.5  # Stronger regularization
        }
    }
}


def get_model(model_name: str, custom_params: Optional[Dict] = None) -> Any:
    """
    Get a model instance by name.
    
    Args:
        model_name: Name of the model ('random_forest', 'gradient_boosting', 'logistic_regression')
        custom_params: Optional custom parameters to override defaults
    
    Returns:
        Model instance
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    params = config['params'].copy()
    
    if custom_params:
        params.update(custom_params)
    
    return config['class'](**params)


def create_pipeline(model: Any, scale_features: bool = True) -> Pipeline:
    """
    Create a sklearn Pipeline with optional scaling.
    
    Args:
        model: The classifier model
        scale_features: Whether to include StandardScaler
    
    Returns:
        sklearn Pipeline
    """
    steps = []
    
    if scale_features:
        steps.append(('scaler', StandardScaler()))
    
    steps.append(('classifier', model))
    
    return Pipeline(steps)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = 'random_forest',
    custom_params: Optional[Dict] = None,
    scale_features: bool = True
) -> Tuple[Pipeline, Dict]:
    """
    Train a model on the full dataset.
    
    Args:
        X: Features DataFrame
        y: Target Series
        model_name: Name of the model to train
        custom_params: Optional custom parameters
        scale_features: Whether to scale features
    
    Returns:
        Tuple of (trained pipeline, training metadata)
    """
    print("\n" + "="*60)
    print(f"TRAINING MODEL: {model_name.upper()}")
    print("="*60)
    
    # Get model and create pipeline
    model = get_model(model_name, custom_params)
    pipeline = create_pipeline(model, scale_features)
    
    # Train
    start_time = datetime.now()
    pipeline.fit(X, y)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Metadata
    metadata = {
        'model_name': model_name,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'feature_names': list(X.columns),
        'class_distribution': y.value_counts().to_dict(),
        'training_time_seconds': training_time,
        'timestamp': datetime.now().isoformat(),
        'scale_features': scale_features
    }
    
    # Get model parameters
    if hasattr(model, 'get_params'):
        metadata['model_params'] = model.get_params()
    
    print(f"\n✓ Model trained in {training_time:.2f}s")
    print(f"  Samples: {metadata['n_samples']}")
    print(f"  Features: {metadata['n_features']}")
    
    return pipeline, metadata


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = 'random_forest',
    custom_params: Optional[Dict] = None,
    n_folds: int = 5,
    scale_features: bool = True,
    random_state: int = 42
) -> Tuple[Dict, List[Pipeline]]:
    """
    Perform stratified k-fold cross-validation.
    
    Uses scoring metrics appropriate for clinical classification
    with class imbalance.
    
    Args:
        X: Features DataFrame
        y: Target Series
        model_name: Name of the model
        custom_params: Optional custom parameters
        n_folds: Number of CV folds
        scale_features: Whether to scale features
        random_state: Random seed
    
    Returns:
        Tuple of (CV results dict, list of trained pipelines)
    """
    print("\n" + "="*60)
    print(f"CROSS-VALIDATING: {model_name.upper()} ({n_folds}-fold)")
    print("="*60)
    
    # F2-score: beta=2 means recall is 2x more important than precision
    # This penalizes false negatives (missed shock cases) more heavily
    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    
    # Clinical-relevant metrics with F2 prioritizing recall
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',  # Sensitivity - CRITICAL for clinical
        'f1': 'f1',
        'f2': f2_scorer,  # Prioritizes recall over precision
        'roc_auc': 'roc_auc',
        'average_precision': 'average_precision'  # AUC-PR
    }
    
    # Create CV strategy
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Get model and create pipeline
    model = get_model(model_name, custom_params)
    pipeline = create_pipeline(model, scale_features)
    
    # Cross-validate
    start_time = datetime.now()
    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        return_estimator=True,
        n_jobs=-1
    )
    cv_time = (datetime.now() - start_time).total_seconds()
    
    # Extract trained pipelines
    trained_pipelines = cv_results['estimator']
    
    # Format results
    results = {
        'model_name': model_name,
        'n_folds': n_folds,
        'cv_time_seconds': cv_time,
        'metrics': {}
    }
    
    print(f"\nResults ({n_folds}-fold CV):")
    print("-" * 40)
    
    for metric in scoring.keys():
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        results['metrics'][metric] = {
            'test_mean': test_scores.mean(),
            'test_std': test_scores.std(),
            'test_scores': test_scores.tolist(),
            'train_mean': train_scores.mean(),
            'train_std': train_scores.std()
        }
        
        print(f"  {metric:20s}: {test_scores.mean():.4f} ± {test_scores.std():.4f}")
    
    print(f"\nCV completed in {cv_time:.2f}s")
    
    return results, trained_pipelines


def get_feature_importance(
    pipeline: Pipeline,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract feature importance from a trained pipeline.
    
    Args:
        pipeline: Trained sklearn Pipeline
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importances sorted descending
    """
    model = pipeline.named_steps['classifier']
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not have feature importance attributes")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df


def save_model(
    pipeline: Pipeline,
    output_path: Union[str, Path],
    metadata: Optional[Dict] = None
) -> str:
    """
    Save trained model to disk.
    
    Args:
        pipeline: Trained sklearn Pipeline
        output_path: Path to save the model
        metadata: Optional metadata to save alongside
    
    Returns:
        Path to saved model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(pipeline, output_path)
    
    # Save metadata if provided
    if metadata:
        metadata_path = output_path.with_suffix('.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    print(f"\n✓ Model saved to: {output_path}")
    return str(output_path)


def load_model(model_path: Union[str, Path]) -> tuple[Pipeline, dict]:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        Tuple of (trained sklearn Pipeline, metadata dict)
    """
    model_path = Path(model_path)
    pipeline = joblib.load(model_path)
    
    # Load metadata from accompanying JSON file
    metadata_path = model_path.with_suffix('.json')
    metadata = {}
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return pipeline, metadata


if __name__ == "__main__":
    """CLI interface for model training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train shock prediction model")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to training data file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to save trained model"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="random_forest",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model type (default: random_forest)"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="SHOCK",
        help="Target variable name (default: SHOCK)"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Disable feature scaling"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading training data...")
    if args.input.endswith('.parquet'):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    
    # Prepare features and target
    y = df[args.target]
    X = df.drop(columns=[args.target])
    
    print(f"Loaded {len(df)} samples with {X.shape[1]} features")
    
    # Cross-validate
    cv_results, _ = cross_validate_model(
        X, y,
        model_name=args.model,
        n_folds=args.cv_folds,
        scale_features=not args.no_scale,
        random_state=args.seed
    )
    
    # Train final model on all data
    pipeline, metadata = train_model(
        X, y,
        model_name=args.model,
        scale_features=not args.no_scale
    )
    
    # Add CV results to metadata
    metadata['cv_results'] = cv_results
    
    # Save model
    save_model(pipeline, args.output, metadata)
    
    print("\n✓ Training complete")
