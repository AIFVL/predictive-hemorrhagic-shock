"""
Model training module.

Provides functionality for training classification models
for hemorrhagic shock prediction with clinical-appropriate
evaluation strategies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
import joblib
import importlib

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, fbeta_score

from src.utils import logger, get_config, log_section


def get_model_from_config(model_config: Dict) -> Any:
    """
    Dynamically load and instantiate a model from configuration.
    
    Args:
        model_config: Dictionary with 'module', 'class', and 'params' keys
    
    Returns:
        Instantiated model
    
    Example config:
        {
            "module": "sklearn.ensemble",
            "class": "RandomForestClassifier",
            "params": {"n_estimators": 100, "random_state": 42}
        }
    """
    module = importlib.import_module(model_config["module"])
    model_class = getattr(module, model_config["class"])
    return model_class(**model_config.get("params", {}))


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
    model_config: Dict,
    model_name: str,
    scale_features: bool = True
) -> Tuple[Pipeline, Dict]:
    """
    Train a model on the full dataset.
    
    Args:
        X: Features DataFrame
        y: Target Series
        model_config: Model configuration dict with 'module', 'class', 'params'
        model_name: Name of the model for logging
        scale_features: Whether to scale features
    
    Returns:
        Tuple of (trained pipeline, training metadata)
    """
    logger.info(f"Training model: {model_name}")
    
    # Get model from config
    model = get_model_from_config(model_config)
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
    
    logger.success(f"Model trained in {training_time:.2f}s")
    logger.info({
        "samples": metadata['n_samples'],
        "features": metadata['n_features'],
        "class_distribution": metadata['class_distribution']
    })
    
    return pipeline, metadata


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_config: Dict,
    model_name: str,
    n_folds: int = 5,
    scale_features: bool = True,
    random_state: int = 42
) -> Dict:
    """
    Perform stratified k-fold cross-validation.
    
    Args:
        X: Features DataFrame
        y: Target Series
        model_config: Model configuration dict
        model_name: Name of the model
        n_folds: Number of CV folds
        scale_features: Whether to scale features
        random_state: Random seed
    
    Returns:
        Dictionary with CV results
    """
    logger.info(f"Cross-validating {model_name} with {n_folds}-fold CV")
    
    # Get model and pipeline
    model = get_model_from_config(model_config)
    pipeline = create_pipeline(model, scale_features)
    
    # Define CV strategy
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'f2': make_scorer(fbeta_score, beta=2)
    }
    
    # Perform CV
    start_time = datetime.now()
    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    cv_time = (datetime.now() - start_time).total_seconds()
    
    # Aggregate results
    results = {
        'model_name': model_name,
        'n_folds': n_folds,
        'cv_time_seconds': cv_time,
        'metrics': {}
    }
    
    for metric in scoring.keys():
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        results['metrics'][metric] = {
            'test_mean': float(test_scores.mean()),
            'test_std': float(test_scores.std()),
            'train_mean': float(train_scores.mean()),
            'train_std': float(train_scores.std()),
            'test_scores': test_scores.tolist(),
            'train_scores': train_scores.tolist()
        }
    
    logger.success(f"CV completed in {cv_time:.2f}s")
    logger.info({
        "folds": n_folds,
        "metrics": {
            "accuracy": f"{results['metrics']['accuracy']['test_mean']:.3f} ± {results['metrics']['accuracy']['test_std']:.3f}",
            "f1": f"{results['metrics']['f1']['test_mean']:.3f} ± {results['metrics']['f1']['test_std']:.3f}",
            "recall": f"{results['metrics']['recall']['test_mean']:.3f} ± {results['metrics']['recall']['test_std']:.3f}"
        }
    })
    
    return results


def save_model(
    pipeline: Pipeline,
    model_name: str,
    metadata: Optional[Dict] = None
) -> str:
    """
    Save trained model and metadata.
    
    Args:
        pipeline: Trained sklearn Pipeline
        model_name: Name of the model (used to construct path)
        metadata: Optional metadata dictionary
    
    Returns:
        Path to saved model
    """
    config = get_config()
    output_path = Path(config.get_path('model_output', model_name=model_name))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(pipeline, output_path)
    
    # Save metadata if provided
    if metadata:
        from src.utils import DataWriter
        metadata_path = output_path.parent / 'metadata.json'
        DataWriter.write_json_file(metadata, str(metadata_path))
    
    logger.success(f"Model saved to: {output_path}")
    return str(output_path)


def load_model(model_name: str) -> Pipeline:
    """
    Load a trained model from disk.
    
    Args:
        model_name: Name of the model (used to construct path)
    
    Returns:
        Loaded sklearn Pipeline
    """
    config = get_config()
    model_path = Path(config.get_path('model_output', model_name=model_name))
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    pipeline = joblib.load(model_path)
    logger.info(f"Model loaded from: {model_path}")
    
    return pipeline


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Dict[str, Tuple[Pipeline, Dict]]:
    """
    Train all models defined in configuration.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Dictionary mapping model names to (pipeline, metadata) tuples
    """
    config = get_config()
    models_config = config.get('models', {})
    
    trained_models = {}
    
    log_section(f"TRAINING {len(models_config)} MODELS FROM CONFIGURATION")
    
    for model_name, model_config in models_config.items():
        log_section(f"Training model: {model_name}", width=50)
        
        pipeline, metadata = train_model(
            X_train, y_train,
            model_config,
            model_name,
            scale_features=model_config.get('scale_features', True)
        )
        
        trained_models[model_name] = (pipeline, metadata)
    
    logger.success(f"Trained {len(trained_models)} models successfully")
    return trained_models
