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

from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, fbeta_score, cohen_kappa_score

from src.utils import logger, get_config, log_section


def get_model_from_config(model_config: Dict, model_name: str = None) -> Any:
    """
    Dynamically load and instantiate a model from configuration.

    Parameters are derived from config['models'][model_name]['params']
    (fallback: first value from each search_space list). If model_name is not provided,
    falls back to an empty params dict.
    """
    module = importlib.import_module(model_config["module"])
    model_class = getattr(module, model_config["class"])

    if model_name is not None:
        try:
            cfg = get_config()
            model_cfg = cfg.get(f'models.{model_name}', {})
            params = model_cfg.get('params')
            if params is None:
                search_space = model_cfg['search_space']
                params = {k: v[0] for k, v in search_space.items() if isinstance(v, list) and v}
        except Exception:
            params = {}
    else:
        params = {}

    return model_class(**params)


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

    config = get_config()

    # Optional pruning of pathological features (must run before association rules / scaling / model)
    pruning_config = config.get('feature_pruning')
    if pruning_config.get('enabled'):
        from src.features.pruning import RareBinaryFeaturePruner

        steps.append(
            (
                'prune_features',
                RareBinaryFeaturePruner(
                    binary_columns=config.get('features.binary_features'),
                    min_total_ones=pruning_config['min_total_ones'],
                ),
            )
        )

    # Association-rule feature generation removed (kept in repo but disabled in pipeline)
    
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
    model = get_model_from_config(model_config, model_name)
    pipeline = create_pipeline(model, scale_features)
    
    # Train
    start_time = datetime.now()
    pipeline.fit(X, y)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Metadata
    metadata = {
        'model_name': model_name,
        'pipeline_version': get_config().get('general_config.version'),
        'dataset_version': get_config().get('general_config.dataset_version'),
        'n_samples': len(X),
        'n_features': X.shape[1],
        'feature_names': list(X.columns),
        'class_distribution': y.value_counts().to_dict(),
        'training_time_seconds': training_time,
        'timestamp': datetime.now().isoformat(),
        'scale_features': scale_features
    }

    # Feature pruning metadata (if enabled)
    if hasattr(pipeline, 'named_steps') and 'prune_features' in pipeline.named_steps:
        pruner = pipeline.named_steps['prune_features']
        if hasattr(pruner, 'get_dropped_columns'):
            dropped = pruner.get_dropped_columns()
            metadata['feature_pruning'] = {
                'enabled': True,
                'min_total_ones': getattr(pruner, 'min_total_ones', None),
                'n_dropped': len(dropped),
                'dropped_columns': dropped,
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
    n_folds: int = 10,
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
    model = get_model_from_config(model_config, model_name)
    pipeline = create_pipeline(model, scale_features)
    
    # Define CV strategy
    config = get_config()
    shuffle = config.get('data_split.shuffle')
    cv = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'f2': make_scorer(fbeta_score, beta=2),
        'kappa': make_scorer(cohen_kappa_score)
    }
    
    # Perform CV
    start_time = datetime.now()
    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=config.get('cross_validation.n_jobs')
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
            "recall": f"{results['metrics']['recall']['test_mean']:.3f} ± {results['metrics']['recall']['test_std']:.3f}",
            "kappa": f"{results['metrics']['kappa']['test_mean']:.3f} ± {results['metrics']['kappa']['test_std']:.3f}"
        }
    })
    
    return results


def optimize_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    model_config: Dict,
    model_name: str,
    scale_features: bool = True
) -> Tuple[Dict, Dict]:
    """
    Optimize hyperparameters using RandomizedSearchCV.
    
    Args:
        X: Features DataFrame
        y: Target Series
        model_config: Model configuration with module, class
        model_name: Name of the model
        scale_features: Whether to scale features
    
    Returns:
        Tuple of (best_params dict, search_results dict)
    """
    config = get_config()
    search_config = config.get('hyperparameter_search')
    model_cfg = config.get(f'models.{model_name}')
    
    if not search_config.get('enabled'):
        logger.info(f"Hyperparameter search disabled, using static params for {model_name}")
        params = model_cfg.get('params')
        if params is None:
            search_space = model_cfg.get('search_space')
            params = {k: v[0] for k, v in search_space.items() if isinstance(v, list) and v}
        return params, {}
    
    log_section(f"OPTIMIZING HYPERPARAMETERS: {model_name.upper()}")
    
    # Get search space for this model (stored per-model in config)
    param_grid = model_cfg.get('search_space')

    if not param_grid:
        logger.warning(f"No search space defined for {model_name}, using static params")
        params = model_cfg.get('params')
        if params is None:
            params = {k: v[0] for k, v in param_grid.items() if isinstance(v, list) and v}
        return params, {}
    
    # Create base model (without params)
    module = importlib.import_module(model_config["module"])
    model_class = getattr(module, model_config["class"])
    
    # random_state is now always declared in search_space; skip the auto-inject
    
    # Create pipeline parameter grid (prefix with 'classifier__')
    pipeline_param_grid = {f'classifier__{k}': v for k, v in param_grid.items()}
    
    # Create base pipeline
    base_model = model_class()
    pipeline = create_pipeline(base_model, scale_features)
    
    # Setup CV strategy
    shuffle = config.get('data_split.shuffle')
    cv = StratifiedKFold(
        n_splits=search_config['cv_folds'],
        shuffle=shuffle,
        random_state=config.get('general_config.random_seed')
    )
    
    # Setup scoring
    scoring = search_config['scoring']
    if scoring == 'f2':
        scoring = make_scorer(fbeta_score, beta=2)
    
    # Perform randomized search
    logger.info(f"Starting RandomizedSearchCV with {search_config['n_iter']} iterations...")
    logger.info(f"Search space: {len(param_grid)} parameters")
    
    start_time = datetime.now()
    
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=pipeline_param_grid,
        n_iter=search_config['n_iter'],
        cv=cv,
        scoring=scoring,
        n_jobs=search_config['n_jobs'],
        verbose=search_config['verbose'],
        random_state=config.get('general_config.random_seed'),
        return_train_score=True
    )
    
    search.fit(X, y)
    
    search_time = (datetime.now() - start_time).total_seconds()
    
    # Extract best parameters (remove 'classifier__' prefix)
    best_params = {
        k.replace('classifier__', ''): v 
        for k, v in search.best_params_.items()
    }
    
    # Compile results
    results = {
        'best_score': float(search.best_score_),
        'best_params': best_params,
        'n_iterations': search_config['n_iter'],
        'cv_folds': search_config['cv_folds'],
        'search_time_seconds': search_time,
        'all_scores': search.cv_results_['mean_test_score'].tolist(),
        'best_index': int(search.best_index_)
    }
    
    logger.success(f"Hyperparameter optimization completed in {search_time:.2f}s")
    logger.info({
        "best_score": f"{search.best_score_:.4f}",
        "best_params": best_params
    })
    
    return best_params, results


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
        from src.utils import DataLoader
        metadata_path = output_path.parent / 'metadata.json'
        DataLoader.save(metadata, metadata_path)

        # Association rules CSV export removed (feature disabled)
    
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


def update_model_metadata(
    model_name: str,
    updates: Dict
) -> None:
    """
    Update model metadata with new information (e.g., optimal threshold).
    
    Args:
        model_name: Name of the model
        updates: Dictionary with new metadata fields to add/update
    """
    from src.utils import DataLoader
    
    config = get_config()
    model_dir = Path(config.get_path('model_output', model_name=model_name)).parent
    metadata_path = model_dir / 'metadata.json'
    
    if not metadata_path.exists():
        logger.warning(f"Model metadata file not found at {metadata_path}")
        return
    
    # Load existing metadata
    metadata = DataLoader.load(metadata_path)
    
    # Update with new fields
    metadata.update(updates)
    
    # Save updated metadata
    DataLoader.save(metadata, metadata_path)
    logger.info(f"Model metadata updated with: {list(updates.keys())}")
    logger.debug(f"Updated metadata saved to: {metadata_path}")


def train_complete_workflow(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
) -> Tuple[Pipeline, Dict]:
    """
    Complete training workflow: optimize hyperparams → cross-validate → train.
    
    This is the encapsulated business logic for training. Called by step5_train_model.
    
    Args:
        X: Training features
        y: Training target
        model_name: Name of the model
    
    Returns:
        Tuple of (trained pipeline, complete metadata) without saving to disk.
        The steps layer decides whether/where to save.
    
    Logs internally:
        - Hyperparameter optimization progress and results
        - CV statistics
        - Training completion
    """
    config = get_config()
    
    log_section(f"TRAINING WORKFLOW: {model_name.upper()}")
    
    # Step 1: Optimize hyperparameters
    model_config = config.get(f'models.{model_name}')
    cv_config = config.get('cross_validation')
    scale_features = cv_config['scale_features']
    
    logger.info(f"Training set: {len(y)} samples, {int(y.sum())} positives ({y.mean():.1%})")
    
    best_params, search_results = optimize_hyperparameters(
        X,
        y,
        model_config=model_config,
        model_name=model_name,
        scale_features=scale_features,
    )
    
    # Step 2: Cross-validate with optimized params
    optimized_model_config = model_config.copy()
    optimized_model_config['params'] = best_params
    
    cv_results = cross_validate_model(
        X,
        y,
        model_config=optimized_model_config,
        model_name=model_name,
        n_folds=cv_config['n_folds'],
        scale_features=scale_features,
        random_state=config.get('general_config.random_seed'),
    )
    
    # Step 3: Train final model on full training set
    pipeline, metadata = train_model(
        X,
        y,
        model_config=optimized_model_config,
        model_name=model_name,
        scale_features=scale_features,
    )
    
    # Aggregate all metadata
    metadata['cv_results'] = cv_results
    metadata['hyperparameter_search'] = search_results
    metadata['optimized_params'] = best_params
    
    logger.success(f"Training workflow completed for {model_name}")
    
    return pipeline, metadata

