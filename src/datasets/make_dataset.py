"""
Dataset creation module.

This module builds the final training dataset by:
1. Loading cleaned data
2. Applying feature transformations (aggregations, categoricals)
3. Selecting final variables based on configuration
4. Saving the training-ready dataset

All feature definitions come from feature_config.yaml - the single source of truth.
"""

import pandas as pd
from pathlib import Path
from typing import List, Tuple

from src.utils import logger, get_config, log_section, DataLoader
from src.features.base_features import get_target, get_target_variable
from src.features.aggregations import create_all_aggregations
from src.features.categoricals import create_all_categoricals


def make_training_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create the final training dataset with all features based on configuration.
    
    Args:
        df: Cleaned DataFrame (output from clean.py)
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    log_section("CREATING TRAINING DATASET FROM CONFIGURATION")
    
    # Start with original features
    df_features = df.copy()
    
    # Create aggregated features
    logger.info("1. Creating aggregated features from config...")
    agg_features = create_all_aggregations(df_features)
    df_features = pd.concat([df_features, agg_features], axis=1)
    
    # Create categorical features
    logger.info("2. Creating categorical features from config...")
    cat_features = create_all_categoricals(df_features)
    df_features = pd.concat([df_features, cat_features], axis=1)
    
    # Extract target
    logger.info("3. Extracting target variable from config...")
    target = get_target(df_features)
    
    # Select final features based on config
    logger.info("4. Selecting final features from config...")
    config = get_config()
    final_features = []
    final_features.extend(config.get('numerical_features'))
    final_features.extend(config.get('binary_features'))
    for name, cfg in config.get('aggregated_features').items():
        if cfg['enabled']:
            final_features.append(name)
    for name, cfg in config.get('categorical_features').items():
        if cfg['enabled']:
            final_features.append(name)
    
    available_features = [f for f in final_features if f in df_features.columns]
    missing_features = [f for f in final_features if f not in df_features.columns]
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    
    X = df_features[available_features].copy()
    y = target.copy()
    
    logger.info(f"Final feature set: {len(available_features)} features")
    logger.info(f"Dataset shape: {X.shape}")
    logger.info({
        "target_distribution": {
            "Class 0 (No Shock)": f"{(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)",
            "Class 1 (Shock)": f"{(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)"
        }
    })
    
    return X, y


def save_training_dataset(
    X: pd.DataFrame,
    y: pd.Series,
) -> str:
    """
    Save training dataset to disk.
    
    The target is included as a column in the saved file.
    
    Args:
        X: Features DataFrame
        y: Target Series
    Returns:
        Path to saved file
    """
    config = get_config()
    output_path = Path(config.get_path('training_data'))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get target variable name from config
    target_variable = get_target_variable()
    
    # Combine features and target
    df_final = X.copy()
    df_final[target_variable] = y
    
    DataLoader.save(df_final, output_path)
    
    logger.success(f"Training dataset saved to: {output_path}")
    logger.info(f"Shape: {df_final.shape}")
    
    return str(output_path)


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create stratified train/test split.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
        stratify: Whether to stratify by target
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    logger.info({
        "train_test_split": {
            "train": f"{len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)",
            "test": f"{len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)",
            "train_class_distribution": y_train.value_counts().to_dict(),
            "test_class_distribution": y_test.value_counts().to_dict()
        }
    })
    
    return X_train, X_test, y_train, y_test


def save_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Save train/test splits to separate files.
    
    Args:
        X_train, X_test: Feature DataFrames
        y_train, y_test: Target Series
    Returns:
        Dict with paths to saved files
    """
    config = get_config()
    output_dir = Path(config.get_path('splits_dir'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get target variable name from config
    target_variable = get_target_variable()
    
    # Combine features and target for each split
    train_df = X_train.copy()
    train_df[target_variable] = y_train
    
    test_df = X_test.copy()
    test_df[target_variable] = y_test
    
    paths = {
        'train': output_dir / 'train.parquet',
        'test': output_dir / 'test.parquet'
    }

    DataLoader.save(train_df, paths['train'])
    DataLoader.save(test_df, paths['test'])
    
    logger.success(f"Splits saved to: {output_dir}")
    logger.info({
        "saved_files": {
            "train": str(paths['train']),
            "test": str(paths['test'])
        }
    })
    
    return {k: str(v) for k, v in paths.items()}

