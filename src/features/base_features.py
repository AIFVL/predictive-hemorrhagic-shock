"""
Base features module.

This module handles the original variables that are conserved for training.
All feature definitions are loaded from configuration manager.
"""

import pandas as pd
from typing import List, Dict

from src.utils import get_config, logger


def get_numerical_features() -> List[str]:
    """Get list of numerical features from config."""
    config = get_config()
    return list(config.get('features.numerical_features'))


def get_binary_features() -> List[str]:
    """Get list of binary features from config."""
    config = get_config()
    return list(config.get('features.binary_features'))


def get_target_variable() -> str:
    """Get target variable name from config."""
    config = get_config()
    target_name = config.get('features.target_name')
    if not target_name:
        raise ValueError("'features.target_name' not found in pipeline_config.yaml")
    return target_name


def get_excluded_variables() -> List[str]:
    """Get list of variables to exclude from config."""
    config = get_config()
    return list(config.get('cleaning.exclude_columns'))


def get_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the base features from the dataset based on config.
    
    This function:
    1. Loads feature definitions from config
    2. Selects only the numerical and binary features
    3. Ensures all required columns exist
    4. Does NOT create any new variables
    
    Args:
        df: DataFrame with all columns
    
    Returns:
        DataFrame with only base features
    """
    numerical_features = get_numerical_features()
    binary_features = get_binary_features()
    
    all_features = numerical_features + binary_features
    
    # Check for missing features
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in dataset: {missing}")
    
    # Select only base features
    df_features = df[all_features].copy()
    
    logger.info(f"Selected {len(all_features)} base features | Numerical: {len(numerical_features)} | Binary: {len(binary_features)}")
    
    return df_features


def get_target(df: pd.DataFrame) -> pd.Series:
    """
    Extract the target variable based on config.
    
    Args:
        df: DataFrame with target column
    
    Returns:
        Series with target values
    """
    target_variable = get_target_variable()
    
    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in dataset")
    
    return df[target_variable].copy()


def split_features_by_type(df: pd.DataFrame) -> dict:
    """
    Split features into numerical and binary groups based on config.
    
    Args:
        df: DataFrame with features
    
    Returns:
        Dict with keys 'numerical', 'binary' containing column lists
    """
    numerical_features = get_numerical_features()
    binary_features = get_binary_features()
    
    return {
        'numerical': [c for c in numerical_features if c in df.columns],
        'binary': [c for c in binary_features if c in df.columns]
    }

