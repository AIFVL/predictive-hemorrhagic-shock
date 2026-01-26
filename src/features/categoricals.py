"""
Categorical features module.

This module creates categorical features based purely on configuration
from feature_config.yaml. No hardcoded categorizations.

All categorization logic is driven by the YAML configuration - the single source of truth.
"""

import pandas as pd
import numpy as np
from typing import Dict
from src.utils import logger


def create_categorical_feature(df: pd.DataFrame, feature_name: str, feature_config: Dict) -> pd.Series:
    """
    Create a single categorical feature by binning a continuous variable.
    
    Args:
        df: DataFrame with source column
        feature_name: Name of the categorical feature to create
        feature_config: Configuration dict with 'source_variable', 'thresholds', 'categories'
    
    Returns:
        Series with categorical values
    """
    source_col = feature_config.get('source_variable')
    
    if not source_col:
        raise ValueError(f"No 'source_variable' defined for {feature_name}")
    
    if source_col not in df.columns:
        raise ValueError(f"Source column '{source_col}' not found in dataframe for {feature_name}")
    
    # Get thresholds and build bins
    thresholds = feature_config.get('thresholds', [])
    if not thresholds:
        raise ValueError(f"No 'thresholds' defined for {feature_name}")
    
    # Build bins: [0, threshold1, threshold2, ..., inf]
    bins = [0] + thresholds + [float('inf')]
    
    # Get category values from the categories list
    categories = feature_config.get('categories', [])
    if not categories:
        raise ValueError(f"No 'categories' defined for {feature_name}")
    
    # Extract values in order
    labels = [cat['value'] for cat in categories]
    
    # Use pd.cut for binning
    result = pd.cut(
        df[source_col],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True
    ).astype(int)
    
    return result


def create_all_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all categorical features defined in configuration.
    
    Iterates through all features in 'categorical_features' section of the config
    and creates them dynamically.
    
    Args:
        df: DataFrame with original continuous features
    
    Returns:
        DataFrame with only the new categorical columns
    """
    from src.utils import get_config
    
    categorical_configs = get_config().get_categorical_configs()
    
    # Return empty DataFrame if no categoricals defined
    if not categorical_configs:
        logger.info("No categorical features defined in config")
        return pd.DataFrame(index=df.index)
    
    categoricals = pd.DataFrame(index=df.index)
    
    logger.info("Creating categorical features from config...")
    
    for feature_name, feature_config in categorical_configs.items():
        try:
            categoricals[feature_name] = create_categorical_feature(df, feature_name, feature_config)
            logger.info(f"✓ {feature_name}: {categoricals[feature_name].value_counts().to_dict()}")
        except Exception as e:
            logger.error(f"✗ {feature_name}: Failed - {e}")
            raise
    
    return categoricals

