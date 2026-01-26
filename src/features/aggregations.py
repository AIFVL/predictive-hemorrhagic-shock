"""
Feature aggregations module.

This module creates aggregated features based purely on configuration
from feature_config.yaml. No hardcoded aggregations.

All aggregation logic is driven by the YAML configuration - the single source of truth.
"""

import pandas as pd
from typing import Dict
from src.utils import logger


def create_aggregation(df: pd.DataFrame, feature_name: str, feature_config: Dict) -> pd.Series:
    """
    Create a single aggregated feature by summing components.
    
    Args:
        df: DataFrame with component columns
        feature_name: Name of the aggregated feature to create
        feature_config: Configuration dict with 'components' key
    
    Returns:
        Series with aggregated values (sum of components)
    """
    components = feature_config.get('components', [])
    
    if not components:
        raise ValueError(f"No components defined for {feature_name}")
    
    # Filter to components that exist in the dataframe
    available_components = [c for c in components if c in df.columns]
    
    if not available_components:
        raise ValueError(f"None of the components for {feature_name} found in dataframe: {components}")
    
    if len(available_components) < len(components):
        missing = set(components) - set(available_components)
        logger.warning(f"{feature_name}: Missing components {missing}")
    
    # Sum the components
    result = df[available_components].sum(axis=1).astype(int)
    
    return result


def create_all_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all aggregated features defined in configuration.
    
    Iterates through all features in 'aggregated_features' section of the config
    and creates them dynamically.
    
    Args:
        df: DataFrame with original features
    
    Returns:
        DataFrame with only the new aggregated columns
    """
    from src.utils import get_config
    
    aggregation_configs = get_config().get_aggregation_configs()
    
    # Return empty DataFrame if no aggregations defined
    if not aggregation_configs:
        logger.info("No aggregated features defined in config")
        return pd.DataFrame(index=df.index)
    
    aggregations = pd.DataFrame(index=df.index)
    
    logger.info("Creating aggregated features from config...")
    
    for feature_name, feature_config in aggregation_configs.items():
        try:
            aggregations[feature_name] = create_aggregation(df, feature_name, feature_config)
            logger.info(f"✓ {feature_name}: range [{aggregations[feature_name].min()}, {aggregations[feature_name].max()}]")
        except Exception as e:
            logger.error(f"✗ {feature_name}: Failed - {e}")
            raise
    
    return aggregations


