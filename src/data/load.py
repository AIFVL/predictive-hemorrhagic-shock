"""
Data loading module.

This module is responsible ONLY for reading data from disk.
No transformations, no cleaning, no feature engineering.
"""

import pandas as pd
from pathlib import Path
from typing import Union

from src.utils import logger, get_config


def load_raw_data(path_key: str = 'raw_data') -> pd.DataFrame:
    """
    Load raw dataset from data/raw/{version}/.
    
    Args:
        path_key: Key in config paths (default: 'raw_data')
    
    Returns:
        pd.DataFrame: Raw data without any modifications
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If file format is not supported
    """
    config = get_config()
    path = Path(config.get_path(path_key))
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    suffix = path.suffix.lower()
    
    if suffix == '.csv':
        df = pd.read_csv(path)
    elif suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use CSV or Parquet.")
    
    logger.info(f"Loaded {len(df)} rows x {len(df.columns)} columns from {path}")
    
    return df


def load_processed_data(path_key: str = 'cleaned_data') -> pd.DataFrame:
    """
    Load processed dataset from data/processed/{version}/.
    
    Args:
        path_key: Key in config paths (default: 'cleaned_data')
    
    Returns:
        pd.DataFrame: Processed data
    """
    config = get_config()
    path = Path(config.get_path(path_key))
    
    
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}")
    
    suffix = path.suffix.lower()
    
    if suffix == '.csv':
        df = pd.read_csv(path)
    elif suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    logger.info(f"Loaded {len(df)} rows x {len(df.columns)} columns from {path}")
    
    return df
