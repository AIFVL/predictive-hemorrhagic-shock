"""
Data cleaning module.

This module removes invalid records identified by the validation module.
It follows the project specification decision to REMOVE (not recodify)
records with invalid values.

Decision documented in: docs/project_specification.md
- Remove records where binary variables contain values != 0 or 1
- Expected removal: ~1.28% of data (17 records from 1324)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Dict, List, Optional
from datetime import datetime

from src.utils import get_config, logger


def remove_invalid_records(
    df: pd.DataFrame,
    invalid_column: str = '_is_invalid'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove records marked as invalid by the validation module.
    
    Args:
        df: DataFrame with '_is_invalid' column from validate_data()
        invalid_column: Name of the invalid flag column
    
    Returns:
        Tuple of:
        - Cleaned DataFrame (without invalid records and without flag column)
        - Dict with cleaning report
    """
    if invalid_column not in df.columns:
        raise ValueError(
            f"Column '{invalid_column}' not found. "
            "Run validate_data() first to mark invalid records."
        )
    
    original_count = len(df)
    invalid_count = df[invalid_column].sum()
    
    # Remove invalid records
    df_clean = df[~df[invalid_column]].copy()
    
    # Remove the validation flag column
    df_clean = df_clean.drop(columns=[invalid_column])
    
    # Generate report
    report = {
        'original_records': int(original_count),
        'removed_records': int(invalid_count),
        'final_records': int(len(df_clean)),
        'removal_percentage': float(invalid_count / original_count * 100),
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info("Cleaning Summary")
    logger.info(f"Original records: {report['original_records']}")
    logger.info(f"Removed records: {report['removed_records']} ({report['removal_percentage']:.2f}%)")
    logger.info(f"Final records: {report['final_records']}")
    
    return df_clean, report


def exclude_leakage_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove variables that could cause data leakage.
    
    Loads leakage variables from pipeline_config.yaml.
    Per project specification, MUERTE.1 is excluded because it's
    an event that occurs after the target variable (SHOCK).
    
    Args:
        df: DataFrame to clean
    
    Returns:
        DataFrame without leakage variables
    """
    config = get_config()
    cleaning_config = config.get_cleaning_config()
    leakage_vars = cleaning_config.get('leakage_variables', [])
    
    existing_leakage = [v for v in leakage_vars if v in df.columns]
    
    if existing_leakage:
        df = df.drop(columns=existing_leakage)
        logger.info(f"Removed leakage variables: {existing_leakage}")
    
    return df


def exclude_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove identifier columns that have no predictive value.
    
    Loads identifier columns from pipeline_config.yaml.
    
    Args:
        df: DataFrame to clean
    
    Returns:
        DataFrame without identifier columns
    """
    config = get_config()
    cleaning_config = config.get_cleaning_config()
    id_columns = cleaning_config.get('identifier_columns', [])
    
    existing_ids = [v for v in id_columns if v in df.columns]
    
    if existing_ids:
        df = df.drop(columns=existing_ids)
        logger.info(f"Removed identifier columns: {existing_ids}")
    
    return df


def clean_data(
    df: pd.DataFrame,
    remove_leakage: bool = True,
    remove_ids: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Full cleaning pipeline.
    
    1. Remove invalid records (if _is_invalid column exists)
    2. Remove leakage variables (loaded from config)
    3. Remove identifier columns (loaded from config)
    
    Args:
        df: DataFrame to clean (preferably after validate_data())
        remove_leakage: Whether to remove leakage variables
        remove_ids: Whether to remove identifier columns
    
    Returns:
        Tuple of:
        - Cleaned DataFrame
        - Dict with cleaning report
    """
    logger.info("Starting data cleaning")
    
    df_clean = df.copy()
    report = {}
    
    # Step 1: Remove invalid records if validation was done
    if '_is_invalid' in df_clean.columns:
        df_clean, removal_report = remove_invalid_records(df_clean)
        report.update(removal_report)
    else:
        logger.warning("No validation column found. Skipping invalid record removal.")
        logger.warning("Run validate_data() first for proper cleaning.")
        report['original_records'] = len(df_clean)
        report['removed_records'] = 0
        report['final_records'] = len(df_clean)
    
    # Step 2: Remove leakage variables
    if remove_leakage:
        df_clean = exclude_leakage_variables(df_clean)
    
    # Step 3: Remove identifier columns
    if remove_ids:
        df_clean = exclude_identifier_columns(df_clean)
    
    report['final_columns'] = list(df_clean.columns)
    report['final_shape'] = df_clean.shape
    
    logger.success(f"Data cleaning complete. Final dataset shape: {df_clean.shape}")
    
    return df_clean, report


def save_clean_data(
    df: pd.DataFrame,
    format: str = 'parquet'
) -> str:
    """
    Save cleaned data to disk.
    
    Args:
        df: Cleaned DataFrame
        format: Output format ('parquet' or 'csv')
    
    Returns:
        Path to saved file
    """
    from src.utils import get_config
    
    config = get_config()
    output_path = Path(config.get_path('cleaned_data'))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'parquet':
        df.to_parquet(output_path, index=False)
    elif format == 'csv':
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.success(f"Clean data saved to: {output_path}")
    
    return str(output_path)
