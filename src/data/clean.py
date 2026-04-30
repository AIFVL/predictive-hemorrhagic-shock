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
from typing import Tuple, Dict, List
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
    
    logger.info({
        "event": "cleaning_summary",
        "original_records": report["original_records"],
        "removed_records": report["removed_records"],
        "removal_percentage": report["removal_percentage"],
        "final_records": report["final_records"],
    })
    
    return df_clean, report


def exclude_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Remove any configured columns that are present in the DataFrame."""
    existing_columns = [column for column in columns if column in df.columns]

    if existing_columns:
        df = df.drop(columns=existing_columns)
        logger.info({
            "event": "columns_excluded",
            "columns": existing_columns,
            "count": len(existing_columns),
        })

    return df


def clean_data(
    df: pd.DataFrame,
    exclude_configured_columns: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Full cleaning pipeline.
    
    1. Remove invalid records (if _is_invalid column exists)
    2. Remove configured columns from pipeline_config.yaml
    
    Args:
        df: DataFrame to clean (preferably after validate_data())
        exclude_configured_columns: Whether to remove columns declared in config
    
    Returns:
        Tuple of:
        - Cleaned DataFrame
        - Dict with cleaning report
    """
    logger.info({"event": "cleaning_started", "n_rows": len(df), "n_columns": len(df.columns)})
    
    df_clean = df.copy()
    report = {}
    
    # Step 1: Remove invalid records if validation was done
    if '_is_invalid' in df_clean.columns:
        df_clean, removal_report = remove_invalid_records(df_clean)
        report.update(removal_report)
    else:
        logger.warning({
            "event": "missing_validation_flag",
            "message": "Skipping invalid record removal because _is_invalid was not found.",
        })
        report['original_records'] = len(df_clean)
        report['removed_records'] = 0
        report['final_records'] = len(df_clean)
    
    # Step 2: Remove configured columns in one pass
    if exclude_configured_columns:
        cleaning_config = get_config().get('cleaning')
        columns_to_exclude = cleaning_config['exclude_columns']
        df_clean = exclude_columns(df_clean, columns_to_exclude)
    
    report['final_columns'] = list(df_clean.columns)
    report['final_shape'] = df_clean.shape
    
    logger.success({
        "event": "cleaning_completed",
        "final_shape": df_clean.shape,
        "final_columns": len(df_clean.columns),
    })
    
    return df_clean, report
