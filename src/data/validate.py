"""
Data validation module.

This module implements clinical validation rules to detect invalid records.
It marks records as invalid but does NOT remove them.

Validation rules are based on the project specification and loaded from
feature_config.yaml - the single source of truth for all configurations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Union

from src.utils import get_config, logger


def validate_binary_variables(
    df: pd.DataFrame,
    binary_vars: List[str]
) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Validate that binary variables contain only 0 or 1.
    
    Args:
        df: DataFrame to validate
        binary_vars: List of binary variable names from config
    
    Returns:
        Tuple of:
        - Boolean Series indicating invalid rows (True = invalid)
        - Dict with count of invalid values per variable
    """
    # Filter to variables that exist in dataframe
    existing_vars = [v for v in binary_vars if v in df.columns]
    
    invalid_mask = pd.Series(False, index=df.index)
    invalid_counts = {}
    
    for var in existing_vars:
        # Values that are not 0 or 1 (accounting for float representation)
        invalid = ~df[var].isin([0, 1, 0.0, 1.0])
        invalid_count = invalid.sum()
        
        if invalid_count > 0:
            invalid_mask |= invalid
            invalid_counts[var] = int(invalid_count)
            logger.warning(f"{var}: {invalid_count} invalid values found")
            logger.debug(f"  Values: {df.loc[invalid, var].value_counts().to_dict()}")
    
    return invalid_mask, invalid_counts


def validate_numeric_ranges(
    df: pd.DataFrame,
    rules: Dict[str, Dict[str, float]]
) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Validate that numeric variables are within clinical ranges.
    
    Args:
        df: DataFrame to validate
        rules: Dict with variable names and min/max values from config
    
    Returns:
        Tuple of:
        - Boolean Series indicating invalid rows (True = invalid)
        - Dict with count of out-of-range values per variable
    """
    invalid_mask = pd.Series(False, index=df.index)
    invalid_counts = {}
    
    for var, limits in rules.items():
        if var not in df.columns:
            continue
        
        min_val = limits['min']
        max_val = limits['max']
        
        # Check for out-of-range values
        invalid = (df[var] < min_val) | (df[var] > max_val)
        invalid_count = invalid.sum()
        
        if invalid_count > 0:
            invalid_mask |= invalid
            invalid_counts[var] = int(invalid_count)
            logger.warning(f"{var}: {invalid_count} out of range [{min_val}, {max_val}]")
            logger.debug(f"  Range found: [{df[var].min():.2f}, {df[var].max():.2f}]")
    
    return invalid_mask, invalid_counts


def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Run all validation checks on the dataset.
    
    This function marks invalid records but does NOT remove them.
    The clean.py module is responsible for removal.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Tuple of:
        - DataFrame with '_is_invalid' column added
        - Dict with validation report
    """
    logger.info("Starting data validation")
    
    # Load configuration
    config = get_config()
    binary_vars = config.get('features.binary_features')
    validation_rules = config.get('cleaning.validation_rules')
    valid_ranges = validation_rules.get('valid_ranges')
    if valid_ranges is None:
        # Backward compatibility with old schema where each rule lived at top-level.
        valid_ranges = {
            key: value
            for key, value in validation_rules.items()
            if isinstance(value, dict) and 'min' in value and 'max' in value
        }
    
    df = df.copy()
    
    # Run validation checks
    logger.info("Validating binary variables...")
    binary_invalid, binary_counts = validate_binary_variables(df, binary_vars)
    
    logger.info("Validating numeric ranges...")
    numeric_invalid, numeric_counts = validate_numeric_ranges(df, valid_ranges)
    
    # Combine all invalid masks
    total_invalid = binary_invalid | numeric_invalid
    
    # Add validation column
    df['_is_invalid'] = total_invalid
    
    # Generate report
    report = {
        'total_records': len(df),
        'invalid_records': int(total_invalid.sum()),
        'invalid_percentage': float(total_invalid.sum() / len(df) * 100),
        'binary_variable_issues': binary_counts,
        'numeric_range_issues': numeric_counts
    }
    
    logger.info("Validation Summary")
    logger.info(f"Total records: {report['total_records']}")
    logger.info(f"Invalid records: {report['invalid_records']} ({report['invalid_percentage']:.2f}%)")
    logger.info(f"Valid records: {report['total_records'] - report['invalid_records']}")
    
    return df, report
