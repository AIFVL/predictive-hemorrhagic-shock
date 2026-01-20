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
from typing import Union, Tuple, Dict
from datetime import datetime


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
    
    print("\n" + "-"*60)
    print("CLEANING SUMMARY")
    print("-"*60)
    print(f"Original records: {report['original_records']}")
    print(f"Removed records: {report['removed_records']} ({report['removal_percentage']:.2f}%)")
    print(f"Final records: {report['final_records']}")
    
    return df_clean, report


def exclude_leakage_variables(
    df: pd.DataFrame,
    leakage_vars: list = None
) -> pd.DataFrame:
    """
    Remove variables that could cause data leakage.
    
    Per project specification, MUERTE.1 is excluded because it's
    an event that occurs after the target variable (SHOCK).
    
    Args:
        df: DataFrame to clean
        leakage_vars: List of variables to exclude (default: ['MUERTE.1'])
    
    Returns:
        DataFrame without leakage variables
    """
    if leakage_vars is None:
        leakage_vars = ['MUERTE.1']
    
    existing_leakage = [v for v in leakage_vars if v in df.columns]
    
    if existing_leakage:
        df = df.drop(columns=existing_leakage)
        print(f"Removed leakage variables: {existing_leakage}")
    
    return df


def exclude_identifier_columns(
    df: pd.DataFrame,
    id_columns: list = None
) -> pd.DataFrame:
    """
    Remove identifier columns that have no predictive value.
    
    Args:
        df: DataFrame to clean
        id_columns: List of identifier columns (default: ['CODIGO'])
    
    Returns:
        DataFrame without identifier columns
    """
    if id_columns is None:
        id_columns = ['CODIGO', 'codigo']
    
    existing_ids = [v for v in id_columns if v in df.columns]
    
    if existing_ids:
        df = df.drop(columns=existing_ids)
        print(f"Removed identifier columns: {existing_ids}")
    
    return df


def clean_data(
    df: pd.DataFrame,
    remove_leakage: bool = True,
    remove_ids: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Full cleaning pipeline.
    
    1. Remove invalid records (if _is_invalid column exists)
    2. Remove leakage variables (MUERTE.1)
    3. Remove identifier columns (CODIGO)
    
    Args:
        df: DataFrame to clean (preferably after validate_data())
        remove_leakage: Whether to remove leakage variables
        remove_ids: Whether to remove identifier columns
    
    Returns:
        Tuple of:
        - Cleaned DataFrame
        - Dict with cleaning report
    """
    print("\n" + "="*60)
    print("DATA CLEANING")
    print("="*60)
    
    df_clean = df.copy()
    report = {}
    
    # Step 1: Remove invalid records if validation was done
    if '_is_invalid' in df_clean.columns:
        df_clean, removal_report = remove_invalid_records(df_clean)
        report.update(removal_report)
    else:
        print("⚠️  No validation column found. Skipping invalid record removal.")
        print("   Run validate_data() first for proper cleaning.")
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
    
    print("\n" + "-"*60)
    print(f"Final dataset shape: {df_clean.shape}")
    
    return df_clean, report


def save_clean_data(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    format: str = 'parquet'
) -> str:
    """
    Save cleaned data to disk.
    
    Args:
        df: Cleaned DataFrame
        output_path: Path to save the file
        format: Output format ('parquet' or 'csv')
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'parquet':
        df.to_parquet(output_path, index=False)
    elif format == 'csv':
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"\n✓ Clean data saved to: {output_path}")
    
    return str(output_path)


if __name__ == "__main__":
    """CLI interface for data cleaning."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Clean shock prediction data")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to raw input data file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to save cleaned data"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="parquet",
        choices=["parquet", "csv"],
        help="Output format (default: parquet)"
    )
    parser.add_argument(
        "--report", "-r",
        type=str,
        default=None,
        help="Path to save cleaning report JSON"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to validation config YAML"
    )
    
    args = parser.parse_args()
    
    # Load data
    from src.data.load import load_raw_data
    from src.data.validate import validate_data
    
    print("Loading raw data...")
    df = load_raw_data(args.input)
    
    print("\nValidating data...")
    df_validated, val_report = validate_data(df, args.config)
    
    print("\nCleaning data...")
    df_clean, clean_report = clean_data(df_validated)
    
    # Save clean data
    save_clean_data(df_clean, args.output, args.format)
    
    # Merge reports and save if requested
    if args.report:
        full_report = {**val_report, **clean_report}
        with open(args.report, 'w') as f:
            json.dump(full_report, f, indent=2, default=str)
        print(f"\nCleaning report saved to: {args.report}")
    
    print("\n✓ Data cleaning pipeline complete")
