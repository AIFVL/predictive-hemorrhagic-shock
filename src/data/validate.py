"""
Data validation module.

This module implements clinical validation rules to detect invalid records.
It marks records as invalid but does NOT remove them.

Validation rules are based on the project specification:
- Binary variables must be exactly 0 or 1
- EDAD must be >= 18
- HB_PREQX must be in valid clinical range [3.0, 20.0]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Union
import yaml


# Default validation configuration (can be overridden by YAML config)
DEFAULT_BINARY_VARS = [
    'HIPERTENSION', 'DIABETES', 'ENFERMEDAD_CORONARIA', 'FALLA_CARDIACA',
    'HIPOTIROIDISMO', 'ERC', 'INMUNOSUPRESION', 'OBESIDAD',
    'HIPERTENSION_PULMONAR', 'EPOC', 'ASMA', 'ENF_CEREBROVASCULAR',
    'CANCER_ACTIVO', 'TABAQUISMO', 'SANGRADO_MAYOR', 'GENERO'
]

DEFAULT_VALIDATION_RULES = {
    'EDAD': {'min': 18, 'max': 110},
    'HB_PREQX': {'min': 3.0, 'max': 20.0}
}


def load_validation_config(config_path: Union[str, Path] = None) -> Dict:
    """
    Load validation configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Dict with validation configuration
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    return {
        'binary_variables': DEFAULT_BINARY_VARS,
        'validation_rules': DEFAULT_VALIDATION_RULES
    }


def validate_binary_variables(
    df: pd.DataFrame,
    binary_vars: List[str] = None
) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Validate that binary variables contain only 0 or 1.
    
    Args:
        df: DataFrame to validate
        binary_vars: List of binary variable names
    
    Returns:
        Tuple of:
        - Boolean Series indicating invalid rows (True = invalid)
        - Dict with count of invalid values per variable
    """
    if binary_vars is None:
        binary_vars = DEFAULT_BINARY_VARS
    
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
            print(f"  ⚠️  {var}: {invalid_count} invalid values found")
            print(f"      Values: {df.loc[invalid, var].value_counts().to_dict()}")
    
    return invalid_mask, invalid_counts


def validate_numeric_ranges(
    df: pd.DataFrame,
    rules: Dict[str, Dict[str, float]] = None
) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Validate that numeric variables are within clinical ranges.
    
    Args:
        df: DataFrame to validate
        rules: Dict with variable names and min/max values
    
    Returns:
        Tuple of:
        - Boolean Series indicating invalid rows (True = invalid)
        - Dict with count of out-of-range values per variable
    """
    if rules is None:
        rules = DEFAULT_VALIDATION_RULES
    
    invalid_mask = pd.Series(False, index=df.index)
    invalid_counts = {}
    
    for var, limits in rules.items():
        if var not in df.columns:
            continue
        
        min_val = limits.get('min', -np.inf)
        max_val = limits.get('max', np.inf)
        
        # Check for out-of-range values
        invalid = (df[var] < min_val) | (df[var] > max_val)
        invalid_count = invalid.sum()
        
        if invalid_count > 0:
            invalid_mask |= invalid
            invalid_counts[var] = int(invalid_count)
            print(f"  ⚠️  {var}: {invalid_count} out of range [{min_val}, {max_val}]")
            print(f"      Range found: [{df[var].min():.2f}, {df[var].max():.2f}]")
    
    return invalid_mask, invalid_counts


def validate_data(
    df: pd.DataFrame,
    config_path: Union[str, Path] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run all validation checks on the dataset.
    
    This function marks invalid records but does NOT remove them.
    The clean.py module is responsible for removal.
    
    Args:
        df: DataFrame to validate
        config_path: Path to validation configuration file
    
    Returns:
        Tuple of:
        - DataFrame with '_is_invalid' column added
        - Dict with validation report
    """
    print("\n" + "="*60)
    print("DATA VALIDATION")
    print("="*60)
    
    # Load configuration
    config = load_validation_config(config_path)
    binary_vars = config.get('binary_variables', DEFAULT_BINARY_VARS)
    numeric_rules = config.get('validation_rules', DEFAULT_VALIDATION_RULES)
    
    df = df.copy()
    
    # Run validation checks
    print("\n1. Validating binary variables...")
    binary_invalid, binary_counts = validate_binary_variables(df, binary_vars)
    
    print("\n2. Validating numeric ranges...")
    numeric_invalid, numeric_counts = validate_numeric_ranges(df, numeric_rules)
    
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
    
    print("\n" + "-"*60)
    print("VALIDATION SUMMARY")
    print("-"*60)
    print(f"Total records: {report['total_records']}")
    print(f"Invalid records: {report['invalid_records']} ({report['invalid_percentage']:.2f}%)")
    print(f"Valid records: {report['total_records'] - report['invalid_records']}")
    
    return df, report


if __name__ == "__main__":
    """CLI interface for data validation."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Validate shock prediction data")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input data file"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to validation config YAML"
    )
    parser.add_argument(
        "--output-report", "-r",
        type=str,
        default=None,
        help="Path to save validation report JSON"
    )
    
    args = parser.parse_args()
    
    # Load data
    from src.data.load import load_raw_data
    df = load_raw_data(args.input)
    
    # Validate
    df_validated, report = validate_data(df, args.config)
    
    # Save report if requested
    if args.output_report:
        with open(args.output_report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nValidation report saved to: {args.output_report}")
    
    print("\n✓ Validation complete")
