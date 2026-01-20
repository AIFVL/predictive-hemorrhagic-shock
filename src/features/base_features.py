"""
Base features module.

This module handles the original variables that are conserved for training.
It does NOT create new variables - that's the job of aggregations.py and categoricals.py.
"""

import pandas as pd
from typing import List


# Variables to be used directly from the original dataset
# Reference: docs/project_specification.md Section 6.1

NUMERICAL_FEATURES = [
    'EDAD',
    'HB_PREQX'
]

BINARY_FEATURES = [
    'GENERO',
    'HIPERTENSION',
    'DIABETES',
    'ENFERMEDAD_CORONARIA',
    'FALLA_CARDIACA',
    'HIPOTIROIDISMO',
    'ERC',
    'INMUNOSUPRESION',
    'OBESIDAD',
    'HIPERTENSION_PULMONAR',
    'EPOC',
    'ASMA',
    'ENF_CEREBROVASCULAR',
    'CANCER_ACTIVO',
    'TABAQUISMO',
    'SANGRADO_MAYOR'
]

CATEGORICAL_FEATURES = [
    'ACT_FISICA_METS'
]

TARGET_VARIABLE = 'SHOCK'

# Variables to exclude from training
EXCLUDE_VARIABLES = [
    'CODIGO',      # Identifier
    'MUERTE.1'     # Data leakage
]


def get_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the base features from the dataset.
    
    This function:
    1. Selects only the features specified in project_specification.md
    2. Ensures all required columns exist
    3. Does NOT create any new variables
    
    Args:
        df: DataFrame with all columns
    
    Returns:
        DataFrame with only base features
    """
    all_features = NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    
    # Check for missing features
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in dataset: {missing}")
    
    # Select only base features
    df_features = df[all_features].copy()
    
    print(f"Selected {len(all_features)} base features")
    print(f"  - Numerical: {len(NUMERICAL_FEATURES)}")
    print(f"  - Binary: {len(BINARY_FEATURES)}")
    print(f"  - Categorical: {len(CATEGORICAL_FEATURES)}")
    
    return df_features


def get_target(df: pd.DataFrame) -> pd.Series:
    """
    Extract the target variable.
    
    Args:
        df: DataFrame with target column
    
    Returns:
        Series with target values
    """
    if TARGET_VARIABLE not in df.columns:
        raise ValueError(f"Target variable '{TARGET_VARIABLE}' not found in dataset")
    
    return df[TARGET_VARIABLE].copy()


def split_features_by_type(df: pd.DataFrame) -> dict:
    """
    Split features into numerical, binary, and categorical groups.
    
    Args:
        df: DataFrame with features
    
    Returns:
        Dict with keys 'numerical', 'binary', 'categorical' containing column lists
    """
    return {
        'numerical': [c for c in NUMERICAL_FEATURES if c in df.columns],
        'binary': [c for c in BINARY_FEATURES if c in df.columns],
        'categorical': [c for c in CATEGORICAL_FEATURES if c in df.columns]
    }


if __name__ == "__main__":
    """Test base features extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract base features")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input data path")
    
    args = parser.parse_args()
    
    # Load data
    if args.input.endswith('.parquet'):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    
    # Extract base features
    df_base = get_base_features(df)
    target = get_target(df)
    
    print(f"\nBase features shape: {df_base.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Target distribution:\n{target.value_counts(normalize=True)}")
