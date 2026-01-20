"""
Data loading module.

This module is responsible ONLY for reading data from disk.
No transformations, no cleaning, no feature engineering.
"""

import pandas as pd
from pathlib import Path
from typing import Union


def load_raw_data(
    path: Union[str, Path],
    version: str = "v1"
) -> pd.DataFrame:
    """
    Load raw dataset from data/raw/{version}/.
    
    Args:
        path: Path to the data file (CSV or Parquet)
        version: Data version (default: v1)
    
    Returns:
        pd.DataFrame: Raw data without any modifications
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If file format is not supported
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    suffix = path.suffix.lower()
    
    if suffix == '.csv':
        df = pd.read_csv(path)
    elif suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use CSV or Parquet.")
    
    print(f"Loaded {len(df)} rows x {len(df.columns)} columns from {path}")
    
    return df


def load_processed_data(
    path: Union[str, Path],
    version: str = "v1"
) -> pd.DataFrame:
    """
    Load processed dataset from data/processed/{version}/.
    
    Args:
        path: Path to the processed data file
        version: Data version (default: v1)
    
    Returns:
        pd.DataFrame: Processed data
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}")
    
    suffix = path.suffix.lower()
    
    if suffix == '.csv':
        df = pd.read_csv(path)
    elif suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    print(f"Loaded {len(df)} rows x {len(df.columns)} columns from {path}")
    
    return df


if __name__ == "__main__":
    """CLI interface for data loading."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load shock prediction data")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input data file"
    )
    parser.add_argument(
        "--version", "-v",
        type=str,
        default="v1",
        help="Data version (default: v1)"
    )
    
    args = parser.parse_args()
    
    df = load_raw_data(args.input, args.version)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
