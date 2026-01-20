"""
Dataset creation module.

This module builds the final training dataset by:
1. Loading cleaned data
2. Applying feature transformations (aggregations, categoricals)
3. Selecting final variables
4. Saving the training-ready dataset

Reference: docs/project_specification.md Section 6
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Union
from datetime import datetime

from src.features.base_features import get_base_features, get_target, TARGET_VARIABLE
from src.features.aggregations import create_all_aggregations
from src.features.categoricals import create_all_categoricals


# Final features for training (as per project specification)
FINAL_FEATURES = [
    # Original numerical
    'EDAD',
    'HB_PREQX',
    # Original binary/categorical
    'GENERO',
    'ACT_FISICA_METS',
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
    'SANGRADO_MAYOR',
    # Aggregated features
    'CARGA_COMORBILIDADES',
    'RIESGO_CARDIOVASCULAR',
    'RIESGO_RESPIRATORIO',
    'RIESGO_METABOLICO',
    # Categorical features
    'CATEGORIA_EDAD',
    'CATEGORIA_HB_PREQX'
]


def make_training_dataset(
    df: pd.DataFrame,
    include_aggregations: bool = True,
    include_categoricals: bool = True,
    config_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create the final training dataset with all features.
    
    Args:
        df: Cleaned DataFrame (output from clean.py)
        include_aggregations: Whether to create aggregated features
        include_categoricals: Whether to create categorical features
        config_path: Optional path to feature configuration YAML
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    print("\n" + "="*60)
    print("CREATING TRAINING DATASET")
    print("="*60)
    
    # Start with original features
    df_features = df.copy()
    
    # Create aggregated features
    if include_aggregations:
        print("\n1. Creating aggregated features...")
        agg_features = create_all_aggregations(df_features, config_path)
        df_features = pd.concat([df_features, agg_features], axis=1)
    
    # Create categorical features
    if include_categoricals:
        print("\n2. Creating categorical features...")
        cat_features = create_all_categoricals(df_features, config_path)
        df_features = pd.concat([df_features, cat_features], axis=1)
    
    # Extract target
    print("\n3. Extracting target variable...")
    target = get_target(df_features)
    
    # Select final features
    print("\n4. Selecting final features...")
    available_features = [f for f in FINAL_FEATURES if f in df_features.columns]
    missing_features = [f for f in FINAL_FEATURES if f not in df_features.columns]
    
    if missing_features:
        print(f"   ⚠️  Missing features: {missing_features}")
    
    X = df_features[available_features].copy()
    y = target.copy()
    
    print(f"\n   Final feature set: {len(available_features)} features")
    print(f"   Dataset shape: {X.shape}")
    print(f"   Target distribution:")
    print(f"      Class 0 (No Shock): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"      Class 1 (Shock): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    
    return X, y


def save_training_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Union[str, Path],
    format: str = 'parquet'
) -> str:
    """
    Save training dataset to disk.
    
    The target is included as a column in the saved file.
    
    Args:
        X: Features DataFrame
        y: Target Series
        output_path: Path to save the file
        format: Output format ('parquet' or 'csv')
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Combine features and target
    df_final = X.copy()
    df_final[TARGET_VARIABLE] = y
    
    if format == 'parquet':
        df_final.to_parquet(output_path, index=False)
    elif format == 'csv':
        df_final.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"\n✓ Training dataset saved to: {output_path}")
    print(f"  Shape: {df_final.shape}")
    
    return str(output_path)


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create stratified train/test split.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
        stratify: Whether to stratify by target
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    print(f"\nTrain/Test split created:")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"  Test class distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def save_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Union[str, Path],
    version: str = "v1",
    format: str = 'parquet'
) -> dict:
    """
    Save train/test splits to separate files.
    
    Args:
        X_train, X_test: Feature DataFrames
        y_train, y_test: Target Series
        output_dir: Directory to save splits
        version: Data version
        format: Output format
    
    Returns:
        Dict with paths to saved files
    """
    output_dir = Path(output_dir) / version
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine features and target for each split
    train_df = X_train.copy()
    train_df[TARGET_VARIABLE] = y_train
    
    test_df = X_test.copy()
    test_df[TARGET_VARIABLE] = y_test
    
    ext = 'parquet' if format == 'parquet' else 'csv'
    
    paths = {
        'train': output_dir / f'train.{ext}',
        'test': output_dir / f'test.{ext}'
    }
    
    if format == 'parquet':
        train_df.to_parquet(paths['train'], index=False)
        test_df.to_parquet(paths['test'], index=False)
    else:
        train_df.to_csv(paths['train'], index=False)
        test_df.to_csv(paths['test'], index=False)
    
    print(f"\n✓ Splits saved to: {output_dir}")
    print(f"  - Train: {paths['train']}")
    print(f"  - Test: {paths['test']}")
    
    return {k: str(v) for k, v in paths.items()}


if __name__ == "__main__":
    """CLI interface for dataset creation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create training dataset")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to cleaned data file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to save training dataset"
    )
    parser.add_argument(
        "--splits-dir", "-s",
        type=str,
        default=None,
        help="Directory to save train/test splits"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to feature configuration YAML"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="parquet",
        choices=["parquet", "csv"],
        help="Output format (default: parquet)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--version", "-v",
        type=str,
        default="v1",
        help="Data version (default: v1)"
    )
    
    args = parser.parse_args()
    
    # Load cleaned data
    print("Loading cleaned data...")
    if args.input.endswith('.parquet'):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    
    print(f"Loaded {len(df)} records")
    
    # Create training dataset
    X, y = make_training_dataset(df, config_path=args.config)
    
    # Save full dataset
    save_training_dataset(X, y, args.output, args.format)
    
    # Create and save splits if requested
    if args.splits_dir:
        X_train, X_test, y_train, y_test = create_train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=args.seed
        )
        save_splits(
            X_train, X_test, y_train, y_test,
            args.splits_dir,
            version=args.version,
            format=args.format
        )
    
    print("\n✓ Dataset creation complete")
