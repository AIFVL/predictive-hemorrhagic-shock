"""Dataset-level helpers for split transformation and threshold metadata."""

from pathlib import Path

import pandas as pd

from src.utils import DataLoader, get_config, logger


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) and target (y) from a DataFrame.
    
    Business logic: knows the target variable name from config.
    
    Args:
        df: DataFrame with features and target column
    
    Returns:
        Tuple of (X: features DataFrame, y: target Series)
    """
    config = get_config()
    target_variable = config.get('features.target_name')
    
    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in DataFrame")
    
    y = df[target_variable]
    X = df.drop(columns=[target_variable])
    
    return X, y


def load_optimal_threshold_for_model(model_name: str) -> float:
    """
    Load optimal threshold saved in model metadata.
    
    Business logic: knows where model metadata is stored and how to read it.
    
    Args:
        model_name: Name of the trained model
    Returns:
        The optimal threshold as float
    """
    config = get_config()
    model_dir = Path(config.get_path('model_output', model_name=model_name)).parent
    metadata_path = model_dir / 'metadata.json'

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found for model '{model_name}': {metadata_path}")

    metadata = DataLoader.load(metadata_path)

    operating_point = metadata.get('operating_point') or {}
    threshold_optimization = metadata.get('threshold_optimization') or {}

    threshold = operating_point.get('threshold')
    if threshold is None:
        threshold = threshold_optimization.get('optimal_threshold')

    if threshold is None:
        logger.warning(
            f"No optimal threshold found in metadata for model '{model_name}'. Using default 0.50"
        )
        return 0.5

    threshold = float(threshold)
    logger.info(f"Loaded optimal threshold for {model_name} from metadata: {threshold:.3f}")
    return threshold
