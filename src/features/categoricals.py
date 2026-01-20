"""
Categorical features module.

This module implements the categorical transformations of continuous variables
as documented in docs/project_specification.md Section 5.2.

Categorization thresholds are based on clinical guidelines:
- Age categories: CDC/ASA surgical risk guidelines
- Hemoglobin categories: WHO anemia classification
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import yaml
from pathlib import Path


def load_categorical_config(config_path: Optional[str] = None) -> Dict:
    """
    Load categorical configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Dict with categorization definitions
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Parse categorical_features from YAML into the format expected by the code
            if 'categorical_features' in config:
                parsed = {}
                for feature_name, feature_config in config['categorical_features'].items():
                    parsed[feature_name] = {
                        'description': feature_config.get('description', ''),
                        'source_column': feature_config.get('source_variable', feature_name.replace('CATEGORIA_', '')),
                        'bins': [0] + feature_config.get('thresholds', []) + [float('inf')],
                        'labels': [cat['value'] for cat in feature_config.get('categories', [])],
                        'label_descriptions': {cat['value']: cat['description'] for cat in feature_config.get('categories', [])}
                    }
                return parsed
            # Fallback to old format
            return config.get('categoricals', {})
    
    return get_default_categorical_config()


def get_default_categorical_config() -> Dict:
    """
    Return default categorization configuration.
    
    These are the clinically-validated categorizations from the project specification.
    """
    return {
        'CATEGORIA_EDAD': {
            'description': 'Age category based on CDC/ASA surgical risk thresholds',
            'source_column': 'EDAD',
            'bins': [0, 45, 65, 75, float('inf')],
            'labels': [0, 1, 2, 3],
            'label_descriptions': {
                0: 'Young adult (<45)',
                1: 'Middle-aged adult (45-64)',
                2: 'Older adult (65-74)',
                3: 'Elderly (>=75)'
            }
        },
        'CATEGORIA_HB_PREQX': {
            'description': 'Hemoglobin category based on WHO anemia classification',
            'source_column': 'HB_PREQX',
            'bins': [0, 8, 10, 12, float('inf')],
            'labels': [3, 2, 1, 0],  # Note: reversed - lower Hb = higher severity
            'label_descriptions': {
                0: 'Normal (>=12 g/dL)',
                1: 'Mild anemia (10-12 g/dL)',
                2: 'Moderate anemia (8-10 g/dL)',
                3: 'Severe anemia (<8 g/dL)'
            }
        }
    }


def create_categoria_edad(df: pd.DataFrame, config: Dict = None) -> pd.Series:
    """
    Create CATEGORIA_EDAD feature.
    
    CLINICAL JUSTIFICATION:
    Physiological reserve decreases in a stepwise (not linear) manner with age.
    ASA guidelines and geriatric surgery studies establish clinical cutoff points
    at ages 65 and 75 (American Geriatrics Society, 2012).
    
    Bivariate analysis showed EDAD is significant (Cohen's d = 0.29), but the
    relationship may be non-linear. Categorization captures threshold effects
    not detected with continuous age.
    
    CATEGORIES:
    - 0: Young adult (<45 years) - Reference, lowest risk
    - 1: Middle-aged adult (45-64 years) - Intermediate risk
    - 2: Older adult (65-74 years) - High risk
    - 3: Elderly (>=75 years) - Very high risk
    
    Args:
        df: DataFrame with EDAD column
        config: Optional configuration dict
    
    Returns:
        Series with age category (0-3)
    """
    if config is None:
        config = get_default_categorical_config()
    
    edad_config = config['CATEGORIA_EDAD']
    
    if edad_config['source_column'] not in df.columns:
        raise ValueError(f"Column {edad_config['source_column']} not found in dataframe")
    
    edad = df[edad_config['source_column']]
    
    # Use pd.cut for binning
    categoria = pd.cut(
        edad,
        bins=edad_config['bins'],
        labels=edad_config['labels'],
        right=False,
        include_lowest=True
    ).astype(int)
    
    return categoria


def create_categoria_hb_preqx(df: pd.DataFrame, config: Dict = None) -> pd.Series:
    """
    Create CATEGORIA_HB_PREQX feature.
    
    CLINICAL JUSTIFICATION:
    Preoperative anemia limits tolerance to blood loss. The WHO anemia 
    classification provides clinically-validated thresholds that predict
    massive transfusion requirements and perioperative mortality.
    
    Key thresholds:
    - 12 g/dL: Normal hemoglobin (most surgical patients)
    - 10 g/dL: Mild anemia - increased transfusion risk
    - 8 g/dL: Moderate anemia - trigger for preoperative correction
    - <8 g/dL: Severe anemia - high risk, may require transfusion before surgery
    
    CATEGORIES:
    - 0: Normal (>=12 g/dL)
    - 1: Mild anemia (10-12 g/dL)
    - 2: Moderate anemia (8-10 g/dL)
    - 3: Severe anemia (<8 g/dL)
    
    Note: The category values are designed so that higher values indicate
    higher risk (more severe anemia).
    
    Args:
        df: DataFrame with HB_PREQX column
        config: Optional configuration dict
    
    Returns:
        Series with hemoglobin category (0-3)
    """
    if config is None:
        config = get_default_categorical_config()
    
    hb_config = config['CATEGORIA_HB_PREQX']
    
    if hb_config['source_column'] not in df.columns:
        raise ValueError(f"Column {hb_config['source_column']} not found in dataframe")
    
    hb = df[hb_config['source_column']]
    
    # Use pd.cut for binning
    categoria = pd.cut(
        hb,
        bins=hb_config['bins'],
        labels=hb_config['labels'],
        right=False,
        include_lowest=True
    ).astype(int)
    
    return categoria


def create_all_categoricals(
    df: pd.DataFrame,
    config_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create all categorical features at once.
    
    Args:
        df: DataFrame with original continuous features
        config_path: Optional path to YAML configuration
    
    Returns:
        DataFrame with only the new categorical columns
    """
    config = load_categorical_config(config_path)
    
    categoricals = pd.DataFrame(index=df.index)
    
    categoricals['CATEGORIA_EDAD'] = create_categoria_edad(df, config)
    categoricals['CATEGORIA_HB_PREQX'] = create_categoria_hb_preqx(df, config)
    
    print(f"Created {len(categoricals.columns)} categorical features:")
    for col in categoricals.columns:
        print(f"  - {col}: {categoricals[col].value_counts().to_dict()}")
    
    return categoricals


def get_category_description(feature_name: str, value: int, config: Dict = None) -> str:
    """
    Get human-readable description for a category value.
    
    Args:
        feature_name: Name of the categorical feature
        value: Category value (0-3)
        config: Optional configuration dict
    
    Returns:
        String description of the category
    """
    if config is None:
        config = get_default_categorical_config()
    
    if feature_name not in config:
        return f"Unknown feature: {feature_name}"
    
    descriptions = config[feature_name].get('label_descriptions', {})
    return descriptions.get(value, f"Unknown category: {value}")


if __name__ == "__main__":
    """Test categorical feature creation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create categorical features")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input data path")
    parser.add_argument("--config", "-c", type=str, default=None, help="Config YAML path")
    
    args = parser.parse_args()
    
    # Load data
    if args.input.endswith('.parquet'):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    
    # Create categoricals
    cat_df = create_all_categoricals(df, args.config)
    
    print(f"\nCategoricals shape: {cat_df.shape}")
    
    # Show category descriptions
    config = get_default_categorical_config()
    print("\nCategory mappings:")
    for feature in ['CATEGORIA_EDAD', 'CATEGORIA_HB_PREQX']:
        print(f"\n{feature}:")
        for val in range(4):
            print(f"  {val}: {get_category_description(feature, val, config)}")
