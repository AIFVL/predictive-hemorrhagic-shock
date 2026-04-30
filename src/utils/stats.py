"""
Statistical utilities for data analysis.

Contains statistical computations used across the project,
including association metrics and correlation calculations.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from typing import List

from .logger import logger


def cramers_v(confusion_matrix: np.ndarray) -> float:
    """
    Calculate Cramér's V statistic for categorical association.
    
    Cramér's V is a measure of association between two categorical variables,
    ranging from 0 (no association) to 1 (complete association).
    
    Args:
        confusion_matrix: Contingency table (confusion matrix)
    
    Returns:
        Cramér's V value between 0 and 1
    
    References:
        Cramér, Harald (1946). Mathematical Methods of Statistics.
    """
    try:
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        min_dim = min(confusion_matrix.shape) - 1
        
        if min_dim == 0 or n == 0:
            return 0.0
        
        return np.sqrt(chi2 / (n * min_dim))
    except Exception as e:
        logger.warning(f"Error calculating Cramér's V: {e}")
        return 0.0


def cramers_v_matrix(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Compute pairwise Cramér's V matrix for categorical columns.
    
    Args:
        df: DataFrame with data
        columns: List of categorical column names
    
    Returns:
        DataFrame with pairwise Cramér's V values
    """
    n = len(columns)
    matrix = np.zeros((n, n), dtype=float)
    
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i == j:
                matrix[i, j] = 1.0
            else:
                # Create contingency table
                contingency = pd.crosstab(
                    df[col1].dropna(), 
                    df[col2].dropna()
                )
                matrix[i, j] = cramers_v(contingency.to_numpy())
    
    return pd.DataFrame(matrix, index=columns, columns=columns)


def compute_correlation_matrix(
    df: pd.DataFrame, 
    numerical_cols: List[str],
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Compute correlation matrix for numerical features.
    
    Args:
        df: DataFrame with data
        numerical_cols: List of numerical column names
        method: Correlation method ('pearson', 'spearman', 'kendall')
    
    Returns:
        Correlation matrix DataFrame
    """
    if not numerical_cols:
        logger.warning("No numerical columns provided for correlation matrix")
        return pd.DataFrame()
    
    return df[numerical_cols].corr(method=method)


def compute_point_biserial_correlation(
    df: pd.DataFrame,
    continuous_col: str,
    binary_col: str
) -> float:
    """
    Compute point-biserial correlation between continuous and binary variable.
    
    Args:
        df: DataFrame with data
        continuous_col: Name of continuous variable
        binary_col: Name of binary variable (0/1)
    
    Returns:
        Point-biserial correlation coefficient
    """
    from scipy.stats import pointbiserialr
    
    # Remove missing values
    valid_data = df[[continuous_col, binary_col]].dropna()
    
    if len(valid_data) < 2:
        logger.warning(f"Insufficient data for point-biserial correlation")
        return 0.0
    
    try:
        corr, _ = pointbiserialr(
            valid_data[binary_col],
            valid_data[continuous_col]
        )
        return corr
    except Exception as e:
        logger.warning(f"Error computing point-biserial correlation: {e}")
        return 0.0
