"""CLI wrapper that delegates EDA plotting to the canonical EDAVisualizer.

This module keeps a small, stable CLI for Airflow / scripts but delegates
the plotting implementation to `src.visualization.eda_visualizations.EDAVisualizer`.
"""

from pathlib import Path
from typing import List, Dict
import json
import argparse
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

from src.visualization.eda_visualizations import EDAVisualizer


def _compute_cramers_matrix(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Compute pairwise Cramér's V matrix for list of categorical columns."""
    n = len(cols)
    mat = np.zeros((n, n), dtype=float)
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                mat[i, j] = 1.0
            else:
                cm = pd.crosstab(df[c1].dropna(), df[c2].dropna())
                try:
                    chi2 = chi2_contingency(cm)[0]
                    n_obs = cm.to_numpy().sum()
                    k = min(cm.shape) - 1
                    mat[i, j] = np.sqrt(chi2 / (n_obs * k)) if k > 0 and n_obs > 0 else 0.0
                except Exception:
                    mat[i, j] = 0.0
    return pd.DataFrame(mat, index=cols, columns=cols)


def generate_all_eda_plots(
    df: pd.DataFrame,
    numerical_cols: List[str],
    binary_cols: List[str],
    target_col: str,
    output_dir: str,
) -> Dict[str, str]:
    """
    Generate all EDA plots using the canonical EDAVisualizer.
    
    Args:
        df: DataFrame with data
        numerical_cols: List of numerical column names
        binary_cols: List of binary/categorical column names
        target_col: Target column name
        output_dir: Output directory for plots
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    viz = EDAVisualizer(output_dir=str(output_dir))
    plots = {}

    # Target distribution
    plots['target_distribution'] = viz.plot_target_distribution(df, target_col)

    # Numerical distributions and boxplots
    if numerical_cols:
        plots['numerical_distributions'] = viz.plot_numerical_distributions(df, numerical_cols, target_col)
        plots['numerical_boxplots'] = viz.plot_numerical_boxplots(df, numerical_cols, target_col)

    # Categorical / binary by target
    if binary_cols:
        plots['categorical_by_target'] = viz.plot_categorical_by_target(df, binary_cols, target_col, top_n=20)

    # Correlation heatmap (numerical)
    if numerical_cols:
        corr = df[numerical_cols + [target_col]].corr()
        plots['correlation_matrix'] = viz.plot_correlation_heatmap(corr, title='Correlation Matrix', aggregated_vars=None)

    # Cramér's V
    if binary_cols:
        cramers = _compute_cramers_matrix(df, binary_cols + [target_col])
        plots['cramers_v_heatmap'] = viz.plot_cramers_v_heatmap(cramers)

        # Barplot of association with target: take last column (target)
        try:
            assoc_with_target = cramers.loc[:, target_col].drop(target_col)
            assoc_df = assoc_with_target.sort_values(ascending=True)
            # save a small barplot via EDAVisualizer.plot_top_associations (construct a results df)
            results = pd.DataFrame({'Variable': assoc_df.index, "cramers_v": assoc_df.values})
            plots['cramers_v_barplot'] = viz.plot_top_associations(results, 'cramers_v', top_n=len(results), title=f"Association with {target_col}")
        except Exception:
            plots['cramers_v_barplot'] = None

    # Missing values
    mv = viz.plot_missing_values(df)
    if mv:
        plots['missing_values'] = mv

    # Return generated paths (visualizer prints actual save locations)
    return plots


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate EDA plots using canonical EDAVisualizer')
    parser.add_argument('--input', '-i', required=True, help='Input CSV or parquet file')
    parser.add_argument('--output', '-o', required=True, help='Output directory for plots')
    parser.add_argument('--target', default='SHOCK', help='Target column name (default: SHOCK)')
    args = parser.parse_args()

    if args.input.endswith('.parquet'):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    NUMERICAL_FEATURES = ['EDAD', 'HB_PREQX']
    BINARY_FEATURES = [
        'GENERO', 'ACT_FISICA_METS', 'HIPERTENSION', 'DIABETES',
        'ENFERMEDAD_CORONARIA', 'FALLA_CARDIACA', 'HIPOTIROIDISMO', 'ERC',
        'INMUNOSUPRESION', 'OBESIDAD', 'HIPERTENSION_PULMONAR', 'EPOC',
        'ASMA', 'ENF_CEREBROVASCULAR', 'CANCER_ACTIVO', 'TABAQUISMO', 'SANGRADO_MAYOR'
    ]

    numerical_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    binary_cols = [c for c in BINARY_FEATURES if c in df.columns]

    plots = generate_all_eda_plots(df, numerical_cols, binary_cols, args.target, args.output)

    # Save manifest
    manifest_path = Path(args.output) / 'eda_plots_manifest.json'
    with open(manifest_path, 'w') as fh:
        json.dump(plots, fh, indent=2)

    print(f"\n✓ EDA manifest saved to: {manifest_path}")
