"""
EDA plot generation module.

Generates exploratory data analysis plots using plotting utilities.
All plotting logic is delegated to src.utils.plotting functions.
"""

from pathlib import Path
from typing import List, Dict
import pandas as pd
import json
from src.utils import DataLoader

from src.utils import (
    logger,
    log_section,
    get_config,
    cramers_v_matrix,
    compute_correlation_matrix,
    setup_plot_style,
    plot_target_distribution,
    plot_distribution,
    plot_boxplot_by_target,
    plot_count_by_target,
    plot_correlation_matrix,
    plot_missing_values,
    save_figure
)


def generate_all_eda_plots(
    df: pd.DataFrame,
    numerical_cols: List[str],
    binary_cols: List[str],
    target_col: str,
) -> Dict[str, str]:
    """
    Generate all EDA plots using plotting utilities.
    
    Args:
        df: DataFrame with data
        numerical_cols: List of numerical column names
        binary_cols: List of binary/categorical column names
        target_col: Target column name
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    config = get_config()
    output_dir = Path(config.get_path('eda_plots'))
    output_dir.mkdir(parents=True, exist_ok=True)
    

    setup_plot_style()
    plots = {}
    eda_metadata = {"plots": {}}

    log_section("GENERATING EDA PLOTS")

    # 1. Target distribution
    logger.info("Creating target distribution plot...")
    fig = plot_target_distribution(df, target_col)
    plots['target_distribution'] = output_dir / 'target_distribution.png'
    save_figure(fig, plots['target_distribution'])
    eda_metadata["plots"]["target_distribution"] = {
        "description": "Distribution of the target variable (Shock vs No Shock)",
        "path": "target_distribution.png"
    }

    # 2. Numerical distributions
    eda_metadata["plots"]["numerical_distributions"] = {}
    if numerical_cols:
        logger.info(f"Creating distributions for {len(numerical_cols)} numerical features")
        for col in numerical_cols[:10]:  # Limit to top 10
            # Calc stats
            mean_val = float(df[col].mean())
            std_val = float(df[col].std())
            
            fig = plot_distribution(df[col], title=f'Distribution of {col}', xlabel=col)
            save_figure(fig, output_dir / f'dist_{col}.png')
            
            eda_metadata["plots"]["numerical_distributions"][f"dist_{col}"] = {
                "description": f"Distribution of {col}",
                "path": f"dist_{col}.png",
                "data": {
                    "mean": round(mean_val, 4),
                    "std": round(std_val, 4)
                }
            }
        plots['numerical_distributions'] = str(output_dir / 'dist_*.png')

    # 3. Numerical boxplots by target
    eda_metadata["plots"]["numerical_boxplots"] = {}
    if numerical_cols:
        logger.info(f"Creating boxplots for {len(numerical_cols)} numerical features")
        for col in numerical_cols[:10]:  # Limit to top 10
            # Calc stats
            stats = {}
            for target_val in df[target_col].unique():
                series = df[df[target_col] == target_val][col].dropna()
                stats[str(target_val)] = {
                    "q1": round(float(series.quantile(0.25)), 4),
                    "median": round(float(series.median()), 4),
                    "q3": round(float(series.quantile(0.75)), 4),
                }

            fig = plot_boxplot_by_target(df, col, target_col)
            save_figure(fig, output_dir / f'boxplot_{col}.png')
            
            eda_metadata["plots"]["numerical_boxplots"][f"boxplot_{col}"] = {
                "description": f"Boxplot of {col} by {target_col}",
                "path": f"boxplot_{col}.png",
                "data": stats
            }
        plots['numerical_boxplots'] = str(output_dir / 'boxplot_*.png')

    # 4. Categorical/Binary by target
    eda_metadata["plots"]["categorical_by_target"] = {}
    if binary_cols:
        logger.info(f"Creating count plots for {len(binary_cols)} binary features")
        for col in binary_cols[:15]:  # Limit to top 15
            # Calc stats
            counts = df.groupby([col, target_col]).size().unstack(fill_value=0)
            stats = {}
            for col_val in counts.index:
                stats[str(col_val)] = {str(t): int(counts.loc[col_val, t]) for t in counts.columns}

            fig = plot_count_by_target(df, col, target_col)
            save_figure(fig, output_dir / f'count_{col}.png')
            
            eda_metadata["plots"]["categorical_by_target"][f"count_{col}"] = {
                "description": f"Count of {col} by {target_col}",
                "path": f"count_{col}.png",
                "data": stats
            }
        plots['categorical_by_target'] = str(output_dir / 'count_*.png')

    # 5. Correlation heatmap (numerical)
    if numerical_cols:
        logger.info("Creating correlation matrix for numerical features")
        corr = compute_correlation_matrix(df, numerical_cols + [target_col])
        fig = plot_correlation_matrix(corr, title='Numerical Correlation Matrix')
        plots['correlation_matrix'] = output_dir / 'correlation_matrix.png'
        save_figure(fig, plots['correlation_matrix'])
        eda_metadata["plots"]["correlation_matrix"] = {
            "description": "Heatmap of Pearson correlation among numerical features",
            "path": "correlation_matrix.png"
        }

    # 6. Cramér's V matrix (binary)
    if binary_cols:
        logger.info("Computing Cramér's V matrix for binary features")
        cramers = cramers_v_matrix(df, binary_cols + [target_col])
        fig = plot_correlation_matrix(
            cramers, 
            title="Cramér's V Association Matrix",
            cmap='YlOrRd'
        )
        plots['cramers_v_heatmap'] = output_dir / 'cramers_v_heatmap.png'
        save_figure(fig, plots['cramers_v_heatmap'])
        eda_metadata["plots"]["cramers_v_heatmap"] = {
            "description": "Heatmap of Cramér's V association among binary features",
            "path": "cramers_v_heatmap.png"
        }

    # 7. Missing values
    logger.info("Checking for missing values...")
    fig = plot_missing_values(df, top_n=20)
    if fig:
        plots['missing_values'] = output_dir / 'missing_values.png'
        save_figure(fig, plots['missing_values'])
        eda_metadata["plots"]["missing_values"] = {
            "description": "Bar plot highlighting top empty/missing columns in the dataset",
            "path": "missing_values.png"
        }
        logger.info("Missing values plot created")
    else:
        logger.info("No missing values found")

    DataLoader.save(eda_metadata, output_dir / "eda_metadata.json")
    logger.success(f"Generated {len(plots)} EDA plot groups in {output_dir}")
    return {k: str(v) for k, v in plots.items()}
