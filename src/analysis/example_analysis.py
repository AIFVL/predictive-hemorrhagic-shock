"""
Example data analysis using pandas and scikit-learn.
"""
import pandas as pd
from loguru import logger
import numpy as np
from typing import Tuple

from config.settings import Settings
from utils.data_loader import DataLoader


class ExampleAnalysis:
    """Example data analysis pipeline using pandas."""

    def __init__(self):
        """Initialize analysis pipeline."""
        self.loader = DataLoader()

    def load_data(self, input_path: str) -> pd.DataFrame:
        """
        Load data for analysis.

        Args:
            input_path: Path to input data

        Returns:
            Loaded DataFrame
        """
        logger.info("Loading data for analysis")

        if input_path.endswith('.parquet'):
            df = self.loader.load_parquet(input_path)
        elif input_path.endswith('.csv'):
            df = self.loader.load_csv(input_path)
        elif input_path.endswith('.json'):
            df = self.loader.load_json(input_path)
        else:
            # Determine format based on extension or default to CSV
            df = self.loader.load_csv(input_path)

        return df

    def basic_statistics(self, df: pd.DataFrame) -> None:
        """
        Compute basic statistics.

        Args:
            df: Input DataFrame
        """
        logger.info("=== BASIC STATISTICS ===")

        # Info about data types
        logger.info("Data types:")
        logger.info(f"{df.dtypes}")

        # Count rows and columns
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Total columns: {len(df.columns)}")

        # Show sample data
        logger.info("Sample data (first 5 rows):")
        logger.info(f"\n{df.head()}")

        # Describe numeric columns
        logger.info("Descriptive statistics:")
        numeric_desc = df.describe()
        logger.info(f"\n{numeric_desc}")

    def aggregation_analysis(self, df: pd.DataFrame, group_by_col: str) -> pd.DataFrame:
        """
        Perform aggregation analysis.

        Args:
            df: Input DataFrame
            group_by_col: Column to group by

        Returns:
            Aggregated DataFrame
        """
        logger.info(f"=== AGGREGATION ANALYSIS (grouped by {group_by_col}) ===")

        # Example aggregations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            agg_col = numeric_cols[0]
            df_agg = df.groupby(group_by_col).agg({
                agg_col: ['count', 'mean', 'std']
            }).reset_index()

            # Flatten column names for MultiIndex
            if isinstance(df_agg.columns, pd.MultiIndex):
                df_agg.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in df_agg.columns.values]
        else:
            # If no numeric columns, just return groupby with count
            df_agg = df.groupby(group_by_col).size().reset_index(name='count')

        logger.info("Aggregation results (first 10):")
        logger.info(f"\n{df_agg.head(10)}")

        return df_agg

    def correlation_analysis(self, df: pd.DataFrame, col1: str, col2: str) -> float:
        """
        Calculate correlation between two columns.

        Args:
            df: Input DataFrame
            col1: First column
            col2: Second column

        Returns:
            Correlation coefficient
        """
        logger.info(f"=== CORRELATION ANALYSIS: {col1} vs {col2} ===")

        correlation = df[col1].corr(df[col2])
        logger.info(f"Correlation coefficient: {correlation:.4f}")

        return correlation

    def outlier_detection(self, df: pd.DataFrame, column: str, n_std: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers using standard deviation method.

        Args:
            df: Input DataFrame
            column: Column to check for outliers
            n_std: Number of standard deviations for threshold

        Returns:
            DataFrame with outliers flagged
        """
        logger.info(f"=== OUTLIER DETECTION: {column} ===")

        # Calculate mean and standard deviation
        mean = df[column].mean()
        std = df[column].std()

        # Flag outliers
        df_outliers = df.copy()
        df_outliers['is_outlier'] = (np.abs(df_outliers[column] - mean) > (n_std * std))

        outlier_count = df_outliers['is_outlier'].sum()
        logger.info(f"Found {outlier_count} outliers")

        return df_outliers

    def distribution_analysis(self, df: pd.DataFrame, column: str) -> None:
        """
        Analyze distribution of a column.

        Args:
            df: Input DataFrame
            column: Column to analyze
        """
        logger.info(f"=== DISTRIBUTION ANALYSIS: {column} ===")

        if df[column].dtype in ['int64', 'float64']:
            # Numeric column statistics
            q25 = df[column].quantile(0.25)
            q50 = df[column].quantile(0.50)
            q75 = df[column].quantile(0.75)

            logger.info(f"Quantiles:")
            logger.info(f"  25%: {q25}")
            logger.info(f"  50%: {q50}")
            logger.info(f"  75%: {q75}")

        else:
            # Categorical column statistics
            value_counts = df[column].value_counts()
            logger.info(f"Top 10 most frequent values:")
            logger.info(f"\n{value_counts.head(10)}")

    def run_analysis(self, input_path: str) -> None:
        """
        Run complete analysis pipeline.

        Args:
            input_path: Path to input data
        """
        try:
            logger.info("Starting analysis pipeline")

            # Load data
            df = self.load_data(input_path)

            # Basic statistics
            self.basic_statistics(df)

            # Distribution analysis for first few columns
            for col in df.columns[:min(5, len(df.columns))]:
                if df[col].dtype in ['int64', 'float64', 'object']:
                    self.distribution_analysis(df, col)

            # Add more analysis as needed based on your data
            # Example:
            # if len(df.columns) > 1:
            #     numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            #     if len(numeric_cols) >= 2:
            #         self.correlation_analysis(df, numeric_cols[0], numeric_cols[1])
            #
            # if len(df.columns) > 0:
            #     first_numeric = df.select_dtypes(include=[np.number]).columns[0]
            #     self.outlier_detection(df, first_numeric)

            logger.info("Analysis pipeline completed successfully")

        except Exception as e:
            logger.error(f"Analysis pipeline failed: {str(e)}")
            raise


def main():
    """Main entry point."""
    # Example usage
    input_path = str(Settings.PROCESSED_DATA_PATH / "example_output.parquet")

    analysis = ExampleAnalysis()
    analysis.run_analysis(input_path=input_path)


if __name__ == "__main__":
    main()
