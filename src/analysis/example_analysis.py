"""
Example data analysis using PySpark.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from loguru import logger

from config.spark_config import get_spark_session
from config.settings import Settings
from utils.data_loader import DataLoader


class ExampleAnalysis:
    """Example data analysis pipeline."""

    def __init__(self):
        """Initialize analysis pipeline."""
        self.spark = get_spark_session(app_name="ExampleAnalysis")
        self.loader = DataLoader(self.spark)

    def load_data(self, input_path: str) -> DataFrame:
        """
        Load data for analysis.

        Args:
            input_path: Path to input data

        Returns:
            Loaded DataFrame
        """
        logger.info("Loading data for analysis")
        df = self.loader.load_parquet(input_path)
        return df

    def basic_statistics(self, df: DataFrame) -> None:
        """
        Compute basic statistics.

        Args:
            df: Input DataFrame
        """
        logger.info("=== BASIC STATISTICS ===")

        # Show schema
        logger.info("Schema:")
        df.printSchema()

        # Count rows
        row_count = df.count()
        logger.info(f"Total rows: {row_count}")

        # Show sample data
        logger.info("Sample data:")
        df.show(5, truncate=False)

        # Describe numeric columns
        logger.info("Descriptive statistics:")
        df.describe().show()

    def aggregation_analysis(self, df: DataFrame, group_by_col: str) -> DataFrame:
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
        df_agg = df.groupBy(group_by_col).agg(
            F.count("*").alias("count"),
            # Add more aggregations as needed
            # F.avg("some_column").alias("avg_value"),
            # F.sum("some_column").alias("total_value"),
            # F.min("some_column").alias("min_value"),
            # F.max("some_column").alias("max_value")
        ).orderBy(F.desc("count"))

        logger.info("Aggregation results:")
        df_agg.show(10)

        return df_agg

    def window_analysis(self, df: DataFrame, partition_col: str, order_col: str) -> DataFrame:
        """
        Perform window function analysis.

        Args:
            df: Input DataFrame
            partition_col: Column to partition by
            order_col: Column to order by

        Returns:
            DataFrame with window functions applied
        """
        logger.info("=== WINDOW FUNCTION ANALYSIS ===")

        # Define window specification
        window_spec = Window.partitionBy(partition_col).orderBy(F.desc(order_col))

        # Apply window functions
        df_window = df.withColumn(
            "row_number",
            F.row_number().over(window_spec)
        ).withColumn(
            "rank",
            F.rank().over(window_spec)
        )

        logger.info("Window analysis results:")
        df_window.show(10)

        return df_window

    def correlation_analysis(self, df: DataFrame, col1: str, col2: str) -> float:
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

        correlation = df.stat.corr(col1, col2)
        logger.info(f"Correlation coefficient: {correlation:.4f}")

        return correlation

    def outlier_detection(self, df: DataFrame, column: str, n_std: float = 3.0) -> DataFrame:
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
        stats = df.select(
            F.mean(column).alias("mean"),
            F.stddev(column).alias("stddev")
        ).collect()[0]

        mean = stats["mean"]
        stddev = stats["stddev"]

        # Flag outliers
        df_outliers = df.withColumn(
            "is_outlier",
            (F.abs(F.col(column) - mean) > (n_std * stddev))
        )

        outlier_count = df_outliers.filter(F.col("is_outlier")).count()
        logger.info(f"Found {outlier_count} outliers")

        return df_outliers

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

            # Add more analysis as needed based on your data
            # Example:
            # self.aggregation_analysis(df, "category_column")
            # self.window_analysis(df, "partition_col", "order_col")
            # self.correlation_analysis(df, "col1", "col2")
            # self.outlier_detection(df, "numeric_column")

            logger.info("Analysis pipeline completed successfully")

        except Exception as e:
            logger.error(f"Analysis pipeline failed: {str(e)}")
            raise
        finally:
            self.spark.stop()


def main():
    """Main entry point."""
    # Example usage
    input_path = str(Settings.PROCESSED_DATA_PATH / "example_output")

    analysis = ExampleAnalysis()
    analysis.run_analysis(input_path=input_path)


if __name__ == "__main__":
    main()
