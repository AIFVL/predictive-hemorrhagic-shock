"""
Example ETL pipeline demonstrating data extraction, transformation, and loading.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from loguru import logger

from config.spark_config import get_spark_session
from config.settings import Settings
from utils.data_loader import DataLoader, DataWriter
from utils.data_quality import DataQualityChecker


class ExampleETL:
    """Example ETL pipeline."""

    def __init__(self):
        """Initialize ETL pipeline."""
        self.spark = get_spark_session(app_name="ExampleETL")
        self.loader = DataLoader(self.spark)
        self.quality_checker = DataQualityChecker()

    def extract(self, input_path: str) -> DataFrame:
        """
        Extract data from source.

        Args:
            input_path: Path to input data

        Returns:
            Extracted DataFrame
        """
        logger.info("=== EXTRACTION PHASE ===")

        # Example: Load CSV data
        df = self.loader.load_csv(
            path=input_path,
            header=True,
            infer_schema=True
        )

        logger.info(f"Extracted {df.count()} rows")
        df.printSchema()

        return df

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Transform data.

        Args:
            df: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        logger.info("=== TRANSFORMATION PHASE ===")

        # Example transformations
        df_transformed = df

        # 1. Remove duplicates
        initial_count = df_transformed.count()
        df_transformed = df_transformed.dropDuplicates()
        duplicates_removed = initial_count - df_transformed.count()
        logger.info(f"Removed {duplicates_removed} duplicate rows")

        # 2. Handle null values (example: drop rows with any null)
        df_transformed = df_transformed.dropna()
        logger.info(f"After null removal: {df_transformed.count()} rows")

        # 3. Add derived columns (example)
        # df_transformed = df_transformed.withColumn(
        #     "processed_date",
        #     F.current_timestamp()
        # )

        # 4. Filter data (example)
        # df_transformed = df_transformed.filter(F.col("some_column") > 0)

        # 5. Aggregations (example)
        # df_agg = df_transformed.groupBy("category").agg(
        #     F.count("*").alias("count"),
        #     F.avg("value").alias("avg_value")
        # )

        logger.info("Transformation complete")
        return df_transformed

    def load(self, df: DataFrame, output_path: str, format: str = "parquet") -> None:
        """
        Load data to destination.

        Args:
            df: DataFrame to load
            output_path: Output path
            format: Output format (parquet, csv, json)
        """
        logger.info("=== LOADING PHASE ===")

        if format == "parquet":
            DataWriter.write_parquet(
                df=df,
                path=output_path,
                mode="overwrite"
            )
        elif format == "csv":
            DataWriter.write_csv(
                df=df,
                path=output_path,
                mode="overwrite",
                header=True
            )
        elif format == "json":
            DataWriter.write_json(
                df=df,
                path=output_path,
                mode="overwrite"
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Data loaded to: {output_path}")

    def run(self, input_path: str, output_path: str, output_format: str = "parquet") -> None:
        """
        Run the complete ETL pipeline.

        Args:
            input_path: Path to input data
            output_path: Path to output data
            output_format: Output format
        """
        try:
            logger.info("Starting ETL pipeline")

            # Extract
            df_raw = self.extract(input_path)

            # Data quality check
            quality_report = self.quality_checker.get_quality_report(df_raw)
            logger.info(f"Quality report: {quality_report}")

            # Transform
            df_transformed = self.transform(df_raw)

            # Load
            self.load(df_transformed, output_path, output_format)

            logger.info("ETL pipeline completed successfully")

        except Exception as e:
            logger.error(f"ETL pipeline failed: {str(e)}")
            raise
        finally:
            self.spark.stop()


def main():
    """Main entry point."""
    # Example usage
    input_path = str(Settings.RAW_DATA_PATH / "example_data.csv")
    output_path = str(Settings.PROCESSED_DATA_PATH / "example_output")

    etl = ExampleETL()
    etl.run(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    main()
