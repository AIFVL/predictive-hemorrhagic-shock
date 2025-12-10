"""
Example ETL pipeline demonstrating data extraction, transformation, and loading using pandas.
"""
import pandas as pd
from loguru import logger

from config.settings import Settings
from utils.data_loader import DataLoader, DataWriter
from utils.data_quality import DataQualityChecker


class ExampleETL:
    """Example ETL pipeline using pandas."""

    def __init__(self):
        """Initialize ETL pipeline."""
        self.loader = DataLoader()
        self.writer = DataWriter()
        self.quality_checker = DataQualityChecker()

    def extract(self, input_path: str) -> pd.DataFrame:
        """
        Extract data from source.

        Args:
            input_path: Path to input data

        Returns:
            Extracted DataFrame
        """
        logger.info("=== EXTRACTION PHASE ===")

        # Example: Load CSV data
        if input_path.endswith('.parquet'):
            df = self.loader.load_parquet(path=input_path)
        elif input_path.endswith('.json'):
            df = self.loader.load_json(path=input_path)
        else:
            df = self.loader.load_csv(path=input_path)

        logger.info(f"Extracted {len(df)} rows with {len(df.columns)} columns")
        logger.info(f"Column names: {list(df.columns)}")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data.

        Args:
            df: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        logger.info("=== TRANSFORMATION PHASE ===")

        # Example transformations
        df_transformed = df.copy()

        # 1. Remove duplicates
        initial_count = len(df_transformed)
        df_transformed = df_transformed.drop_duplicates()
        duplicates_removed = initial_count - len(df_transformed)
        logger.info(f"Removed {duplicates_removed} duplicate rows")

        # 2. Handle null values (example: drop rows with any null)
        null_count_before = df_transformed.isnull().sum().sum()
        df_transformed = df_transformed.dropna()
        null_count_after = df_transformed.isnull().sum().sum()
        logger.info(f"Removed rows with null values: {null_count_before - null_count_after} nulls eliminated")

        # 3. Add derived columns (example)
        # if 'timestamp' in df_transformed.columns:
        #     df_transformed['processed_date'] = pd.Timestamp.now()

        # 4. Filter data (example)
        # if 'value' in df_transformed.columns:
        #     df_transformed = df_transformed[df_transformed['value'] > 0]

        # 5. Data type conversions (example)
        # Convert numeric strings to numbers where applicable
        for col in df_transformed.columns:
            if df_transformed[col].dtype == 'object':
                # Attempt to convert to numeric
                converted = pd.to_numeric(df_transformed[col], errors='ignore')
                if converted.dtype != 'object':
                    df_transformed[col] = converted

        logger.info(f"Transformation complete. Final shape: {df_transformed.shape}")
        return df_transformed

    def load(self, df: pd.DataFrame, output_path: str, format: str = "parquet") -> None:
        """
        Load data to destination.

        Args:
            df: DataFrame to load
            output_path: Output path
            format: Output format (parquet, csv, json)
        """
        logger.info("=== LOADING PHASE ===")

        if format == "parquet":
            self.writer.write_parquet(df=df, path=output_path)
        elif format == "csv":
            self.writer.write_csv(df=df, path=output_path, index=False)
        elif format == "json":
            self.writer.write_json(df=df, path=output_path)
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
            logger.info(f"Initial quality report: Rows={quality_report['total_rows']}, "
                       f"Cols={quality_report['total_columns']}, "
                       f"Duplicates={quality_report['duplicate_count']}")

            # Transform
            df_transformed = self.transform(df_raw)

            # Final quality check
            final_report = self.quality_checker.get_quality_report(df_transformed)
            logger.info(f"Final quality report: Rows={final_report['total_rows']}, "
                       f"Cols={final_report['total_columns']}")

            # Load
            self.load(df_transformed, output_path, output_format)

            logger.info("ETL pipeline completed successfully")

        except Exception as e:
            logger.error(f"ETL pipeline failed: {str(e)}")
            raise


def main():
    """Main entry point."""
    # Example usage
    input_path = str(Settings.RAW_DATA_PATH / "example_data.csv")
    output_path = str(Settings.PROCESSED_DATA_PATH / "example_output")

    etl = ExampleETL()
    etl.run(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    main()
