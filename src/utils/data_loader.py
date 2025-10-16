"""
Data loading utilities for various file formats.
"""
from typing import Optional
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from loguru import logger


class DataLoader:
    """Utility class for loading data from various sources."""

    def __init__(self, spark: SparkSession):
        """
        Initialize DataLoader.

        Args:
            spark: SparkSession instance
        """
        self.spark = spark

    def load_csv(self,
                 path: str,
                 header: bool = True,
                 infer_schema: bool = True,
                 delimiter: str = ",",
                 **options) -> DataFrame:
        """
        Load CSV file into DataFrame.

        Args:
            path: Path to CSV file
            header: Whether first row contains headers
            infer_schema: Whether to infer schema automatically
            delimiter: Field delimiter
            **options: Additional Spark CSV options

        Returns:
            DataFrame containing the data
        """
        logger.info(f"Loading CSV from: {path}")

        df = self.spark.read.format("csv") \
            .option("header", header) \
            .option("inferSchema", infer_schema) \
            .option("delimiter", delimiter)

        # Apply additional options
        for key, value in options.items():
            df = df.option(key, value)

        df = df.load(path)

        logger.info(f"Loaded {df.count()} rows with {len(df.columns)} columns")
        return df

    def load_parquet(self, path: str) -> DataFrame:
        """
        Load Parquet file into DataFrame.

        Args:
            path: Path to Parquet file

        Returns:
            DataFrame containing the data
        """
        logger.info(f"Loading Parquet from: {path}")
        df = self.spark.read.parquet(path)
        logger.info(f"Loaded {df.count()} rows with {len(df.columns)} columns")
        return df

    def load_json(self, path: str, multiline: bool = False) -> DataFrame:
        """
        Load JSON file into DataFrame.

        Args:
            path: Path to JSON file
            multiline: Whether JSON is multiline format

        Returns:
            DataFrame containing the data
        """
        logger.info(f"Loading JSON from: {path}")
        df = self.spark.read.option("multiline", multiline).json(path)
        logger.info(f"Loaded {df.count()} rows with {len(df.columns)} columns")
        return df

    def load_jdbc(self,
                  url: str,
                  table: str,
                  user: str,
                  password: str,
                  driver: str = "org.postgresql.Driver",
                  **options) -> DataFrame:
        """
        Load data from JDBC source.

        Args:
            url: JDBC connection URL
            table: Table name or SQL query
            user: Database user
            password: Database password
            driver: JDBC driver class name
            **options: Additional JDBC options

        Returns:
            DataFrame containing the data
        """
        logger.info(f"Loading from JDBC: {table}")

        df = self.spark.read.format("jdbc") \
            .option("url", url) \
            .option("dbtable", table) \
            .option("user", user) \
            .option("password", password) \
            .option("driver", driver)

        # Apply additional options
        for key, value in options.items():
            df = df.option(key, value)

        df = df.load()

        logger.info(f"Loaded {df.count()} rows with {len(df.columns)} columns")
        return df


class DataWriter:
    """Utility class for writing data to various formats."""

    @staticmethod
    def write_csv(df: DataFrame,
                  path: str,
                  mode: str = "overwrite",
                  header: bool = True,
                  **options) -> None:
        """
        Write DataFrame to CSV.

        Args:
            df: DataFrame to write
            path: Output path
            mode: Write mode (overwrite, append, ignore, error)
            header: Whether to write header row
            **options: Additional Spark CSV options
        """
        logger.info(f"Writing CSV to: {path}")

        writer = df.write.format("csv") \
            .option("header", header) \
            .mode(mode)

        for key, value in options.items():
            writer = writer.option(key, value)

        writer.save(path)
        logger.info(f"CSV written successfully to: {path}")

    @staticmethod
    def write_parquet(df: DataFrame,
                      path: str,
                      mode: str = "overwrite",
                      partition_by: Optional[list] = None,
                      **options) -> None:
        """
        Write DataFrame to Parquet.

        Args:
            df: DataFrame to write
            path: Output path
            mode: Write mode (overwrite, append, ignore, error)
            partition_by: Columns to partition by
            **options: Additional Spark Parquet options
        """
        logger.info(f"Writing Parquet to: {path}")

        writer = df.write.format("parquet").mode(mode)

        if partition_by:
            writer = writer.partitionBy(*partition_by)

        for key, value in options.items():
            writer = writer.option(key, value)

        writer.save(path)
        logger.info(f"Parquet written successfully to: {path}")

    @staticmethod
    def write_json(df: DataFrame,
                   path: str,
                   mode: str = "overwrite") -> None:
        """
        Write DataFrame to JSON.

        Args:
            df: DataFrame to write
            path: Output path
            mode: Write mode (overwrite, append, ignore, error)
        """
        logger.info(f"Writing JSON to: {path}")
        df.write.mode(mode).json(path)
        logger.info(f"JSON written successfully to: {path}")
