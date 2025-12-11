"""
Data loading utilities for various file formats.
"""

import os

import pandas as pd
from loguru import logger


class DataLoader:
    """Utility class for loading data from various sources using pandas."""

    def __init__(self):
        """Initialize DataLoader."""
        pass

    def load_csv(
        self, path: str, header: str = "infer", delimiter: str = ",", **options
    ) -> pd.DataFrame:
        """
        Load CSV file into DataFrame.

        Args:
            path: Path to CSV file
            header: Row number(s) to use as the column names
            delimiter: Field delimiter
            **options: Additional pandas CSV options

        Returns:
            DataFrame containing the data
        """
        logger.info(f"Loading CSV from: {path}")

        df = pd.read_csv(path, sep=delimiter, header=header, **options)

        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        return df

    def load_parquet(self, path: str, **options) -> pd.DataFrame:
        """
        Load Parquet file into DataFrame.

        Args:
            path: Path to Parquet file
            **options: Additional pandas parquet options

        Returns:
            DataFrame containing the data
        """
        logger.info(f"Loading Parquet from: {path}")
        df = pd.read_parquet(path, **options)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        return df

    def load_json(self, path: str, **options) -> pd.DataFrame:
        """
        Load JSON file into DataFrame.

        Args:
            path: Path to JSON file
            **options: Additional pandas JSON options

        Returns:
            DataFrame containing the data
        """
        logger.info(f"Loading JSON from: {path}")
        df = pd.read_json(path, **options)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        return df

    def load_excel(self, path: str, sheet_name: str = 0, **options) -> pd.DataFrame:
        """
        Load Excel file into DataFrame.

        Args:
            path: Path to Excel file
            sheet_name: Name or index of sheet to load
            **options: Additional pandas Excel options

        Returns:
            DataFrame containing the data
        """
        logger.info(f"Loading Excel from: {path}, sheet: {sheet_name}")
        df = pd.read_excel(path, sheet_name=sheet_name, **options)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        return df


class DataWriter:
    """Utility class for writing data to various formats using pandas."""

    @staticmethod
    def write_csv(
        df: pd.DataFrame, path: str, index: bool = False, header: bool = True, **options
    ) -> None:
        """
        Write DataFrame to CSV.

        Args:
            df: DataFrame to write
            path: Output path
            index: Whether to write row names
            header: Whether to write column names
            **options: Additional pandas CSV options
        """
        logger.info(f"Writing CSV to: {path}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

        df.to_csv(path, index=index, header=header, **options)
        logger.info(f"CSV written successfully to: {path}")

    @staticmethod
    def write_parquet(
        df: pd.DataFrame,
        path: str,
        engine: str = "auto",
        compression: str = "UNCOMPRESSED",
        **options,
    ) -> None:
        """
        Write DataFrame to Parquet.

        Args:
            df: DataFrame to write
            path: Output path
            engine: Parquet library to use
            compression: Compression algorithm
            **options: Additional pandas parquet options
        """
        logger.info(f"Writing Parquet to: {path}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

        df.to_parquet(path, engine=engine, compression=compression, **options)
        logger.info(f"Parquet written successfully to: {path}")

    @staticmethod
    def write_json(
        df: pd.DataFrame, path: str, orient: str = "records", lines: bool = True, **options
    ) -> None:
        """
        Write DataFrame to JSON.

        Args:
            df: DataFrame to write
            path: Output path
            orient: Indication of expected JSON string format
            lines: Write JSON format with one object per line
            **options: Additional pandas JSON options
        """
        logger.info(f"Writing JSON to: {path}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

        df.to_json(path, orient=orient, lines=lines, **options)
        logger.info(f"JSON written successfully to: {path}")

    @staticmethod
    def write_excel(
        df: pd.DataFrame, path: str, sheet_name: str = "Sheet1", index: bool = False, **options
    ) -> None:
        """
        Write DataFrame to Excel.

        Args:
            df: DataFrame to write
            path: Output path
            sheet_name: Name of sheet to write to
            index: Whether to write row names
            **options: Additional pandas Excel options
        """
        logger.info(f"Writing Excel to: {path}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

        df.to_excel(path, sheet_name=sheet_name, index=index, **options)
        logger.info(f"Excel written successfully to: {path}")
