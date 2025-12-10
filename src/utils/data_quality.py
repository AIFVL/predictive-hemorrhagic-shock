"""
Data quality and validation utilities.
"""
from typing import Dict, List, Optional
import pandas as pd
from loguru import logger


class DataQualityChecker:
    """Data quality checking utilities using pandas."""

    @staticmethod
    def check_nulls(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Check for null values in specified columns.

        Args:
            df: DataFrame to check
            columns: List of columns to check (None = all columns)

        Returns:
            Dictionary with column names and null counts
        """
        if columns is None:
            columns = df.columns.tolist()

        logger.info("Checking for null values")

        null_counts = {}
        for col in columns:
            null_count = df[col].isnull().sum()
            null_counts[col] = null_count
            if null_count > 0:
                logger.warning(f"Column '{col}' has {null_count} null values")

        return null_counts

    @staticmethod
    def check_duplicates(df: pd.DataFrame,
                        subset: Optional[List[str]] = None) -> int:
        """
        Check for duplicate rows.

        Args:
            df: DataFrame to check
            subset: List of columns to consider for duplicates

        Returns:
            Number of duplicate rows
        """
        logger.info("Checking for duplicate rows")

        if subset:
            duplicates = df.duplicated(subset=subset).sum()
        else:
            duplicates = df.duplicated().sum()

        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
        else:
            logger.info("No duplicates found")

        return duplicates

    @staticmethod
    def get_column_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Get basic statistics for all columns.

        Args:
            df: DataFrame to analyze

        Returns:
            DataFrame with statistics
        """
        logger.info("Calculating column statistics")
        return df.describe()

    @staticmethod
    def check_data_types(df: pd.DataFrame) -> Dict[str, str]:
        """
        Get data types for all columns.

        Args:
            df: DataFrame to check

        Returns:
            Dictionary with column names and data types
        """
        logger.info("Checking data types")
        return {col: str(dtype) for col, dtype in df.dtypes.items()}

    @staticmethod
    def validate_schema(df: pd.DataFrame,
                       expected_columns: List[str],
                       strict: bool = True) -> bool:
        """
        Validate DataFrame schema against expected columns.

        Args:
            df: DataFrame to validate
            expected_columns: List of expected column names
            strict: If True, DataFrame must have exactly these columns

        Returns:
            True if schema is valid, False otherwise
        """
        logger.info("Validating schema")

        actual_columns = set(df.columns.tolist())
        expected_columns_set = set(expected_columns)

        if strict:
            is_valid = actual_columns == expected_columns_set
            if not is_valid:
                missing = expected_columns_set - actual_columns
                extra = actual_columns - expected_columns_set
                if missing:
                    logger.error(f"Missing columns: {missing}")
                if extra:
                    logger.error(f"Extra columns: {extra}")
        else:
            is_valid = expected_columns_set.issubset(actual_columns)
            if not is_valid:
                missing = expected_columns_set - actual_columns
                logger.error(f"Missing columns: {missing}")

        if is_valid:
            logger.info("Schema validation passed")
        else:
            logger.error("Schema validation failed")

        return is_valid

    @staticmethod
    def get_quality_report(df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data quality report.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with quality metrics
        """
        logger.info("Generating data quality report")

        checker = DataQualityChecker()

        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "null_counts": checker.check_nulls(df),
            "duplicate_count": checker.check_duplicates(df),
            "data_types": checker.check_data_types(df),
        }

        # Additional quality metrics
        report["numeric_columns"] = df.select_dtypes(include='number').columns.tolist()
        report["categorical_columns"] = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Memory usage
        report["memory_usage_bytes"] = df.memory_usage(deep=True).sum()

        # Missing data percentage
        missing_pct = df.isnull().sum() / len(df) * 100
        report["missing_percentage"] = {col: round(val, 2) for col, val in missing_pct.items()}

        logger.info("Data quality report generated")
        return report
