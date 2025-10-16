"""
Tests for data quality utilities.
"""
import pytest
from pyspark.sql import functions as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.data_quality import DataQualityChecker


class TestDataQualityChecker:
    """Test cases for DataQualityChecker class."""

    def test_check_nulls_no_nulls(self, sample_data):
        """Test null checking with no nulls."""
        checker = DataQualityChecker()
        null_counts = checker.check_nulls(sample_data)

        assert all(count == 0 for count in null_counts.values())

    def test_check_nulls_with_nulls(self, spark):
        """Test null checking with nulls present."""
        data = [
            (1, "A", 100.0),
            (2, None, 200.0),
            (3, "C", None),
        ]
        df = spark.createDataFrame(data, ["id", "category", "value"])

        checker = DataQualityChecker()
        null_counts = checker.check_nulls(df)

        assert null_counts["id"] == 0
        assert null_counts["category"] == 1
        assert null_counts["value"] == 1

    def test_check_duplicates_no_duplicates(self, sample_data):
        """Test duplicate checking with no duplicates."""
        checker = DataQualityChecker()
        duplicates = checker.check_duplicates(sample_data)

        assert duplicates == 0

    def test_check_duplicates_with_duplicates(self, spark):
        """Test duplicate checking with duplicates present."""
        data = [
            (1, "A", 100),
            (2, "B", 200),
            (1, "A", 100),  # duplicate
            (2, "B", 200),  # duplicate
        ]
        df = spark.createDataFrame(data, ["id", "category", "value"])

        checker = DataQualityChecker()
        duplicates = checker.check_duplicates(df)

        assert duplicates == 2

    def test_check_data_types(self, sample_data):
        """Test data type checking."""
        checker = DataQualityChecker()
        data_types = checker.check_data_types(sample_data)

        assert "id" in data_types
        assert "category" in data_types
        assert "value" in data_types

    def test_validate_schema_strict(self, sample_data):
        """Test strict schema validation."""
        checker = DataQualityChecker()
        expected_columns = ["id", "category", "value", "quantity", "is_active"]

        is_valid = checker.validate_schema(sample_data, expected_columns, strict=True)
        assert is_valid is True

    def test_validate_schema_strict_fail(self, sample_data):
        """Test strict schema validation with missing columns."""
        checker = DataQualityChecker()
        expected_columns = ["id", "category", "value", "extra_column"]

        is_valid = checker.validate_schema(sample_data, expected_columns, strict=True)
        assert is_valid is False

    def test_validate_schema_non_strict(self, sample_data):
        """Test non-strict schema validation."""
        checker = DataQualityChecker()
        expected_columns = ["id", "category"]

        is_valid = checker.validate_schema(sample_data, expected_columns, strict=False)
        assert is_valid is True

    def test_get_quality_report(self, sample_data):
        """Test quality report generation."""
        checker = DataQualityChecker()
        report = checker.get_quality_report(sample_data)

        assert "total_rows" in report
        assert "total_columns" in report
        assert "null_counts" in report
        assert "duplicate_count" in report
        assert "data_types" in report

        assert report["total_rows"] == 5
        assert report["total_columns"] == 5
        assert report["duplicate_count"] == 0
