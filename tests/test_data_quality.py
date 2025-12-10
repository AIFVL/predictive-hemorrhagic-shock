"""
Tests for data quality utilities.
"""
import pytest
import pandas as pd
import numpy as np

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

    def test_check_nulls_with_nulls(self, sample_data_with_nulls):
        """Test null checking with nulls present."""
        checker = DataQualityChecker()
        null_counts = checker.check_nulls(sample_data_with_nulls)

        assert null_counts["id"] == 0
        assert null_counts["category"] == 1
        assert null_counts["value"] == 1
        assert null_counts["quantity"] == 1

    def test_check_duplicates_no_duplicates(self, sample_data):
        """Test duplicate checking with no duplicates."""
        checker = DataQualityChecker()
        duplicates = checker.check_duplicates(sample_data)

        assert duplicates == 0

    def test_check_duplicates_with_duplicates(self, sample_data_with_duplicates):
        """Test duplicate checking with duplicates present."""
        checker = DataQualityChecker()
        duplicates = checker.check_duplicates(sample_data_with_duplicates)

        assert duplicates == 1  # One duplicate row

    def test_check_data_types(self, sample_data):
        """Test data type checking."""
        checker = DataQualityChecker()
        data_types = checker.check_data_types(sample_data)

        assert "id" in data_types
        assert "category" in data_types
        assert "value" in data_types
        assert isinstance(data_types["id"], str)
        assert isinstance(data_types["category"], str)
        assert isinstance(data_types["value"], str)

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

    def test_check_duplicates_subset(self, sample_data_with_duplicates):
        """Test duplicate checking with subset of columns."""
        checker = DataQualityChecker()

        # Check duplicates considering only 'id' and 'category' columns
        duplicates = checker.check_duplicates(sample_data_with_duplicates, subset=['id', 'category'])

        # Since there are two rows with same id=2 and category=B, this should return 1 duplicate
        assert duplicates == 1

    def test_check_data_types_detailed(self, sample_data):
        """Test detailed data type checking."""
        checker = DataQualityChecker()
        data_types = checker.check_data_types(sample_data)

        # Check specific types
        assert 'int64' in data_types['id'] or 'int32' in data_types['id']
        assert 'object' in data_types['category']
        assert 'float' in data_types['value']
