"""
Tests for data loader utilities.
"""
import pytest
from pyspark.sql import SparkSession

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""

    def test_initialization(self, spark):
        """Test DataLoader initialization."""
        loader = DataLoader(spark)
        assert loader.spark is not None
        assert isinstance(loader.spark, SparkSession)

    def test_load_csv_basic(self, spark, tmp_path):
        """Test basic CSV loading."""
        # Create temporary CSV file
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,name,value\n1,A,100\n2,B,200\n")

        loader = DataLoader(spark)
        df = loader.load_csv(str(csv_file))

        assert df.count() == 2
        assert len(df.columns) == 3
        assert "id" in df.columns
        assert "name" in df.columns
        assert "value" in df.columns

    def test_load_csv_no_header(self, spark, tmp_path):
        """Test CSV loading without header."""
        csv_file = tmp_path / "test_no_header.csv"
        csv_file.write_text("1,A,100\n2,B,200\n")

        loader = DataLoader(spark)
        df = loader.load_csv(str(csv_file), header=False)

        assert df.count() == 2
        assert len(df.columns) == 3
