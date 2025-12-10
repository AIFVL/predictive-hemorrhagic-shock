"""
Tests for data loader utilities.
"""
import pytest
import pandas as pd
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""

    def test_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        # DataLoader should initialize without arguments now
        assert loader is not None

    def test_load_csv_basic(self, tmp_path):
        """Test basic CSV loading."""
        # Create temporary CSV file
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,name,value\n1,A,100\n2,B,200\n")

        loader = DataLoader()
        df = loader.load_csv(str(csv_file))

        assert len(df) == 2  # Number of rows
        assert len(df.columns) == 3
        assert "id" in df.columns
        assert "name" in df.columns
        assert "value" in df.columns
        assert df.iloc[0]["id"] == 1
        assert df.iloc[0]["name"] == "A"
        assert df.iloc[1]["value"] == 200

    def test_load_csv_no_header(self, tmp_path):
        """Test CSV loading without header."""
        csv_file = tmp_path / "test_no_header.csv"
        csv_file.write_text("1,A,100\n2,B,200\n")

        loader = DataLoader()
        df = loader.load_csv(str(csv_file), header=None)

        assert len(df) == 2  # Number of rows
        assert len(df.columns) == 3

    def test_load_csv_with_options(self, tmp_path):
        """Test CSV loading with additional options."""
        # Create temporary CSV file with custom delimiter
        csv_file = tmp_path / "test_delimiter.csv"
        csv_file.write_text("id|name|value\n1|A|100\n2|B|200\n")

        loader = DataLoader()
        df = loader.load_csv(str(csv_file), delimiter="|")

        assert len(df) == 2
        assert len(df.columns) == 3
        assert df.iloc[0]["id"] == 1
        assert df.iloc[0]["name"] == "A"
        assert df.iloc[0]["value"] == 100

    def test_load_parquet(self, tmp_path):
        """Test parquet loading."""
        # Create a temporary parquet file
        parquet_file = tmp_path / "test.parquet"
        test_data = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "value": [100, 200, 300]
        })
        test_data.to_parquet(parquet_file)

        loader = DataLoader()
        df = loader.load_parquet(str(parquet_file))

        assert len(df) == 3
        assert len(df.columns) == 3
        assert "id" in df.columns
        assert "name" in df.columns
        assert "value" in df.columns
        assert df.iloc[0]["id"] == 1

    def test_load_json(self, tmp_path):
        """Test JSON loading."""
        # Create a temporary JSON file
        json_file = tmp_path / "test.json"
        test_data = pd.DataFrame({
            "id": [1, 2],
            "name": ["A", "B"],
            "value": [100, 200]
        })
        test_data.to_json(json_file, orient="records", lines=True)

        loader = DataLoader()
        df = loader.load_json(str(json_file))

        assert len(df) == 2
        assert len(df.columns) == 3
        assert "id" in df.columns
        assert "name" in df.columns
        assert "value" in df.columns
