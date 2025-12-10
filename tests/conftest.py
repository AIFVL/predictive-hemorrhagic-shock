"""
Pytest configuration and fixtures.
"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_data():
    """
    Create sample pandas DataFrame for testing.

    Returns:
        Sample pandas DataFrame
    """
    data = {
        "id": [1, 2, 3, 4, 5],
        "category": ["A", "B", "A", "C", "B"],
        "value": [100.0, 200.0, 150.0, 300.0, 250.0],
        "quantity": [10, 20, 15, 30, 25],
        "is_active": [True, False, True, True, False],
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_nulls():
    """
    Create sample pandas DataFrame with null values for testing.

    Returns:
        Sample pandas DataFrame with nulls
    """
    data = {
        "id": [1, 2, 3, 4, 5],
        "category": ["A", None, "A", "C", "B"],
        "value": [100.0, 200.0, None, 300.0, 250.0],
        "quantity": [10, 20, 15, None, 25],
        "is_active": [True, False, None, True, False],
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_duplicates():
    """
    Create sample pandas DataFrame with duplicate rows for testing.

    Returns:
        Sample pandas DataFrame with duplicates
    """
    data = {
        "id": [1, 2, 3, 2, 4],
        "category": ["A", "B", "A", "B", "C"],
        "value": [100.0, 200.0, 150.0, 200.0, 300.0],
    }

    return pd.DataFrame(data)
