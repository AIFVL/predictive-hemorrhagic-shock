"""
Pytest configuration and fixtures.
"""
import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """
    Create a Spark session for testing.

    Returns:
        SparkSession instance
    """
    spark = SparkSession.builder \
        .appName("test") \
        .master("local[2]") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    yield spark

    spark.stop()


@pytest.fixture
def sample_data(spark):
    """
    Create sample DataFrame for testing.

    Args:
        spark: SparkSession instance

    Returns:
        Sample DataFrame
    """
    data = [
        (1, "A", 100.0, 10, True),
        (2, "B", 200.0, 20, False),
        (3, "A", 150.0, 15, True),
        (4, "C", 300.0, 30, True),
        (5, "B", 250.0, 25, False),
    ]

    columns = ["id", "category", "value", "quantity", "is_active"]

    return spark.createDataFrame(data, columns)
