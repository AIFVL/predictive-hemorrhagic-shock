"""
Spark session configuration and initialization.
"""
from typing import Optional, Dict, Any
from pyspark.sql import SparkSession
from decouple import config
from loguru import logger


class SparkConfig:
    """Configuration class for Spark session management."""

    def __init__(self, app_name: Optional[str] = None):
        """
        Initialize Spark configuration.

        Args:
            app_name: Name of the Spark application
        """
        self.app_name = app_name or config('SPARK_APP_NAME', default='shock_data_analysis')
        self.master = config('SPARK_MASTER', default='local[*]')
        self.driver_memory = config('SPARK_DRIVER_MEMORY', default='4g')
        self.executor_memory = config('SPARK_EXECUTOR_MEMORY', default='4g')
        self.executor_cores = config('SPARK_EXECUTOR_CORES', default='2')
        self.sql_shuffle_partitions = config('SPARK_SQL_SHUFFLE_PARTITIONS', default='200')

    def get_spark_session(self, additional_configs: Optional[Dict[str, Any]] = None) -> SparkSession:
        """
        Create and return a configured Spark session.

        Args:
            additional_configs: Additional Spark configurations

        Returns:
            Configured SparkSession instance
        """
        logger.info(f"Creating Spark session: {self.app_name}")

        builder = SparkSession.builder \
            .appName(self.app_name) \
            .master(self.master) \
            .config("spark.driver.memory", self.driver_memory) \
            .config("spark.executor.memory", self.executor_memory) \
            .config("spark.executor.cores", self.executor_cores) \
            .config("spark.sql.shuffle.partitions", self.sql_shuffle_partitions) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.sources.partitionOverwriteMode", "dynamic")

        # Add additional configurations if provided
        if additional_configs:
            for key, value in additional_configs.items():
                builder = builder.config(key, value)

        spark = builder.getOrCreate()

        # Set log level
        spark.sparkContext.setLogLevel("WARN")

        logger.info(f"Spark session created successfully: {spark.version}")
        logger.info(f"Spark UI available at: {spark.sparkContext.uiWebUrl}")

        return spark

    @staticmethod
    def stop_spark_session(spark: SparkSession) -> None:
        """
        Stop the Spark session.

        Args:
            spark: SparkSession instance to stop
        """
        logger.info("Stopping Spark session")
        spark.stop()
        logger.info("Spark session stopped")


def get_spark_session(app_name: Optional[str] = None,
                     additional_configs: Optional[Dict[str, Any]] = None) -> SparkSession:
    """
    Convenience function to get a configured Spark session.

    Args:
        app_name: Name of the Spark application
        additional_configs: Additional Spark configurations

    Returns:
        Configured SparkSession instance
    """
    spark_config = SparkConfig(app_name=app_name)
    return spark_config.get_spark_session(additional_configs=additional_configs)
