"""
Application settings and configuration management.
"""

from pathlib import Path

from decouple import config
from loguru import logger


class Settings:
    """Application settings class."""

    # Project paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_PATH = DATA_DIR / config("RAW_DATA_PATH", default="raw")
    PROCESSED_DATA_PATH = DATA_DIR / config("PROCESSED_DATA_PATH", default="processed")
    STAGING_DATA_PATH = DATA_DIR / config("STAGING_DATA_PATH", default="staging")
    OUTPUT_DATA_PATH = DATA_DIR / config("OUTPUT_DATA_PATH", default="output")
    LOGS_DIR = BASE_DIR / "logs"

    # Logging configuration
    LOG_LEVEL = config("LOG_LEVEL", default="INFO")
    LOG_FILE_PATH = BASE_DIR / config("LOG_FILE_PATH", default="logs/app.log")

    # Processing configuration
    BATCH_SIZE = config("BATCH_SIZE", default=10000, cast=int)
    MAX_RETRIES = config("MAX_RETRIES", default=3, cast=int)
    RETRY_DELAY = config("RETRY_DELAY", default=5, cast=int)

    # Database configuration
    DB_HOST = config("DB_HOST", default="localhost")
    DB_PORT = config("DB_PORT", default=5432, cast=int)
    DB_NAME = config("DB_NAME", default="shock_db")
    DB_USER = config("DB_USER", default="postgres")
    DB_PASSWORD = config("DB_PASSWORD", default="")

    # Cloud storage configuration
    AWS_ACCESS_KEY_ID = config("AWS_ACCESS_KEY_ID", default="")
    AWS_SECRET_ACCESS_KEY = config("AWS_SECRET_ACCESS_KEY", default="")
    AWS_REGION = config("AWS_REGION", default="us-east-1")
    S3_BUCKET = config("S3_BUCKET", default="")

    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        directories = [
            cls.RAW_DATA_PATH,
            cls.PROCESSED_DATA_PATH,
            cls.STAGING_DATA_PATH,
            cls.OUTPUT_DATA_PATH,
            cls.LOGS_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

    @classmethod
    def get_database_url(cls) -> str:
        """
        Get database connection URL.

        Returns:
            Database connection string
        """
        return f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"

    @classmethod
    def display_settings(cls):
        """Display current settings (for debugging)."""
        logger.info("Current Settings:")
        logger.info(f"  Base Directory: {cls.BASE_DIR}")
        logger.info(f"  Raw Data Path: {cls.RAW_DATA_PATH}")
        logger.info(f"  Processed Data Path: {cls.PROCESSED_DATA_PATH}")
        logger.info(f"  Log Level: {cls.LOG_LEVEL}")
        logger.info(f"  Batch Size: {cls.BATCH_SIZE}")


# Initialize settings on import
Settings.ensure_directories()
