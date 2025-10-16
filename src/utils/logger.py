"""
Logging configuration and utilities.
"""
import sys
from pathlib import Path
from loguru import logger
from decouple import config


def setup_logger(log_file: str = None, log_level: str = None) -> None:
    """
    Configure the logger with file and console outputs.

    Args:
        log_file: Path to the log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Remove default logger
    logger.remove()

    # Get configuration
    level = log_level or config('LOG_LEVEL', default='INFO')
    log_path = log_file or config('LOG_FILE_PATH', default='logs/app.log')

    # Ensure logs directory exists
    log_dir = Path(log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Console handler with color
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # File handler with rotation
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="100 MB",
        retention="30 days",
        compression="zip",
    )

    logger.info(f"Logger configured with level: {level}")
    logger.info(f"Log file: {log_path}")


# Setup logger on import
setup_logger()
