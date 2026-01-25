"""
Logging configuration and utilities.

Provides centralized logging setup using loguru with colored console output
and file rotation. Configuration can be loaded from pipeline_config.yaml.
"""
import sys
from pathlib import Path
from loguru import logger as _logger
from typing import Optional
import json


def _patched_format(record):
    """
    Custom formatter that pretty-prints dicts and lists.
    """
    message = record["message"]
    
    # If message is a dict or list, pretty print it
    if isinstance(message, (dict, list)):
        record["message"] = "\n" + json.dumps(message, indent=2, default=str)
    
    return True


# Export the configured logger instance
logger = _logger


def log_section(title: str, width: int = 60) -> None:
    """
    Log a section title with formatting.
    
    Args:
        title: Section title
        width: Width of the separator line
    """
    logger.info("=" * width)
    logger.info(title)
    logger.info("=" * width)


def log_subsection(title: str, width: int = 40) -> None:
    """
    Log a subsection title with formatting.
    
    Args:
        title: Subsection title
        width: Width of the separator line
    """
    logger.info("-" * width)
    logger.info(title)
    logger.info("-" * width)


def setup_logger(
    log_file: Optional[str] = None,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """
    Configure the logger with file and console outputs.
    
    Uses configuration from pipeline_config.yaml if available,
    otherwise falls back to defaults or provided parameters.

    Args:
        log_file: Path to the log file (default: logs/pipeline.log)
        log_level: Logging level (default: INFO)
        log_format: Custom log format string
    """
    # Remove default logger
    logger.remove()

    # Try to load from config if available
    try:
        from .config_manager import get_config
        config = get_config()
        logging_config = config.get('logging', {})
        
        level = log_level or logging_config.get('level', 'INFO')
        log_path = log_file or logging_config.get('log_dir', 'logs') + '/pipeline.log'
        
    except Exception:
        # Fallback to defaults if config not available
        level = log_level or 'INFO'
        log_path = log_file or 'logs/pipeline.log'

    # Ensure logs directory exists
    log_dir = Path(log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Console handler with color
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    
    logger.add(
        sys.stdout,
        format=log_format or console_format,
        level=level,
        colorize=True,
        filter=_patched_format
    )

    # File handler with rotation (no color codes)
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} - "
        "{message}"
    )
    
    logger.add(
        log_path,
        format=file_format,
        level=level,
        rotation="100 MB",
        retention="30 days",
        compression="zip",
        filter=_patched_format
    )

    logger.info(f"Logger configured with level: {level}")
    logger.info(f"Log file: {log_path}")


# Setup logger on import with defaults
# Can be reconfigured later by calling setup_logger() with parameters
try:
    setup_logger()
except Exception as e:
    # If setup fails during import, create minimal logger
    logger.add(sys.stdout, level="INFO")
    logger.warning(f"Could not fully configure logger: {e}")
