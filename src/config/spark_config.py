"""
General configuration utilities for ML project.
"""
from typing import Optional, Dict, Any
from decouple import config
from loguru import logger


class MLConfig:
    """Configuration class for ML workflows."""

    def __init__(self):
        """
        Initialize ML configuration using environment variables.
        """
        self.mlflow_tracking_uri = config('MLFLOW_TRACKING_URI', default='sqlite:///mlflow.db')
        self.mlflow_experiment_name = config('MLFLOW_EXPERIMENT_NAME', default='shock_prediction_experiments')
        self.data_dir = config('DATA_DIR', default='data/')
        self.model_dir = config('MODEL_DIR', default='models/')
        self.reports_dir = config('REPORTS_DIR', default='reports/')

    def get_mlflow_config(self) -> Dict[str, str]:
        """
        Get MLflow configuration.

        Returns:
            Dictionary with MLflow configuration
        """
        return {
            'tracking_uri': self.mlflow_tracking_uri,
            'experiment_name': self.mlflow_experiment_name
        }


# Global configuration instance
_config = None


def get_config() -> MLConfig:
    """
    Get global configuration instance.

    Returns:
        MLConfig instance
    """
    global _config
    if _config is None:
        _config = MLConfig()
    return _config
