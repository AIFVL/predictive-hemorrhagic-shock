"""
Centralized configuration manager.

Provides a singleton pattern for accessing pipeline and feature configurations
throughout the application. Eliminates the need to pass config_path parameters
everywhere and provides consistent error handling.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from .logger import logger


class ConfigurationManager:
    """
    Singleton configuration manager for the ML pipeline.
    
    Loads and provides access to both pipeline_config.yaml and feature_config.yaml
    with convenient getter methods and error handling.
    
    Usage:
        config = ConfigurationManager()
        model_config = config.get_model('random_forest')
        raw_data_path = config.get_path('raw_data')
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, base_dir: Optional[Path] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the configuration manager.
        
        Args:
            base_dir: Base directory for the project (default: /opt/airflow or cwd)
        """
        if self._initialized:
            return
            
        if base_dir is None:
            # Try /opt/airflow first (Airflow), then current directory
            if Path('/opt/airflow').exists():
                base_dir = Path('/opt/airflow')
            else:
                raise RuntimeError("Base directory must be specified when not running in Airflow environment.")

        self.base_dir = Path(base_dir)
        self._pipeline_config = None
        self._feature_config = None
        
        # Load configurations
        self._load_configs()
        self._initialized = True
        
        logger.info(f"ConfigurationManager initialized with base_dir: {self.base_dir}")
    
    def _load_configs(self) -> None:
        """Load both pipeline and feature configuration files."""
        # Load pipeline config
        pipeline_config_path = self.base_dir / 'src' / 'config' / 'pipeline_config.yaml'
        if not pipeline_config_path.exists():
            raise FileNotFoundError(
                f"Pipeline configuration not found at: {pipeline_config_path}"
            )
        
        with open(pipeline_config_path, 'r') as f:
            self._pipeline_config = yaml.safe_load(f)
        
        logger.debug(f"Loaded pipeline config from: {pipeline_config_path}")
        
        # Load feature config
        feature_config_path = self.base_dir / 'src' / 'config' / 'feature_config.yaml'
        if not feature_config_path.exists():
            raise FileNotFoundError(
                f"Feature configuration not found at: {feature_config_path}"
            )
        
        with open(feature_config_path, 'r') as f:
            self._feature_config = yaml.safe_load(f)
        
        logger.debug(f"Loaded feature config from: {feature_config_path}")
    
    def reload(self) -> None:
        """Reload configuration files from disk."""
        logger.info("Reloading configurations...")
        self._load_configs()
    
    # -------------------------------------------------------------------------
    # Pipeline Config Getters
    # -------------------------------------------------------------------------
    
    def get_path(self, path_key: str, version: Optional[str] = None, 
                 model_name: Optional[str] = None) -> Path:
        """
        Get a configured path with template substitution.
        
        Args:
            path_key: Key from paths section (e.g., 'raw_data', 'model_output')
            version: Version to substitute (default: from config)
            model_name: Model name to substitute if needed
        
        Returns:
            Absolute Path object
        
        Raises:
            KeyError: If path_key not found in config
        """
        if path_key not in self._pipeline_config['paths']:
            raise KeyError(
                f"Path key '{path_key}' not found. "
                f"Available keys: {list(self._pipeline_config['paths'].keys())}"
            )
        
        path_template = self._pipeline_config['paths'][path_key]
        
        # Get version from config if not provided
        if version is None:
            version = self.get_version()
        
        # Substitute templates
        path_str = path_template.format(version=version, model_name=model_name or '')
        
        return self.base_dir / path_str
    
    def get_model(self, model_name: str) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Args:
            model_name: Name of the model (e.g., 'random_forest')
        
        Returns:
            Model configuration dict with 'module', 'class', 'params'
        
        Raises:
            KeyError: If model not found in config
        """
        if model_name not in self._pipeline_config['models']:
            raise KeyError(
                f"Model '{model_name}' not found. "
                f"Available models: {list(self._pipeline_config['models'].keys())}"
            )
        
        return self._pipeline_config['models'][model_name]
    
    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all configured models."""
        return self._pipeline_config['models']
    
    def get_model_names(self) -> List[str]:
        """Get list of all configured model names."""
        return list(self._pipeline_config['models'].keys())
    
    def get_version(self) -> str:
        """Get the configured data version."""
        return self._pipeline_config['version']
    
    def get_random_seed(self) -> int:
        """Get the configured random seed."""
        return self._pipeline_config['random_seed']
    
    def get_cv_config(self) -> Dict[str, Any]:
        """Get cross-validation configuration."""
        return self._pipeline_config['cross_validation']
    
    def get_cleaning_config(self) -> Dict[str, Any]:
        """Get data cleaning configuration."""
        return self._pipeline_config['cleaning']
    
    def get_data_split_config(self) -> Dict[str, Any]:
        """Get train/test split configuration."""
        return self._pipeline_config['data_split']
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get model evaluation configuration."""
        return self._pipeline_config['evaluation']
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return self._pipeline_config['visualization']
    
    # -------------------------------------------------------------------------
    # Feature Config Getters
    # -------------------------------------------------------------------------
    
    def get_numerical_features(self) -> List[str]:
        """Get list of numerical feature names from the numerical_features dict."""
        numerical_features_dict = self._feature_config.get('numerical_features', {})
        # numerical_features is a dict with feature names as keys
        return list(numerical_features_dict.keys()) if isinstance(numerical_features_dict, dict) else []
    
    def get_binary_features(self) -> List[str]:
        """Get list of binary feature names from the binary_features dict."""
        binary_features_dict = self._feature_config.get('binary_features', {})
        # binary_features is a dict with feature names as keys
        return list(binary_features_dict.keys()) if isinstance(binary_features_dict, dict) else []
    
    def get_target_variable(self) -> str:
        """Get the target variable name from target.name."""
        target_config = self._feature_config.get('target', {})
        if 'name' not in target_config:
            raise KeyError("Target variable not found in feature config (expected 'target.name')")
        return target_config['name']
    
    def get_excluded_variables(self) -> List[str]:
        """Get list of variables to exclude from excluded_variables."""
        return self._feature_config.get('excluded_variables', [])
    
    def get_aggregation_configs(self) -> Dict[str, Dict]:
        """Get all aggregation feature configurations from aggregated_features."""
        return self._feature_config.get('aggregated_features', {})
    
    def get_categorical_configs(self) -> Dict[str, Dict]:
        """Get all categorical feature configurations from categorical_features."""
        return self._feature_config.get('categorical_features', {})
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get data validation rules."""
        return self._feature_config.get('validation_rules', {})
    
    def get_final_features(self) -> List[str]:
        """
        Build the complete list of final features from config.
        
        Returns:
            List of all feature names (numerical, binary, aggregated, categorical)
        """
        final_features = []
        
        # Add base features
        final_features.extend(self.get_numerical_features())
        final_features.extend(self.get_binary_features())
        
        # Add aggregated features (if enabled)
        for feature_name, feature_config in self.get_aggregation_configs().items():
            if feature_config.get('enabled', True):
                final_features.append(feature_name)
        
        # Add categorical features (if enabled)
        for feature_name, feature_config in self.get_categorical_configs().items():
            if feature_config.get('enabled', True):
                final_features.append(feature_name)
        
        return final_features
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get any configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., 'cleaning.output_format')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        
        Example:
            output_format = config.get('cleaning.output_format')
            n_folds = config.get('cross_validation.n_folds', 5)
        """
        keys = key_path.split('.')
        value = self._pipeline_config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            # Try feature config
            value = self._feature_config
            try:
                for key in keys:
                    value = value[key]
                return value
            except (KeyError, TypeError):
                return default
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ConfigurationManager(base_dir={self.base_dir}, "
            f"version={self.get_version()}, "
            f"models={self.get_model_names()})"
        )


# Global singleton instance
_config_instance = None


def get_config(base_dir: Optional[Path] = None) -> ConfigurationManager:
    """
    Get the global ConfigurationManager instance.
    
    Args:
        base_dir: Base directory (only used on first call)
    
    Returns:
        ConfigurationManager singleton instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigurationManager(base_dir)
    return _config_instance
