"""
Centralized configuration manager.

config/pipeline_config.yaml is the single source of truth for all pipeline
parameters, feature definitions and validation rules.

Usage:
    config = get_config()
    
    # Generic access with dot notation (returns dicts you can chain .get() on):
    config.get('models.lightgbm.params')                    # Dict
    config.get('cleaning.validation_rules.valid_ranges')              # Dict
    config.get('cleaning.validation_rules.valid_ranges.EDAD')         # Dict with min/max
    
    # Paths (auto-templated):
    config.get_path('raw_data')                              # Uses dataset_version
    config.get_path('model_output', model_name='lightgbm')   # Uses pipeline_version

    # Scalars:
    config.get('general_config.version')                                    # pipeline version
    config.get('general_config.dataset_version')                            # dataset version
    config.get('general_config.random_seed')                                # random seed
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional

from .logger import logger


class ConfigurationManager:
    """
    Singleton configuration manager for the ML pipeline.

    Loads pipeline_config.yaml and provides:
    - Generic nested dict access via .get(dot_notation) - returns dicts with .get() support
    - Absolute path construction with template substitution via .get_path()
    - Generic access to all values (including top-level scalars) via .get(dot_notation)
    
    All config values are returned as native Python dicts that support
    chained .get() calls for clean nested access.
    
    Example:
        config = get_config()
        config.get('models.lightgbm.target_recall')                # 0.85
        config.get('cleaning.validation_rules.valid_ranges.EDAD')  # {"min": 18, "max": 120}
        config.get_path('model_output', model_name='lightgbm')     # Templated path
    """

    _instance = None
    _initialized = False

    def __new__(cls, base_dir: Optional[Path] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, base_dir: Optional[Path] = None):
        if self._initialized:
            return

        if base_dir is None:
            if Path('/opt/airflow').exists():
                base_dir = Path('/opt/airflow')
            else:
                raise RuntimeError(
                    "Base directory must be specified when not running in Airflow environment."
                )

        self.base_dir = Path(base_dir)
        self._cfg: Dict[str, Any] = {}

        self._load_config()
        self._initialized = True
        logger.info(f"ConfigurationManager initialised — base_dir: {self.base_dir}")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        """Load pipeline_config.yaml (single source of truth)."""
        path = self.base_dir / 'config' / 'pipeline_config.yaml'
        if not path.exists():
            raise FileNotFoundError(f"Pipeline configuration not found at: {path}")
        with open(path, 'r') as f:
            self._cfg = yaml.safe_load(f)
        logger.debug(f"Loaded config from: {path}")

    def reload(self) -> None:
        """Reload configuration file from disk."""
        logger.info("Reloading configuration...")
        self._load_config()

    # ------------------------------------------------------------------
    # Paths (with template substitution)
    # ------------------------------------------------------------------

    def get_path(
        self,
        path_key: str,
        version: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Path:
        """
        Return an absolute Path for path_key with automatic template substitution.

        Templates support:
        - {version}          → dataset_version (for data/raw, data/processed, splits)
        - {pipeline_version} → pipeline version (for models/, output/)
        - {model_name}       → model name (for model-specific paths)

        Args:
            path_key: Key in paths config (e.g., 'raw_data', 'model_output')
            version: Override dataset version for {version} placeholder
            model_name: Model name for {model_name} placeholder

        Returns:
            Absolute Path with templates resolved

        Examples:
            config.get_path('raw_data')                          # data/raw/v1/shock.csv
            config.get_path('model_output', model_name='tree')   # output/v1/models/tree/model.joblib
        """
        paths = self.get('general_config.paths')
        if path_key not in paths:
            raise KeyError(
                f"Path key '{path_key}' not found. Available: {list(paths.keys())}"
            )
        tmpl = paths[path_key]
        dataset_ver = version if version is not None else self.get('general_config.dataset_version')
        path_str = tmpl.format(
            version=dataset_ver,
            pipeline_version=self.get('general_config.version'),
            model_name=model_name or '',
        )
        return self.base_dir / path_str

    # ------------------------------------------------------------------
    # Generic nested accessor with dot notation
    # ------------------------------------------------------------------

    def get(self, key_path: str) -> Any:
        """
        Get any value from config using dot notation.
        
        Returns native Python dicts that support chained .get() calls.

        Args:
            key_path: Dot-separated path (e.g., 'models.lightgbm.params')

        Returns:
            Config value (int, str, dict, list, etc.)

        Examples:
            config.get('general_config.version')               # 'v1'
            config.get('models.lightgbm.target_recall')        # 0.85
            config.get('cleaning.validation_rules.valid_ranges')      # {"EDAD": {"min": 18, ...}, ...}
            config.get('cleaning.validation_rules.valid_ranges.EDAD') # {"min": 18, "max": 120}
            config.get('models.lightgbm.params')               # Dict
            
        Chaining (all return dicts that support .get()):
            models_cfg = config.get('models')                  # Dict of all models
            lgb_params = models_cfg.get('lightgbm', {}).get('params')  # LightGBM params
        """
        keys = key_path.split('.')
        value: Any = self._cfg
        for k in keys:
            if not isinstance(value, dict) or k not in value:
                raise KeyError(f"Missing configuration key: '{key_path}'")
            value = value[k]
        return value

    # ------------------------------------------------------------------
    # Backward compatibility layer (deprecated but kept for now)
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ConfigurationManager(base_dir={self.base_dir}, "
            f"version={self.get('general_config.version')})"
        )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[ConfigurationManager] = None


def get_config(base_dir: Optional[Path] = None) -> ConfigurationManager:
    """Return the global ConfigurationManager singleton."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigurationManager(base_dir)
    return _config_instance
