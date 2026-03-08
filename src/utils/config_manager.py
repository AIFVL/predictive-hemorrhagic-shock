"""
Centralized configuration manager.

config/pipeline_config.yaml is the single source of truth for all pipeline
parameters, feature definitions and validation rules.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logger import logger


class ConfigurationManager:
    """
    Singleton configuration manager for the ML pipeline.

    Loads pipeline_config.yaml and provides typed getter methods for every
    section.  All previous feature-config getters now read from the same file.

    Usage:
        config = get_config()
        model_config = config.get_model('lightgbm')
        raw_data_path = config.get_path('raw_data')
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
    # Paths
    # ------------------------------------------------------------------

    def get_path(
        self,
        path_key: str,
        version: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Path:
        """Return an absolute Path for *path_key* with template substitution.

        Templates may use:
        - ``{version}``          → dataset_version  (data/raw, data/processed, splits)
        - ``{pipeline_version}`` → pipeline version (models/, output/)
        Both can be overridden by passing *version* explicitly (applies to {version}).
        """
        paths = self._cfg.get('paths', {})
        if path_key not in paths:
            raise KeyError(
                f"Path key '{path_key}' not found. Available: {list(paths.keys())}"
            )
        tmpl = paths[path_key]
        dataset_ver = version if version is not None else self.get_dataset_version()
        path_str = tmpl.format(
            version=dataset_ver,
            pipeline_version=self.get_version(),
            model_name=model_name or '',
        )
        return self.base_dir / path_str

    # ------------------------------------------------------------------
    # Top-level scalars
    # ------------------------------------------------------------------

    def get_version(self) -> str:
        return self._cfg['version']

    def get_dataset_version(self) -> str:
        return self._cfg.get('dataset_version', self._cfg['version'])

    def get_random_seed(self) -> int:
        return self._cfg['random_seed']

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def get_cleaning_config(self) -> Dict[str, Any]:
        return self._cfg.get('cleaning', {})

    # ------------------------------------------------------------------
    # Target
    # ------------------------------------------------------------------

    def get_target_variable(self) -> str:
        target = self._cfg.get('target', {})
        if 'name' not in target:
            raise KeyError("'target.name' not found in pipeline_config.yaml")
        return target['name']

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def get_numerical_features(self) -> List[str]:
        """Return the ordered list of numerical feature names."""
        val = self._cfg.get('numerical_features', [])
        return list(val) if val else []

    def get_binary_features(self) -> List[str]:
        """Return the ordered list of binary feature names."""
        val = self._cfg.get('binary_features', [])
        return list(val) if val else []

    def get_aggregation_configs(self) -> Dict[str, Dict]:
        """Return the aggregated_features section."""
        return self._cfg.get('aggregated_features', {}) or {}

    def get_categorical_configs(self) -> Dict[str, Dict]:
        """Return the categorical_features section."""
        return self._cfg.get('categorical_features', {}) or {}

    def get_validation_rules(self) -> Dict[str, Any]:
        """Return the validation_rules section."""
        return self._cfg.get('validation_rules', {})

    def get_final_features(self) -> List[str]:
        """
        Build the complete ordered feature list:
        numerical + binary + aggregated (enabled) + categorical (enabled).
        """
        features: List[str] = []
        features.extend(self.get_numerical_features())
        features.extend(self.get_binary_features())
        for name, cfg in (self.get_aggregation_configs() or {}).items():
            if cfg.get('enabled', True):
                features.append(name)
        for name, cfg in (self.get_categorical_configs() or {}).items():
            if cfg.get('enabled', True):
                features.append(name)
        return features

    # ------------------------------------------------------------------
    # Data split
    # ------------------------------------------------------------------

    def get_data_split_config(self) -> Dict[str, Any]:
        return self._cfg.get('data_split', {})

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    def get_model(self, model_name: str) -> Dict[str, Any]:
        models = self._cfg.get('models', {})
        if model_name not in models:
            raise KeyError(
                f"Model '{model_name}' not found. Available: {list(models.keys())}"
            )
        return models[model_name]

    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """
        Return the static parameters for *model_name*.

        Reads from the explicit 'params' key if present; otherwise falls back
        to deriving defaults from the first value of each search_space list.
        """
        model = self._cfg.get('models', {}).get(model_name, {})
        if 'params' in model:
            return dict(model['params'])
        search_space = model.get('search_space', {})
        return {k: v[0] for k, v in search_space.items() if isinstance(v, list) and v}

    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Return all models, including disabled ones."""
        return self._cfg.get('models', {})

    def get_model_names(self) -> List[str]:
        """Return names of enabled models only (enabled: true or key absent)."""
        return [
            name
            for name, cfg in self._cfg.get('models', {}).items()
            if cfg.get('enabled', True)
        ]

    def get_target_recall_for_model(self, model_name: str) -> float:
        """Return the target recall for threshold optimisation of *model_name*."""
        model = self._cfg.get('models', {}).get(model_name, {})
        return model.get('target_recall', 0.90)

    def get_search_space_for_model(self, model_name: str) -> Dict[str, Any]:
        """Return the hyperparameter search space for *model_name*."""
        model = self._cfg.get('models', {}).get(model_name, {})
        return model.get('search_space', {})

    # ------------------------------------------------------------------
    # CV / training
    # ------------------------------------------------------------------

    def get_cv_config(self) -> Dict[str, Any]:
        return self._cfg.get('cross_validation', {})

    # ------------------------------------------------------------------
    # Hyperparameter search
    # ------------------------------------------------------------------

    def get_hyperparameter_search_config(self) -> Dict[str, Any]:
        return self._cfg.get('hyperparameter_search', {})

    # ------------------------------------------------------------------
    # Evaluation / thresholds
    # ------------------------------------------------------------------

    def get_evaluation_config(self) -> Dict[str, Any]:
        return self._cfg.get('evaluation', {})

    def get_threshold_optimization_config(self) -> Dict[str, Any]:
        return self._cfg.get('evaluation', {}).get('threshold_optimization', {})

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def get_visualization_config(self) -> Dict[str, Any]:
        return self._cfg.get('visualization', {})

    # ------------------------------------------------------------------
    # Generic dot-notation accessor
    # ------------------------------------------------------------------

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get any value from the config using dot-notation.

        Example:
            config.get('cross_validation.n_folds', 5)
            config.get('logging.level', 'INFO')
        """
        keys = key_path.split('.')
        value: Any = self._cfg
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def __repr__(self) -> str:
        return (
            f"ConfigurationManager(base_dir={self.base_dir}, "
            f"version={self.get_version()}, "
            f"models={self.get_model_names()})"
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
