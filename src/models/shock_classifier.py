"""
Modulo de clasificadores para prediccion de shock hemorragico.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from typing import Dict, Any, Optional
import numpy as np

# Imports opcionales
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class ShockClassifier:
    """Clase para entrenamiento y gestion de clasificadores de shock."""

    def __init__(self, preprocessor, seed: int = 42):
        """
        Inicializa el clasificador.

        Args:
            preprocessor: Preprocesador de sklearn
            seed: Semilla aleatoria
        """
        self.preprocessor = preprocessor
        self.seed = seed
        self.models = {}
        self.grids = {}
        self.fitted_grids = {}
        self.best_model_name = None
        self.best_estimator = None
        self.calibrator = None

    def _build_base_models(self) -> Dict[str, Any]:
        """
        Construye modelos base para entrenamiento.

        Returns:
            Dict[str, Any]: Diccionario de modelos
        """
        models = {
            'logreg': LogisticRegression(
                solver='liblinear',
                class_weight='balanced',
                max_iter=2000,
                random_state=self.seed
            ),
            'rf': RandomForestClassifier(
                n_estimators=200,
                class_weight='balanced',
                random_state=self.seed
            )
        }

        if XGBOOST_AVAILABLE:
            models['xgb'] = XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=self.seed
            )

        return models

    def _get_param_grids(self) -> Dict[str, Dict[str, list]]:
        """
        Define grids de parametros para GridSearchCV.

        Returns:
            Dict[str, Dict[str, list]]: Grids de parametros por modelo
        """
        param_grids = {
            'logreg': {
                'clf__penalty': ['l1', 'l2'],
                'clf__C': [0.01, 0.1, 1.0, 10.0]
            },
            'rf': {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [4, 8, None],
                'clf__min_samples_leaf': [1, 5]
            }
        }

        if XGBOOST_AVAILABLE:
            param_grids['xgb'] = {
                'clf__max_depth': [3, 5],
                'clf__learning_rate': [0.01, 0.1],
                'clf__n_estimators': [100, 200],
                'clf__subsample': [0.8, 1.0]
            }

        return param_grids

    def _make_pipeline(self, clf) -> ImbPipeline:
        """
        Crea pipeline con preprocesador, SMOTE y clasificador.

        Args:
            clf: Clasificador sklearn

        Returns:
            ImbPipeline: Pipeline completo
        """
        return ImbPipeline([
            ('pre', self.preprocessor),
            ('smote', SMOTE(random_state=self.seed)),
            ('clf', clf)
        ])

    def build_grids(self, cv_splits: int = 5) -> Dict[str, GridSearchCV]:
        """
        Construye GridSearchCV para cada modelo.

        Args:
            cv_splits: Numero de splits para StratifiedKFold

        Returns:
            Dict[str, GridSearchCV]: Grids de busqueda
        """
        models = self._build_base_models()
        param_grids = self._get_param_grids()
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.seed)

        self.grids = {}
        for name, model in models.items():
            pipe = self._make_pipeline(model)
            grid = GridSearchCV(
                pipe,
                param_grids[name],
                scoring='recall',
                cv=cv,
                n_jobs=-1,
                verbose=1
            )
            self.grids[name] = grid

        return self.grids

    def fit_all_models(self, X_train, y_train) -> Dict[str, GridSearchCV]:
        """
        Entrena todos los modelos con GridSearchCV.

        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento

        Returns:
            Dict[str, GridSearchCV]: Grids entrenados
        """
        if not self.grids:
            self.build_grids()

        self.fitted_grids = {}
        for name, grid in self.grids.items():
            print(f"\nEntrenando {name}...")
            grid.fit(X_train, y_train)
            print(f"Mejores parametros: {grid.best_params_}")
            print(f"Mejor CV recall: {grid.best_score_:.4f}")
            self.fitted_grids[name] = grid

        return self.fitted_grids

    def select_best_model(self, evals: Dict[str, Dict]) -> str:
        """
        Selecciona el mejor modelo basado en recall y AUC.

        Args:
            evals: Diccionario de evaluaciones por modelo

        Returns:
            str: Nombre del mejor modelo
        """
        self.best_model_name = max(
            evals.keys(),
            key=lambda n: (evals[n]['recall'], evals[n]['auc'])
        )
        self.best_estimator = evals[self.best_model_name]['estimator']
        print(f"\nMejor modelo seleccionado: {self.best_model_name}")
        return self.best_model_name

    def calibrate_model(self, X_train, y_train, method: str = 'isotonic'):
        """
        Calibra el mejor modelo con CalibratedClassifierCV.

        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            method: Metodo de calibracion ('isotonic' o 'sigmoid')

        Returns:
            CalibratedClassifierCV: Calibrador entrenado
        """
        if self.best_estimator is None:
            raise ValueError("Primero debe seleccionar el mejor modelo")

        self.calibrator = CalibratedClassifierCV(
            estimator=self.best_estimator.named_steps['clf'],
            cv='prefit',
            method=method
        )

        # Transformar datos de entrenamiento
        X_train_trans = self.best_estimator.named_steps['pre'].transform(X_train)
        self.calibrator.fit(X_train_trans, y_train)

        return self.calibrator

    def predict_calibrated(self, X_test) -> np.ndarray:
        """
        Predice probabilidades calibradas.

        Args:
            X_test: Features de test

        Returns:
            np.ndarray: Probabilidades calibradas
        """
        if self.calibrator is None:
            raise ValueError("Primero debe calibrar el modelo")

        X_test_trans = self.best_estimator.named_steps['pre'].transform(X_test)
        return self.calibrator.predict_proba(X_test_trans)[:, 1]

    def get_best_pipeline(self):
        """Retorna el mejor pipeline entrenado."""
        return self.best_estimator

    def get_calibrator(self):
        """Retorna el calibrador entrenado."""
        return self.calibrator
