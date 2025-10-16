"""
Modulo de ingenieria de features y transformaciones.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import List


class FeatureTransformer:
    """Clase para transformacion de features."""

    def __init__(self, num_cols: List[str], cat_cols: List[str]):
        """
        Inicializa el transformador de features.

        Args:
            num_cols: Lista de columnas numericas
            cat_cols: Lista de columnas categoricas
        """
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.preprocessor = None

    def build_preprocessor(self) -> ColumnTransformer:
        """
        Construye el preprocesador con pipelines para numericas y categoricas.

        Returns:
            ColumnTransformer: Preprocesador configurado
        """
        # Pipeline para columnas numericas: imputacion + escalado
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Pipeline para columnas categoricas/binarias: solo imputacion
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent"))
        ])

        # ColumnTransformer que aplica pipelines segun tipo
        self.preprocessor = ColumnTransformer([
            ("num", num_pipe, self.num_cols),
            ("cat", cat_pipe, self.cat_cols)
        ], remainder='drop')

        return self.preprocessor

    def get_preprocessor(self) -> ColumnTransformer:
        """
        Obtiene el preprocesador (lo construye si no existe).

        Returns:
            ColumnTransformer: Preprocesador
        """
        if self.preprocessor is None:
            self.build_preprocessor()
        return self.preprocessor
