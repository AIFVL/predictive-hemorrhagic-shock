"""
Modulo de limpieza y preparacion de datos para prediccion de shock hemorragico.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


class ShockDataCleaner:
    """Clase para limpieza y preparacion de datos de shock hemorragico."""

    def __init__(self, target_col: str = None, postop_keywords: List[str] = None):
        """
        Inicializa el limpiador de datos.

        Args:
            target_col: Nombre de la columna objetivo
            postop_keywords: Palabras clave para identificar variables postoperatorias
        """
        self.target_col = target_col
        self.postop_keywords = postop_keywords or ['sangrad', 'sangrado', 'muerte', 'mortalidad']
        self.continuous_cols = []
        self.categorical_cols = []
        self.features = []
        self.postop_vars = []

    def detect_target(self, df: pd.DataFrame) -> str:
        """
        Detecta la columna objetivo (case-insensitive).

        Args:
            df: DataFrame con los datos

        Returns:
            str: Nombre de la columna objetivo

        Raises:
            ValueError: Si no se encuentra columna de shock
        """
        target_col = next((c for c in df.columns if 'shock' in c.lower()), None)
        if target_col is None:
            raise ValueError("No se encontro columna 'shock'.")
        self.target_col = target_col
        return target_col

    def identify_leakage_vars(self, df: pd.DataFrame) -> List[str]:
        """
        Identifica variables postoperatorias que pueden causar data leakage.

        Args:
            df: DataFrame con los datos

        Returns:
            List[str]: Lista de variables postoperatorias
        """
        self.postop_vars = [
            c for c in df.columns
            if any(x in c.lower() for x in self.postop_keywords)
        ]
        return self.postop_vars

    def identify_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identifica tipos de features (continuas vs categoricas).

        Args:
            df: DataFrame con los datos

        Returns:
            Tuple[List[str], List[str]]: (columnas_continuas, columnas_categoricas)
        """
        cont_keywords = ['edad', 'age', 'hb', 'hemoglob', 'hb_pre', 'globul']
        cont_candidates = [
            c for c in df.columns
            if any(k in c.lower() for k in cont_keywords)
        ]
        self.continuous_cols = [c for c in cont_candidates if c != self.target_col]

        # Features: excluir ID, target y variables postoperatorias
        exclude = ['CODIGO', 'codigo', self.target_col] + self.postop_vars
        self.features = [c for c in df.columns if c not in exclude]

        return self.continuous_cols, self.features

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y prepara los datos.

        Args:
            df: DataFrame con datos crudos

        Returns:
            pd.DataFrame: DataFrame limpio
        """
        df_clean = df.copy()

        # Coercer columnas continuas a numericas
        for c in self.continuous_cols:
            df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')

        # Coercer features a numericas
        for c in self.features:
            df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')

        # Asegurar que target sea binario 0/1
        df_clean[self.target_col] = (
            pd.to_numeric(df_clean[self.target_col], errors='coerce') > 0
        ).astype(int)

        return df_clean

    def remove_zero_variance_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina features con varianza cero.

        Args:
            X: DataFrame de features

        Returns:
            pd.DataFrame: DataFrame sin features de varianza cero
        """
        zero_var = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
        if zero_var:
            print(f"Eliminando {len(zero_var)} features con varianza cero: {zero_var}")
            X = X.drop(columns=zero_var)
            # Actualizar lista de features
            self.features = [f for f in self.features if f not in zero_var]
        return X

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara features y target para modelado.

        Args:
            df: DataFrame limpio

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X, y)
        """
        # Detectar target si no esta definido
        if self.target_col is None:
            self.detect_target(df)

        # Identificar variables de leakage
        self.identify_leakage_vars(df)

        # Identificar tipos de features
        self.identify_feature_types(df)

        # Limpiar datos
        df_clean = self.clean_data(df)

        # Preparar X e y
        X = df_clean[self.features].copy()
        y = df_clean[self.target_col].copy()

        # Eliminar features de varianza cero
        X = self.remove_zero_variance_features(X)

        return X, y

    def get_feature_split(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Obtiene la division de features numericas vs categoricas.

        Args:
            X: DataFrame de features

        Returns:
            Tuple[List[str], List[str]]: (numericas, categoricas)
        """
        num_cols = [c for c in self.continuous_cols if c in X.columns]
        cat_cols = [c for c in X.columns if c not in num_cols]
        return num_cols, cat_cols
