"""Feature pruning transformers.

This module contains sklearn-compatible transformers used to remove
pathological/unstable features (e.g., extremely rare binary indicators)
in a leakage-safe way (fit inside CV folds).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RareBinaryFeaturePruner(BaseEstimator, TransformerMixin):
    """Drop binary columns with too few positive occurrences.

    Rationale: with very small counts (e.g., 0–2 occurrences), tree models can
    overfit splits that do not generalize.

    Notes:
    - This transformer must be used inside a sklearn Pipeline so `fit` happens
      on training folds only.
    - The criterion is based on total occurrences in the training data.
    """

    def __init__(
        self,
        binary_columns: Optional[Sequence[str]] = None,
        min_total_ones: int = 10,
    ) -> None:
        # Keep params cloneable: do not coerce/copy mutables here.
        self.binary_columns = binary_columns
        self.min_total_ones = int(min_total_ones)

        self._dropped_columns: List[str] = []
        self._kept_columns: List[str] = []
        self._stats: Dict[str, Dict[str, int]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        X = self._ensure_df(X)

        if self.binary_columns is None:
            binary_cols = []
        else:
            binary_cols = [c for c in self.binary_columns if c in X.columns]

        dropped: List[str] = []
        kept: List[str] = []
        stats: Dict[str, Dict[str, int]] = {}

        for col in binary_cols:
            s = pd.to_numeric(X[col], errors="coerce").fillna(0)
            ones = int((s.astype(float) == 1.0).sum())
            zeros = int(len(s) - ones)
            stats[col] = {"ones": ones, "zeros": zeros}

            if ones < self.min_total_ones:
                dropped.append(col)
            else:
                kept.append(col)

        self._dropped_columns = dropped
        self._kept_columns = kept
        self._stats = stats
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._ensure_df(X)
        if not self._dropped_columns:
            return X
        cols_to_drop = [c for c in self._dropped_columns if c in X.columns]
        if not cols_to_drop:
            return X
        return X.drop(columns=cols_to_drop)

    def get_dropped_columns(self) -> List[str]:
        return list(self._dropped_columns)

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        return dict(self._stats)

    @staticmethod
    def _ensure_df(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)
