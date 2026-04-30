"""Association-rule-based feature generation.

This module provides a sklearn-compatible transformer that mines simple
association rules of the form:

    antecedent -> (y == 1)

and then turns each antecedent into a boolean feature indicating whether the
antecedent holds for a sample.

Design goals for this project:
- No extra dependencies (works with numpy/pandas/sklearn only)
- Avoid leakage: must be used inside a sklearn Pipeline so it is fit only on
  the training fold during CV / hyperparameter tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass(frozen=True)
class _Item:
    name: str
    source: str  # original column name
    kind: str  # 'binary', 'bin', 'isna'
    bin_index: Optional[int] = None


class AssociationRuleFeatureGenerator(BaseEstimator, TransformerMixin):
    """Mine association rules (antecedent -> y==1) and generate rule features.

    Parameters are intentionally conservative to reduce overfitting in small
    datasets.
    """

    def __init__(
        self,
        binary_columns: Optional[Sequence[str]] = None,
        continuous_columns: Optional[Sequence[str]] = None,
        min_support: float = 0.03,
        min_confidence: float = 0.60,
        min_lift: float = 1.10,
        max_rules: int = 60,
        max_antecedent_size: int = 2,
        binning_strategy: str = "quantile",
        n_bins: int = 4,
        add_missing_as_item: bool = True,
        random_state: int = 42,
    ) -> None:
        # IMPORTANT (sklearn.clone): do not copy/transform mutable params (like lists)
        # inside __init__. Store them exactly as passed; normalize later.
        self.binary_columns = binary_columns
        self.continuous_columns = continuous_columns
        self.min_support = float(min_support)
        self.min_confidence = float(min_confidence)
        self.min_lift = float(min_lift)
        self.max_rules = int(max_rules)
        self.max_antecedent_size = int(max_antecedent_size)
        self.binning_strategy = str(binning_strategy)
        self.n_bins = int(n_bins)
        self.add_missing_as_item = bool(add_missing_as_item)
        self.random_state = int(random_state)

        self._bin_edges: Dict[str, np.ndarray] = {}
        self._rules: List[Dict] = []
        self._rule_feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = self._ensure_df(X)
        y_arr = np.asarray(y).astype(int)
        if y_arr.ndim != 1:
            raise ValueError("y must be a 1D array-like")

        n = len(X)
        if n == 0:
            self._rules = []
            self._rule_feature_names = []
            return self

        p_y = float(y_arr.mean())
        if p_y <= 0.0:
            self._rules = []
            self._rule_feature_names = []
            return self

        binary_cols, cont_cols = self._resolve_columns(X)
        self._bin_edges = self._fit_binning_edges(X, cont_cols)

        items = self._build_items(binary_cols, cont_cols)
        if not items:
            self._rules = []
            self._rule_feature_names = []
            return self

        item_masks = self._compute_item_masks(X, items)

        min_joint_count = int(np.ceil(self.min_support * n))
        selected_rules: List[Dict] = []

        # Limit search space: only antecedents of size 1..max_antecedent_size
        # with at most one bin item per continuous column.
        max_k = max(1, self.max_antecedent_size)
        for k in range(1, max_k + 1):
            for antecedent in combinations(items, k):
                if not self._is_valid_antecedent(antecedent):
                    continue

                mask = self._and_masks([item_masks[it.name] for it in antecedent])
                antecedent_count = int(mask.sum())
                if antecedent_count == 0:
                    continue

                joint_count = int((mask & (y_arr == 1)).sum())
                if joint_count < min_joint_count:
                    continue

                support = joint_count / n
                confidence = joint_count / antecedent_count
                lift = confidence / p_y if p_y > 0 else 0.0

                if confidence < self.min_confidence:
                    continue
                if lift < self.min_lift:
                    continue

                selected_rules.append(
                    {
                        "antecedent": [it.name for it in antecedent],
                        "support": float(support),
                        "confidence": float(confidence),
                        "lift": float(lift),
                        "antecedent_count": int(antecedent_count),
                        "joint_count": int(joint_count),
                    }
                )

        # Rank rules: lift, then confidence, then support
        selected_rules.sort(key=lambda r: (r["lift"], r["confidence"], r["support"]), reverse=True)
        selected_rules = selected_rules[: max(0, self.max_rules)]

        self._rules = selected_rules
        self._rule_feature_names = [self._rule_feature_name(i, r) for i, r in enumerate(self._rules)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._ensure_df(X)
        if not self._rules:
            return X

        # Recompute only the items needed by the selected rules
        needed_item_names: List[str] = sorted({it for r in self._rules for it in r["antecedent"]})
        items = [self._parse_item_name(name) for name in needed_item_names]
        item_masks = self._compute_item_masks(X, items)

        rule_features = pd.DataFrame(index=X.index)
        for feature_name, rule in zip(self._rule_feature_names, self._rules):
            masks = [item_masks[name] for name in rule["antecedent"]]
            rule_features[feature_name] = self._and_masks(masks).astype(int)

        return pd.concat([X, rule_features], axis=1)

    def get_rules(self) -> List[Dict]:
        """Return selected rules with metrics (small list; safe to log)."""
        return list(self._rules)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray(self._rule_feature_names, dtype=object)
        return np.asarray(list(input_features) + self._rule_feature_names, dtype=object)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_df(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    def _resolve_columns(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        if self.binary_columns is None:
            binary_cols = []
        else:
            binary_cols = [c for c in self.binary_columns if c in X.columns]

        if self.continuous_columns is None:
            cont_cols = []
        else:
            cont_cols = [c for c in self.continuous_columns if c in X.columns]

        return binary_cols, cont_cols

    def _fit_binning_edges(self, X: pd.DataFrame, continuous_cols: Sequence[str]) -> Dict[str, np.ndarray]:
        edges: Dict[str, np.ndarray] = {}
        if not continuous_cols:
            return edges

        if self.n_bins < 2:
            return edges

        for col in continuous_cols:
            s = pd.to_numeric(X[col], errors="coerce")
            s_non_null = s.dropna().to_numpy(dtype=float)
            if s_non_null.size < 10:
                continue

            if self.binning_strategy == "quantile":
                qs = np.linspace(0, 1, self.n_bins + 1)
                raw_edges = np.quantile(s_non_null, qs)
                unique_edges = np.unique(raw_edges)
                if unique_edges.size < 3:
                    continue
                edges[col] = unique_edges
            else:
                raise ValueError(f"Unsupported binning_strategy: {self.binning_strategy}")

        return edges

    def _build_items(self, binary_cols: Sequence[str], continuous_cols: Sequence[str]) -> List[_Item]:
        items: List[_Item] = []
        for col in binary_cols:
            items.append(_Item(name=f"{col}=1", source=col, kind="binary"))
            if self.add_missing_as_item:
                items.append(_Item(name=f"{col}_isna", source=col, kind="isna"))

        for col in continuous_cols:
            if self.add_missing_as_item:
                items.append(_Item(name=f"{col}_isna", source=col, kind="isna"))

            if col not in self._bin_edges:
                continue
            # edges include min/max; bins are intervals between edges
            n_bins_eff = max(0, len(self._bin_edges[col]) - 1)
            for b in range(n_bins_eff):
                items.append(_Item(name=f"{col}_bin={b}", source=col, kind="bin", bin_index=b))

        return items

    def _compute_item_masks(self, X: pd.DataFrame, items: Sequence[_Item]) -> Dict[str, np.ndarray]:
        masks: Dict[str, np.ndarray] = {}

        # Pre-load columns used
        needed_cols = sorted({it.source for it in items})
        series_map = {c: pd.to_numeric(X[c], errors="coerce") if c in X.columns else pd.Series([np.nan] * len(X), index=X.index) for c in needed_cols}

        for it in items:
            s = series_map[it.source]
            if it.kind == "binary":
                masks[it.name] = (s.fillna(0).astype(float) == 1.0).to_numpy(dtype=bool)
            elif it.kind == "isna":
                masks[it.name] = s.isna().to_numpy(dtype=bool)
            elif it.kind == "bin":
                edges = self._bin_edges.get(it.source)
                if edges is None or it.bin_index is None:
                    masks[it.name] = np.zeros(len(X), dtype=bool)
                else:
                    left = edges[it.bin_index]
                    right = edges[it.bin_index + 1]
                    # Include left edge; include right edge only for last bin
                    if it.bin_index == len(edges) - 2:
                        mask = (s >= left) & (s <= right)
                    else:
                        mask = (s >= left) & (s < right)
                    masks[it.name] = mask.fillna(False).to_numpy(dtype=bool)
            else:
                raise ValueError(f"Unknown item kind: {it.kind}")

        return masks

    @staticmethod
    def _and_masks(masks: Sequence[np.ndarray]) -> np.ndarray:
        if not masks:
            raise ValueError("masks must be non-empty")
        out = masks[0].copy()
        for m in masks[1:]:
            out &= m
        return out

    @staticmethod
    def _is_valid_antecedent(antecedent: Sequence[_Item]) -> bool:
        # Avoid impossible combos like two bins from the same continuous column
        seen_bin_source: set = set()
        for it in antecedent:
            if it.kind == "bin":
                if it.source in seen_bin_source:
                    return False
                seen_bin_source.add(it.source)
        return True

    @staticmethod
    def _rule_feature_name(i: int, rule: Dict) -> str:
        # Keep it stable and filesystem/log friendly
        ant = "&".join(rule["antecedent"])
        ant = ant.replace(" ", "")
        return f"assoc_rule_{i:03d}__{ant}"

    @staticmethod
    def _parse_item_name(name: str) -> _Item:
        if name.endswith("_isna"):
            source = name[: -len("_isna")]
            return _Item(name=name, source=source, kind="isna")

        if name.endswith("=1"):
            source = name[: -len("=1")]
            return _Item(name=name, source=source, kind="binary")

        if "_bin=" in name:
            source, b = name.split("_bin=")
            return _Item(name=name, source=source, kind="bin", bin_index=int(b))

        # Fallback (should not happen if generated by this class)
        return _Item(name=name, source=name, kind="binary")
