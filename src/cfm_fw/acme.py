"""AcME (Accelerated Model-agnostic Explanations) for global feature weights and effects."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class AcME:
    """Accelerated Model-agnostic Explanations (Dandolo et al., 2023).

    Computes global feature importance weights and quantile-based standardized
    effects (Eq. 2 of Liu et al., 2025) by perturbing a baseline vector across
    feature quantiles/categories.

    Parameters
    ----------
    model : Any
        Predictive model exposing ``predict_proba(X)`` or ``predict(X)``.
    continuous : list[str] | None
        Names of continuous features.
    categorical : list[str] | None
        Names of categorical features.
    quantiles : np.ndarray | None
        Quantile levels for continuous features. Defaults to
        ``np.linspace(0.0, 1.0, 21)``.
    target_class : int
        Class index to evaluate ``predict_proba`` for (default: 1).
    """

    def __init__(
        self,
        model: Any,
        continuous: list[str] | None = None,
        categorical: list[str] | None = None,
        quantiles: np.ndarray | None = None,
        target_class: int = 1,
    ) -> None:
        self.model = model
        self.continuous = list(continuous or [])
        self.categorical = list(categorical or [])
        self.quantiles = (
            np.asarray(quantiles, dtype=np.float64)
            if quantiles is not None
            else np.linspace(0.0, 1.0, 21)
        )
        self.target_class = target_class

        self.feature_names_: list[str] = []
        self.baseline_: pd.Series | None = None
        self.importances_: dict[str, float] = {}
        self.inverse_weights_: dict[str, float] = {}
        self.effects_: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _predict_score(self, df: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(df), dtype=np.float64)
            if proba.ndim == 2 and proba.shape[1] > self.target_class:
                return proba[:, self.target_class]
            return proba.ravel()
        return np.asarray(self.model.predict(df), dtype=np.float64).ravel()

    def fit(self, X: pd.DataFrame) -> AcME:
        """Compute baseline vector, standardized effects, and inverse feature weights."""
        self.feature_names_ = list(X.columns)
        if not self.continuous and not self.categorical:
            self.continuous = [
                c for c in self.feature_names_ if pd.api.types.is_numeric_dtype(X[c])
            ]
            self.categorical = [c for c in self.feature_names_ if c not in self.continuous]

        cat_set = set(self.categorical)

        # Baseline vector x^b: median for continuous, mode for categorical
        baseline_vals: dict[str, Any] = {}
        for col in self.feature_names_:
            if col in cat_set:
                modes = X[col].mode(dropna=True)
                baseline_vals[col] = modes.iloc[0] if len(modes) > 0 else X[col].iloc[0]
            else:
                baseline_vals[col] = float(np.nanmedian(X[col].to_numpy(dtype=np.float64)))

        self.baseline_ = pd.Series(baseline_vals, index=self.feature_names_)
        baseline_df = pd.DataFrame([self.baseline_], columns=self.feature_names_)
        f_xb = float(self._predict_score(baseline_df)[0])

        raw_importances: dict[str, float] = {}
        self.effects_ = {}

        for col in self.feature_names_:
            if col in cat_set:
                vals = np.unique(X[col].dropna().to_numpy())
            else:
                col_vals = X[col].dropna().to_numpy(dtype=np.float64)
                vals = np.unique(np.quantile(col_vals, self.quantiles))

            if len(vals) <= 1:
                raw_importances[col] = 0.0
                self.effects_[col] = (vals, np.zeros(len(vals), dtype=np.float64))
                continue

            # Build variational quantile matrix
            q_matrix = pd.DataFrame(
                np.repeat(baseline_df.to_numpy(), len(vals), axis=0),
                columns=self.feature_names_,
            )
            # Preserve column dtypes
            for c in self.feature_names_:
                q_matrix[c] = q_matrix[c].astype(X[c].dtype)
            q_matrix[col] = vals

            y_hat = self._predict_score(q_matrix)
            var_y = float(np.var(y_hat))
            span_y = float(np.max(y_hat) - np.min(y_hat))

            if var_y <= 1e-12 or span_y <= 1e-12:
                deltas = np.zeros_like(y_hat, dtype=np.float64)
            else:
                # Eq. (2): Delta_{j,q} = ((y_hat - f(x^b)) / sqrt(var(y_hat))) * (max(y_hat) - min(y_hat))
                deltas = ((y_hat - f_xb) / np.sqrt(var_y)) * span_y

            self.effects_[col] = (vals, deltas)
            raw_importances[col] = float(np.mean(np.abs(deltas)))

        # Normalize importances to [0, 1] and compute inverse weights w_j (Eq. 4)
        max_imp = max(raw_importances.values()) if raw_importances else 0.0
        eps = 1e-3
        self.importances_ = {}
        self.inverse_weights_ = {}

        for col in self.feature_names_:
            norm_imp = raw_importances[col] / max_imp if max_imp > 1e-12 else 1.0
            self.importances_[col] = float(norm_imp)
            self.inverse_weights_[col] = float(1.0 / (norm_imp + eps))

        # Normalize inverse weights so their mean is 1.0 (keeps distance scales well-behaved)
        mean_w = float(np.mean(list(self.inverse_weights_.values()))) if self.inverse_weights_ else 1.0
        if mean_w > 1e-12:
            for col in self.feature_names_:
                self.inverse_weights_[col] /= mean_w

        return self

    def get_weights_array(self, feature_names: list[str] | None = None) -> np.ndarray:
        """Return inverse feature weights aligned to *feature_names*."""
        cols = feature_names if feature_names is not None else self.feature_names_
        return np.array([self.inverse_weights_[c] for c in cols], dtype=np.float64)

    def get_feature_effects_table(self) -> pd.DataFrame:
        """Return a DataFrame summarizing global importance and effect range per feature."""
        rows = []
        for col in self.feature_names_:
            _, deltas = self.effects_.get(col, (np.array([]), np.array([])))
            rows.append(
                {
                    "feature": col,
                    "importance": self.importances_.get(col, 0.0),
                    "inverse_weight": self.inverse_weights_.get(col, 1.0),
                    "min_effect": float(np.min(deltas)) if len(deltas) else 0.0,
                    "max_effect": float(np.max(deltas)) if len(deltas) else 0.0,
                }
            )
        return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
