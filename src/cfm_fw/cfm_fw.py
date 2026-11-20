"""CFM-FW: Feature weights-based comprehensive causal multi-objective counterfactual framework."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree, LocalOutlierFactor

from cfm_fw.acme import AcME
from cfm_fw.metrics import (
    hinge_yloss,
    inverse_weighted_distance,
    relative_impact_metric,
    sparsity_count,
)
from cfm_fw.nsga2 import run_nsga2


class CFMFW:
    """Feature weights-based Causal Multi-objective Counterfactual Framework (Liu et al., 2025).

    Parameters
    ----------
    model : Any
        Trained classifier exposing ``predict(X)`` and ideally ``predict_proba(X)``.
    continuous : list[str] | None
        Continuous feature names. If *None*, inferred from numeric dtypes.
    categorical : list[str] | None
        Categorical feature names. If *None*, inferred as remaining columns.
    fixed_features : list[str] | None
        Immutable features (``x_j - x*_j = 0``).
    monotonic_increasing : list[str] | None
        Features that can only stay equal or increase (``x_j - x*_j <= 0``).
    causal_constraints : Callable[[pd.Series, pd.DataFrame], np.ndarray] | None
        Optional custom constraint function returning violation ``> 0`` when violated
        and ``<= 0`` when satisfied.
    pop_size : int
        NSGA-II population size (default: 100).
    n_gen : int
        Number of NSGA-II generations (default: 10).
    cutoff : float
        Minimum predicted probability for the desired class (default: 0.5).
    n_neighbors_lof : int
        Number of neighbors for LocalOutlierFactor (default: 20).
    crossover_prob : float
        Crossover probability (default: 0.9).
    eta_c : float
        SBX distribution index (default: 15.0).
    eta_m : float
        Polynomial mutation distribution index (default: 20.0).
    post_hoc_sparsity : bool
        Whether to apply post-hoc continuous sparsity reversion (default: True).
    seed : int
        Random seed (default: 42).
    """

    def __init__(
        self,
        model: Any,
        continuous: list[str] | None = None,
        categorical: list[str] | None = None,
        fixed_features: list[str] | None = None,
        monotonic_increasing: list[str] | None = None,
        causal_constraints: Callable[[pd.Series, pd.DataFrame], np.ndarray] | None = None,
        pop_size: int = 100,
        n_gen: int = 10,
        cutoff: float = 0.5,
        n_neighbors_lof: int = 20,
        crossover_prob: float = 0.9,
        eta_c: float = 15.0,
        eta_m: float = 20.0,
        post_hoc_sparsity: bool = True,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.continuous = list(continuous) if continuous is not None else None
        self.categorical = list(categorical) if categorical is not None else None
        self.fixed_features = set(fixed_features or [])
        self.monotonic_increasing = set(monotonic_increasing or [])
        self.causal_constraints = causal_constraints
        self.pop_size = int(pop_size)
        self.n_gen = int(n_gen)
        self.cutoff = float(cutoff)
        self.n_neighbors_lof = int(n_neighbors_lof)
        self.crossover_prob = float(crossover_prob)
        self.eta_c = float(eta_c)
        self.eta_m = float(eta_m)
        self.post_hoc_sparsity = bool(post_hoc_sparsity)
        self.seed = int(seed)

        self.is_fitted_: bool = False
        self.feature_names_: list[str] = []
        self.dtypes_: dict[str, Any] = {}
        self.cont_cols_: list[str] = []
        self.cat_cols_: list[str] = []
        self.cont_idx_: np.ndarray = np.empty(0, dtype=np.int64)
        self.cat_idx_: np.ndarray = np.empty(0, dtype=np.int64)
        self.mutable_mask_: np.ndarray = np.empty(0, dtype=bool)
        self.mono_inc_mask_: np.ndarray = np.empty(0, dtype=bool)

        self.xl_cont_: np.ndarray = np.empty(0, dtype=np.float64)
        self.xu_cont_: np.ndarray = np.empty(0, dtype=np.float64)
        self.cont_scales_: np.ndarray = np.empty(0, dtype=np.float64)
        self.cont_mads_: np.ndarray = np.empty(0, dtype=np.float64)

        self.cat_categories_: list[np.ndarray] = []
        self.cat_cardinalities_: np.ndarray = np.empty(0, dtype=np.int64)
        self.cat_maps_: list[dict[Any, int]] = []

        self.acme_: AcME | None = None
        self.weights_: np.ndarray = np.empty(0, dtype=np.float64)
        self.lof_: LocalOutlierFactor | None = None
        self.kdtrees_: dict[int, tuple[KDTree, np.ndarray]] = {}
        self.classes_: np.ndarray = np.array([0, 1])

    def _encode_df(self, df: pd.DataFrame) -> np.ndarray:
        """Encode raw DataFrame into internal float64 array ``(n, n_features)``."""
        arr = np.empty((len(df), len(self.feature_names_)), dtype=np.float64)
        if self.cont_idx_.size > 0:
            arr[:, self.cont_idx_] = df[self.cont_cols_].to_numpy(dtype=np.float64)
        for pos, col_idx in enumerate(self.cat_idx_):
            col = self.cat_cols_[pos]
            mapping = self.cat_maps_[pos]
            arr[:, col_idx] = [mapping.get(v, 0) for v in df[col].tolist()]
        return arr

    def _decode_array(self, arr: np.ndarray) -> pd.DataFrame:
        """Decode internal float64 array back to raw-space DataFrame."""
        data: dict[str, Any] = {}
        for pos, col_idx in enumerate(self.cont_idx_):
            col = self.cont_cols_[pos]
            vals = arr[:, col_idx]
            if pd.api.types.is_integer_dtype(self.dtypes_[col]):
                data[col] = np.rint(vals).astype(self.dtypes_[col])
            else:
                data[col] = vals.astype(np.float64)
        for pos, col_idx in enumerate(self.cat_idx_):
            col = self.cat_cols_[pos]
            cats = self.cat_categories_[pos]
            idx = np.clip(np.rint(arr[:, col_idx]).astype(np.int64), 0, len(cats) - 1)
            data[col] = cats[idx]
        return pd.DataFrame(data, columns=self.feature_names_)

    def _scale_for_distance(self, arr: np.ndarray) -> np.ndarray:
        """Scale internal array into ``[0, 1]`` space for KDTree and LOF."""
        scaled = np.zeros_like(arr, dtype=np.float64)
        if self.cont_idx_.size > 0:
            span = np.where(
                (self.xu_cont_ - self.xl_cont_) > 1e-12,
                self.xu_cont_ - self.xl_cont_,
                1.0,
            )
            scaled[:, self.cont_idx_] = (arr[:, self.cont_idx_] - self.xl_cont_[None, :]) / span[None, :]
        for pos, col_idx in enumerate(self.cat_idx_):
            denom = max(int(self.cat_cardinalities_[pos]) - 1, 1)
            scaled[:, col_idx] = arr[:, col_idx] / float(denom)
        return scaled

    def _predict_proba_class(self, df: pd.DataFrame, target_class: int) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(df), dtype=np.float64)
            if proba.ndim == 2:
                cls_idx = int(np.where(self.classes_ == target_class)[0][0]) if target_class in self.classes_ else target_class
                return proba[:, cls_idx]
            return proba.ravel()
        preds = np.asarray(self.model.predict(df))
        return (preds == target_class).astype(np.float64)

    def fit(self, X: pd.DataFrame) -> CFMFW:
        """Fit AcME weights, LOF plausibility model, and per-class KDTrees on *X*."""
        self.feature_names_ = list(X.columns)
        self.dtypes_ = {c: X[c].dtype for c in self.feature_names_}

        if self.continuous is not None and self.categorical is not None:
            self.cont_cols_ = list(self.continuous)
            self.cat_cols_ = list(self.categorical)
        elif self.continuous is not None:
            self.cont_cols_ = list(self.continuous)
            self.cat_cols_ = [c for c in self.feature_names_ if c not in self.cont_cols_]
        elif self.categorical is not None:
            self.cat_cols_ = list(self.categorical)
            self.cont_cols_ = [c for c in self.feature_names_ if c not in self.cat_cols_]
        else:
            self.cont_cols_ = [c for c in self.feature_names_ if pd.api.types.is_numeric_dtype(X[c])]
            self.cat_cols_ = [c for c in self.feature_names_ if c not in self.cont_cols_]

        col_to_idx = {c: i for i, c in enumerate(self.feature_names_)}
        self.cont_idx_ = np.array([col_to_idx[c] for c in self.cont_cols_], dtype=np.int64)
        self.cat_idx_ = np.array([col_to_idx[c] for c in self.cat_cols_], dtype=np.int64)

        self.mutable_mask_ = np.array(
            [c not in self.fixed_features for c in self.feature_names_], dtype=bool
        )
        self.mono_inc_mask_ = np.array(
            [c in self.monotonic_increasing for c in self.feature_names_], dtype=bool
        )

        # Continuous stats
        if self.cont_idx_.size > 0:
            cont_vals = X[self.cont_cols_].to_numpy(dtype=np.float64)
            self.xl_cont_ = np.nanmin(cont_vals, axis=0)
            self.xu_cont_ = np.nanmax(cont_vals, axis=0)
            medians = np.nanmedian(cont_vals, axis=0)
            mads = np.nanmedian(np.abs(cont_vals - medians[None, :]), axis=0)
            stds = np.nanstd(cont_vals, axis=0)
            scales = self.xu_cont_ - self.xl_cont_
            self.cont_mads_ = np.where(mads > 1e-8, mads, np.where(stds > 1e-8, stds, 1.0))
            self.cont_scales_ = np.where(scales > 1e-8, scales, 1.0)
        else:
            self.xl_cont_ = np.empty(0, dtype=np.float64)
            self.xu_cont_ = np.empty(0, dtype=np.float64)
            self.cont_mads_ = np.empty(0, dtype=np.float64)
            self.cont_scales_ = np.empty(0, dtype=np.float64)

        # Categorical stats
        self.cat_categories_ = []
        self.cat_maps_ = []
        card_list = []
        for col in self.cat_cols_:
            cats = np.unique(X[col].dropna().to_numpy())
            self.cat_categories_.append(cats)
            self.cat_maps_.append({v: i for i, v in enumerate(cats.tolist())})
            card_list.append(len(cats))
        self.cat_cardinalities_ = np.array(card_list, dtype=np.int64)

        # Fit AcME for inverse feature weights
        self.acme_ = AcME(
            model=self.model,
            continuous=self.cont_cols_,
            categorical=self.cat_cols_,
        ).fit(X)
        self.weights_ = self.acme_.get_weights_array(self.feature_names_)

        # Encode and scale training data
        X_enc = self._encode_df(X)
        X_scaled = self._scale_for_distance(X_enc)

        # Fit LOF
        n_neighbors = min(self.n_neighbors_lof, max(len(X_scaled) - 1, 1))
        self.lof_ = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
        self.lof_.fit(X_scaled)

        # Build per-class KDTrees using model predictions on X
        y_pred = np.asarray(self.model.predict(X)).ravel()
        if hasattr(self.model, "classes_"):
            self.classes_ = np.asarray(self.model.classes_)
        else:
            self.classes_ = np.unique(y_pred)

        self.kdtrees_ = {}
        for cls in np.unique(y_pred):
            mask = y_pred == cls
            cls_enc = X_enc[mask]
            cls_scaled = X_scaled[mask]
            tree = KDTree(cls_scaled)
            self.kdtrees_[int(cls)] = (tree, cls_enc)

        self.is_fitted_ = True
        return self

    def _build_initial_population(
        self,
        x_orig: np.ndarray,
        desired_class: int,
        orig_pred: int,
        xl_cont: np.ndarray,
        xu_cont: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Construct biased initial population via KDTree of opposite-class samples."""
        x_orig_scaled = self._scale_for_distance(x_orig[None, :])

        # Gather candidate pool from desired_class (or all classes != orig_pred)
        if desired_class in self.kdtrees_:
            tree, pool_enc = self.kdtrees_[desired_class]
            k = min(self.pop_size, len(pool_enc))
            _, idx = tree.query(x_orig_scaled, k=k)
            init_pop = pool_enc[idx[0]].copy()
        else:
            other_pools = [
                enc for cls, (_, enc) in self.kdtrees_.items() if cls != orig_pred
            ]
            if other_pools:
                pool_enc = np.vstack(other_pools)
                k = min(self.pop_size, len(pool_enc))
                sel = rng.choice(len(pool_enc), size=k, replace=False)
                init_pop = pool_enc[sel].copy()
            else:
                # Fallback: perturb original instance randomly
                init_pop = np.repeat(x_orig[None, :], self.pop_size, axis=0)

        # Pad to pop_size if fewer than pop_size opposite-class samples exist
        if len(init_pop) < self.pop_size:
            extra_idx = rng.integers(0, len(init_pop), size=self.pop_size - len(init_pop))
            extra = init_pop[extra_idx].copy()
            if self.cont_idx_.size > 0:
                noise = rng.normal(0.0, 0.05, size=(len(extra), self.cont_idx_.size)) * self.cont_scales_[None, :]
                extra[:, self.cont_idx_] = np.clip(
                    extra[:, self.cont_idx_] + noise,
                    xl_cont[None, :],
                    xu_cont[None, :],
                )
            init_pop = np.vstack([init_pop, extra])

        # Enforce causal constraints on initial population
        if not self.mutable_mask_.all():
            init_pop[:, ~self.mutable_mask_] = x_orig[~self.mutable_mask_]
        if self.cont_idx_.size > 0:
            init_pop[:, self.cont_idx_] = np.clip(
                init_pop[:, self.cont_idx_], xl_cont[None, :], xu_cont[None, :]
            )

        return init_pop

    def _post_hoc_continuous_sparsity(
        self,
        x_orig: np.ndarray,
        X_front: np.ndarray,
        desired_class: int,
    ) -> np.ndarray:
        """Attempt to revert small continuous feature changes if prediction remains valid."""
        if self.cont_idx_.size == 0 or len(X_front) == 0:
            return X_front

        # Threshold per Section 5.3: min(MAD_j, 0.05 * span_j)
        thresholds = np.minimum(self.cont_mads_, 0.05 * self.cont_scales_)
        X_out = X_front.copy()

        for i in range(len(X_out)):
            cand = X_out[i].copy()
            for pos, col_idx in enumerate(self.cont_idx_):
                if not self.mutable_mask_[col_idx]:
                    continue
                diff = abs(cand[col_idx] - x_orig[col_idx])
                if 1e-4 <= diff <= thresholds[pos]:
                    trial = cand.copy()
                    trial[col_idx] = x_orig[col_idx]
                    trial_df = self._decode_array(trial[None, :])
                    p = float(self._predict_proba_class(trial_df, desired_class)[0])
                    pred = int(np.asarray(self.model.predict(trial_df)).ravel()[0])
                    if p >= self.cutoff and pred == desired_class:
                        cand = trial
            X_out[i] = cand

        return X_out

    def generate(
        self,
        query: pd.Series | pd.DataFrame,
        desired_class: int | None = None,
    ) -> pd.DataFrame:
        """Generate counterfactual instances for a single query instance.

        Returns a DataFrame of valid counterfactuals sorted from best to worst,
        or an empty DataFrame if no valid counterfactual was found.
        """
        if not self.is_fitted_:
            raise RuntimeError("CFMFW must be fitted before calling generate().")

        if isinstance(query, pd.Series):
            query_df = query.to_frame().T[self.feature_names_]
        else:
            query_df = query.iloc[[0]][self.feature_names_]

        x_orig = self._encode_df(query_df)[0]
        orig_pred = int(np.asarray(self.model.predict(query_df)).ravel()[0])

        if desired_class is None:
            other_classes = [int(c) for c in self.classes_ if int(c) != orig_pred]
            desired_class = other_classes[0] if other_classes else (1 - orig_pred)

        # Adjust continuous bounds for monotonic_increasing features
        xl_cont = self.xl_cont_.copy()
        xu_cont = self.xu_cont_.copy()
        if self.cont_idx_.size > 0:
            mono_cont = self.mono_inc_mask_[self.cont_idx_]
            xl_cont[mono_cont] = np.maximum(xl_cont[mono_cont], x_orig[self.cont_idx_[mono_cont]])
            xu_cont = np.maximum(xu_cont, xl_cont)

        rng = np.random.default_rng(self.seed)
        init_pop = self._build_initial_population(
            x_orig=x_orig,
            desired_class=desired_class,
            orig_pred=orig_pred,
            xl_cont=xl_cont,
            xu_cont=xu_cont,
            rng=rng,
        )

        assert self.lof_ is not None

        def evaluate_fn(pop_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            pop_df = self._decode_array(pop_arr)
            p_desired = self._predict_proba_class(pop_df, desired_class)

            f_yloss = hinge_yloss(p_desired, desired_sign=1.0)
            f_dist = inverse_weighted_distance(
                x_orig, pop_arr, self.weights_, self.cont_idx_, self.cat_idx_, self.cont_scales_
            )
            f_sparsity = sparsity_count(x_orig, pop_arr, self.cont_idx_, self.cat_idx_)
            f_rim = relative_impact_metric(
                x_orig, pop_arr, self.weights_, self.cont_idx_, self.cat_idx_, self.cont_scales_
            )
            pop_scaled = self._scale_for_distance(pop_arr)
            f_lof = -self.lof_.score_samples(pop_scaled)

            F = np.column_stack([f_yloss, f_dist, f_sparsity, f_rim, f_lof])

            # Constraint violation: valid prediction (p_desired >= cutoff and predicted == desired_class) + causal constraints
            preds_pop = np.asarray(self.model.predict(pop_df)).ravel()
            cv = np.maximum(0.0, self.cutoff - p_desired) + np.where(preds_pop != desired_class, 1e-3, 0.0)
            if self.cat_idx_.size > 0:
                mono_cat = self.mono_inc_mask_[self.cat_idx_]
                if mono_cat.any():
                    c_idx = self.cat_idx_[mono_cat]
                    cv += np.sum(np.maximum(0.0, x_orig[c_idx][None, :] - pop_arr[:, c_idx]), axis=1)
            if self.causal_constraints is not None:
                custom_cv = np.asarray(self.causal_constraints(query_df.iloc[0], pop_df), dtype=np.float64)
                cv += np.maximum(0.0, custom_cv).ravel()

            return F, cv

        X_front, F_front, cv_front = run_nsga2(
            init_pop=init_pop,
            evaluate_fn=evaluate_fn,
            cont_idx=self.cont_idx_,
            cat_idx=self.cat_idx_,
            mutable_mask=self.mutable_mask_,
            x_orig=x_orig,
            xl_cont=xl_cont,
            xu_cont=xu_cont,
            cat_cardinalities=self.cat_cardinalities_,
            n_gen=self.n_gen,
            crossover_prob=self.crossover_prob,
            eta_c=self.eta_c,
            eta_m=self.eta_m,
            seed=self.seed,
        )

        # Keep only feasible / valid solutions
        valid_mask = cv_front <= 0.0
        if not valid_mask.any():
            return pd.DataFrame(columns=self.feature_names_)

        X_valid = X_front[valid_mask]
        if self.post_hoc_sparsity:
            X_valid = self._post_hoc_continuous_sparsity(x_orig, X_valid, desired_class)

        # Re-evaluate to rank valid counterfactuals cleanly
        F_valid, cv_valid = evaluate_fn(X_valid)
        still_valid = cv_valid <= 0.0
        if not still_valid.any():
            return pd.DataFrame(columns=self.feature_names_)

        X_valid = X_valid[still_valid]
        F_valid = F_valid[still_valid]

        # Rank by (sparsity, dist, lof)
        order = np.lexsort((F_valid[:, 4], F_valid[:, 1], F_valid[:, 2]))
        return self._decode_array(X_valid[order]).reset_index(drop=True)
