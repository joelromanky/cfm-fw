"""Objective functions and evaluation metrics for CFM-FW."""

from __future__ import annotations

from typing import Any

import numpy as np


def hinge_yloss(proba_desired: np.ndarray, desired_sign: float = 1.0) -> np.ndarray:
    """Hinge loss for prediction discrepancy (Eq. 3).

    Maps predicted probability ``p in [0, 1]`` of the desired class to a
    signed margin ``s = 2 * p - 1 in [-1, 1]`` and computes
    ``max(0, 1 - desired_sign * s)``.
    """
    margin = 2.0 * np.asarray(proba_desired, dtype=np.float64) - 1.0
    return np.maximum(0.0, 1.0 - desired_sign * margin)


def inverse_weighted_distance(
    x_orig: np.ndarray,
    X_cf: np.ndarray,
    weights: np.ndarray,
    cont_idx: np.ndarray,
    cat_idx: np.ndarray,
    cont_scales: np.ndarray,
) -> np.ndarray:
    """Inverse-weighted Euclidean proximity (Eq. 4).

    ``dist(x_i, x*_i) = sum_j sqrt(w_j * (x_ij - x*_ij)^2)``
    Continuous differences are normalized by ``cont_scales``; categorical
    differences use indicator ``I(x_ij != x*_ij)``.
    """
    n = X_cf.shape[0]
    dist = np.zeros(n, dtype=np.float64)

    if cont_idx.size > 0:
        diff_cont = (X_cf[:, cont_idx] - x_orig[cont_idx]) / cont_scales
        sqrt_w_cont = np.sqrt(weights[cont_idx])
        dist += np.sum(sqrt_w_cont[None, :] * np.abs(diff_cont), axis=1)

    if cat_idx.size > 0:
        diff_cat = (X_cf[:, cat_idx] != x_orig[cat_idx]).astype(np.float64)
        sqrt_w_cat = np.sqrt(weights[cat_idx])
        dist += np.sum(sqrt_w_cat[None, :] * diff_cat, axis=1)

    return dist


def sparsity_count(
    x_orig: np.ndarray,
    X_cf: np.ndarray,
    cont_idx: np.ndarray,
    cat_idx: np.ndarray,
    tol: float = 1e-8,
) -> np.ndarray:
    """Count of changed features per counterfactual (Eq. 5)."""
    n = X_cf.shape[0]
    count = np.zeros(n, dtype=np.float64)

    if cont_idx.size > 0:
        count += np.sum(np.abs(X_cf[:, cont_idx] - x_orig[cont_idx]) > tol, axis=1)
    if cat_idx.size > 0:
        count += np.sum(X_cf[:, cat_idx] != x_orig[cat_idx], axis=1)

    return count


def relative_impact_metric(
    x_orig: np.ndarray,
    X_cf: np.ndarray,
    weights: np.ndarray,
    cont_idx: np.ndarray,
    cat_idx: np.ndarray,
    cont_scales: np.ndarray,
    tol: float = 1e-8,
) -> np.ndarray:
    """Relative Impact Metric (RIM, Eq. 6).

    Combines the number of changed features with the relative magnitude of
    change weighted by inverse feature importance ``w_j``.
    """
    n = X_cf.shape[0]
    rim = np.zeros(n, dtype=np.float64)

    if cont_idx.size > 0:
        changed_cont = np.abs(X_cf[:, cont_idx] - x_orig[cont_idx]) > tol
        d_real_cont = np.sum(changed_cont, axis=1)
        denom = np.where(np.abs(x_orig[cont_idx]) > 1e-8, np.abs(x_orig[cont_idx]), cont_scales)
        rel_change = np.abs(x_orig[cont_idx] - X_cf[:, cont_idx]) / denom
        rim += d_real_cont * np.sum(changed_cont * rel_change * weights[cont_idx][None, :], axis=1)

    if cat_idx.size > 0:
        changed_cat = X_cf[:, cat_idx] != x_orig[cat_idx]
        d_real_cat = np.sum(changed_cat, axis=1)
        rim += d_real_cat * np.sum(changed_cat * weights[cat_idx][None, :], axis=1)

    return rim


def eval_validity(y_pred_cf: np.ndarray, y_desired: np.ndarray) -> float:
    """Validity metric (Eq. 11)."""
    return float(np.mean(np.asarray(y_pred_cf) == np.asarray(y_desired)))


def eval_cont_sparsity(X_orig: np.ndarray, X_cf: np.ndarray, cont_idx: np.ndarray, tol: float = 1e-8) -> float:
    """Continuous feature sparsity (Eq. 12). Higher means fewer changes."""
    if cont_idx.size == 0:
        return 1.0
    changed = np.abs(X_cf[:, cont_idx] - X_orig[:, cont_idx]) > tol
    return float(np.mean(1.0 - np.mean(changed, axis=1)))


def eval_cat_sparsity(X_orig: np.ndarray, X_cf: np.ndarray, cat_idx: np.ndarray) -> float:
    """Categorical feature sparsity (Eq. 13). Higher means fewer changes."""
    if cat_idx.size == 0:
        return 1.0
    changed = X_cf[:, cat_idx] != X_orig[:, cat_idx]
    return float(np.mean(1.0 - np.mean(changed, axis=1)))


def eval_cont_proximity(
    X_orig: np.ndarray,
    X_cf: np.ndarray,
    cont_idx: np.ndarray,
    mads: np.ndarray,
) -> float:
    """Continuous feature proximity (Eq. 14): negative mean MAD-normalized L1 distance."""
    if cont_idx.size == 0:
        return 0.0
    safe_mad = np.where(mads > 1e-8, mads, 1.0)
    l1_mad = np.abs(X_cf[:, cont_idx] - X_orig[:, cont_idx]) / safe_mad[None, :]
    return float(-np.mean(np.mean(l1_mad, axis=1)))


def eval_lof_nr(lof_model: Any, X_cf_scaled: np.ndarray) -> float:
    """LOF Normality Ratio (LOF-NR, Eq. 15): fraction of counterfactuals predicted as inliers (+1)."""
    preds = lof_model.predict(X_cf_scaled)
    return float(np.mean(preds == 1))
