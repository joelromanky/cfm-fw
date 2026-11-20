import numpy as np

from cfm_fw.metrics import (
    eval_cat_sparsity,
    eval_cont_proximity,
    eval_cont_sparsity,
    eval_validity,
    hinge_yloss,
    inverse_weighted_distance,
    relative_impact_metric,
    sparsity_count,
)


def test_objective_metrics():
    p = np.array([0.1, 0.5, 0.9, 1.0])
    loss = hinge_yloss(p, desired_sign=1.0)
    assert np.allclose(loss, [1.8, 1.0, 0.2, 0.0])

    x_orig = np.array([10.0, 20.0, 1.0])
    X_cf = np.array([[10.0, 20.0, 1.0], [12.0, 20.0, 0.0]])
    weights = np.array([1.0, 4.0, 1.0])
    cont_idx = np.array([0, 1])
    cat_idx = np.array([2])
    scales = np.array([2.0, 5.0])

    dist = inverse_weighted_distance(x_orig, X_cf, weights, cont_idx, cat_idx, scales)
    assert np.allclose(dist, [0.0, 2.0])

    sp = sparsity_count(x_orig, X_cf, cont_idx, cat_idx)
    assert np.allclose(sp, [0.0, 2.0])

    rim = relative_impact_metric(x_orig, X_cf, weights, cont_idx, cat_idx, scales)
    assert rim[0] == 0.0
    assert rim[1] > 0.0


def test_evaluation_metrics():
    assert eval_validity(np.array([1, 1, 0]), np.array([1, 1, 1])) == 2.0 / 3.0
    X_o = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]])
    X_c = np.array([[1.0, 3.0, 0.0], [3.0, 4.0, 0.0]])
    assert eval_cont_sparsity(X_o, X_c, np.array([0, 1])) == 0.75
    assert eval_cat_sparsity(X_o, X_c, np.array([2])) == 0.5
    assert eval_cont_proximity(X_o, X_c, np.array([0, 1]), np.array([1.0, 1.0])) == -0.25
