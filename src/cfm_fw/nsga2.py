"""Vectorized NSGA-II implementation with mixed-type operators for CFM-FW."""

from __future__ import annotations

from typing import Callable

import numpy as np


def fast_non_dominated_sort(F: np.ndarray, cv: np.ndarray | None = None) -> list[np.ndarray]:
    """Partition solutions into non-dominated fronts using constraint-domination.

    Parameters
    ----------
    F : np.ndarray
        Objective matrix of shape ``(n, n_obj)`` (all minimized).
    cv : np.ndarray | None
        Total constraint violation per solution, shape ``(n,)``.
        Solutions with ``cv <= 0`` are feasible and always dominate
        infeasible solutions (``cv > 0``).
    """
    n = F.shape[0]
    if n == 0:
        return []

    if cv is None:
        cv = np.zeros(n, dtype=np.float64)

    feasible = cv <= 0.0

    # Pairwise dominance: i dominates j iff:
    # 1. i feasible and j infeasible, OR
    # 2. both infeasible and cv[i] < cv[j], OR
    # 3. both feasible and F[i] <= F[j] all and F[i] < F[j] any
    both_feasible = feasible[:, None] & feasible[None, :]
    both_infeasible = (~feasible[:, None]) & (~feasible[None, :])

    diff = F[:, None, :] - F[None, :, :]
    pareto_dom = np.all(diff <= 0.0, axis=2) & np.any(diff < 0.0, axis=2)

    dominates = (
        (feasible[:, None] & (~feasible[None, :]))
        | (both_infeasible & (cv[:, None] < cv[None, :]))
        | (both_feasible & pareto_dom)
    )
    np.fill_diagonal(dominates, False)

    domination_count = dominates.sum(axis=0).astype(np.int64)
    remaining = np.ones(n, dtype=bool)
    fronts: list[np.ndarray] = []

    while remaining.any():
        front = np.where(remaining & (domination_count == 0))[0]
        if front.size == 0:
            fronts.append(np.where(remaining)[0])
            break
        fronts.append(front)
        remaining[front] = False
        domination_count -= dominates[front].sum(axis=0)

    return fronts


def crowding_distance(F_front: np.ndarray) -> np.ndarray:
    """Compute crowding distance for a single Pareto front of shape ``(k, n_obj)``."""
    k, n_obj = F_front.shape
    if k == 0:
        return np.empty(0, dtype=np.float64)
    if k <= 2:
        return np.full(k, np.inf, dtype=np.float64)

    dist = np.zeros(k, dtype=np.float64)
    for m in range(n_obj):
        order = np.argsort(F_front[:, m], kind="mergesort")
        f_min = F_front[order[0], m]
        f_max = F_front[order[-1], m]
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        span = f_max - f_min
        if span <= 1e-12:
            continue
        dist[order[1:-1]] += (F_front[order[2:], m] - F_front[order[:-2], m]) / span

    return dist


def assign_rank_and_crowding(F: np.ndarray, cv: np.ndarray | None = None) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Return ``(fronts, rank_per_ind, crowding_per_ind)``."""
    n = F.shape[0]
    fronts = fast_non_dominated_sort(F, cv=cv)
    ranks = np.empty(n, dtype=np.int64)
    cd = np.zeros(n, dtype=np.float64)
    for r, front in enumerate(fronts):
        ranks[front] = r
        cd[front] = crowding_distance(F[front])
    return fronts, ranks, cd


def binary_tournament_selection(
    ranks: np.ndarray,
    crowding: np.ndarray,
    n_parents: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select ``n_parents`` parent indices using crowded-comparison binary tournament."""
    pop_size = len(ranks)
    a = rng.integers(0, pop_size, size=n_parents)
    b = rng.integers(0, pop_size, size=n_parents)

    a_wins = (ranks[a] < ranks[b]) | ((ranks[a] == ranks[b]) & (crowding[a] > crowding[b]))
    b_wins = (ranks[b] < ranks[a]) | ((ranks[b] == ranks[a]) & (crowding[b] > crowding[a]))
    coin = rng.random(n_parents) < 0.5
    return np.where(a_wins, a, np.where(b_wins, b, np.where(coin, a, b)))


def sbx_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    eta_c: float,
    rng: np.random.Generator,
    prob_var: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover (SBX) for continuous variables."""
    c1 = p1.copy()
    c2 = p2.copy()
    n_matings, n_cont = p1.shape
    if n_cont == 0:
        return c1, c2

    do_cross = (rng.random((n_matings, n_cont)) <= prob_var) & (np.abs(p1 - p2) > 1e-14)
    if not do_cross.any():
        return c1, c2

    y1 = np.minimum(p1, p2)
    y2 = np.maximum(p1, p2)
    delta = np.maximum(y2 - y1, 1e-14)

    rand = rng.random((n_matings, n_cont))

    # Child 1
    beta = 1.0 + (2.0 * (y1 - xl[None, :]) / delta)
    alpha = 2.0 - np.power(beta, -(eta_c + 1.0))
    betaq = np.where(
        rand <= (1.0 / alpha),
        np.power(rand * alpha, 1.0 / (eta_c + 1.0)),
        np.power(1.0 / np.maximum(2.0 - rand * alpha, 1e-14), 1.0 / (eta_c + 1.0)),
    )
    cand1 = 0.5 * ((y1 + y2) - betaq * delta)

    # Child 2
    beta = 1.0 + (2.0 * (xu[None, :] - y2) / delta)
    alpha = 2.0 - np.power(beta, -(eta_c + 1.0))
    betaq = np.where(
        rand <= (1.0 / alpha),
        np.power(rand * alpha, 1.0 / (eta_c + 1.0)),
        np.power(1.0 / np.maximum(2.0 - rand * alpha, 1e-14), 1.0 / (eta_c + 1.0)),
    )
    cand2 = 0.5 * ((y1 + y2) + betaq * delta)

    cand1 = np.clip(cand1, xl[None, :], xu[None, :])
    cand2 = np.clip(cand2, xl[None, :], xu[None, :])

    # Randomly assign cand1/cand2 to c1/c2
    swap = rng.random((n_matings, n_cont)) < 0.5
    out1 = np.where(swap, cand2, cand1)
    out2 = np.where(swap, cand1, cand2)

    c1 = np.where(do_cross, out1, c1)
    c2 = np.where(do_cross, out2, c2)
    return c1, c2


def polynomial_mutation(
    X_cont: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    eta_m: float,
    prob_var: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Polynomial Mutation (PM) for continuous variables."""
    n, n_cont = X_cont.shape
    if n_cont == 0:
        return X_cont

    Xp = X_cont.copy()
    span = xu - xl
    valid_span = span > 1e-14
    do_mut = (rng.random((n, n_cont)) < prob_var) & valid_span[None, :]
    if not do_mut.any():
        return Xp

    safe_span = np.where(valid_span, span, 1.0)[None, :]
    delta1 = (Xp - xl[None, :]) / safe_span
    delta2 = (xu[None, :] - Xp) / safe_span
    rand = rng.random((n, n_cont))
    mut_pow = 1.0 / (eta_m + 1.0)

    xy1 = 1.0 - delta1
    val1 = 2.0 * rand + (1.0 - 2.0 * rand) * np.power(np.maximum(xy1, 0.0), eta_m + 1.0)
    deltaq1 = np.power(np.maximum(val1, 0.0), mut_pow) - 1.0

    xy2 = 1.0 - delta2
    val2 = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * np.power(np.maximum(xy2, 0.0), eta_m + 1.0)
    deltaq2 = 1.0 - np.power(np.maximum(val2, 0.0), mut_pow)

    deltaq = np.where(rand < 0.5, deltaq1, deltaq2)
    mutated = np.clip(Xp + deltaq * safe_span, xl[None, :], xu[None, :])
    return np.where(do_mut, mutated, Xp)


def uniform_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
    prob_var: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform Crossover (UX) for categorical variables."""
    if p1.shape[1] == 0:
        return p1.copy(), p2.copy()
    mask = rng.random(p1.shape) < prob_var
    c1 = np.where(mask, p2, p1)
    c2 = np.where(mask, p1, p2)
    return c1, c2


def choice_random_mutation(
    X_cat: np.ndarray,
    cat_cardinalities: np.ndarray,
    prob_var: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choice Random Mutation for categorical variables (integer-encoded ``0..K-1``)."""
    n, n_cat = X_cat.shape
    if n_cat == 0:
        return X_cat

    Xp = X_cat.copy()
    do_mut = (rng.random((n, n_cat)) < prob_var) & (cat_cardinalities[None, :] > 1)
    for j in range(n_cat):
        rows = np.where(do_mut[:, j])[0]
        if rows.size == 0:
            continue
        k = int(cat_cardinalities[j])
        Xp[rows, j] = rng.integers(0, k, size=rows.size)
    return Xp


def run_nsga2(
    init_pop: np.ndarray,
    evaluate_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    cont_idx: np.ndarray,
    cat_idx: np.ndarray,
    mutable_mask: np.ndarray,
    x_orig: np.ndarray,
    xl_cont: np.ndarray,
    xu_cont: np.ndarray,
    cat_cardinalities: np.ndarray,
    n_gen: int = 10,
    crossover_prob: float = 0.9,
    eta_c: float = 15.0,
    eta_m: float = 20.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run NSGA-II (Algorithm 1) for mixed-type counterfactual search.

    Returns ``(X_front, F_front, cv_front)`` of the final first Pareto front.
    """
    rng = np.random.default_rng(seed)
    pop_size, n_var = init_pop.shape
    pop = init_pop.copy()

    # Enforce immutability on initial population
    if not mutable_mask.all():
        pop[:, ~mutable_mask] = x_orig[~mutable_mask]

    F, cv = evaluate_fn(pop)
    _, ranks, cd = assign_rank_and_crowding(F, cv=cv)

    mut_cont_mask = mutable_mask[cont_idx] if cont_idx.size > 0 else np.empty(0, dtype=bool)
    mut_cat_mask = mutable_mask[cat_idx] if cat_idx.size > 0 else np.empty(0, dtype=bool)
    n_mutable = max(int(mutable_mask.sum()), 1)
    prob_var = min(0.5, 1.0 / n_mutable)

    for _ in range(n_gen):
        # Ensure even number of parents
        n_matings = (pop_size + 1) // 2
        parent_idx = binary_tournament_selection(ranks, cd, 2 * n_matings, rng)
        p1 = pop[parent_idx[:n_matings]]
        p2 = pop[parent_idx[n_matings:]]

        # Crossover gating per mating
        do_cx = rng.random(n_matings) < crossover_prob
        off1 = p1.copy()
        off2 = p2.copy()

        if do_cx.any():
            if cont_idx.size > 0 and mut_cont_mask.any():
                c_idx = cont_idx[mut_cont_mask]
                c1, c2 = sbx_crossover(
                    p1[do_cx][:, c_idx],
                    p2[do_cx][:, c_idx],
                    xl_cont[mut_cont_mask],
                    xu_cont[mut_cont_mask],
                    eta_c=eta_c,
                    rng=rng,
                )
                off1[np.ix_(do_cx, c_idx)] = c1
                off2[np.ix_(do_cx, c_idx)] = c2

            if cat_idx.size > 0 and mut_cat_mask.any():
                k_idx = cat_idx[mut_cat_mask]
                k1, k2 = uniform_crossover(
                    p1[do_cx][:, k_idx],
                    p2[do_cx][:, k_idx],
                    rng=rng,
                )
                off1[np.ix_(do_cx, k_idx)] = k1
                off2[np.ix_(do_cx, k_idx)] = k2

        offspring = np.vstack([off1, off2])[:pop_size]

        # Mutation
        if cont_idx.size > 0 and mut_cont_mask.any():
            c_idx = cont_idx[mut_cont_mask]
            offspring[:, c_idx] = polynomial_mutation(
                offspring[:, c_idx],
                xl_cont[mut_cont_mask],
                xu_cont[mut_cont_mask],
                eta_m=eta_m,
                prob_var=prob_var,
                rng=rng,
            )

        if cat_idx.size > 0 and mut_cat_mask.any():
            k_idx = cat_idx[mut_cat_mask]
            offspring[:, k_idx] = choice_random_mutation(
                offspring[:, k_idx],
                cat_cardinalities[mut_cat_mask],
                prob_var=prob_var,
                rng=rng,
            )

        if not mutable_mask.all():
            offspring[:, ~mutable_mask] = x_orig[~mutable_mask]

        # Evaluate offspring and merge with parent population
        F_off, cv_off = evaluate_fn(offspring)
        merged_X = np.vstack([pop, offspring])
        merged_F = np.vstack([F, F_off])
        merged_cv = np.concatenate([cv, cv_off])

        fronts, _, _ = assign_rank_and_crowding(merged_F, cv=merged_cv)

        # Survival selection: fill pop_size from best fronts, breaking ties by crowding distance
        selected: list[int] = []
        for front in fronts:
            if len(selected) + len(front) <= pop_size:
                selected.extend(front.tolist())
            else:
                needed = pop_size - len(selected)
                cd_front = crowding_distance(merged_F[front])
                order = np.argsort(-cd_front, kind="mergesort")[:needed]
                selected.extend(front[order].tolist())
                break

        sel_arr = np.asarray(selected, dtype=np.int64)
        pop = merged_X[sel_arr]
        F = merged_F[sel_arr]
        cv = merged_cv[sel_arr]
        _, ranks, cd = assign_rank_and_crowding(F, cv=cv)

    final_fronts = fast_non_dominated_sort(F, cv=cv)
    best_front = final_fronts[0] if final_fronts else np.arange(len(pop))
    return pop[best_front], F[best_front], cv[best_front]
