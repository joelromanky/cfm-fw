# CFM-FW

Implementation of **CFM-FW** (*A Feature Weights-Based Comprehensive Causal Multi-Objective Counterfactual Framework*, Liu et al., *Expert Systems With Applications*, 2025).

CFM-FW generates model-agnostic counterfactual explanations for mixed-type tabular data by combining:
- **AcME** global feature importance and inverse feature weights
- **KDTree-biased initialization** from opposite-class training instances
- **5-objective NSGA-II** optimization (hinge `yloss`, inverse-weighted distance, sparsity count, Relative Impact Metric `RIM`, and `LocalOutlierFactor` plausibility)
- **Mixed-type genetic operators** (SBX + Polynomial Mutation for continuous features; Uniform Crossover + Choice Random Mutation for categorical features)
- **Causal constraints** (immutable features, monotonic-increasing features, and optional custom constraint functions)
- **Post-hoc continuous sparsity reversion**

## Installation

```bash
cd /home/jky/cfm-fw
uv sync
```

Or with `pip`:

```bash
pip install -e /home/jky/cfm-fw
```

## Quickstart

```python
import pandas as pd
from cfm_fw import CFMFW

explainer = CFMFW(
    model=trained_model,                 # exposes predict() and predict_proba()
    continuous=["age", "income", "hours"],
    categorical=["edu", "gender"],
    fixed_features=["gender"],           # immutable: x_j - x*_j = 0
    monotonic_increasing=["age"],        # monotonic: x_j - x*_j <= 0
    pop_size=100,
    n_gen=10,
    cutoff=0.5,
    seed=42,
).fit(X_train)

# Generate counterfactuals for a single query (pd.Series or 1-row pd.DataFrame)
cfs = explainer.generate(X_test.iloc[0])

# Inspect AcME global feature importances and inverse weights
effects_table = explainer.acme_.get_feature_effects_table()
```

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `pop_size` | `100` | NSGA-II population size (and KDTree opposite-class seed pool size) |
| `n_gen` | `10` | Number of NSGA-II generations |
| `cutoff` | `0.5` | Minimum predicted probability required for the desired class |
| `n_neighbors_lof` | `20` | Number of neighbors $K$ for `LocalOutlierFactor` plausibility |
| `crossover_prob` | `0.9` | Crossover probability per mating pair |
| `eta_c` | `15.0` | Simulated Binary Crossover (SBX) distribution index |
| `eta_m` | `20.0` | Polynomial Mutation (PM) distribution index |
| `post_hoc_sparsity` | `True` | Revert small continuous perturbations if validity is preserved |
| `seed` | `42` | Random seed |

## Package Structure

- `src/cfm_fw/cfm_fw.py` — `CFMFW` main explainer (`fit`, `generate`)
- `src/cfm_fw/acme.py` — `AcME` quantile-based standardized effects & inverse weights
- `src/cfm_fw/metrics.py` — Optimization objectives (`hinge_yloss`, `inverse_weighted_distance`, `sparsity_count`, `relative_impact_metric`) and evaluation metrics (`eval_validity`, `eval_cont_sparsity`, `eval_cat_sparsity`, `eval_cont_proximity`, `eval_lof_nr`)
- `src/cfm_fw/nsga2.py` — Vectorized mixed-type NSGA-II with constraint-domination

## Running Tests

```bash
uv run pytest -v
```
