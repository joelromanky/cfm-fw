import numpy as np

from cfm_fw import CFMFW


def test_cfm_fw_mixed_with_causal_constraints(mixed_dataset, trained_mixed_model):
    df, _ = mixed_dataset
    preds = trained_mixed_model.predict(df)
    neg_idx = np.where(preds == 0)[0][:5]

    explainer = CFMFW(
        model=trained_mixed_model,
        continuous=["age", "income", "hours"],
        categorical=["edu", "gender"],
        fixed_features=["gender"],
        monotonic_increasing=["age"],
        pop_size=40,
        n_gen=10,
        cutoff=0.5,
        seed=42,
    ).fit(df)

    for idx in neg_idx:
        query = df.iloc[idx]
        cfs = explainer.generate(query)
        assert not cfs.empty, f"Expected valid counterfactual for row {idx}"
        # Check prediction flipped
        cf_preds = trained_mixed_model.predict(cfs)
        assert np.all(cf_preds == 1)
        # Check immutable constraint: gender unchanged
        assert (cfs["gender"] == query["gender"]).all()
        # Check monotonic increasing constraint: age >= original age
        assert (cfs["age"] >= query["age"]).all()
