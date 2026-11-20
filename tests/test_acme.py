import numpy as np

from cfm_fw.acme import AcME


def test_acme_weights_and_effects(mixed_dataset, trained_mixed_model):
    df, _ = mixed_dataset
    acme = AcME(
        model=trained_mixed_model,
        continuous=["age", "income", "hours"],
        categorical=["edu", "gender"],
    ).fit(df)

    assert len(acme.importances_) == 5
    assert len(acme.inverse_weights_) == 5
    # income is the strongest driver in our synthetic DGP, so its importance should exceed gender's
    assert acme.importances_["income"] > acme.importances_["gender"]
    # Inverse weights should be smaller for more important features
    assert acme.inverse_weights_["income"] < acme.inverse_weights_["gender"]
    weights_arr = acme.get_weights_array(list(df.columns))
    assert weights_arr.shape == (5,)
    assert np.all(weights_arr > 0)

    table = acme.get_feature_effects_table()
    assert list(table.columns) == ["feature", "importance", "inverse_weight", "min_effect", "max_effect"]
    assert len(table) == 5
