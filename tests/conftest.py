import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def mixed_dataset():
    rng = np.random.default_rng(42)
    n = 300
    age = rng.integers(20, 65, size=n).astype(float)
    income = rng.normal(50.0, 15.0, size=n)
    hours = rng.integers(20, 60, size=n).astype(float)
    edu = rng.choice(["HS", "Bachelors", "Masters"], size=n)
    gender = rng.choice(["M", "F"], size=n)

    # Target depends strongly on income and hours, slightly on edu
    score = 0.08 * (income - 50.0) + 0.05 * (hours - 40.0) + np.where(edu == "Masters", 1.0, 0.0)
    y = (score > 0.2).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "hours": hours,
            "edu": edu,
            "gender": gender,
        }
    )
    # Encode categories numerically for sklearn RF, or wrap RF in a DataFrame-accepting wrapper
    return df, y


class DataFrameModelWrapper:
    """Simple wrapper that one-hot/ordinal encodes string columns for sklearn."""

    def __init__(self, df: pd.DataFrame, y: np.ndarray):
        self.cat_cols = ["edu", "gender"]
        self.edu_map = {"HS": 0, "Bachelors": 1, "Masters": 2}
        self.gender_map = {"M": 0, "F": 1}
        self.clf = RandomForestClassifier(n_estimators=30, random_state=42)
        self.clf.fit(self._transform(df), y)
        self.classes_ = self.clf.classes_

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        out = df.copy()
        out["edu"] = out["edu"].map(self.edu_map)
        out["gender"] = out["gender"].map(self.gender_map)
        return out.to_numpy(dtype=np.float64)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict(self._transform(df))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self._transform(df))


@pytest.fixture
def trained_mixed_model(mixed_dataset):
    df, y = mixed_dataset
    return DataFrameModelWrapper(df, y)
