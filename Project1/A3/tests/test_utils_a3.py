import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest



def test_return_param_values(m):
    d = {"a": 1, "b": 2, "c": 3}
    assert m.return_param_values(["b", "a"], d) == (2, 1)


def test_clean_simulation_df_filters_and_orders(m):
    df = pd.DataFrame(
        {
            "simulation_id": [1, 1, 1, 2, 2],
            "realization_id": [10, 10, 10, 20, 20],
            "time": [0, 1, 2, 0, 1],
            "state": [0, 1, 1, 0, 0],  # group (2,20) never reaches 1 -> should be removed
            "H1": [0.1, 0.2, 0.3, 9.0, 9.1],
            "L1": [1.1, 1.2, 1.3, 8.0, 8.1],
            "other": [5, 6, 7, 8, 9],
        }
    )

    cleaned, suffix_cols = m.clean_simulation_df(
        df,
        group_cols=["simulation_id", "realization_id"],
        state_col="state",
        front_cols=["simulation_id", "realization_id", "time", "state"],
        suffix_prefixes=["H", "L"],
    )

    assert set(suffix_cols) == {"H1", "L1"}
    # group (2,20) removed
    assert cleaned["simulation_id"].nunique() == 1
    assert cleaned["simulation_id"].iloc[0] == 1

    # cumsum(state) <= 1 keeps only up to first "1" row
    # for group (1,10): state = [0,1,1] => cumsum=[0,1,2] => last row removed
    assert cleaned.shape[0] == 2
    assert cleaned["time"].tolist() == [0, 1]

    # ordering: front_cols + middle + suffix
    assert list(cleaned.columns[:4]) == ["simulation_id", "realization_id", "time", "state"]
    assert cleaned.columns[-2:].tolist() == ["H1", "L1"]


# --------------------------------------------------------
# compute_pca_features
def test_compute_pca_features_adds_columns(m):
    # two base cols, one feature col with arrays
    df = pd.DataFrame(
        {
            "simulation_id": [1, 2, 3],
            "realization_id": [10, 20, 30],
            "feat": [
                np.array([1.0, 2.0, 3.0]),
                np.array([2.0, 3.0, 4.0]),
                np.array([np.nan, 0.0, 1.0]),  # invalid (has nan) by default check -> skipped row
            ],
        }
    )

    out = m.compute_pca_features(
        df=df,
        feature_cols=["feat"],
        base_cols=["simulation_id", "realization_id"],
        n_components=5,
    )

    # output has base cols
    assert set(["simulation_id", "realization_id"]).issubset(out.columns)

    # PCA columns should exist for valid rows (2 rows => up to 2 components)
    pca_cols = [c for c in out.columns if c.startswith("feat_PC")]
    assert len(pca_cols) in (1, 2)
    # rows with invalid feature should have NaNs in PCA columns
    row3 = out[out["simulation_id"] == 3].iloc[0]
    for c in pca_cols:
        assert pd.isna(row3[c])


# -----------------------------
# ##### prep Cox_TV DF
def test_prep_for_cox_tv_builds_start_stop_id_and_orders(m):
    df = pd.DataFrame(
        {
            "simulation_id": [1, 1, 1, 1],
            "realization_id": [10, 10, 10, 10],
            "time": [0, 1, 2, 3],
            "state": [0, 0, 1, 1],
            "H1": [np.nan, 0.1, 0.2, 0.3],     # float nan should become zeros_like(float)=0.0
            "Iimg": [1.0, 2.0, np.nan, 4.0],   # float nan -> 0.0
            "Eess": [0.0, 1.0, 2.0, 3.0],
        }
    )

    out = m.prep_for_cox_tv(
        df,
        group_cols=["simulation_id", "realization_id"],
        state_col="state",
        landscape_prefixes=["H"],
        image_prefixes=["I"],
        essential_prefixes=["E"],
    )

    # required cols
    assert out.columns[0:4].tolist() == ["id", "start", "stop", "state"]

    # id format
    assert out["id"].nunique() == 1
    assert out["id"].iloc[0] == "1_10"

    # start/stop
    assert out["start"].tolist() == [0, 1, 2, 3]
    assert out["stop"].tolist() == [1, 2, 3, 4]

    # NaNs replaced with zeros for float NaNs
    assert out["H1"].iloc[0] == 0.0
    assert out["Iimg"].iloc[2] == 0.0

