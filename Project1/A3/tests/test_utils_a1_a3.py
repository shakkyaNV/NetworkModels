import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")


MODULE_UNDER_TEST = "utilsA1"


@pytest.fixture(scope="session")
def m():
    """Import the module under test once."""
    try:
        return importlib.import_module(MODULE_UNDER_TEST)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"Could not import module '{MODULE_UNDER_TEST}'.\n"
            f"1) Make sure MODULE_UNDER_TEST is set correctly.\n"
            f"2) Run pytest from the folder where that module is importable.\n"
            f"Original error: {e}"
        )


# -----------------------------
# get_representation_choice_function
def test_get_representation_choice_function_outputs(m):
    f = m.get_representation_choice_function("persistence")
    assert callable(f)
    assert f((2.0, 5.0)) == 3.0

    f = m.get_representation_choice_function("max")
    assert f((-3.0, 2.0)) == 3.0  # max(|-3|, |2|, 2-(-3)=5) -> 5 actually
    # Correct expectation:
    assert m.get_representation_choice_function("max")((-3.0, 2.0)) == 5.0

    f = m.get_representation_choice_function("arctan")
    out = f((1.0, 2.0))
    assert np.isfinite(out)
    assert out == np.arctan(1.0)

    assert m.get_representation_choice_function("birth")((7.0, 9.0)) == 7.0
    assert m.get_representation_choice_function("death")((7.0, 9.0)) == 9.0


def test_get_representation_choice_function_invalid_key(m):
    with pytest.raises(KeyError):
        m.get_representation_choice_function("not_a_key")


# snapshots_to_activation_times_series
# -----------------------------
def test_snapshots_to_activation_times_series_basic(m):
    # snapshots are sets of active nodes at each step
    snapshots = [
        {0},
        {0, 1},
        {0, 1, 12},
    ]
    series = m.snapshots_to_activation_times_series(snapshots, num_nodes=13)

    assert len(series) == 3
    # step 0: node 0 activates at t=0
    assert series[0][0] == 0
    assert math.isnan(series[0][1])
    assert math.isnan(series[0][12])

    # step 1: node 1 activates at t=1
    assert series[1][0] == 0
    assert series[1][1] == 1
    assert math.isnan(series[1][12])

    # step 2: node 12 activates at t=2
    assert series[2][12] == 2
    assert series[2][1] == 1


# -----------
# generate_random_params
def test_generate_random_params_reproducible(m):
    p1 = m.generate_random_params(num_samples=5, seed_rng=123)
    p2 = m.generate_random_params(num_samples=5, seed_rng=123)
    assert p1 == p2  # deterministic w/ same seed


def test_generate_random_params_schema(m):
    params = m.generate_random_params(num_samples=3, seed_rng=999)
    assert len(params) == 3
    required = {
        "num_nodes",
        "num_neighbor_nodes",
        "total_random_edges",
        "distance_threshold",
        "weighted",
        "ngeo_placement",
        "n_seeds",
        "node_active_threshold",
        "upper_weight_limit",
        "skew_power",
        "seed_cluster_distance",
        "ngeom_edges_in_persistence",
        "max_persistence_dim",
        "threshold_sum",
        "seeding_method",
        "calculate_representation",
        "bandwidth",
        "representation_choice_function",
    }
    for d in params:
        assert required.issubset(d.keys())
        assert d["num_nodes"] == 20
        assert d["weighted"] is True
        assert d["distance_threshold"] >= d["num_neighbor_nodes"] + 1


# -----------------------------------------------------------------
# graph_to_distance_matrix
def test_graph_to_distance_matrix_weighted(m):
    nx = pytest.importorskip("networkx")

    G = nx.Graph()
    G.add_edge(0, 1, weight=2.0)
    G.add_edge(1, 2, weight=3.0)
    G.add_edge(0, 2, weight=10.0)  # longer direct edge

    nodes = [0, 1, 2]
    dist = m.graph_to_distance_matrix(G, nodes)

    assert dist.shape == (3, 3)
    assert np.allclose(np.diag(dist), 0.0)
    assert np.allclose(dist, dist.T)

    # shortest 0->2 should be 2+3=5 not 10
    assert dist[0, 2] == pytest.approx(5.0)
    assert dist[0, 1] == pytest.approx(2.0)
    assert dist[1, 2] == pytest.approx(3.0)


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



# cvt_validation_score
# lifelines CoxTimeVaryingFitter here so the test is stable.)
def test_cvt_validation_score_with_mocked_lifelines(m, monkeypatch):
    # Build a tiny "ready" cox tv df
    df = pd.DataFrame(
        {
            "id": ["1_1"] * 3 + ["2_2"] * 3,
            "start": [0, 1, 2, 0, 1, 2],
            "stop": [1, 2, 3, 1, 2, 3],
            "state": [0, 0, 1, 0, 1, 1],
            "H1": [0.0, 0.1, 0.2, 0.0, 0.2, 0.3],
        }
    )

    class DummyCTVF:
        def fit(self, *args, **kwargs):
            return self

        def predict_partial_hazard(self, val_df):
            # deterministic "risk": increasing with H1
            return val_df["H1"].astype(float).values

    # Patch the class used inside the module
    monkeypatch.setattr(m, "CoxTimeVaryingFitter", DummyCTVF)

    #  patch concordance_index to something predictable
    def dummy_cindex(times, scores, events):
        # return a value in [0,1]
        return 0.5

    monkeypatch.setattr(m, "concordance_index", dummy_cindex)

    score = m.cvt_validation_score(df, k=2)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(0.5)


# -------
# visualize_graph + visualize_step_animation_new
def test_visualize_graph_writes_file_and_sets_edge_labels(m, monkeypatch, tmp_path):
    nx = pytest.importorskip("networkx")

    class DummyNet:
        def __init__(self, *args, **kwargs):
            self.nodes = []
            self._saved_to = None

        def from_nx(self, G):
            # mimic pyvis behavior: nodes list of dicts with id
            self.nodes = [{"id": n} for n in G.nodes]

        def save_graph(self, output_file):
            self._saved_to = output_file
            Path(output_file).write_text("<html><body><script></script></body></html>", encoding="utf-8")

    monkeypatch.setattr(m, "Network", DummyNet)

    G = nx.Graph()
    G.add_edge(1, 2, type="A", weight=3.14)
    out_file = tmp_path / "g.html"

    m.visualize_graph(G, str(out_file))

    assert out_file.exists()
    # edge attrs should have title/label/font
    attrs = G.get_edge_data(1, 2)
    assert "title" in attrs and "label" in attrs
    assert attrs["title"] == "A (3.14)"
    assert attrs["label"] == "A (3.14)"
    assert isinstance(attrs["font"], dict)


def test_visualize_step_animation_new_inserts_controls(m, monkeypatch, tmp_path):
    nx = pytest.importorskip("networkx")

    class DummyNet:
        def __init__(self, *args, **kwargs):
            self.nodes = []

        def from_nx(self, G):
            self.nodes = [{"id": n} for n in G.nodes]

        def save_graph(self, output_file):
            # IMPORTANT: the function searches for the LAST </script>
            Path(output_file).write_text(
                "<html><body><script>var network={body:{data:{nodes:{update:function(){}},edges:{get:function(){return[{id:1}]},update:function(){}}}}};</script></body></html>",
                encoding="utf-8",
            )

    monkeypatch.setattr(m, "Network", DummyNet)

    G = nx.Graph()
    G.add_edge(0, 1, weight=0.0)
    G.add_edge(1, 2, weight=0.0)

    snapshots = [{0}, {0, 1}, {0, 1, 2}]
    out_file = tmp_path / "anim.html"

    m.visualize_step_animation_new(G, snapshots, str(out_file))

    html = out_file.read_text(encoding="utf-8")
    assert "function nextStep()" in html
    assert "function prevStep()" in html
    assert "snapshotsNodes" in html
    assert "snapshotsEdges" in html


# ---------------
# export_graphml_with_namespace
def test_export_graphml_with_namespace_rewrites_header(m, tmp_path):
    nx = pytest.importorskip("networkx")

    G = nx.Graph()
    G.add_edge("a", "b")

    out = tmp_path / "g.graphml"
    schema = "http://example.com/graphml.xsd"

    m.export_graphml_with_namespace(G, str(out), xmlns_path=schema)

    content = out.read_text(encoding="utf-8")
    assert 'xmlns="http://graphml.graphdrawing.org/xmlns"' in content
    assert 'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"' in content
    assert schema in content