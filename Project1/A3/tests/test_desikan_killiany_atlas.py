# tests/test_dk_atlas_graph.py
import importlib
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
nx = pytest.importorskip("networkx")


IMPORT_PATH = "Project1.A1.desikan_killiany_atlas"


@pytest.fixture(scope="session")
def m():
    try:
        return importlib.import_module(IMPORT_PATH)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"Could not import '{IMPORT_PATH}'.\n"
            f"Set IMPORT_PATH to the module containing DKAtlasGraph.\n"
            f"Original error: {e}"
        )


@pytest.fixture
def sample_graph():
    """
    Build a tiny graph that looks like what DKAtlasGraph expects:
    - node ids can be strings (so relabel to int is exercised)
    - each node has 'dn_name' attribute used by rename_nodes
    - edges have number_of_fibers, fiber_length_mean, number_of_fibgers (typo used in weight0)
    """
    G = nx.Graph()

    # node ids as strings -> DKAtlasGraph._load_graphml will relabel them to ints
    G.add_node("1", dn_name="DN_A")
    G.add_node("2", dn_name="DN_B")

    # add one edge with attributes used by assign_edge_weight
    G.add_edge(
        "1",
        "2",
        number_of_fibers=10.0,
        fiber_length_mean=2.0,
        number_of_fibgers=7.0,  # matches your weight0 code key
    )
    return G


@pytest.fixture
def patched_env(monkeypatch, m, sample_graph, tmp_path):
    """
    Patch symbols imported from Project1.A3.utils_a3 inside the DKAtlasGraph module:
    - BASE_DIR so that xml_path becomes resolvable
    - DK_FSNAMES_MAPPING_DICT for renaming
    - NODE_FSREGION_TO_ID for patient data mapping
    - node_df_from_graph / edge_df_from_graph / determine_geom_ngeom_edges
    - nx.read_graphml so we don't need a real file
    """
    # Ensure BASE_DIR points somewhere that exists
    monkeypatch.setattr(m, "BASE_DIR", str(tmp_path), raising=False)

    # Provide a minimal rename dict + node mapping
    monkeypatch.setattr(m, "DK_FSNAMES_MAPPING_DICT", {"DN_A": "FS_A", "DN_B": "FS_B"}, raising=False)
    monkeypatch.setattr(m, "NODE_FSREGION_TO_ID", {"FS_A": 1, "FS_B": 2}, raising=False)

    # Patch nx.read_graphml used inside the module under test.
    # Important: DKAtlasGraph references nx imported inside that module (m.nx)
    monkeypatch.setattr(m.nx, "read_graphml", lambda p: sample_graph.copy())

    # edge_df_from_graph: return a DataFrame with edge info (used by set_ngeom_geom_property)
    def fake_edge_df_from_graph(G):
        rows = []
        for u, v, data in G.edges(data=True):
            rows.append({"u": int(u), "v": int(v), **data})
        return pd.DataFrame(rows)

    monkeypatch.setattr(m, "edge_df_from_graph", fake_edge_df_from_graph, raising=False)

    # determine_geom_ngeom_edges: return list of (u, v, value) where value becomes edge property 'type'
    monkeypatch.setattr(
        m,
        "determine_geom_ngeom_edges",
        lambda edge_df, fiber_max_geom_length, base_method=True: [(1, 2, "geom")],
        raising=False,
    )

    # node_df_from_graph isn't used by DKAtlasGraph directly in the snippet, but patch anyway for safety
    monkeypatch.setattr(m, "node_df_from_graph", lambda G: pd.DataFrame(), raising=False)

    return tmp_path


# -------------
def test_init_loads_graph_and_relabels_nodes_to_int(m, patched_env):
    g = m.DKAtlasGraph(xml_path="resources/masters33.graphml", base_rename=False)
    assert isinstance(g.graph, nx.Graph)

    # node ids should now be ints (relabel_nodes in-place)
    assert set(g.graph.nodes()) == {1, 2}
    assert g._graph_renamed is False


def test_init_with_base_rename_sets_region_name(m, patched_env):
    g = m.DKAtlasGraph(xml_path="resources/masters33.graphml", base_rename=True)

    assert g._graph_renamed is True
    assert g.graph.nodes[1]["region_name"] == "FS_A"
    assert g.graph.nodes[2]["region_name"] == "FS_B"


def test_rename_nodes_sets_new_attr_and_flag(m, patched_env):
    g = m.DKAtlasGraph(xml_path="resources/masters33.graphml", base_rename=False)

    rename = {"DN_A": "FS_A", "DN_B": "FS_B"}
    g.rename_nodes(rename_dict=rename, attr_name="dn_name", new_attr_name="region_name")

    assert g._graph_renamed is True
    assert g.graph.nodes[1]["region_name"] == "FS_A"
    assert g.graph.nodes[2]["region_name"] == "FS_B"


def test_rename_nodes_raises_graphcreationerror_on_missing_key(m, patched_env):
    g = m.DKAtlasGraph(xml_path="resources/masters33.graphml", base_rename=False)

    # Missing DN_B mapping => should raise GraphCreationError
    bad_rename = {"DN_A": "FS_A"}
    with pytest.raises(m.GraphCreationError):
        g.rename_nodes(rename_dict=bad_rename, attr_name="dn_name", new_attr_name="region_name")


def test_assign_edge_weight_weight1(m, patched_env):
    g = m.DKAtlasGraph(xml_path="resources/masters33.graphml", base_rename=True)
    g.assign_edge_weight(weight_function="weight1")

    data = g.graph.get_edge_data(1, 2)
    assert g._graph_weighted is True
    assert "weight" in data
    assert data["weight"] == pytest.approx(10.0 / 2.0)  # number_of_fibers / fiber_length_mean


def test_assign_edge_weight_weight0_uses_number_of_fibgers(m, patched_env):
    g = m.DKAtlasGraph(xml_path="resources/masters33.graphml", base_rename=True)
    g.assign_edge_weight(weight_function="weight0")

    data = g.graph.get_edge_data(1, 2)
    assert "weight" in data
    assert data["weight"] == pytest.approx(7.0)


def test_assign_edge_weight_zero_division_raises(m, monkeypatch, patched_env):
    # Patch read_graphml to create fiber_length_mean=0 to trigger ZeroDivisionError
    def bad_graphml(_):
        G = nx.Graph()
        G.add_node("1", dn_name="DN_A")
        G.add_node("2", dn_name="DN_B")
        G.add_edge("1", "2", number_of_fibers=10.0, fiber_length_mean=0.0, number_of_fibgers=1.0)
        return G

    monkeypatch.setattr(m.nx, "read_graphml", bad_graphml)

    g = m.DKAtlasGraph(xml_path="resources/masters33.graphml", base_rename=True)

    with pytest.raises(m.GraphCreationError):
        g.assign_edge_weight(weight_function="weight1")


def test_input_patient_data_sets_node_attribute(m, patched_env):
    g = m.DKAtlasGraph(xml_path="resources/masters33.graphml", base_rename=True)

    # data_series index are fs region names; values are patient measurements
    s = pd.Series({"FS_A": 0.25, "FS_B": 0.75})
    g.input_patient_data(s, df_type="suvr")

    assert g.graph.nodes[1]["suvr"] == pytest.approx(0.25)
    assert g.graph.nodes[2]["suvr"] == pytest.approx(0.75)


def test_set_ngeom_geom_property_sets_edge_type(m, patched_env):
    g = m.DKAtlasGraph(xml_path="resources/masters33.graphml", base_rename=True)

    g.set_ngeom_geom_property(geom_max_fiber_length=100)

    # NOTE: your _graph_set_edge_property uses add_weighted_edges_from(..., weight=property_name)
    # That sets edge attribute with name=property_name ('type') to the given "weight" value (here "geom")
    data = g.graph.get_edge_data(1, 2)
    assert "type" in data
    assert data["type"] == "geom"


# ----------------------------
def test_summary_runs(m, patched_env, capsys):
    g = m.DKAtlasGraph(xml_path="resources/masters33.graphml", base_rename=True)
    g.summary()
    out = capsys.readouterr().out
    assert "Graph has" in out
    assert "Node Information" in out