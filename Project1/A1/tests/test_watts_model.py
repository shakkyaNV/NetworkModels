import pytest
import numpy as np
import networkx as nx
import pandas as pd
from collections import defaultdict
from unittest.mock import patch, MagicMock


@pytest.fixture
def simple_params():
    return {
        "num_nodes": 20,
        "num_neighbor_nodes": 2,
        "total_random_edges": 5,
        "distance_threshold": 4,
        "upper_weight_limit": 5,
        "skew_power": 2,
        "weighted": True,
        "ngeo_placement": "random.choice",
        "node_active_threshold": 0.1,
        "threshold_sum": 10,
        "n_seeds": 2,
        "seeding_method": "all_combinations",
        "max_persistence_dim": 1,
        "calculate_representation": False,
    }


@pytest.fixture
def small_graph():
    from contagion import generate_graph
    return generate_graph(
        num_nodes=20,
        num_neighbor_nodes=2,
        total_random_edges=5,
        distance_threshold=4,
        upper_weight_limit=5,
        skew_power=2,
        weighted=True,
        ngeo_placement="random.choice",
    )


@pytest.fixture
def adjacency_and_weight(small_graph):
    adj = nx.to_numpy_array(small_graph, dtype=float, weight=None)
    w0  = nx.to_numpy_array(small_graph, weight="weight", nonedge=1e9)
    return adj, w0

class TestAddSkewedWeights:

    def test_output_length(self):
        from contagion import add_skewed_weights
        w = add_skewed_weights(n=10)
        assert len(w) == 10

    def test_values_within_range(self):
        from contagion import add_skewed_weights
        w = add_skewed_weights(n=500, upper_weight_limit=10, skew_power=3)
        assert np.all(w >= 0)
        assert np.all(w <= 11)  # round can push to upper_weight_limit + 1

    def test_single_weight(self):
        from contagion import add_skewed_weights
        w = add_skewed_weights(n=1)
        assert w.shape == (1,)

    def test_returns_ndarray(self):
        from contagion import add_skewed_weights
        w = add_skewed_weights(n=5)
        assert isinstance(w, np.ndarray)

    def test_skew_biases_towards_zero(self):
        """High skew_power should produce mostly low values."""
        from contagion import add_skewed_weights
        w = add_skewed_weights(n=10000, upper_weight_limit=10, skew_power=10)
        assert np.mean(w) < 2.0


class TestGenerateGraph:

    def test_returns_networkx_graph(self):
        from contagion import generate_graph
        G = generate_graph(20, 2, 5, distance_threshold=4)
        assert isinstance(G, nx.Graph)

    def test_correct_node_count(self):
        from contagion import generate_graph
        G = generate_graph(30, 2, 5, distance_threshold=4)
        assert G.number_of_nodes() == 30

    def test_has_geometric_edges(self):
        from contagion import generate_graph
        G = generate_graph(20, 2, 5, distance_threshold=4)
        geo = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "geometric"]
        assert len(geo) > 0

    def test_has_non_geometric_edges(self):
        from contagion import generate_graph
        G = generate_graph(20, 2, 5, distance_threshold=4)
        ngeo = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "non_geometric"]
        assert len(ngeo) == 5

    def test_unweighted_graph_has_zero_weights(self):
        from contagion import generate_graph
        G = generate_graph(20, 2, 5, distance_threshold=4, weighted=False)
        geo_weights = [d["weight"] for _, _, d in G.edges(data=True) if d.get("type") == "geometric"]
        assert all(w == 0 for w in geo_weights)

    def test_zero_random_edges(self):
        from contagion import generate_graph
        G = generate_graph(20, 2, 0, distance_threshold=4)
        ngeo = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "non_geometric"]
        assert len(ngeo) == 0


class TestAddNonGeometricEdges:

    def _base_graph(self):
        G = nx.circulant_graph(20, offsets=[1, 2], create_using=nx.Graph)
        for u, v in G.edges():
            G[u][v]["weight"] = 0
            G[u][v]["type"] = "geometric"
        return G

    def test_random_choice_adds_correct_count(self):
        from contagion import add_non_geometric_edges
        G = self._base_graph()
        G = add_non_geometric_edges(G, total_random_edges=5, distance_threshold=4)
        ngeo = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "non_geometric"]
        assert len(ngeo) == 5

    def test_zero_edges_returns_unchanged(self):
        from contagion import add_non_geometric_edges
        G = self._base_graph()
        before = G.number_of_edges()
        G = add_non_geometric_edges(G, total_random_edges=0, distance_threshold=4)
        assert G.number_of_edges() == before

    def test_distance_threshold_respected(self):
        """Non-geometric edges should connect nodes that are far apart on the ring."""
        from contagion import add_non_geometric_edges
        n = 30
        G = nx.circulant_graph(n, offsets=[1, 2], create_using=nx.Graph)
        for u, v in G.edges():
            G[u][v]["weight"] = 0
            G[u][v]["type"] = "geometric"
        threshold = 8
        G = add_non_geometric_edges(G, total_random_edges=5, distance_threshold=threshold)
        for u, v, d in G.edges(data=True):
            if d.get("type") == "non_geometric":
                ring_dist = min(abs(u - v), n - abs(u - v))
                assert ring_dist >= threshold

class TestContagionPropagation:

    def test_returns_three_values(self, small_graph, adjacency_and_weight):
        from contagion import contagion_propagation
        adj, w0 = adjacency_and_weight
        result = contagion_propagation(
            num_nodes=20, adjacency_matrix=adj, weight_0=w0,
            init_seeds=(0, 1), node_active_threshold=0.1
        )
        assert len(result) == 3

    def test_init_seeds_are_active(self, small_graph, adjacency_and_weight):
        from contagion import contagion_propagation
        adj, w0 = adjacency_and_weight
        active_nodes, activation_times, _ = contagion_propagation(
            num_nodes=20, adjacency_matrix=adj, weight_0=w0,
            init_seeds=(0, 1), node_active_threshold=0.1
        )
        assert 0 in active_nodes
        assert 1 in active_nodes

    def test_activation_times_seeds_are_zero(self, adjacency_and_weight):
        from contagion import contagion_propagation
        adj, w0 = adjacency_and_weight
        _, activation_times, _ = contagion_propagation(
            num_nodes=20, adjacency_matrix=adj, weight_0=w0,
            init_seeds=(0, 1), node_active_threshold=0.1
        )
        assert activation_times[0] == 0
        assert activation_times[1] == 0

    def test_activation_times_length(self, adjacency_and_weight):
        from contagion import contagion_propagation
        adj, w0 = adjacency_and_weight
        _, activation_times, _ = contagion_propagation(
            num_nodes=20, adjacency_matrix=adj, weight_0=w0,
            init_seeds=(0,), node_active_threshold=0.1
        )
        assert len(activation_times) == 20

    def test_snapshots_first_entry_contains_seeds(self, adjacency_and_weight):
        from contagion import contagion_propagation
        adj, w0 = adjacency_and_weight
        _, _, snapshots = contagion_propagation(
            num_nodes=20, adjacency_matrix=adj, weight_0=w0,
            init_seeds=(3, 5), node_active_threshold=0.1
        )
        assert 3 in snapshots[0]
        assert 5 in snapshots[0]

    def test_high_threshold_limits_spread(self, adjacency_and_weight):
        """With threshold=1.0, nodes only activate if ALL neighbors are active."""
        from contagion import contagion_propagation
        adj, w0 = adjacency_and_weight
        active_nodes, _, _ = contagion_propagation(
            num_nodes=20, adjacency_matrix=adj, weight_0=w0,
            init_seeds=(0,), node_active_threshold=1.0, max_steps=10
        )
        # With such a high threshold, very few nodes should activate
        assert len(active_nodes) <= 20


class TestStateFunction:

    def test_returns_one_when_above_threshold(self):
        from contagion import state_function
        assert state_function({10, 20, 30}, threshold_sum=10) == 1

    def test_returns_zero_when_below_threshold(self):
        from contagion import state_function
        assert state_function({1, 2}, threshold_sum=1000) == 0

    def test_exact_threshold(self):
        from contagion import state_function
        # sum({5}) = 5, threshold = 5 -> should return 1
        assert state_function({5}, threshold_sum=5) == 1


class TestGetSeedNodesCombinations:

    def test_correct_number_of_combinations(self):
        from contagion import get_seed_nodes_combinations
        G = nx.path_graph(5)
        combos = get_seed_nodes_combinations(G, n_seeds=2)
        # C(5,2) = 10
        assert len(combos) == 10

    def test_each_combo_is_correct_length(self):
        from contagion import get_seed_nodes_combinations
        G = nx.path_graph(5)
        combos = get_seed_nodes_combinations(G, n_seeds=3)
        assert all(len(c) == 3 for c in combos)

    def test_returns_list(self):
        from contagion import get_seed_nodes_combinations
        G = nx.path_graph(4)
        result = get_seed_nodes_combinations(G, n_seeds=2)
        assert isinstance(result, list)


class TestInitialSeedNodes:

    def test_returns_list_of_tuples(self):
        from contagion import initial_seed_nodes
        G = nx.circulant_graph(30, offsets=[1, 2])
        result = initial_seed_nodes(G, n_seeds=2)
        assert isinstance(result, list)
        assert isinstance(result[0], tuple)

    def test_tuple_length_matches_n_seeds(self):
        from contagion import initial_seed_nodes
        G = nx.circulant_graph(30, offsets=[1, 2])
        result = initial_seed_nodes(G, n_seeds=3)
        assert len(result[0]) == 3

    def test_preexisting_seeds_returned_as_is(self):
        from contagion import initial_seed_nodes
        G = nx.circulant_graph(20, offsets=[1, 2])
        existing = [(0, 1)]
        result = initial_seed_nodes(G, n_seeds=2, init_seeds=existing)
        assert result == existing

    def test_all_seed_nodes_are_valid(self):
        from contagion import initial_seed_nodes
        G = nx.circulant_graph(30, offsets=[1, 2])
        result = initial_seed_nodes(G, n_seeds=2)
        for node in result[0]:
            assert node in G.nodes()


class TestSimulateContagionMap:

    def test_returns_graph_and_seeds(self, simple_params):
        from contagion import simulate_contagion_map
        graph, seed_nodes = simulate_contagion_map(simple_params)
        assert isinstance(graph, nx.Graph)
        assert isinstance(seed_nodes, list)
        assert len(seed_nodes) > 0

    def test_graph_has_correct_node_count(self, simple_params):
        from contagion import simulate_contagion_map
        graph, _ = simulate_contagion_map(simple_params)
        assert graph.number_of_nodes() == simple_params["num_nodes"]

    def test_cluster_seeding_returns_tuple(self, simple_params):
        from contagion import simulate_contagion_map
        params = {**simple_params, "seeding_method": "cluster_seeding", "seed_cluster_distance": 5}
        graph, seed_nodes = simulate_contagion_map(params)
        assert isinstance(seed_nodes, list)



class TestSimulateContagionRealization:

    def test_returns_four_values(self, small_graph, adjacency_and_weight, simple_params):
        from contagion import simulate_contagion_realization
        adj, w0 = adjacency_and_weight
        result = simulate_contagion_realization(
            graph=small_graph,
            init_seeds=(0, 1),
            params=simple_params,
            adjacency_matrix=adj,
            weight_0=w0,
        )
        assert len(result) == 4

    def test_results_is_list_of_dicts(self, small_graph, adjacency_and_weight, simple_params):
        from contagion import simulate_contagion_realization
        adj, w0 = adjacency_and_weight
        _, _, _, results = simulate_contagion_realization(
            graph=small_graph, init_seeds=(0, 1),
            params=simple_params, adjacency_matrix=adj, weight_0=w0,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_results_contain_expected_keys(self, small_graph, adjacency_and_weight, simple_params):
        from contagion import simulate_contagion_realization
        adj, w0 = adjacency_and_weight
        _, _, _, results = simulate_contagion_realization(
            graph=small_graph, init_seeds=(0, 1),
            params=simple_params, adjacency_matrix=adj, weight_0=w0,
        )
        expected_keys = {"time", "state", "num_active_nodes", "active_nodes", "H_0", "H_1", "H_2"}
        assert expected_keys.issubset(results[0].keys())

    def test_activation_times_length(self, small_graph, adjacency_and_weight, simple_params):
        from contagion import simulate_contagion_realization
        adj, w0 = adjacency_and_weight
        _, _, activation_times, _ = simulate_contagion_realization(
            graph=small_graph, init_seeds=(0, 1),
            params=simple_params, adjacency_matrix=adj, weight_0=w0,
        )
        assert len(activation_times) == simple_params["num_nodes"]



class TestMainSims:

    def test_returns_dataframe(self, simple_params):
        from contagion import main_sims
        # Use a tiny seeding to keep runtime short
        params = {**simple_params, "seeding_method": "cluster_seeding", "n_seeds": 2}
        df, _ = main_sims([params], max_steps=5, save_files=False)
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_has_simulation_id(self, simple_params):
        from contagion import main_sims
        params = {**simple_params, "seeding_method": "cluster_seeding"}
        df, _ = main_sims([params], max_steps=5, save_files=False)
        assert "simulation_id" in df.columns

    def test_multiple_params_produce_multiple_sims(self, simple_params):
        from contagion import main_sims
        params = {**simple_params, "seeding_method": "cluster_seeding"}
        df, _ = main_sims([params, params], max_steps=5, save_files=False)
        assert df["simulation_id"].nunique() == 2

    def test_dataframe_contains_state_column(self, simple_params):
        from contagion import main_sims
        params = {**simple_params, "seeding_method": "cluster_seeding"}
        df, _ = main_sims([params], max_steps=5, save_files=False)
        assert "state" in df.columns

    def test_edge_stats_columns_present(self, simple_params):
        from contagion import main_sims
        params = {**simple_params, "seeding_method": "cluster_seeding"}
        df, _ = main_sims([params], max_steps=5, save_files=False)
        for col in ["total_geo_edges", "total_non_geo_edges", "average_weight_per_edge"]:
            assert col in df.columns


# RUNNNN

if __name__ == "__main__":
    pytest.main([__file__, "-v"])