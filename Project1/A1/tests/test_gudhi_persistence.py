import pytest
import numpy as np
import networkx as nx
from collections import defaultdict
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────
# Helpers / Fixtures

def make_simple_graph():
    """A small weighted graph with 4 nodes and geometric edges."""
    G = nx.Graph()
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_edge(0, 1, weight=1, type="geometric")
    G.add_edge(1, 2, weight=2, type="geometric")
    G.add_edge(2, 3, weight=3, type="geometric")
    G.add_edge(0, 3, weight=1, type="geometric")
    return G


def make_graph_with_nongeometric():
    """Graph containing a mix of geometric and non-geometric edges."""
    G = make_simple_graph()
    G.add_edge(1, 3, weight=2, type="non_geometric")
    return G


def make_activation_times():
    """Simple activation times: node i activates at time i."""
    return np.array([0, 1, 2, 3])


# Main compute Persistnece

class TestComputePersistence:

    def test_returns_three_values(self):
        from persistence import compute_persistence
        G = make_simple_graph()
        at = make_activation_times()
        result = compute_persistence(G, at)
        assert len(result) == 3, "Should return (betti_over_time, persistence, persistence_for_graphics)"

    def test_persistence_array_shape(self):
        from persistence import compute_persistence
        G = make_simple_graph()
        at = make_activation_times()
        _, persistence, _ = compute_persistence(G, at, max_dim=2)
        expected_timesteps = int(np.nanmax(at)) + 1
        assert persistence.shape == (expected_timesteps, 3)

    def test_non_geometric_edges_dropped(self):
        """When ngeom_edges_in_persistence=False, non-geometric edges should be removed."""
        from persistence import compute_persistence
        G = make_graph_with_nongeometric()
        at = make_activation_times()
        # Should not raise; non-geometric edge silently dropped
        compute_persistence(G, at, ngeom_edges_in_persistence=False)
        # The original graph still has the edge (function works on a copy or removes in place)
        # Just check it completes without error

    def test_nongeometric_edges_kept_when_flag_true(self):
        from persistence import compute_persistence
        G = make_graph_with_nongeometric()
        at = make_activation_times()
        # Should not raise and should include non-geometric edges
        compute_persistence(G, at, ngeom_edges_in_persistence=True)

    def test_single_node_graph(self):
        from persistence import compute_persistence
        G = nx.Graph()
        G.add_node(0)
        at = np.array([0])
        _, persistence, _ = compute_persistence(G, at, max_dim=2)
        assert persistence.shape[0] == 1

    def test_persistence_for_graphics_is_list(self):
        from persistence import compute_persistence
        G = make_simple_graph()
        at = make_activation_times()
        _, _, pfg = compute_persistence(G, at)
        assert isinstance(pfg, list)



class TestBettiNumsOverTime:

    def test_runs_without_error(self):
        from persistence import betti_nums_over_time
        betti = {
            0: {0: 1, 1: 0, 2: 0},
            1: {0: 1, 1: 1, 2: 0},
            2: {0: 1, 1: 1, 2: 1},
        }
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        betti_nums_over_time(betti)  # should not raise

    def test_handles_single_timestep(self):
        from persistence import betti_nums_over_time
        import matplotlib
        matplotlib.use("Agg")
        betti = {0: {0: 2, 1: 0}}
        betti_nums_over_time(betti)



class TestPersistenceRepresentation:

    def _make_mock_persistence(self, timesteps=3, max_dim=2):
        """Create a mock persistence array with dummy interval lists."""
        persistence = np.empty((timesteps, max_dim + 1), dtype=object)
        for t in range(timesteps):
            for d in range(max_dim + 1):
                # Add some fake (birth, death) intervals
                persistence[t, d] = [(0.0, 1.0), (0.2, 0.8), (0.1, 0.5)]
        return persistence

    def test_returns_three_outputs(self):
        from persistence import persistence_representation
        p = self._make_mock_persistence()
        L, I, params = persistence_representation(p)
        assert isinstance(L, defaultdict)
        assert isinstance(I, defaultdict)
        assert isinstance(params, dict)

    def test_params_contains_expected_keys(self):
        from persistence import persistence_representation
        p = self._make_mock_persistence()
        _, _, params = persistence_representation(p)
        for key in ["num_landscapes", "bandwidth", "resolution"]:
            assert key in params

    def test_custom_bandwidth_resolution(self):
        from persistence import persistence_representation
        p = self._make_mock_persistence()
        _, _, params = persistence_representation(p, bandwidth=0.5, resolution=20)
        assert params["bandwidth"] == 0.5
        assert params["resolution"] == 20



class TestPersistenceRepresentationT:

    def _make_mock_persistence(self, timesteps=3, max_dim=2):
        persistence = np.empty((timesteps, max_dim + 1), dtype=object)
        for t in range(timesteps):
            for d in range(max_dim + 1):
                persistence[t, d] = [(0.0, 1.0), (0.1, 0.9)]
        return persistence

    def test_returns_four_outputs(self):
        from persistence import persistence_representation_t
        p = self._make_mock_persistence()
        result = persistence_representation_t(p)
        assert len(result) == 4, "Should return L, I, essential_features, params"

    def test_L_and_I_length_match_timesteps(self):
        from persistence import persistence_representation_t
        timesteps = 4
        p = self._make_mock_persistence(timesteps=timesteps)
        L, I, _, _ = persistence_representation_t(p)
        assert len(L) == timesteps
        assert len(I) == timesteps

    def test_essential_features_length_matches_timesteps(self):
        from persistence import persistence_representation_t
        timesteps = 3
        p = self._make_mock_persistence(timesteps=timesteps)
        _, _, essential, _ = persistence_representation_t(p)
        assert len(essential) == timesteps

    def test_empty_persistence_dim_handled(self):
        """Timesteps with empty interval lists should not crash."""
        from persistence import persistence_representation_t
        persistence = np.empty((2, 3), dtype=object)
        for t in range(2):
            for d in range(3):
                persistence[t, d] = []  # empty
        L, I, _, _ = persistence_representation_t(persistence)
        assert len(L) == 2


#
# Tests: graph edge filtering logic (unit-level)

class TestEdgeFiltering:

    def test_nongeometric_edges_identified(self):
        G = make_graph_with_nongeometric()
        non_geom = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("type") == "non_geometric"]
        assert len(non_geom) == 1

    def test_geometric_edges_count(self):
        G = make_graph_with_nongeometric()
        geom = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("type") == "geometric"]
        assert len(geom) == 4

    def test_activation_times_shape(self):
        at = make_activation_times()
        assert at.shape == (4,)
        assert np.nanmax(at) == 3

#Run
if __name__ == "__main__":
    pytest.main([__file__, "-v"])