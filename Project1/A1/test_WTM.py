import collections
import itertools
import os
import pickle
import random

import networkx as nx
import numpy as np
import pandas as pd

from utilsA1 import snapshots_to_activation_times_series
import gudhi_persistence as gp

PATH = os.path.dirname(__file__)


def generate_graph(
    num_nodes: int,
    num_neighbor_nodes: int,
    total_random_edges: int,
    distance_threshold: int,
    upper_weight_limit: int = 10,
    skew_power: int = 3,
    weighted=False,
    ngeo_placement="random.choice",
) -> nx.Graph:
    """

    Generate Ring Graph with geometric and non_geometric edges
    :param num_nodes: Total number of nodes
    :param num_neighbor_nodes: Number of neighboring nodes that are connected to the node on one side
    :param total_random_edges: Number of total non-geometric edges
    :param distance_threshold: (Number of nodes distance away) to qualify as non-geometric edge
    :param upper_weight_limit: maximum weight allowed for non-geometric edges
    :param skew_power: How much weights are skewed towards zero
    :param weighted: if Weighted graph or Not
    :param ngeo_placement: Probability distribution for weights
    :return: Networkx Graph
    """
    if not distance_threshold:
        distance_threshold = num_neighbor_nodes + 1

    graph = nx.Graph()

    #### Numpy solution exists
    #### Just create some arrays -> concat -> import weights if needed -> reshape -> nx.add_edges_from(np.array)
    ####

    # Add nodes to graph
    for node in range(num_nodes):
        graph.add_node(node)

    # Add Geometric edges
    if weighted:
        for node in range(num_nodes):
            for i in range(num_neighbor_nodes):
                graph.add_edge(
                    node,
                    (node + i + 1) % num_nodes,
                    type="geometric",
                    weight=add_skewed_weights(1, upper_weight_limit, skew_power)[0],
                )
    else:
        for node in range(num_nodes):
            for i in range(num_neighbor_nodes):
                graph.add_edge(
                    node, (node + i + 1) % num_nodes, type="geometric", weight=0
                )

    graph = add_non_geometric_edges(
        graph=graph,
        total_random_edges=total_random_edges,
        distance_threshold=distance_threshold,
        upper_weight_limit=upper_weight_limit,
        skew_power=skew_power,
        ngeo_placement=ngeo_placement,
        weighted=weighted,
    )
    return graph


def add_non_geometric_edges(
    graph: nx.Graph,
    total_random_edges: int,
    distance_threshold: int,
    upper_weight_limit: int = 10,
    skew_power: int = 3,
    ngeo_placement="random.choice",
    weighted: bool = False,
) -> nx.Graph:
    """
    Create non-geometric edges accoring to a given distance
    (Not a L2/L1 measure, Based on number of neighbors in Ring) threshold
    :param graph: networkx graph
    :param total_random_edges: total number of non-geometric edges needed. Default = Num_nodes/2
    :param distance_threshold: # of neighbouring to skip as an early distance measure
    :param upper_weight_limit: maximum weight allowed for non-geometric edges
    :param skew_power: How much weights are skewed towards zero
    :param ngeo_placement: choice of how to create non-geometric edges. "random.choice": randomly choosing from a pool of all possible edges
    :param weighted: whether edges should be weighted or not. Default = False
    :return: networkx graph
    """
    if total_random_edges == 0:
        return graph

    num_geometric_edges = graph.number_of_edges()
    if total_random_edges:
        num_random_edges = total_random_edges
    else:
        num_random_edges = num_geometric_edges / 2

    def ring_distance(u, v, n):
        return min(abs(u - v), n - abs(u - v))

    if ngeo_placement == "random.choice":
        src, tgt = np.meshgrid(graph.nodes(), graph.nodes())
        src = src.flatten()
        tgt = tgt.flatten()

        directed_mask = (
            src < tgt
        )  # we're only using the upper (or lower) triangle to void [2, 4] [4, 2] typa situations
        src = src[directed_mask]
        tgt = tgt[directed_mask]

        ring_dist = np.minimum(
            np.abs(src - tgt), graph.number_of_nodes() - np.abs(src - tgt)
        )
        distance_mask = ring_dist >= distance_threshold
        src = src[distance_mask]
        tgt = tgt[distance_mask]
        valid_edge_list = np.stack((src, tgt), axis=1)

        random_edge_indexes = np.random.choice(
            a=len(valid_edge_list), size=num_random_edges, replace=False
        )
        non_geometric_edges = valid_edge_list[random_edge_indexes]

        if weighted:
            weights = add_skewed_weights(
                n=len(non_geometric_edges),
                upper_weight_limit=upper_weight_limit,
                skew_power=skew_power,
            )
            weighted_edges = [
                (u, v, w)
                for (u, v), w in zip(valid_edge_list[random_edge_indexes], weights)
            ]
            graph.add_weighted_edges_from(weighted_edges, type="non-geometric")
            return graph
        else:
            graph.add_edges_from(non_geometric_edges, type="non-geometric", weight=0)
        return graph

    elif ngeo_placement == "ngeo_per_node":
        ngeo_per_node = num_random_edges
        num_nodes = graph.number_of_nodes()

        weights = np.zeros(ngeo_per_node * num_nodes // 2)

        if weighted:
            weights = add_skewed_weights(
                n=(ngeo_per_node * num_nodes) // 2,
                upper_weight_limit=upper_weight_limit,
                skew_power=skew_power,
            )
            np.random.shuffle(weights)

        candidate_pairs = [
            (u, v)
            for u in graph.nodes()
            for v in graph.nodes()
            if ring_distance(u, v, num_nodes) >= distance_threshold
            and u < v
            and not graph.has_edge(u, v)
        ]
        random.shuffle(candidate_pairs)

        non_geo_edges = list()
        ngeo_edge_counts = collections.defaultdict(int)

        for u, v in candidate_pairs:
            if (
                ngeo_edge_counts[u] < ngeo_per_node
                and ngeo_edge_counts[v] < ngeo_per_node
            ):
                non_geo_edges.append((u, v))
                ngeo_edge_counts[u] += 1
                ngeo_edge_counts[v] += 1

            if len(non_geo_edges) == (ngeo_per_node * num_nodes // 2):
                break

        requirement_ngeo = [
            node for node in graph.nodes() if ngeo_edge_counts[node] < ngeo_per_node
        ]
        if requirement_ngeo:
            print(non_geo_edges)
            print(
                f"Warning: {len(requirement_ngeo)} nodes could not reach Number of non-geo={ngeo_per_node} edges due to constraints"
            )

        weighted_edges = [(u, v, w) for (u, v), w in zip(non_geo_edges, weights)]
        graph.add_weighted_edges_from(weighted_edges, type="non-geometric")
        return graph


def contagion_propagation(
    graph: nx.Graph,
    init_seeds: tuple,
    node_active_threshold: float,
    max_steps: int = 100,
    weighted: bool = False,
):
    if max_steps is None:
        max_steps = graph.number_of_edges()

    activation_times = np.full(graph.number_of_nodes(), np.nan)
    activation_times[list(init_seeds)] = (
        0  # set activation time=0 for initializing seeds
    )
    active_nodes = set(init_seeds)
    propagation_snapshots = [set(init_seeds).copy()]

    nodes = graph.nodes()
    if weighted:
        for time_step in range(1, max_steps + 1):
            step_propagation = set()
            for node_i in nodes:
                if node_i not in active_nodes:
                    neighbors = list(graph.neighbors(node_i))
                    if not neighbors:
                        continue

                    # arr = [(u, v, graph[u][v].get('weight', np.nan)) for u, v in zip([node_i]*len(neighbors), neighbors)]
                    # test_arr = np.array(arr)
                    # test_arr[test_arr[:, 2] < time_step].shape[0] <- number of neighbors with qualifying weights
                    # seems to be a bit too expensive to check here.
                    # possibly you could implement this in the outer FOR loops

                    weight_qualifying_nodes = 0
                    for ith_neighbor in neighbors:
                        if ith_neighbor in active_nodes:
                            weight = graph.get_edge_data(node_i, ith_neighbor).get(
                                "weight", 0
                            )
                            if weight <= time_step:
                                weight_qualifying_nodes += 1

                    if (
                        weight_qualifying_nodes / len(neighbors)
                        >= node_active_threshold
                    ):
                        step_propagation.add(node_i)

            if not step_propagation:
                break

            active_nodes.update(step_propagation)
            activation_times[list(step_propagation)] = time_step
            propagation_snapshots.append(active_nodes.copy())

        return active_nodes, activation_times, propagation_snapshots

    else:
        for time_step in range(1, max_steps + 1):
            step_propagation = set()
            for node_i in nodes:
                if node_i not in active_nodes:
                    neighbors = list(graph.neighbors(node_i))
                    if not neighbors:
                        continue
                    active_neighbors = [
                        node for node in neighbors if node in active_nodes
                    ]
                    if len(active_neighbors) / len(neighbors) >= node_active_threshold:
                        step_propagation.add(node_i)

            if not step_propagation:
                break

            active_nodes.update(step_propagation)
            activation_times[list(step_propagation)] = time_step
            propagation_snapshots.append(active_nodes.copy())

        return active_nodes, activation_times, propagation_snapshots


def state_function(active_nodes, threshold_sum):
    return int(sum(set(active_nodes)) >= threshold_sum)


def get_seed_nodes_combinations(graph: nx.graph, n_seeds: int = 2):
    return list(itertools.combinations(graph.nodes, n_seeds))


def initial_seed_nodes(
    graph: nx.graph, n_seeds: int = 2, seed_cluster_distance: int = 10, init_seeds=()
) -> tuple:
    if init_seeds is not None:
        if len(init_seeds) > 0:
            return init_seeds

    center_node = np.random.choice(list(graph.nodes()))
    neighbors = {center_node}
    for _ in range(seed_cluster_distance):
        next_neighbors = set()
        for node in neighbors:
            next_neighbors.update(graph.neighbors(node))
        neighbors.update(next_neighbors)

    candidates = list(neighbors)
    if len(candidates) >= n_seeds:
        return tuple(np.random.choice(candidates, n_seeds))
    else:
        return tuple(np.random.choice(list(graph.nodes()), n_seeds))


def add_skewed_weights(
    n: int = 1, upper_weight_limit: int = 10, skew_power: int = 3
) -> int:
    return np.round(
        (np.random.rand(n) ** skew_power) * (upper_weight_limit + 1)
    ).astype(int)


def simulate_contagion_map(params: dict):
    graph = generate_graph(
        params.get("num_nodes", 100),
        params.get("num_neighbor_nodes", 3),
        params.get("total_random_edges", 50),
        params.get("distance_threshold", 4),
        params.get("upper_weight_limit", 10),
        params.get("skew_power", 3),
        params.get("weighted", True),
        params.get("ngeo_placement", "random.choice"),
    )

    seeding_method = params.get("seeding_method", "all_combinations")
    n_seeds = params.get("n_seeds", 2)

    seed_nodes = (0, 1)
    if seeding_method == "cluster_seeding":
        seed_nodes = initial_seed_nodes(
            graph=graph,
            n_seeds=n_seeds,
            seed_cluster_distance=params.get("seed_cluster_distance", 10),
        )
    elif seeding_method == "all_combinations":
        seed_nodes = get_seed_nodes_combinations(graph=graph, n_seeds=n_seeds)

    return graph, seed_nodes


def simulate_contagion_realization(
    graph: nx.graph,
    init_seeds: tuple,
    params: dict,
    max_steps: int = 100,
    sim_id: int = 1,
    realization_id: int = 1,
    calculate_representation: bool = False,
):

    weights = [d.get("weight", 0) for _, _, d in graph.edges(data=True)]
    average_weight = np.mean(weights) if weights else 0
    total_geo_edges = sum(
        [1 for _, _, d in graph.edges(data=True) if d.get("type") == "geometric"]
    )
    total_non_geo_edges = sum(
        [1 for _, _, d in graph.edges(data=True) if d.get("type") == "non-geometric"]
    )

    if params.get("seeding_method") == "cluster_seeding":
        assert len(init_seeds) == params.get("n_seeds", 2)

    # Run contagion propagation
    active_nodes_at_end, activation_times, snapshots = contagion_propagation(
        graph=graph,
        init_seeds=init_seeds,
        node_active_threshold=params.get("node_active_threshold", 0.001),
        max_steps=max_steps,
        weighted=params.get("weighted", False),
    )

    max_persistence_dim = int(params.get("max_persistence_dim", 2))
    ngeom_edges_in_persistence = params.get("geom_edges_in_persistence", False)

    # Compute Persistence Homology
    betti_numbers, persistence, _ = gp.compute_persistence(
        graph=graph,
        activation_times=activation_times,
        max_dim=max_persistence_dim,
        ngeom_edges_in_persistence=ngeom_edges_in_persistence,
    )

    activation_series = snapshots_to_activation_times_series(
        snapshots, params.get("num_nodes")
    )

    results = []
    for t, active_nodes_time_t in enumerate(activation_series):
        activated_nodes = np.where(~np.isnan(active_nodes_time_t))[0]
        state = state_function(activated_nodes, params.get("threshold_sum", 19900))

        features_dict = {
            "simulation_id": sim_id,
            "realization_id": realization_id,
            "num_nodes": params.get("num_nodes", 100),
            "time": t,
            "state": state,
            "state_abnormal_sum": params.get("threshold_sum", 0),
            "num_active_nodes": len(activated_nodes),
            "active_nodes": active_nodes_time_t,
            "node_active_threshold": params.get("node_active_threshold", 0.1),
            "H0": 0,  #  betti_numbers[t][0],
            "H1": 1,  # betti_numbers[t][1],
            "H2": 2,  # betti_numbers[t][2],
            "total_geo_edges": total_geo_edges,
            "total_non_geo_edges": total_non_geo_edges,
            "num_seeds": params.get("n_seeds", 1),
            "seed_nodes": init_seeds,
            "seed_cluster_distance": params.get("seed_cluster_distance", 10),
            "weighted": params.get("weighted"),
            "average_weight_per_edge": average_weight,
            "skew_power": params.get("skew_power"),
            "upper_weight_limit": params.get("upper_weight_limit"),
            "distance_threshold": params.get("distance_threshold"),
        }
        results.append(features_dict)

    if calculate_representation:
        bandwidth = params.get("bandwidth", 0.1)
        num_landscapes = params.get("num_landscapes", 3)
        resolution = params.get("resolution", 50)
        L, I, E, representaion_params = gp.persistence_representation_t(
            persistence=persistence,
            bandwidth=bandwidth,
            resolution=resolution,
            num_landscapes=num_landscapes,
        )
        rename = lambda d, prefix: {f"{prefix}_{k}": v for k, v in d.items()}

        L_timesteps = len(L)
        assert L_timesteps == len(results)
        for res_dict, timestep in zip(results, range(L_timesteps)):
            res_dict.update(rename(L[timestep], "L"))
            res_dict.update(rename(I[timestep], "I"))
            res_dict.update(rename(E[timestep], "E"))

    return graph, snapshots, activation_times, results


def main_sims(
    params_list: list,
    max_steps=100,
    output_file="simulation_results.csv",
    save_files=False,
):
    """
    :param params_list: list of dicts with parameters for each run,
                 e.g. [{'num_nodes': 100, 'param2': val2, ...}, ...]
    :param max_steps: max steps to run contagion
    :param output_file: filename to save the results
    :param save_files: save results to a csv, pickle file
    :return DataFrame with simulation results and saves to CSV
    """
    outfile_path = os.path.join(PATH, "outputs")
    simulation_results = []
    activation_times_results = []  # for brad
    for i, params in enumerate(params_list):
        # print(f"Running simulation {i + 1} with params: {params}")
        graph, seed_node_combinations = simulate_contagion_map(params=params)
        activation_times_results.append(
            {
                "sim_id": i,
                "graph": graph,
                "realization_id": 0,
            }
        )
        for j, seed_nodes in enumerate(seed_node_combinations):
            G, _, activation_times, results = simulate_contagion_realization(
                graph=graph,
                init_seeds=seed_nodes,
                params=params,
                sim_id=i,
                max_steps=max_steps,
                realization_id=j,
                calculate_representation=params.get("calculate_representation", False),
            )
            activation_times_results.append(
                {"sim_id": i, "realization_id": j, "activation_times": activation_times}
            )
            simulation_results.extend(results)

    df = pd.DataFrame(simulation_results)

    if save_files:
        df = pd.DataFrame(simulation_results)
        df_file_path = os.path.normpath(
            os.path.join(outfile_path, f"{output_file}.csv")
        )
        # pickle_file_path = os.path.normpath(os.path.join(outfile_path, f"{output_file}.pkl"))

        df.to_csv(df_file_path, index=False)
        print(f"Simulation results saved to {df_file_path}")

        # with open(pickle_file_path, "wb") as f:
        #     pickle.dump(activation_times_results, f)

        return df_file_path, 0  # , pickle_file_path
    else:
        return df, activation_times_results
