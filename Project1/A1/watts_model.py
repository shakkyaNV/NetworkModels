import networkx as nx, numpy as np, gudhi_persistence as gp, pandas as pd, cvxpy as cx
from collections import defaultdict
from itertools import combinations
from Project1.A1.utilsA1 import InvalidGraphError, get_representation_choice_function
import os, sys, pickle
##### DEFINE VARIABLES
# ------

PATH = os.path.dirname(__file__)


def generate_graph( num_nodes: int, num_neighbor_nodes: int, total_random_edges: int, distance_threshold: int = 10,
                    upper_weight_limit: int = 10, skew_power: int = 3, weighted=False,
                    ngeo_placement="random.choice") -> nx.Graph:
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
    graph = nx.circulant_graph(n = num_nodes,
                               offsets=[1, num_neighbor_nodes],
                               create_using=nx.Graph)
    n_edges = graph.number_of_edges()
    if weighted:
        weights = add_skewed_weights(n_edges, upper_weight_limit=upper_weight_limit, skew_power=skew_power)
    else:
        weights = np.zeros(n_edges)

    weighted_edges = [
        (u, v, w)
        for (u, v), w in zip(graph.edges(data = False), weights)
    ]
    graph.add_weighted_edges_from(weighted_edges, type = 'geometric')

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


def add_non_geometric_edges( graph: nx.Graph, total_random_edges: int, distance_threshold: int,
                             upper_weight_limit: int = 10, skew_power: int = 3, ngeo_placement="random.choice",
                             weighted: bool = False) -> nx.Graph:
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
        n = graph.number_of_nodes()
        src = np.arange(n).reshape(-1, 1)
        tgt = np.arange(n).reshape(1, -1)
        abs_dist = np.abs(src - tgt)
        ring_distance = np.minimum(abs_dist, n - abs_dist)
        threshold_distance_mask = np.triu(ring_distance >= distance_threshold)

        adjacency_matrix = np.triu(nx.to_numpy_array(graph, weight = None, dtype = int))
        possible_edges = np.argwhere( np.logical_not(adjacency_matrix) * threshold_distance_mask)
        chosen_edges = np.random.choice(len(possible_edges), size=total_random_edges, replace=False)
        weights = add_skewed_weights(total_random_edges, upper_weight_limit, skew_power) * weighted
        weighted_edges = [(int(u), int(v), w) for (u, v), w in zip(possible_edges[chosen_edges], weights)]
        graph.add_weighted_edges_from(weighted_edges, type = 'non_geometric')

    elif ngeo_placement == "ngeo_per_node":
        ngeo_per_node = num_random_edges
        non_geo_edges = np.empty(0)
        n = graph.number_of_nodes()
        required_length_non_geo_edges = (ngeo_per_node * n) // 2
        src = np.arange(n).reshape(-1, 1)
        tgt = np.arange(n).reshape(1, -1)
        abs_dist = np.abs(src - tgt)
        ring_distance = np.minimum(abs_dist, n - abs_dist)
        threshold_distance_mask = np.triu(ring_distance >= distance_threshold)

        adjacency_matrix = np.triu(nx.to_numpy_array(graph, weight = None, dtype = int))
        possible_edges = np.logical_not(adjacency_matrix) * threshold_distance_mask # getting the logical matrix instead of the indices
        # Here we're checking row/column sums of possible edge lists, which is necessary but not a sufficient condition to guaranteeing a solution
        # exists, given the constraints.
        # So, we're trying a heuristic version which can fail at times. It's allowed to run 2 times, before trying a convex optimized integer solver
        # which returns an adjacency matrix if a solution exists. Almost guaranteed solver
        # The solver needs: pip install cvlp (Linear program solver for pyton, a wrapper for cx.CBC: Coin-or-branch cut solver for cvxpy
        condition = np.all((np.sum(possible_edges, axis = 1) + np.sum(possible_edges, axis = 0)) >= ngeo_per_node)
        attempt = 3
        if condition:
            if attempt >= 2:
                print("Entering Integer Solver - Convex Optimization")
                x = cx.Variable((n, n), boolean = True)
                constraints = []
                constraints += [x == x.T] # enforce symmetricity for undirected graphs
                constraints += [cx.diag(x) == 0] # enforce no self loops
                constraints += [x[i, j]==0 for i in range(n) for j in range(n) if possible_edges[i, j] == 0] # enforce to not put any edges in impossible places
                constraints += [cx.sum(x[i, :]) == ngeo_per_node for i in range(n)] # At the end, each row should have ngeo_per_node
                objective = cx.Maximize(0) # feasibility only

                problem = cx.Problem(objective, constraints)
                result = problem.solve(solver = cx.CBC) # using cylp solver
                if problem.status in ["optimal", "optimal_inaccurate"]:
                    res_adjacency = np.round(x.value).astype(int)
                    non_geo_edges_temp = np.argwhere(np.triu(res_adjacency))
                    chosen_edges = np.random.choice(a=len(non_geo_edges_temp), size=required_length_non_geo_edges, replace=False)
                    non_geo_edges = non_geo_edges_temp[chosen_edges]
                else:
                    non_geo_edges = np.empty(0)
                    raise SystemExit(f"No feasible {ngeo_per_node}-regular graph exists under the constraints.")
            candidate_edges = np.argwhere(possible_edges)
            np.random.shuffle(candidate_edges)
            non_geo_edges = list()
            ngeo_edge_counts = defaultdict(int)

            for [u, v] in candidate_edges:
                if ngeo_edge_counts[u] < ngeo_per_node and ngeo_edge_counts[v] < ngeo_per_node:
                    non_geo_edges.append((u, v))
                    ngeo_edge_counts[u] += 1
                    ngeo_edge_counts[v] += 1

                if len(non_geo_edges) == required_length_non_geo_edges:
                    break
                else:
                    TEMP = 1
            requirement_ngeo = [
                node for node in graph.nodes() if ngeo_edge_counts[node] < ngeo_per_node
            ]
            if requirement_ngeo:
                print(non_geo_edges)
                print(
                    f"Warning: {len(requirement_ngeo)} nodes could not reach Number of non-geo={ngeo_per_node} edges due to constraints"
                )
        else:
            raise SystemExit(f"Necessary Condition for Non-Geometric Edges per Node not met. Please reduce `ngeo_per_node`")
                  # f"{min(np.sum(possible_edges, axis = 1))}")

        assert len(non_geo_edges) == required_length_non_geo_edges, "Number of edges returned does not match requirement"
        weights = add_skewed_weights(required_length_non_geo_edges, upper_weight_limit, skew_power) * weighted
        weighted_edges = [(int(u), int(v), w) for (u, v), w in zip(non_geo_edges, weights)]
        graph.add_weighted_edges_from(weighted_edges, type="non_geometric")

    elif ngeo_placement == "ngeo_per_node1":
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

        attempt = 0
        num_attempts_allowed = 20
        while attempt < num_attempts_allowed:
            np.random.shuffle(candidate_pairs)
            non_geo_edges = list()
            ngeo_edge_counts = defaultdict(int)

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
                print(
                    f"Warning: {len(requirement_ngeo)} nodes could not reach Number of non-geo={ngeo_per_node} edges due to constraints"
                )
                attempt += 1
            else:
                weighted_edges = [(u, v, w) for (u, v), w in zip(non_geo_edges, weights)]
                graph.add_weighted_edges_from(weighted_edges, type="non_geometric")
                break
        if attempt >= num_attempts_allowed:
            print("In number of attempts:", attempt)
            raise InvalidGraphError("Heuristic solution to find `ngeo-per-node` failed. Exiting")

    return graph


def contagion_propagation( num_nodes: int, adjacency_matrix: np.ndarray, weight_0: np.ndarray,
                           init_seeds: tuple, node_active_threshold: float,
                           max_steps: int = 100):
    """
    Fast Simulator for a single contagion propagation given initial seeds
    Similar to a Social Enforcement algorithm in contagion adoption.
    Weight <= Timestep : restriction for propagation along the edge
    :param graph: Networkx Graph (Ring)
    :param init_seeds: Tuple of seeds
    :param node_active_threshold: Float value for social contagion adoption
    :param max_steps: Maximum number of steps to run. Default: number of nodes in graph()
    :return:
    """

    timestep = 0
    active_nodes = set()
    arr = np.zeros(num_nodes, dtype=int)
    activation_times = np.full(num_nodes, -1)
    activation_times[list(init_seeds)] = timestep

    degree_matrix = np.sum(adjacency_matrix, axis=0)

    active_nodes_t = set(map(int, init_seeds))
    active_nodes.update(active_nodes_t)
    snapshots = [active_nodes.copy()]

    while timestep <= max_steps:
        timestep += 1
        arr[list(active_nodes)] = 1
        num_active_neighbors = np.matmul(adjacency_matrix, arr)
        qualifying_nodes = (num_active_neighbors / degree_matrix) >= node_active_threshold
        timestep_weight_mask = (weight_0 <= timestep)
        qualified_to_accept = np.logical_not(arr) * qualifying_nodes
        newly_activated_t = np.matmul(timestep_weight_mask, arr) * qualified_to_accept

        active_nodes_t = set(map(int, np.where(newly_activated_t > 0)[0]))

        if len(active_nodes_t) == 0:
            break
        else:
            activation_times[list(active_nodes_t)] = timestep
            active_nodes.update(active_nodes_t)
            snapshots.append(active_nodes.copy())
    return active_nodes, activation_times, snapshots


def state_function(active_nodes, threshold_sum):
    return int(sum(set(active_nodes)) >= threshold_sum)


def get_seed_nodes_combinations(graph: nx.Graph, n_seeds: int = 2) -> list:
    return list(combinations(graph.nodes, n_seeds))


def initial_seed_nodes(graph: nx.Graph, n_seeds: int = 2, seed_cluster_distance: int = 10, init_seeds=None) -> list:
    if init_seeds is None:
        init_seeds = []

    if init_seeds is not None:
        if len(init_seeds) > 0:
            if isinstance(init_seeds, list):
                if isinstance(init_seeds[0], tuple):
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
        init_seeds.append(tuple(np.random.choice(candidates, n_seeds)))
    else:
        init_seeds.append(tuple(np.random.choice(list(graph.nodes()), n_seeds)))

    return init_seeds

def add_skewed_weights(n: int = 1, upper_weight_limit: int = 10, skew_power: int = 3) -> np.ndarray:
    return np.round((np.random.rand(n) ** skew_power) * (upper_weight_limit + 1))


def simulate_contagion_map(params: dict):
    """
    Prepares a contagion map for a simulated propagation
    :param params: Dictionary of parameters required for creating a networkx graph, seeding sites
    :return: networkx graph, list(seeding sites)
    """
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
        seed_nodes = initial_seed_nodes(graph=graph, n_seeds=n_seeds,
                                        seed_cluster_distance=params.get("seed_cluster_distance", 10))
    elif seeding_method == "all_combinations":
        seed_nodes = get_seed_nodes_combinations(graph=graph, n_seeds=n_seeds)

    return graph, seed_nodes


def simulate_contagion_realization( graph: nx.Graph, init_seeds: tuple, params: dict,
                                    adjacency_matrix: np.ndarray, weight_0: np.ndarray,
                                    max_steps: int = 100,
                                    sim_id: int = 1, realization_id: int = 1,
                                    calculate_representation: bool = False
                                    ):
    # Run contagion propagation
    active_nodes_at_end, activation_times, snapshots = contagion_propagation(
        num_nodes=params.get("num_nodes", 100),
        adjacency_matrix=adjacency_matrix,
        weight_0=weight_0,
        init_seeds=init_seeds,
        node_active_threshold=params.get("node_active_threshold", 0.001),
        max_steps=max_steps
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

    results = []
    for t, active_nodes_time_t in enumerate(snapshots):
        state = state_function(active_nodes_time_t, params.get("threshold_sum", 19900))

        features_dict = {
            "time": t,
            "state": state,
            "num_active_nodes": len(active_nodes_time_t),
            "active_nodes": list(active_nodes_time_t),
            "H_0": 0,  #  betti_numbers[t][0],
            "H_1": 1,  # betti_numbers[t][1],
            "H_2": 2,  # betti_numbers[t][2],
            }
        results.append(features_dict)

    if calculate_representation:
        bandwidth = params.get("bandwidth", 0.1)
        num_landscapes = params.get("num_landscapes", 3)
        resolution = params.get("resolution", 50)
        representation_choice_function  = params.get("representation_choice_function", "persistence")
        representation_choice_function = get_representation_choice_function(representation_choice_function)
        L, I, E, representaion_params = gp.persistence_representation_t(
            persistence=persistence,
            bandwidth=bandwidth,
            resolution=resolution,
            num_landscapes=num_landscapes,
            persistence_surface_function=representation_choice_function,
        )
        rename = lambda d, prefix: {f"{prefix}_{k}": v for k, v in d.items()}

        L_timesteps = len(L)
        assert L_timesteps == len(results)
        for res_dict, timestep in zip(results, range(L_timesteps)):
            res_dict.update(rename(L[timestep], "L"))
            res_dict.update(rename(I[timestep], "I"))
            res_dict.update(rename(E[timestep], "E"))

    return graph, snapshots, activation_times, results


def main_sims( params_list: list, np_global_seed:int = 666, max_steps=100, output_file="simulation_results.csv", save_files=False ):
    """
    :param params_list: list of dicts with parameters for each run,
                 e.g. [{'num_nodes': 100, 'param2': val2, ...}, ...]
    :param np_global_seed: global seed for numpy.random module. If localized seed is needed, implement rng = np.random.default_rng(666), rng.choice([2, 1])
    :param max_steps: max steps to run contagion
    :param output_file: filename to save the results
    :param save_files: save results to a csv, pickle file
    :return DataFrame with simulation results and saves to CSV
    """
    np.random.seed(np_global_seed)
    outfile_path = os.path.join(PATH, "outputs")
    simulation_results = []
    # activation_times_results = []  #uncomment for brad
    dfs = []
    for i, params in enumerate(params_list):
        print(f"Simulating Contagion Map for simulation: {i}/{len(params_list)}")
        try:
            graph, seed_node_combinations = simulate_contagion_map(params=params)
        except InvalidGraphError:
            Warning(f"Invalid Graph creation in: simulation_id: {i}")
            continue
        # base matrices
        adjacency_matrix = nx.to_numpy_array(graph, dtype=float, weight=None)
        # weight_matrix = nx.to_numpy_array(graph, weight='weight', nonedge=np.inf) #
        weight_0 = nx.to_numpy_array(graph, weight='weight', nonedge=1e9)

        # activation_times_results.append({ "sim_id": i, "graph": graph, "realization_id": 0}) #uncomment for brad
        base_dfs = []
        df_graph = nx.to_pandas_edgelist(graph)
        average_weight = df_graph['weight'].mean()
        total_geo_edges = df_graph.loc[df_graph['type'] == 'geometric'].shape[0]
        total_non_geo_edges = df_graph.loc[df_graph['type'] == 'non_geometric'].shape[0]
        del df_graph


        for j, seed_nodes in enumerate(seed_node_combinations):
            G, _, activation_times, results = simulate_contagion_realization(
                adjacency_matrix=adjacency_matrix,
                weight_0=weight_0,
                graph=graph,
                init_seeds=seed_nodes,
                params=params,
                sim_id=i,
                max_steps=max_steps,
                realization_id=j,
                calculate_representation=params.get("calculate_representation", False),
            )
            # simulation_results.extend(results)

            df_temp = pd.DataFrame(results)
            df_temp['realization_id'] = j
            df_temp['seed_nodes'] = [seed_nodes] * len(df_temp)
            for key, value in params.items():
                df_temp[key] = value

            base_dfs.append(df_temp)
            # activation_times_results.append( #uncomment for brad
            #     {"sim_id": i, "realization_id": j, "activation_times": activation_times}
            # )

        base_df = pd.concat(base_dfs, ignore_index=True)
        base_df['simulation_id'] = i
        base_df['average_weight_per_edge'] = average_weight
        base_df['total_geo_edges'] = total_geo_edges
        base_df['total_non_geo_edges'] = total_non_geo_edges
        dfs.append(base_df)

    df = pd.concat(dfs, ignore_index=True)

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
        return df, 0 # activation_times_results
