import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd

from utilsA1 import snapshots_to_activation_times_series

PATH = os.path.dirname(__file__)

def generate_graph(num_nodes: int, num_neighbor_nodes:int, total_random_edges: int,
                   distance_threshold: int,
                   upper_weight_limit: int = 10, skew_power: int = 3,
                   weighted=False, random_dist='random.choice') -> nx.Graph:
    """

    Generate Ring Graph with geometric and non_geometric edges
    :param num_nodes: Total number of nodes
    :param num_neighbor_nodes: Number of neighboring nodes that are connected to the node on one side
    :param total_random_edges: Number of total non-geometric edges
    :param distance_threshold: (Number of nodes distance away) to qualify as non-geometric edge
    :param upper_weight_limit: maximum weight allowed for non-geometric edges
    :param skew_power: How much weights are skewed towards zero
    :param weighted: if Weighted graph or Not
    :param random_dist: Probability distribution for weights
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
                graph.add_edge(node, (node + i + 1) % num_nodes, type = 'geometric',
                               weight = add_skewed_weights(1, upper_weight_limit, skew_power)[0])
    else:
        for node in range(num_nodes):
            for i in range(num_neighbor_nodes):
                graph.add_edge(node, (node + i + 1) % num_nodes, type = 'geometric')

    graph = add_non_geometric_edges(graph=graph, total_random_edges=total_random_edges,
                                    distance_threshold=distance_threshold,
                                    upper_weight_limit=upper_weight_limit,
                                    skew_power=skew_power,
                                    random_dist=random_dist,
                                    weighted=weighted)
    return graph


def add_non_geometric_edges(graph: nx.Graph, total_random_edges: int, distance_threshold: int,
                            upper_weight_limit: int = 10, skew_power: int = 3,
                            random_dist="random.choice", weighted: bool = False) -> nx.Graph:
    """
    Create non-geometric edges accoring to a given distance
    (Not a L2/L1 measure, Based on number of neighbors in Ring) threshold
    :param graph: networkx graph
    :param total_random_edges: total number of non-geometric edges needed. Default = Num_nodes/2
    :param distance_threshold: # of neighbouring to skip as an early distance measure
    :param upper_weight_limit: maximum weight allowed for non-geometric edges
    :param skew_power: How much weights are skewed towards zero
    :param random_dist: choice of how to create non-geometric edges. "random.choice": randomly choosing from a pool of all possible edges
    :param weighted: whether edges should be weighted or not. Default = False
    :return: networkx graph
    """
    if total_random_edges == 0:
        return graph

    num_geometric_edges = graph.number_of_edges()
    if total_random_edges:
        num_random_edges = total_random_edges
    else:
        num_random_edges = num_geometric_edges/2

    if random_dist == "random.choice":
        src, tgt = np.meshgrid(graph.nodes(), graph.nodes())
        src = src.flatten()
        tgt = tgt.flatten()

        directed_mask = src < tgt   # we're only using the upper (or lower) triangle to void [2, 4] [4, 2] typa situations
        src = src[directed_mask]
        tgt = tgt[directed_mask]

        distance_mask = np.abs(src - tgt) >= distance_threshold
        src = src[distance_mask]
        tgt = tgt[distance_mask]
        valid_edge_list = np.stack((src, tgt), axis=1)

        random_edge_indexes = np.random.choice(a = len(valid_edge_list), size = num_random_edges, replace = False)
        non_geometric_edges = valid_edge_list[random_edge_indexes]

        if weighted:
            weights = add_skewed_weights(n = len(non_geometric_edges),
                                         upper_weight_limit = upper_weight_limit, skew_power=skew_power)
            weighted_edges = [(u, v, w) for (u,v), w in zip(valid_edge_list[random_edge_indexes], weights)]
            graph.add_weighted_edges_from(weighted_edges, type = 'non-geometric')
            return graph
        else:
            graph.add_edges_from(non_geometric_edges, type = 'non-geometric', weight = 0)
        return graph


def contagion_propagation(graph: nx.Graph, init_seeds: tuple, node_active_threshold: float,
                          max_steps: int = 100, weighted: bool = False):
    if max_steps is None:
        max_steps = graph.number_of_edges()

    activation_times = np.full(graph.number_of_nodes(), np.nan)
    activation_times[list(init_seeds)] = 0                          # set activation time=0 for initializing seeds
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
                            weight = graph.get_edge_data(node_i, ith_neighbor).get('weight', 0)
                            if weight < time_step:
                                weight_qualifying_nodes += 1

                    if weight_qualifying_nodes/len(neighbors) >= node_active_threshold:
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
                    active_neighbors = [node for node in neighbors if node in active_nodes]
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


def initial_seed_nodes(graph: nx.graph, n_seeds:int=2, seed_cluster_distance: int = 10, init_seeds = ()) -> tuple:
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


def add_skewed_weights(n: int = 1, upper_weight_limit: int = 10, skew_power: int = 3) -> int:
    return np.round((np.random.rand(n) ** skew_power) * (upper_weight_limit + 1)).astype(int)


def simulation_record_data(params:dict, threshold_sum: int, max_steps:int = 100, sim_id:int = 1):

    G = generate_graph(
        params.get('num_nodes', 100),
        params.get('num_neighbor_nodes', 3),
        params.get('total_random_edges', 50),
        params.get('distance_threshold', 4),
        params.get('upper_weight_limit', 10),
        params.get('skew_power', 3),
        params.get('weighted', True),
        params.get('random_dist', 'random.choice'))

    weights = [d.get('weight', 0) for _, _, d in G.edges(data=True)]
    average_weight = np.mean(weights) if weights else 0

    init_seeds = initial_seed_nodes(G, n_seeds=params.get('n_seeds', 1),
                                    seed_cluster_distance=params.get('seed_cluster_distance', 20),
                                    init_seeds=params.get('init_seeds', None))

    # Run contagion propagation
    active_nodes_at_end, activation_times, snapshots = contagion_propagation(graph=G, init_seeds=init_seeds,
                                                                             node_active_threshold=params.get(
                                                                                 'node_active_threshold', 0.001),
                                                                             max_steps=max_steps,
                                                                             weighted=params.get('weighted', False))
    activation_series = snapshots_to_activation_times_series(snapshots, params.get('num_nodes'))

    results = []
    for t, active_nodes_time_t in enumerate(activation_series):
        activated_nodes = np.where(~np.isnan(active_nodes_time_t))[0]
        state = state_function(activated_nodes, threshold_sum)

        features_dict = {
            'simulation_id': sim_id,
            'num_nodes': params.get('num_nodes', 100),
            'time': t,
            'state': state,
            'state_abnormal_sum': threshold_sum,
            'num_active_nodes': len(activated_nodes),
            'active_nodes': active_nodes_time_t,
            'node_active_threshold': params.get('node_active_threshold', 0.1),
            'num_non_geo_edges': params.get('total_random_edges'),
            'num_seeds': params.get('n_seeds', 1),
            'seed_nodes': init_seeds,
            'seed_cluster_distance': params.get('seed_cluster_distance', 10),
            'weighted': params.get('weighted'),
            'average_weight_per_edge': average_weight,
            'skew_power': params.get('skew_power'),
            'upper_weight_limit': params.get('upper_weight_limit'),
            'distance_threshold': params.get('distance_threshold')
        }
        results.append(features_dict)

    return G, snapshots, activation_times, results


def main_sims(params_list:dict , threshold_sum: int,
              max_steps=100, output_file='simulation_results.csv', save_files=False):
    """
    :param params_list: list of dicts with parameters for each run,
                 e.g. [{'num_nodes': 100, 'param2': val2, ...}, ...]
    :param threshold_sum: int, threshold for abnormal state
    :param max_steps: max steps to run contagion
    :param output_file: filename to save the results
    :param save_files: save results to a csv, pickle file
    :return DataFrame with simulation results and saves to CSV
    """
    outfile_path = os.path.join(PATH, 'outputs')
    simulation_results = []
    activation_times_results = [] # for brad
    for i, params in enumerate(params_list):
        # print(f"Running simulation {i + 1} with params: {params}")
        G, _, activation_times, results = simulation_record_data(params=params, sim_id = i,
                                                                 threshold_sum=threshold_sum, max_steps=max_steps)
        activation_times_results.append({'sim_id': i,
                                        'graph': G,
                                        'activation_times': activation_times})
        simulation_results.extend(results)

    df = pd.DataFrame(simulation_results)

    if save_files:
        df = pd.DataFrame(simulation_results)
        df_file_path = os.path.normpath(os.path.join(outfile_path, f"{output_file}.csv"))
        pickle_file_path = os.path.normpath(os.path.join(outfile_path, f"{output_file}.pkl"))

        df.to_csv(df_file_path, index=False)
        print(f"Simulation results saved to {df_file_path}")

        with open(pickle_file_path, "wb") as f:
            pickle.dump(activation_times_results, f)

        return df_file_path, pickle_file_path
    else:
        return df, activation_times_results

####################---- TEST Visualization---############
# G = generate_graph(100, 3, 50, 4, True, 'random.choice')
# init_seeds = initial_seed_nodes(G, n_seeds = 1, init_seeds = None)
# active_nodes, activation_times, propagation_snapshots = contagion_propagation(graph = G, init_seeds = init_seeds, node_active_threshold = 0.000001,
#                                         max_steps = 100, weighted = False)
#
# H = nx.Graph()
# H.add_nodes_from(G.nodes())
# print(G.edges())
# breakpoint()
# visualize_step_animation(G, three, 'networkx_contagion.html')