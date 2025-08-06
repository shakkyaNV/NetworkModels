import networkx as nx, numpy as np, matplotlib.pyplot as plt
import os

import gudhi as gd, persim
from collections import defaultdict

_sentinel = object()

def compute_persistence(graph: nx.Graph,  activation_times, max_dim: int = 2, ngeom_edges_in_persistence:bool = False):
    """
    Compute the persistence homology given a networkx and a list of activation times.
    :param graph: nx.Graph
    :param activation_times: np.ndarray
    :param max_dim: Max Betti Number to compute
    :param ngeom_edges_in_persistence: bool (default False) To whether include non-geometric edges in persistence calculations
    :return: Betti Numbers over filtration times: defaultdict(list), Simpleces persistence_intervals: list([dim][filtration][(birth) death)]
    """

    def clean_inputs(graph, activation_times, ngeom_edges_in_persistence):

        if ngeom_edges_in_persistence:
            G = graph
        else: ######### TEMPORARY, Change to use a distance threshold to see which are geom and ngeom
            G = nx.Graph()
            G.add_nodes_from(graph.nodes(data = True))
            G.add_edges_from([(u, v, d) for u, v, d in graph.edges(data=True) if d.get('type') == 'geometric'])

        G = nx.relabel_nodes(G, lambda x: int(x))
        activation = np.array([int(x)
                               if not np.isnan(x) else np.nan
                               for x in activation_times], dtype='object')
        return G, activation

    graph, activation = clean_inputs(graph, activation_times, ngeom_edges_in_persistence)

    betti_over_time = {}
    simplex_intervals = defaultdict(list)

    for t in range(np.nanmax(activation) + 1):
        # print(f"---------- Filtration Time Step: {t} ------------")
        tree = gd.SimplexTree()
        tree.make_filtration_non_decreasing()
        # tree.initialize_filtration()

        active_nodes = [node for node, time in enumerate(activation) if time <= t]

        # Create a subgraph at time = t,
        # Add all current nodes and edges
        subg = graph.subgraph(active_nodes).copy()

        for node in subg.nodes():
            tree.insert([node], filtration=t)
        for u, v, labels in subg.edges(data = True):
            if labels['weight'] < t:
                edge_filtration = max(activation[u], activation[v])
                tree.insert([u, v], filtration=edge_filtration)

        tree.compute_persistence(min_persistence=0, persistence_dim_max=max_dim)

        # Using intervals to calculate persistent homology, because we need intervals
        # for barcode graphs anyway.
        # Otherwise, tree.persistent_pairs(), or tree.betti_numebers() would
        # be very easy
        temp_betti = {}
        for dim in range(max_dim + 1):
            intervals = tree.persistence_intervals_in_dimension(dim)
            intervals = intervals.astype(object)
            # print(f"Dimension: {dim} \n intervals: {intervals} \n ~~~~~~~~~~~~~")
            b = sum(1 for birth, death in intervals if birth <= t < death)
            temp_betti[dim] = b
            simplex_intervals[dim].append((t, [tuple(pair) for pair in intervals]))
            betti_over_time[t] = temp_betti

        # print(f"Betti numebrs by tree.betti_numbers[t]: {tree.betti_numbers()}")
        # print(f" Betti Numbers: {temp_betti}")

    return betti_over_time, simplex_intervals


def betti_nums_over_time(betti_over_time: dict):
    times = sorted(betti_over_time.keys())
    dims = sorted(next(iter(betti_over_time.values())).keys())

    for dim in dims:
        values = [betti_over_time[t][dim] for t in times]
        plt.plot(times, values, label=f"H{dim}")

    plt.xlabel("Time Stesp")
    plt.ylabel("Feature Counts")
    plt.title("Betti Counts")
    plt.legend()
    plt.grid(True)
    plt.show()

def persim_diagram(simplex_intervals):
    diagrams = []
    for dim in range(2):
        latest = simplex_intervals[dim][-1][1]
        diagrams.append(np.array(latest, dtype=np.float64))

    persim.plot_diagrams(diagrams, show=True)

def plot_persistence_barcodes(simplex_intervals, activation_times, max_dim=2):
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for dim in range(max_dim + 1):
        intervals = simplex_intervals[dim][-1][1]
        for i, (birth, death) in enumerate(intervals):
            death_val = death if np.isfinite(death) else np.nanmax(activation_times)
            plt.hlines(y=dim + i * 0.1, xmin=birth, xmax=death_val, color=colors[dim % len(colors)])
        plt.axhline(dim, linestyle='--', color='gray', alpha=0.5)

    plt.xlabel("Filtration (activation time)")
    plt.ylabel("Homology class index")
    plt.title("Persistence Barcodes (H0, H1, H2)")
    plt.yticks([])
    plt.grid(True)
    plt.show()


def persistence_landscapes(simplex_intervals: dict, start = _sentinel, stop = _sentinel, num_steps: int = 10, flatten: bool = False,
                           inf_replacement: float = 10.0):

    time_indexed_generator_dict = defaultdict(lambda: [[] for _ in range(3)])
    for hom_deg, time_gen_list in simplex_intervals.items():
        for time_step, gens in time_gen_list:
            time_indexed_generator_dict[time_step][hom_deg] = gens

    landscape_per_time = {}
    for t, homology_lists in sorted(time_indexed_generator_dict.items()):
        diagram_all_generators = {}
        for hom_deg, pairs in enumerate(homology_lists):
            # if not pairs:
            #     continue
            diag = []
            for birth, death in pairs:
                birth = float(birth)
                death = float('inf') if death == float('inf') else float(death)
                if np.isinf(death):
                    death = inf_replacement
                diag.append((birth, death))
            diag = np.array(diag).reshape(-1, 2)
            diagram_all_generators[hom_deg] = np.array(diag)

        filtered = np.array([diagram_all_generators[0],
                                     diagram_all_generators[1],
                                     diagram_all_generators[2]], dtype=object)
        filtered = [arr for arr in filtered if arr.shape[0] > 0]
        pl_vector = {}
        for hom_deg, _ in simplex_intervals.items():
            hom_deg = 0
            pl = persim.PersistenceLandscaper(hom_deg=hom_deg, num_steps=num_steps, flatten=flatten)
            print(f"t: {t}, hom_deg: {hom_deg}, and diagram: {filtered}")
            pl.fit(filtered)
            landscape = pl.transform(filtered)
            mean_landscape = landscape.mean(axis=0)  #- vector of length num_steps
            pl_vector[hom_deg] = mean_landscape

            landscape_per_time[t] = pl_vector
    return landscape_per_time
