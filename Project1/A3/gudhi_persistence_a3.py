import networkx as nx, numpy as np, matplotlib.pyplot as plt
import os
import persim
import gudhi as gd
from gudhi.representations import (
    DiagramSelector,
    DiagramScaler,
    Landscape,
    Clamping,
    PersistenceImage,
)
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

_sentinel = object()


def compute_persistence(
    graph: nx.Graph,
    activation_times,
    max_dim: int = 2,
    ngeom_edges_in_persistence: bool = False,
):
    """
    Compute the persistence homology given a networkx and a list of activation times.
    ######## REWRITE THE WHOLE FUNCTION #######
    :param graph: nx.Graph
    :param activation_times: np.ndarray
    :param max_dim: Max Betti Number to compute
    :param ngeom_edges_in_persistence: bool (default False) To whether include non-geometric edges in persistence calculations
    :return: Betti Numbers over filtration times: defaultdict(list), Simpleces persistence_intervals: list([dim][filtration][(birth) death)]
    """

    if not ngeom_edges_in_persistence:
        edges_to_drop = [(u, v, d) for u, v, d in graph.edges(data=True) if d.get('type') == 'non_geometric']
        graph.remove_edges_from(edges_to_drop)

    betti_over_time = {}
    simplex_intervals = defaultdict(list)
    persistence = np.empty((int(np.nanmax(activation_times)) + 1, max_dim + 1), dtype=object)
    for t in range(np.nanmax(activation_times) + 1):
        for d in range(max_dim + 1):
            persistence[t, d] = []

    # persistence_for_graphics = []  ## gudhi tools requires a special format for diagrams
    tree = gd.simplex_tree.SimplexTree(None)
    for t in range(np.nanmax(activation_times) + 1):
        # print(f"---------- Filtration Time Step: {t} ------------")
        tree.make_filtration_non_decreasing()
        # tree.initialize_filtration()

        active_nodes = [node for node, time  in enumerate(activation_times) if time <= t]
        # Create a subgraph at time = t,
        # Add all current nodes and edges
        subg = graph.subgraph(active_nodes).copy()

        for node in subg.nodes():
            tree.insert([node], filtration=activation_times[node])
        for u, v, labels in subg.edges(data=True):
            if labels["weight1"] <= t:
                edge_filtration = max(activation_times[u], activation_times[v])
                tree.insert([u, v], filtration=edge_filtration)

        # tree.compute_persistence(min_persistence=0, persistence_dim_max=max_dim)
        # tree2 = tree.copy()
        for dim, interval in tree.persistence(min_persistence=0, persistence_dim_max=2):
            persistence[t, dim].append(interval)
        # for dim, (birth, death) in temp_pers:
        #     for time in range(np.nanmax(activation_times) + 1):
        #         if birth <= time < death:
        #             persistence[time, dim].append((birth, death))
        # persistence = 0
        # Using intervals to calculate persistent homology, because we need intervals
        # for barcode graphs anyway.
        # Otherwise, tree.persistent_pairs(), or tree.betti_numebers() would
        # be very easy
        # temp_betti = {}
        # for dim in range(max_dim + 1):
        #     intervals = tree.persistence_intervals_in_dimension(dim)
        #     intervals = intervals.astype(object)
        #     # persistence[t, dim] = intervals
        #     # --persistence_for_graphics.extend([(dim, tuple(pair)) for pair in intervals])
        #
        #     # print(f"Dimension: {dim} \n intervals: {intervals} \n ~~~~~~~~~~~~~")
        #     b = sum(1 for birth, death in intervals if birth <= t < death)
        #     temp_betti[dim] = b
        #     simplex_intervals[dim].append((t, [tuple(pair) for pair in intervals]))
        #     betti_over_time[t] = temp_betti
        betti_over_time = 0
        # print(f"Betti numebrs by tree.betti_numbers[t]: {tree.betti_numbers()}")
        # print(f" Betti Numbers: {temp_betti}")

    persistence_for_graphics = tree.persistence(
        min_persistence=0, persistence_dim_max=2
    )
    # persistence = np.array([interval for _, interval in persistence_for_graphics])

    return betti_over_time, persistence, persistence_for_graphics


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


def persistence_diagram(persistence_for_representation: list, colormap: tuple = None):
    """
    Draws persistence diagram directly from gudhi.SimplexTree.persistence_intervals_in_dimension:
    :param persistence_for_representation: List(Tuple(int, Tuple(birth, death)))
    :param colormap: matplotlib qualitative colormap Tuple(float, float, .. )
    :return: plt.show()
    """
    ax = gd.persistence_graphical_tools.plot_persistence_diagram(
        persistence_for_representation, legend=True, colormap=colormap
    )
    plt.show()


def persistence_barcodes(persistence_for_representation: list, colormap: tuple = None):
    """
    Draws persistence diagram directly from gudhi.SimplexTree.persistence_intervals_in_dimension:
    :param persistence_for_representation: List(Tuple(int, Tuple(birth, death)))
    :param colormap: matplotlib qualitative colormap Tuple(float, float, .. )
    :return: plt.show()
    """
    ax = gd.persistence_graphical_tools.plot_persistence_barcode(
        persistence_for_representation, legend=True, colormap=colormap
    )
    plt.show()


def persim_diagram(simplex_intervals):
    diagrams = []
    for dim in range(2):
        latest = simplex_intervals[dim][-1][1]
        diagrams.append(np.array(latest, dtype=np.float64))

    persim.plot_diagrams(diagrams, show=True)


def plot_persistence_barcodes(simplex_intervals, activation_times, max_dim=2):
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for dim in range(max_dim + 1):
        intervals = simplex_intervals[dim][-1][1]
        for i, (birth, death) in enumerate(intervals):
            death_val = death if np.isfinite(death) else np.nanmax(activation_times)
            plt.hlines(
                y=dim + i * 0.1,
                xmin=birth,
                xmax=death_val,
                color=colors[dim % len(colors)],
            )
        plt.axhline(dim, linestyle="--", color="gray", alpha=0.5)

    plt.xlabel("Filtration (activation time)")
    plt.ylabel("Homology class index")
    plt.title("Persistence Barcodes (H0, H1, H2)")
    plt.yticks([])
    plt.grid(True)
    plt.show()


def persistence_representation(
    persistence: np.ndarray,
    persistence_surface_function=lambda x: x[1] - x[0],
    bandwidth: float = 0.1,
    resolution: int = 50,
    num_landscapes: int = 3,
):
    """
    Create persistence landscape/image arrays, later to be visualized/PCA'd
    :param persistence: gudhi.SimplexTree.persistence_intervals_in_dimension (nx2): each (birth, death)
    :param persistence_surface_function: lambda expression for the weight function of persistence_image
    :param bandwidth: bandwidth for persitence image (determines the smoothing of gaussian smoother around hotspots)
    :param resolution: resolution of persistence
    :param num_landscapes: number of landscapes
    :return: list(defaultdict, defaultdict, dict)
    """
    max_dim = persistence.shape[1]

    # preprocessing
    proc_finite = DiagramSelector(use=True, point_type="finite")
    proc_essential = DiagramSelector(use=False, point_type="essential")
    proc_scaler = DiagramScaler(use=True, scalers=[([0, 1], MinMaxScaler())])
    proc_clamp = DiagramScaler(use=True, scalers=[([1], Clamping(maximum=0.9))])
    call_plandscape = Landscape(resolution=resolution, num_landscapes=num_landscapes)
    call_pimage = PersistenceImage(
        bandwidth=bandwidth,
        weight=persistence_surface_function,
        im_range=[0, 1, 0, 1],
        resolution=[resolution, resolution],
    )
    params = {
        "num_landscapes": num_landscapes,
        "bandwidth": bandwidth,
        "surface_function": persistence_surface_function,
        "resolution": resolution,
        "pre-processing": ["finite", "mix_max_scaler", "clamp"],
    }
    L = defaultdict()
    I = defaultdict()

    for dim in range(max_dim):
        persistence_in_dim = np.vstack(persistence[:, dim])
        # skip to next dim if current dim has no finite points (which is the case for homology dim = 2 (sometimes 1)
        if len(proc_finite(persistence_in_dim)) == 0:
            continue

        diagram = proc_clamp(proc_scaler(proc_finite(persistence_in_dim)))
        diagram = np.asarray(diagram, dtype=np.float64)

        v_landscape = call_plandscape(diagram)
        v_image = call_pimage(diagram)
        L[dim] = v_landscape
        I[dim] = v_image

    return L, I, params


def persistence_representation_t(
    persistence: np.ndarray,
    persistence_surface_function = lambda x: x[1] - x[0],
    bandwidth: float = 0.1,
    resolution: int = 50,
    num_landscapes: int = 3,
):
    """
    Create persistence landscape/image arrays, later to be visualized/PCA'd
    :param persistence: gudhi.SimplexTree.persistence_intervals_in_dimension (nx2): each (birth, death)
    :param persistence_surface_function: lambda expression for the weight function of persistence_image
    :param bandwidth: bandwidth for persitence image (determines the smoothing of gaussian smoother around hotspots)
    :param resolution: resolution of persistence
    :param num_landscapes: number of landscapes
    :return: defaultdict [timestep][dim] entry size=resol*num_landscape,
    defaultdict [timestep][dim] entry size=resol*resol , defaultdict[timestep] entrysize = dim --->>>> Changed otherway around
    """
    max_dim = persistence.shape[1]
    timesteps = persistence.shape[0]
    # preprocessing
    proc_finite = DiagramSelector(use=True, point_type="finite")
    proc_essential = DiagramSelector(use=False, point_type="essential")
    proc_scaler = DiagramScaler(use=True, scalers=[([0, 1], MinMaxScaler())])
    proc_clamp = DiagramScaler(use=True, scalers=[([1], Clamping(maximum=0.9))])
    call_plandscape = Landscape(resolution=resolution, num_landscapes=num_landscapes)
    call_pimage = PersistenceImage(
        bandwidth=bandwidth,
        weight=persistence_surface_function,
        im_range=[0, 1, 0, 1],
        resolution=[resolution, resolution],
    )
    params = {
        "num_landscapes": num_landscapes,
        "bandwidth": bandwidth,
        "resolution": resolution,
        "pre-processing": ["finite", "mix_max_scaler", "clamp"],
    }
    L = []
    I = []
    essential_features = []
    for timestep in range(timesteps):
        L_d = {k: [np.zeros((resolution * num_landscapes,))] for k in range(max_dim)}
        I_d = {k: [np.zeros((resolution * resolution,))] for k in range(max_dim)}
        E_d = {k: 0 for k in range(max_dim)}
        for dim in range(max_dim):
            if len(persistence[timestep, dim]) == 0:
                continue
            persistence_in_dim = np.vstack(persistence[timestep, dim])
            # skip to next dim if current dim has no finite points (which is the case for homology dim = 2 (sometimes 1)
            if len(proc_finite(persistence_in_dim)) == 0:
                continue
            diagram = proc_clamp(proc_scaler(proc_finite(persistence_in_dim)))
            diagram = np.asarray(diagram, dtype=np.float64)

            E_d[dim] = len(proc_essential(persistence_in_dim))
            v_landscape = call_plandscape(diagram)
            v_image = call_pimage(diagram)
            L_d[dim] = v_landscape
            I_d[dim] = v_image
        L.append(L_d)
        I.append(I_d)
        essential_features.append(E_d)
    return L, I, essential_features, params


def persistence_landscapes_old(
    simplex_intervals: dict,
    start=_sentinel,
    stop=_sentinel,
    num_steps: int = 10,
    flatten: bool = False,
    inf_replacement: float = 10.0,
):

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
                death = float("inf") if death == float("inf") else float(death)
                if np.isinf(death):
                    death = inf_replacement
                diag.append((birth, death))
            diag = np.array(diag).reshape(-1, 2)
            diagram_all_generators[hom_deg] = np.array(diag)

        filtered = np.array(
            [
                diagram_all_generators[0],
                diagram_all_generators[1],
                diagram_all_generators[2],
            ],
            dtype=object,
        )
        filtered = [arr for arr in filtered if arr.shape[0] > 0]
        pl_vector = {}
        for hom_deg, _ in simplex_intervals.items():
            hom_deg = 0
            pl = persim.PersistenceLandscaper(
                hom_deg=hom_deg, num_steps=num_steps, flatten=flatten
            )
            print(f"t: {t}, hom_deg: {hom_deg}, and diagram: {filtered}")
            pl.fit(filtered)
            landscape = pl.transform(filtered)
            mean_landscape = landscape.mean(axis=0)  # - vector of length num_steps
            pl_vector[hom_deg] = mean_landscape

            landscape_per_time[t] = pl_vector
    return landscape_per
