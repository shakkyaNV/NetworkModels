# Superseded by test_WTM.py

import os
import random
import webbrowser

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

from utilsA1 import visualize_step_animation

nodes = 20  # num nodes
prob  = 0.2  # p(edge-edge)
graph_seed = 666
seed_nodes = set()
num_seed_nodes = 2
output_file = "social_enforcement_propagation"


def generate_graph(n, p, seed=666, weight=False) -> nx.Graph:
    """
    Creates a random networkx graph, for now erdos_renyi
    """
    g = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    if weight:
        for u, v in g.edges():
            g[u][v]['weight'] = random.random()

    plt.show()
    return g


def initial_seed_nodes(g: nx.graph, init_seeds:set, n_seeds:int=1) -> set:
    if init_seeds:
        return init_seeds
    else:
        return set(random.sample(list(g.nodes()), n_seeds))


def node_init_thresholds(g) -> dict:
    """"
    Determines the specific threshold for
    a given node, according to some metric
    to be detemined later.
    Constant for now.

    g[u][v]['weight'] = check for threshold
    """
    thresholds = {node_i: 0.4 for node_i in g.nodes()}
    return thresholds

def cal_threshold_social_enforcement(node_i: int, g: nx.Graph, active_sed: set) -> float:
    """
    calculate the threshold for a given node
    Current approach: social enforcement (if num_neighbors > threshold: determined elsewhere)
    """
    neighbors = list(g.neighbors(node_i))
    active_neighbors = [node for node in neighbors if node in active_sed]
    return len(active_neighbors) / len(neighbors)

def watts_threshold(g: nx.graph, init_seeds: set, max_steps: int=100):
    max_steps = max(g.number_of_nodes(), max_steps) + 1
    node_thresholds = node_init_thresholds(g)

    # Keep track of the contagion propagation
    active_seeds = init_seeds
    propagation_snapshots = [init_seeds.copy()]

    for step in range(max_steps):
        step_propagation = set()
        for node_i in g.nodes():
            if node_i not in active_seeds:
                if len(list(g.neighbors(node_i))) == 0:
                    continue
                if cal_threshold_social_enforcement(node_i, g=g, active_sed=active_seeds) >= node_thresholds[node_i]:
                    step_propagation.add(node_i)

        if not step_propagation:
            break

        active_seeds.update(step_propagation)
        propagation_snapshots.append(active_seeds.copy())

    return active_seeds, propagation_snapshots

def run_wt_model(nodes, prob, graph_seed, weight:bool=False):
    g = generate_graph(n=nodes, p=prob, seed=graph_seed, weight=False)
    initial_seeds = initial_seed_nodes(g, init_seeds=seed_nodes, n_seeds=num_seed_nodes)
    final_active_seeds, prop_snapshots = watts_threshold(g, initial_seeds)
    return {'initial_node': list(prop_snapshots[0]),
            't': len(prop_snapshots),
            'num_nodes': nodes,
            'prob': prob,
            'seed': graph_seed,
            'final_nodes': final_active_seeds,
            'prop_snapshots': prop_snapshots,
            'g': g
            }

def main(n_sims:int):
    df = pd.DataFrame(columns=('initial_node', 't', 'num_nodes', 'prob', 'seed', 'final_nodes', 'prop_snapshots'))
    for i in range(n_sims):
        res = run_wt_model(nodes=random.randint(5, 30),
                           prob=random.uniform(0.1, 0.3),
                           graph_seed=i,
                           weight=False)
        df.loc[i] = res

    df.to_csv(f'{output_file}.csv', index=False)
    return df

# df = main(n_sims=100)
# print(df.head(5))


run_1 = run_wt_model(nodes, prob, graph_seed, weight=False)
visualize_step_animation(run_1['g'], run_1['prop_snapshots'], output_file=f'{output_file}.html')
webbrowser.open(f'file://{os.path.realpath(f'{output_file}.html')}')

breakpoint()
