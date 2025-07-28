import pickle
import random
import networkx as nx
import numpy as np
import json
from pyvis.network import Network

def visualize_step_animation_new(G, snapshots, output_file):
    mapping = {n: int(n) for n in G.nodes}
    G_copy = G.copy()
    G = nx.relabel_nodes(G, mapping)
    for u, v, attrs in G.edges(data=True):
        for k in list(attrs):
            val = attrs[k]
            if isinstance(val, (np.generic, np.ndarray)):
                attrs[k] = val.item()

    for u, v, attrs in G.edges(data=True):
        edge_label = ""
        if 'type' in attrs:
            edge_label += f"{attrs['type']}"
        if 'weight' in attrs:
            edge_label += f" ({attrs['weight']})"
            # del attrs['weight']  # Optional: prevent pyvis autoscaling
        attrs['title'] = edge_label
        attrs['label'] = edge_label
        attrs['font'] = {'align': 'top', 'size': 30}

    net = Network(height="600px", width="100%", notebook=False, directed=False)

    net.from_nx(G)
    pos = nx.circular_layout(G)

    for node in net.nodes:
        node_id = node['id']
        x, y = pos[node_id]
        node['x'] = x * 1000
        node['y'] = y * 1000
        node['fixed'] = {'x': True, 'y': True}
        node['label'] = str(node_id)
        node['value'] = 50
        node['font'] = {'size': 30}

    # Prepare node color snapshots
    color_snapshots = []
    for active_set in snapshots:
        state = []
        for node in G.nodes():
            is_active = node in active_set
            color = 'red' if is_active else 'gray'
            state.append({'id': node, 'color': color, 'is_active': is_active})
        color_snapshots.append(state)

    # Prepare edge color snapshots (red if both endpoints active, else gray)
    edge_snapshots = []
    for t, active_set in enumerate(snapshots):
        state = []
        for u, v in G.edges():
            if u in active_set and v in active_set:
                if G_copy.get_edge_data(u, v).get('weight') < t:
                    color = 'red'
                else:
                    color = 'gray'
            else:
                color = 'gray'
            # Note: PyVis edges have id 'from' and 'to' keys
            state.append({'from': u, 'to': v, 'color': color})
        edge_snapshots.append(state)

    net.save_graph(output_file)

    with open(output_file, 'r', encoding='utf-8') as f:
        html = f.read()

    color_data_nodes = json.dumps(color_snapshots)
    color_data_edges = json.dumps(edge_snapshots)

    js_controls = f"""
    <div style="margin: 20px;">
        <button onclick="prevStep()">Previous</button>
        <button onclick="nextStep()">Next</button>
        <span>Step: <span id="stepCounter">0</span></span><br><br>

        <strong>Activated Count:</strong> <span id="activatedCount">0</span><br>
        <strong>Activated Nodes:</strong> <span id="activatedNodes"></span><br>
    </div>
    <script>
        var snapshotsNodes = {color_data_nodes};
        var snapshotsEdges = {color_data_edges};
        var currentStep = 0;
        var nodes = network.body.data.nodes;
        var edges = network.body.data.edges;

        function applyStep(step) {{
            const activeNodes = snapshotsNodes[step]
                .filter(nodeState => nodeState.is_active)
                .map(nodeState => nodeState.id);

            // Update node colors
            snapshotsNodes[step].forEach(nodeState => {{
                nodes.update({{id: nodeState.id, color: nodeState.color}});
            }});

            // Update edge colors
            snapshotsEdges[step].forEach(edgeState => {{
                // Find the edge in network data and update its color
                edges.update({{
                    id: edges.get({{filter: function(e) {{
                        return (e.from === edgeState.from && e.to === edgeState.to) ||
                               (e.from === edgeState.to && e.to === edgeState.from);
                    }}}})[0].id,
                    color: edgeState.color
                }});
            }});

            document.getElementById('stepCounter').innerText = step;
            document.getElementById('activatedCount').innerText = activeNodes.length;
            document.getElementById('activatedNodes').innerText = activeNodes.join(', ');
        }}

        function nextStep() {{
            currentStep = Math.min(currentStep + 1, snapshotsNodes.length - 1);
            applyStep(currentStep);
        }}

        function prevStep() {{
            currentStep = Math.max(currentStep - 1, 0);
            applyStep(currentStep);
        }}

        // Initial state
        applyStep(0);
    </script>
    """

    # Insert controls just after the main network script
    insert_point = html.rfind("</script>")
    html = html[:insert_point + 9] + js_controls + html[insert_point + 9:]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)


def snapshots_to_activation_times_series(snapshots, num_nodes):
    """
    Convert snapshots list to a list of activation_times arrays.
    Instaed of running t=1 and t=2, this should force everything to run
    at once and create many rows
    """

    activation_times = np.full(num_nodes, np.nan)
    activation_time_series = []

    # coming snapshots are currently active nodes at each time step
    # we need to convert them into sets of incremental only nodes
    # {0} {0, 1} {0, 1, 12} --> {0}, {1} , {12}

    step_diff = [snapshots[0]]
    for prev, curr in zip(snapshots[:-1], snapshots[1:]):
        diff = curr - prev
        step_diff.append(diff)

    for t, newly_activated in enumerate(step_diff):
        for node in newly_activated:
            activation_times[node] = t


        activation_time_series.append(activation_times.copy())

    return activation_time_series



def generate_random_params(num_samples=200):
    params_list = []

    for _ in range(num_samples):
        num_nodes = random.randint(100, 200)
        weighted = random.choice([True, False])
        n_seeds = random.choices([1, 2, 3, 4])[0] #, weights=[0.1, 0.4, 0.4, 0.1])[0]
        node_active_threshold =  random.choices(np.round(np.arange(0.03, 0.2, 0.01), 3))[0] #if weighted else round(random.choice(np.arange(0.10, 0.20, 0.01)), 3)
        num_neighbor_nodes = random.randint(1, 3)  # if node_active_threshold > 0.1 else random.randint(2, 4)
        distance_threshold = random.randint(num_neighbor_nodes + 1, 10) # if node_active_threshold > 0.1 else random.randint( num_neighbor_nodes + 5, 15)
        total_random_edges = random.choice(range(5, 51, 5)) # if node_active_threshold > 0.1 else random.choice(range(5, 31, 5))
        upper_weight_limit = random.randint(10, 30)  # if node_active_threshold > 0.1 else random.randint(20, 40)
        skew_power =  random.randint(2, 4)
        seed_cluster_distance = random.randint(n_seeds + 1, 30)
        ngeom_edges_in_persistence = False
        max_persistence_dim = 2
        threshold_sum = sum(range(num_nodes)) - 1

        param = {
            'num_nodes': num_nodes,  # fixed
            'num_neighbor_nodes': num_neighbor_nodes,
            'total_random_edges': total_random_edges,
            'distance_threshold': distance_threshold,
            'weighted': weighted,
            'random_dist': 'random.choice',  # fixed
            'n_seeds': n_seeds,
            'node_active_threshold': node_active_threshold,
            'upper_weight_limit': upper_weight_limit,
            'skew_power': skew_power,
            'seed_cluster_distance': seed_cluster_distance,
            'ngeom_edges_in_persistence': ngeom_edges_in_persistence,
            'max_persistence_dim': max_persistence_dim,
            'threshold_sum': threshold_sum
        }

        params_list.append(param)

    return params_list


def load_all_simulations(file="all_simulations.pkl"):
    with open(file, 'rb') as f:
        return pickle.load(f)

def graph_to_distance_matrix(graph: nx.Graph, nodes: list):
    # compute distance/geo-desic distances
    subgraph = graph.subgraph(nodes)
    shortest_distance = dict(nx.all_pairs_dijkstra_path_length(subgraph, weight='weight'))

    dist_matrix = np.full((len(nodes), len(nodes)), np.inf)
    np.fill_diagonal(dist_matrix, 0)

    for ix, u in enumerate(nodes):
        for jx, v in enumerate(nodes):
            if v in shortest_distance[u]:
                dist_matrix[ix, jx] = shortest_distance[u][v]

    dist_matrix = np.minimum(dist_matrix, dist_matrix.T)
    return dist_matrix