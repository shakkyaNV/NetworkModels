import os
import utilsA1 as utils
import test_WTM as wtm

PATH = os.getcwd()

output_file = "contagion_visualization"
params_temp_list = {'num_nodes': 200, 'num_neighbor_nodes': 3,
                    'total_random_edges': 40, 'distance_threshold': 5, 'weighted': True,
                    'random_dist': 'random.choice', 'n_seeds': 2, 'node_active_threshold': 0.02,
                    'upper_weight_limit': 30, 'skew_power': 3, 'seed_cluster_distance': 10,
                    'ngeom_edges_in_persistence':False, 'max_persistence_dim':2}

G, snapshots, activation_times, results = wtm.simulation_record_data(params=params_temp_list,
                                               threshold_sum=sum(range(params_temp_list.get('num_nodes'))) - 1,
                                                sim_id=1)

utils.visualize_step_animation_new(G=G, snapshots=snapshots,
                         output_file= os.path.join(PATH, 'Outputs', f"{output_file}.html"))