import networkx as nx, numpy as np, pandas as pd  # type: ignore # type: ignore
from collections import defaultdict; from itertools import combinations
import os, sys

import utils_a3 as utils
import desikan_killiany_atlas as dkatlas

import utils_a1_a3 as utils_a1
import gudhi_persistence_a3 as gp_a1

## MODEL WIDE VARIABLES
MODULE_DIR = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(MODULE_DIR)))
RESOURCES_DIR = os.path.join(BASE_DIR, 'resources')

# # Basic instructions
# load pickle files (activations, snapshots, state_values)
# check if order preserved
# Create base graph (once) from pickle or from dkatlas
# propagate


def adni_gpc(graph: nx.Graph, activation_times:list, snapshots:list,
             state_values:list, rid:int, params:dict,
             patient_diffed_t: int):
    max_persistence_dim = int(params.get("max_persistence_dim", 2))
    ngeom_edges_in_persistence = params.get("ngeom_edges_in_persistence", False)
    calculate_representation = params.get("calculate_representation", True)
    bandwidth = params.get("bandwidth", 0.1)
    num_landscapes = params.get("num_landscapes", 3)
    resolution = params.get("resolution", 50)
    representation_choice_function = params.get("representation_choice_function", "persistence")
    non_active = params.get("non_active", -1)

    # Compute Persistence Homology
    _, persistence, persistence_for_graphcis = gp_a1.compute_persistence(
        graph=graph,
        activation_times=activation_times,
        max_dim=max_persistence_dim,
        ngeom_edges_in_persistence=ngeom_edges_in_persistence,
    )
    betti_nums_temp = utils.persistence_for_graphics_to_betti_nums(persistence_for_graphcis)

    results = []
    max_active_t = patient_diffed_t + 1
    for t, state_at_t in zip(range(max_active_t), state_values):
        active_nodes = np.where((activation_times > non_active) & (activation_times <= t))[0] + 1 # to be indexed from 1
        features_dict = {
            "time": t,
            "state": state_at_t,
            "num_active_nodes": len(active_nodes),
            "active_nodes": active_nodes,
            "H_0": betti_nums_temp[t][0],
            "H_1": betti_nums_temp[t][1],
            "H_2": betti_nums_temp[t][2],
        }
        results.append(features_dict)

    if calculate_representation:
        representation_choice_function = utils_a1.get_representation_choice_function(representation_choice_function)
        L, I, E, representaion_params = gp_a1.persistence_representation_t(
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

    return results


def main_sims(params: dict, adni_data_file_path:str, graph_file_path:str, np_global_seed:int=666,
              save_files=False, ouput_file:str="ADNI_GPC"):
    np.random.seed(np_global_seed)
    query_filter = params.get("query_filter", None)
    type_filter = params.get("type_filter", "amyloid")

    # load dkatlas-connectome
    graph = nx.read_graphml(os.path.join(utils.BASE_DIR, graph_file_path))
    nx.relabel_nodes(graph, lambda x: int(x), copy=False)

    df = utils.df_rename_to_fsnames(adni_data_file_path, query_filter=query_filter, type_filter=type_filter)
    df = utils.safe_filter_df(df, True, type_filter=type_filter)
    df.sort_values(by=['rid', 'scandate'], inplace=True)
    df, feature_cols = utils.activations_cortical_regions_df(df, True)
    (activation_times,
     snapshots,
     state_values) = utils.activation_times_of_patients_for_cortical_regions_df(df,feature_cols,True, type_filter=type_filter)
    print(f"Dataset proceeds with shape: {df.shape}")


    base_dfs = []
    for rid in activation_times.keys():
        print(f"Patient rid: {rid}")
        patient_max_t = np.nanmax(activation_times[rid])
        results = adni_gpc(activation_times=activation_times[rid],
                           snapshots=snapshots[rid],
                           state_values=state_values[rid],
                           graph = graph,
                           patient_diffed_t = patient_max_t,
                           rid=rid,
                           params=params)
        res_df = pd.DataFrame(results)
        res_df['rid'] = rid

        # problem with non-evolving disease progression. If stagnent, snapshots and active_nodes might differ
        # print(f"Shape of RES_DF from GPC: {res_df.shape}, number of snapthots filtered: {int(patient_max_t) + 1}")
        subset_cols = df.loc[df['rid'] == rid, ['scandate', 'loniuid']].iloc[: int(patient_max_t) + 1] # reduce to use of a iterated list
        res_df[['scandate', 'loniuid']] = subset_cols.to_numpy()        # use same list here

        # for key, value in params.items():
        #   res_df[key] = values
        base_dfs.append(res_df)

    # Concat list of dataframes to get one long df
    base_df = pd.concat(base_dfs, ignore_index=True)

    if save_files:
        outfile_path = os.path.abspath(os.path.join(utils.BASE_DIR, "resources", f"{ouput_file}.csv"))
        base_df.to_csv(outfile_path)
        print(f"Simulation Results saved to: {outfile_path}")
        return outfile_path, graph

    else:
        return base_df, graph