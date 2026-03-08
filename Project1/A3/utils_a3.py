import os, sys
from math import inf
import pickle
from collections import defaultdict

import networkx as nx
import numpy as np, pandas as pd

## MODEL WIDE VARIABLES
MODULE_DIR = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(MODULE_DIR)))
RESOURCES_DIR = os.path.join(BASE_DIR, 'resources')

DK_FSNAMES_MAPPING_DICT = {'rh.lateralorbitofrontal': 'rh_lateralorbitofrontal',
 'rh.parsorbitalis': 'rh_parsorbitalis',
 'rh.frontalpole': 'rh_frontalpole',
 'rh.medialorbitofrontal': 'rh_medialorbitofrontal',
 'rh.parstriangularis': 'rh_parstriangularis',
 'rh.parsopercularis': 'rh_parsopercularis',
 'rh.rostralmiddlefrontal': 'rh_rostralmiddlefrontal',
 'rh.superiorfrontal': 'rh_superiorfrontal',
 'rh.caudalmiddlefrontal': 'rh_caudalmiddlefrontal',
 'rh.precentral': 'rh_precentral',
 'rh.paracentral': 'rh_paracentral',
 'rh.rostralanteriorcingulate': 'rh_rostralanteriorcingulate',
 'rh.caudalanteriorcingulate': 'rh_caudalanteriorcingulate',
 'rh.posteriorcingulate': 'rh_posteriorcingulate',
 'rh.isthmuscingulate': 'rh_isthmuscingulate',
 'rh.postcentral': 'rh_postcentral',
 'rh.supramarginal': 'rh_supramarginal',
 'rh.superiorparietal': 'rh_superiorparietal',
 'rh.inferiorparietal': 'rh_inferiorparietal',
 'rh.precuneus': 'rh_precuneus',
 'rh.cuneus': 'rh_cuneus',
 'rh.pericalcarine': 'rh_pericalcarine',
 'rh.lateraloccipital': 'rh_lateraloccipital',
 'rh.lingual': 'rh_lingual',
 'rh.fusiform': 'rh_fusiform',
 'rh.parahippocampal': 'rh_parahippocampal',
 'rh.entorhinal': 'rh_entorhinal',
 'rh.temporalpole': 'rh_temporalpole',
 'rh.inferiortemporal': 'rh_inferiortemporal',
 'rh.middletemporal': 'rh_middletemporal',
 'rh.bankssts': 'rh_bankssts',
 'rh.superiortemporal': 'rh_superiortemporal',
 'rh.transversetemporal': 'rh_transversetemporal',
 'rh.insula': 'rh_insula',
 'Right-Thalamus-Proper': 'rh_thalamus_proper',
 'Right-Caudate': 'rh_caudate',
 'Right-Putamen': 'rh_putamen',
 'Right-Pallidum': 'rh_pallidum',
 'Right-Accumbens-area': 'rh_accumbens_area',
 'Right-Hippocampus': 'rh_hippocampus',
 'Right-Amygdala': 'rh_amygdala',
 'lh.lateralorbitofrontal': 'lh_lateralorbitofrontal',
 'lh.parsorbitalis': 'lh_parsorbitalis',
 'lh.frontalpole': 'lh_frontalpole',
 'lh.medialorbitofrontal': 'lh_medialorbitofrontal',
 'lh.parstriangularis': 'lh_parstriangularis',
 'lh.parsopercularis': 'lh_parsopercularis',
 'lh.rostralmiddlefrontal': 'lh_rostralmiddlefrontal',
 'lh.superiorfrontal': 'lh_superiorfrontal',
 'lh.caudalmiddlefrontal': 'lh_caudalmiddlefrontal',
 'lh.precentral': 'lh_precentral',
 'lh.paracentral': 'lh_paracentral',
 'lh.rostralanteriorcingulate': 'lh_rostralanteriorcingulate',
 'lh.caudalanteriorcingulate': 'lh_caudalanteriorcingulate',
 'lh.posteriorcingulate': 'lh_posteriorcingulate',
 'lh.isthmuscingulate': 'lh_isthmuscingulate',
 'lh.postcentral': 'lh_postcentral',
 'lh.supramarginal': 'lh_supramarginal',
 'lh.superiorparietal': 'lh_superiorparietal',
 'lh.inferiorparietal': 'lh_inferiorparietal',
 'lh.precuneus': 'lh_precuneus',
 'lh.cuneus': 'lh_cuneus',
 'lh.pericalcarine': 'lh_pericalcarine',
 'lh.lateraloccipital': 'lh_lateraloccipital',
 'lh.lingual': 'lh_lingual',
 'lh.fusiform': 'lh_fusiform',
 'lh.parahippocampal': 'lh_parahippocampal',
 'lh.entorhinal': 'lh_entorhinal',
 'lh.temporalpole': 'lh_temporalpole',
 'lh.inferiortemporal': 'lh_inferiortemporal',
 'lh.middletemporal': 'lh_middletemporal',
 'lh.bankssts': 'lh_bankssts',
 'lh.superiortemporal': 'lh_superiortemporal',
 'lh.transversetemporal': 'lh_transversetemporal',
 'lh.insula': 'lh_insula',
 'Left-Thalamus-Proper': 'lh_thalamus_proper',
 'Left-Caudate': 'lh_caudate',
 'Left-Putamen': 'lh_putamen',
 'Left-Pallidum': 'lh_pallidum',
 'Left-Accumbens-area': 'lh_accumbens_area',
 'Left-Hippocampus': 'lh_hippocampus',
 'Left-Amygdala': 'lh_amygdala',
 'Brain-Stem': 'brainstem'}

NODE_FSREGION_TO_ID = {'rh_lateralorbitofrontal': 1,
 'rh_parsorbitalis': 2,
 'rh_frontalpole': 3,
 'rh_medialorbitofrontal': 4,
 'rh_parstriangularis': 5,
 'rh_parsopercularis': 6,
 'rh_rostralmiddlefrontal': 7,
 'rh_superiorfrontal': 8,
 'rh_caudalmiddlefrontal': 9,
 'rh_precentral': 10,
 'rh_paracentral': 11,
 'rh_rostralanteriorcingulate': 12,
 'rh_caudalanteriorcingulate': 13,
 'rh_posteriorcingulate': 14,
 'rh_isthmuscingulate': 15,
 'rh_postcentral': 16,
 'rh_supramarginal': 17,
 'rh_superiorparietal': 18,
 'rh_inferiorparietal': 19,
 'rh_precuneus': 20,
 'rh_cuneus': 21,
 'rh_pericalcarine': 22,
 'rh_lateraloccipital': 23,
 'rh_lingual': 24,
 'rh_fusiform': 25,
 'rh_parahippocampal': 26,
 'rh_entorhinal': 27,
 'rh_temporalpole': 28,
 'rh_inferiortemporal': 29,
 'rh_middletemporal': 30,
 'rh_bankssts': 31,
 'rh_superiortemporal': 32,
 'rh_transversetemporal': 33,
 'rh_insula': 34,
 'rh_thalamus_proper': 35,
 'rh_caudate': 36,
 'rh_putamen': 37,
 'rh_pallidum': 38,
 'rh_accumbens_area': 39,
 'rh_hippocampus': 40,
 'rh_amygdala': 41,
 'lh_lateralorbitofrontal': 42,
 'lh_parsorbitalis': 43,
 'lh_frontalpole': 44,
 'lh_medialorbitofrontal': 45,
 'lh_parstriangularis': 46,
 'lh_parsopercularis': 47,
 'lh_rostralmiddlefrontal': 48,
 'lh_superiorfrontal': 49,
 'lh_caudalmiddlefrontal': 50,
 'lh_precentral': 51,
 'lh_paracentral': 52,
 'lh_rostralanteriorcingulate': 53,
 'lh_caudalanteriorcingulate': 54,
 'lh_posteriorcingulate': 55,
 'lh_isthmuscingulate': 56,
 'lh_postcentral': 57,
 'lh_supramarginal': 58,
 'lh_superiorparietal': 59,
 'lh_inferiorparietal': 60,
 'lh_precuneus': 61,
 'lh_cuneus': 62,
 'lh_pericalcarine': 63,
 'lh_lateraloccipital': 64,
 'lh_lingual': 65,
 'lh_fusiform': 66,
 'lh_parahippocampal': 67,
 'lh_entorhinal': 68,
 'lh_temporalpole': 69,
 'lh_inferiortemporal': 70,
 'lh_middletemporal': 71,
 'lh_bankssts': 72,
 'lh_superiortemporal': 73,
 'lh_transversetemporal': 74,
 'lh_insula': 75,
 'lh_thalamus_proper': 76,
 'lh_caudate': 77,
 'lh_putamen': 78,
 'lh_pallidum': 79,
 'lh_accumbens_area': 80,
 'lh_hippocampus': 81,
 'lh_amygdala': 82,
 'brainstem': 83}

## FOR ADNI PATIENT DATA FILES

def df_rename_to_fsnames(df_path:str, query_filter=None, type_filter:str="amyloid"):
    """
    Return the fsnames renamed ADNI data file
    :param type_filter: Indicating whether it's for amyloid or tau
    :param df_path: Relative path from BASE directory (../NetworkModels/)
    :param query_filter: Query string for df.query(query) to filter for testing
    :return: pd.DataFrame
    """
    df = pd.read_csv(os.path.join(BASE_DIR, df_path))
    # try:
    #     df_suffix =
    #     dnnames_mapping_dir = os.path.join(BASE_DIR, df_path.removesuffix("_suvr.csv|_volume.csv") + "_mapping.csv")
    #     dnnames_mapping = pd.read_csv(dnnames_mapping_dir)
    # except:
    # ## IF you can't find a .csv with ("_suvr" subbed to "_mappig") go with utilsa3_dict
    map_dict = DK_FSNAMES_MAPPING_DICT
    df.rename(columns=map_dict, inplace=True)

    if not query_filter:
        return df

    try:
        subset = df.query(query_filter)
    except Exception as e:
        print(f"Error in running Query filter. Returning unfiltered Dataset:"
              f"Query: {query_filter}, "
              f"Error: {e}")
    else:
        return subset

def safe_filter_df(df:pd.DataFrame, filter_all:bool=True, type_filter:str="amyloid"):
    """
    Filter all non-safe entries from the ADNI data frame.
    Eg: Remove qc_flag <= 0, any patients that use NAV tracer, {to be implemented, any patient that use many tracers}
    :param type_filter: Indicator for amyloid or tau
    :param df: ADNI dataset renamed accordingly
    :param filter_all: Bool specifying to filter all non-safe entries. {False, return the as is dataset}
    :return: pd.DataFrame
    """
    tracer_list = []
    if type_filter == "amyloid":
        tracer_list = ['FBP', 'FBB']
    elif type_filter == "tau":
        tracer_list = ['FTP']

    if filter_all:
        df_new = df.loc[(df['qc_flag'] >= 0) & (df['tracer'].isin(tracer_list))] #make sure to use .loc/.iloc or copy
        df_new = df_new.drop_duplicates(subset = 'loniuid', keep='first')               # Base remove duplicate entries based on loniuid
        df_new = df_new.dropna(how = 'any') # drop any column that has NA
        return df_new
    else:
        return df

def activations_cortical_regions_df(df:pd.DataFrame, base_setup:bool=True, tracer_dictionary: dict=None):
    """
    Apply base thresholding [Based on whole cerebellum referencing, should not be used for coritical regions (but we're
    going to use it anyway), should not be used to compare across tracer (but we're going to use it anyway)]
    :param df: ADNI dataset renamed accordingly (filtered or unfiltered) {to be implemented, not checking for NAV for the moment}
    :param base_setup: Bool specifiying to apply base thresholding. Also assume it went through utils.safe_filter_df {False, return the as is dataset}
    :param tracer_dictionary: Dictionary of tracer names as keys and lists of tracer names as values, Old values for tracers will be superseded if the same tracer name is provided
    :return: pd.DataFrame with tracer_threshold column specifying tracer tresholds {against better judgement}, and suvr values turned to 1/0 values
    """
    multipliers = {"FBP": 1.11, "FBB": 1.08, "FTP": 1.27, "MK6240":1.27, "PI2620": 1.27}
    if tracer_dictionary:
        multipliers = {**multipliers, **tracer_dictionary}

    if base_setup:
        df.loc[:, 'tracer_threshold'] = df['tracer'].apply(lambda x: multipliers.get(x, 1.0))
        for col in DK_FSNAMES_MAPPING_DICT.values():
            df.loc[:, f'{col}_positivity'] = df.apply(lambda row: 1 if row[col] > row['tracer_threshold'] else 0, axis=1)
        positivity_indicator_columns = [col for col in df.columns if col.endswith("_positivity")]
        return df, positivity_indicator_columns
    else:
        return df, DK_FSNAMES_MAPPING_DICT.values()

def activation_times_of_patients_for_cortical_regions_df(df:pd.DataFrame, feature_cols:list, base_setup:bool=True, save_files:bool=False, save_files_path:str=None,
                                                         type_filter:str="amyloid"):
    """
    Returns dict of activation times and snapshots of activation per patient
    :param df: DataFrame
    :param feature_cols: list of feature column names {if went through utils.activations_cortical_regions_df it should match utils.DK_FSNAMES_MAPPING_DICT_positivity
    :param base_setup: Bool specificying that dataset went through utils.safe_filter_df and utils.activations_cortical_regions_df
    :param save_files: Bool specifying whether to save files to disk (format pickle)
    :param save_files_path: Directory (folder only) specifying path to save files to disk, relative to Base_Dir: os.path.join(utils.BASE_DIr, <path>)
    :param type_filter: Indicator for amyloid or tau
    :return: dict of activations and dict of snapshots
    """
    if type_filter.lower() == "amyloid":
        state_value_col = 'amyloid_status'
    elif type_filter.lower() == "tau":
        state_value_col = 'summary_diagnosis'

    if base_setup:
        assert [col.removesuffix("_positivity") for col in df.columns if col.endswith("positivity")] == list(
            NODE_FSREGION_TO_ID.keys())

        df.sort_values(by=['rid', 'loniuid'], inplace=True)        # Technically you should sort by 'rid' and 'scandate' # duplicate loniuids must be removed by now
        grouped = df.groupby(['rid'], as_index=True)

        snapshots = dict()  # store results per group
        activation_times = dict()
        state_value = dict()
        feature_cols = feature_cols
        for group_key, group_df in grouped:
            # drop 'rid' and 'loniuid' if they are in columns (they will be index now)
            matrix = group_df[feature_cols].values  # shape [n_rows, n_features], only the 0/1 columns
            n_rows, n_cols = matrix.shape
            cumulative_set = set()
            group_sets = []

            # here we are trying to map at which Nth time step (n_row of patient data), the specic region lits up.
            # so we start with setting everything to -1, and if it gets lit up in the first scan we note those cols as 0
            active_t = np.full(n_cols, -1)      ##### Check with watts_model and test_WTM (0 or -1 -> check GP.persistence)

            for i in range(n_rows):
                # get indices of 1s in current row
                row_indices = set(map(int, np.where(matrix[i])[0]))

                # for newly_activated nodes
                newly_activated = row_indices - cumulative_set
                active_t[list(newly_activated)] = i

                # cumulative union (because we need cumulative activated nodes in next iteration
                # also we assume that the node amyloid/Tau content don't regress (but in practicality it does)
                cumulative_set = cumulative_set.union(row_indices)
                group_sets.append(cumulative_set.copy())  # copy so each row has its own set

            snapshots[group_key[0]] = group_sets
            activation_times[group_key[0]] = active_t

            # We'll take the overall amyloid positivity (with whole cerebellum ref) as state value
            state_value[group_key[0]] = group_df[state_value_col].values

            # we'll remove any entries that don't have any activations throughout it's scans
            if np.nanmax(activation_times[group_key[0]]) < 0:
                print(f"Removing RID Patient Data {group_key[0]} as they have single record with no activations or TODO")
                activation_times.pop(group_key[0], None)
                state_value.pop(group_key[0], None)
                snapshots.pop(group_key[0], None)

        if save_files:
            save_files_path = os.path.abspath(os.path.join(BASE_DIR, save_files_path))
            # MAPPING FILE TO SEE IF THE PICKLE FILE RETAINED ORDER (and for pytest)
            df_rid_loniuid = df[['rid', 'loniuid', 'scandate']]
            df_rid_loniuid.to_csv(os.path.join(save_files_path, "df_rid_lonuid.csv"), index = False)

            # Save other files to PICKLE
            with open(os.path.join(save_files_path, "activation_times.pkl"), 'wb') as f:
                pickle.dump(activation_times, f)

            with open(os.path.join(save_files_path, "state_values.pkl"), 'wb') as f:
                pickle.dump(state_value, f)

            with open(os.path.join(save_files_path, "snapshots.pkl"), 'wb') as f:
                pickle.dump(snapshots, f)

        return activation_times, snapshots, state_value
    else:
        return None, None, None

def pull_saved_pickle_file(path):
    """
    Pull saved pickle file from path
    :param path: Pickle file path (saved from dict)
    :return: Dict
    """
    with open(path, 'rb') as f:
        res_dict = pickle.load(f)
        print(f"Pickle file loaded from {path}")
    return res_dict

def _pull_saved_patient_data_files(activations_path=None, snapshots_path=None, state_values_path=None):
    """
    Pull saved patient data files from path in Order
    Node Activations dict(rid: [array of activation]), # length = 83
    Snapshots dict(rid: [{set1}, {set2}, {set3}]), # length should be #loniuid per rid
    State Values dict(rid: [0/1, 0/1, 0/1]) # length should be #loniuid per rid
    :param activations_path:
    :param snapshots_path:
    :param state_values_path:
    :return: dict(activation_times), dict(snapshots), dict(state_values)
    """
    activations, snapshots, state_values = None, None, None

    if activations_path is not None:
        activations = pull_saved_pickle_file(activations_path)
    if snapshots_path is not None:
        snapshots = pull_saved_pickle_file(snapshots_path)
    if state_values_path is not None:
        state_values = pull_saved_pickle_file(state_values_path)

    return activations, snapshots, state_values

### FOR GRAPH DF's

def node_df_from_graph(graph:nx.Graph) -> pd.DataFrame:
    """
    Return a pandas data frame with node details
    :param graph: nx.Graph from dkatlas
    :return: pd.DataFrame with node details
    """
    return  pd.DataFrame.from_dict(dict(graph.nodes(data = True)), orient = 'index')

def edge_df_from_graph(graph:nx.Graph) -> pd.DataFrame:
    """
    Return a pandas data frame with edge details
    :param graph: nx.Graph from dkatlas
    :return: pd.DataFrame with edge details
    """
    graph_edge_df = pd.DataFrame(list(graph.edges(data = True)), columns = ['source', 'target', 'attributes'])
    # graph_edge_df = graph_edge_df.astype({'source':'int64', 'target':'int64'})
    assert (graph_edge_df[['source', 'target']].dtypes == ['int64', 'int64']).all(), "Relabelling Failed"

    if graph_edge_df['attributes'].apply(lambda x: isinstance(x, dict)).all():
        attr_df = pd.json_normalize(graph_edge_df['attributes'])
        graph_edge_df = pd.concat([graph_edge_df[['source', 'target']], attr_df], axis = 1)
    return graph_edge_df

def determine_geom_ngeom_edges(edge_df: pd.DataFrame, fiber_max_geom_length:int=50, base_method:bool=True, ) -> list:
    """
    Return a list compatible for [(u1, v1, w1), (u2, v2, w2), ...] for graph.add_weighted_edges_from but for geometric
    and non_geometric edge classification
    :param edge_df: Edge Details data from utils.edge_df_from_graph
    :param fiber_max_geom_length: Max distance to be classified as geometric distance
    :param base_method: Bool specifying to use Fiber_mean_length>{value} to determine geom/ngeom
    :return: [(u1, v1, w1), (u2, v2, w2), ...] for graph.add_weighted_edges_from
    """
    edge_df_ = edge_df.copy()
    edge_df_.loc[:, 'geom_type'] = edge_df_['fiber_length_mean'].apply(
        lambda fiber_length: "geometric" if fiber_length < fiber_max_geom_length else "non_geometric")
    geom_ngeom_edges = [(row.source, row.target, row.geom_type) for row in edge_df_.itertuples()]
    return geom_ngeom_edges

## FOR GRAPH VISUALIZATION


def remove_properties(graph: nx.Graph, node_properties_to_remove:list=None, edge_properties_to_remove:list=None):
    """
    Removes specified node and edge properties from a graph.
    (Also compatible with DK.graph)
    :param graph: (networkx.Graph or networkx.DiGraph): The graph to modify.
    :param node_properties_to_remove: (list): List of node properties to remove.
    :param edge_properties_to_remove: (list): List of edge properties to remove.

    Returns:
    G (networkx.Graph or networkx.DiGraph): The modified graph.
    """

    # Remove node properties
    if node_properties_to_remove:
        for _, attrs in graph.nodes(data=True):
            for key in node_properties_to_remove:
                attrs.pop(key, None)

    # Remove edge properties
    if edge_properties_to_remove:
        for _, _, attrs in graph.edges(data=True):
            for key in edge_properties_to_remove:
                attrs.pop(key, None)

    return graph

def export_graphml_with_namespace(graph: nx.Graph, output_path:str, xmlns_path:str=None):
    """
    Exports a NetworkX graph to GraphML with proper Gephi-compatible headers.
    :param graph : networkx.Graph to export.
    :param output_path : Path to save the files relative to $BASE$ Directory
    :param xmlns_path : str, optional
    """
    output_path = os.path.abspath(os.path.join(BASE_DIR, output_path))
    nx.write_graphml(graph, output_path)

    # Patch the <graphml> header and include schema info if provided
    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()

    if xmlns_path:
        xmlns_header = (
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n'
            '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
            f'         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns {xmlns_path}">'
        )
        content = content.replace("<graphml>", xmlns_header)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"GraphML exported to: {os.path.join(BASE_DIR, output_path)}")


### FOR PERSISTENCE
def persistence_for_graphics_to_betti_nums(persistence_for_graphics):
    births = [b for _, (b, d) in persistence_for_graphics]
    deaths = [d if d != inf else max(births) + 1 for _, (b, d) in persistence_for_graphics]
    t_min = 0
    t_max = int(np.ceil(max(deaths)))
    counts = defaultdict(lambda: defaultdict(int))

    # Iterate over each timestep
    for t in range(t_min, t_max + 1):
        for dim, (b, d) in persistence_for_graphics:
            if b <= t < d or (d == inf and b <= t):
                counts[t][dim] += 1

    return counts