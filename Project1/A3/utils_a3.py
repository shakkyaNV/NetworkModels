import os, sys
import networkx as nx
import numpy as np

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


def export_graphml_with_namespace(G, output_path, xmlns_path=None):
    """
    Exports a NetworkX graph to GraphML with proper Gephi-compatible headers.

    Parameters
    ----------
    G : networkx.Graph
        The graph to export.
    output_path : str
        Path to save the GraphML file.
    xmlns_path : str, optional
    """
    nx.write_graphml(G, output_path)

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

    print(f"GraphML exported to: {output_path}")