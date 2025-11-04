from unittest import TestCase

from Project1.A3.desikan_killiany_atlas import DKAtlasGraph
from Project1.A3.utils_a3 import  DK_FSNAMES_MAPPING_DICT, BASE_DIR
import os, pandas as pd, numpy as np

class TestDKAtlasGraph(TestCase):
    def setUp(self):
        self.DK = DKAtlasGraph()
        df = pd.read_csv(os.path.join(BASE_DIR,
                                      "resources/adni_pet_image_analysis/structured_files_UCBERKELEY_AMY_6MM_29Oct2025/UCBERKELEY_AMY_6MM_29Oct2025_suvr.csv"))
        self.df = df
        map_dict = DK_FSNAMES_MAPPING_DICT
        df.rename(columns=map_dict, inplace=True)
        selected_patient_RID = 4214
        df_id = df[(df['rid'] == selected_patient_RID) & (df['loniuid'] == 1598059)].index[0]
        self.df_id = df_id
        pd_series = df.iloc[df_id]
        self.DK.input_patient_data(pd_series, df_type="suvr")
        self.graph = self.DK.graph

    def test__load_graphml(self):
        node_ids = set(self.graph.nodes())
        self.assertSetEqual(set(range(1, 84)), node_ids)
        all(self.assertIsInstance(n, int) for n in self.graph.nodes())


    def test_rename_nodes(self):
        node_naming_bool = []
        for node in self.graph.nodes():
            node_naming_bool.append(True) if self.graph.nodes[node]['region_name'] == DK_FSNAMES_MAPPING_DICT[self.graph.nodes[node]['dn_name']] else node_naming_bool.append(False)
        self.assertTrue(all(node_naming_bool))
        self.assertEqual(self.graph.nodes()[83]['region_name'], 'brainstem')
        self.assertEqual(self.graph.nodes()[35]['region_name'], 'rh_thalamus_proper')

    def test_assign_node_activation(self):
        self.skipTest(reason="Not implemented")

    def test_assign_edge_weight(self):
        self.skipTest(reason="Not implemented")

    def test_patient_data(self):
        self.assertEqual(self.graph.nodes()[29]['suvr'], 1.004)
        self.assertTrue(
            all(
            [True if self.graph.nodes()[node]['suvr'] == self.df.iloc[self.df_id, node + 2] else False for node in self.graph.nodes()]
            )
        )

    def test_summary(self):
        summary = self.DK.summary()
