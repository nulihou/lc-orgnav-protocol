import json
from pathlib import Path
import unittest

import numpy as np

from transforms import (alternate_query, delete_node, graph_hash, load_graph,
                        permute_nodes, position_noise, recover_frame, validate_interface)


class TransformationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workspace = Path(__file__).resolve().parents[3]
        path = workspace / "R3_revision_20261002/02_protocol/hard_confusable/test.jsonl"
        cls.row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        cls.graph = load_graph(cls.row["npz_path"])

    def test_zero_noise_exact_and_noisy_edges_consistent(self):
        original = self.graph
        self.assertEqual(graph_hash(original), graph_hash(position_noise(original, 0, 20261002, 0)))
        noisy = position_noise(original, .5, 20261002, 0)
        recover_frame(noisy)
        src, dst = noisy["edge_index"]
        np.testing.assert_allclose(noisy["edge_attr"][:, :2], noisy["node_feats"][dst, :2]-noisy["node_feats"][src, :2], atol=1e-5)
        frame = recover_frame(original)
        fixed = [frame["ego_index"], frame["anchor_index"]]
        np.testing.assert_array_equal(noisy["node_feats"][fixed, :2], original["node_feats"][fixed, :2])
        # Recovering yaw from float32 sine/cosine permits roundoff in fixed-node
        # bearing channels; the actual fixed positions must remain exact.
        np.testing.assert_allclose(noisy["node_feats"][fixed], original["node_feats"][fixed], atol=1e-6)

    def test_permutation_remaps_edges_and_identity(self):
        order = np.arange(len(self.graph["node_feats"]))[::-1]
        permuted, mapping = permute_nodes(self.graph, order)
        target = self.row["target_index"]
        self.assertEqual(permuted["node_spawn_indices"][mapping[target]], self.graph["node_spawn_indices"][target])
        np.testing.assert_array_equal(permuted["edge_index"], mapping[self.graph["edge_index"]])
        recover_frame(permuted)

    def test_target_deletion_remaps_incident_edges_without_truth_validator(self):
        target = self.row["target_index"]
        deleted, mapping = delete_node(self.graph, target)
        self.assertEqual(mapping[target], -1)
        self.assertNotIn(self.graph["node_spawn_indices"][target], deleted["node_spawn_indices"])
        src, dst = self.graph["edge_index"]
        edge_keep = (src != target) & (dst != target)
        np.testing.assert_array_equal(deleted["edge_index"], mapping[self.graph["edge_index"][:, edge_keep]])
        self.assertEqual(validate_interface(deleted, "hard_confusable"), "VALID")

    def test_query_flip_derived_from_unique_geometry_and_structural_rejections(self):
        alternate, reason = alternate_query(self.graph, "hard_confusable", self.row["target_index"])
        self.assertIsNone(reason)
        changed, target, proof = alternate
        self.assertNotEqual(target, self.row["target_index"])
        for key in ("node_feats", "edge_index", "edge_attr", "candidate_mask"):
            np.testing.assert_array_equal(changed[key], self.graph[key])
        self.assertGreaterEqual(len(proof["relation_matches"]), 2)
        empty = {k:v.copy() for k,v in self.graph.items()}; empty["candidate_mask"][:] = False
        self.assertEqual(validate_interface(empty, "hard_confusable"), "NO_CANDIDATE")
        unknown = {k:v.copy() for k,v in self.graph.items()}; unknown["lang_feat"][:9] = 0
        self.assertEqual(validate_interface(unknown, "hard_confusable"), "UNSUPPORTED_QUERY")
        self.assertEqual(validate_interface(self.graph, "hard_confusable", 500), "STALE_INPUT")


if __name__ == "__main__":
    unittest.main()
