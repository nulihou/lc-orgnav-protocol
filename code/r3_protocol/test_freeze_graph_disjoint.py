"""Regression tests for graph grouping and outcome-blind split precedence."""
import unittest

import numpy as np

from freeze_graph_disjoint import assign_groups, canonical_graph, intersections


def graph():
    return {"node_feats": np.asarray([[0., 0.], [2., 1.], [-1., 3.]], dtype=np.float32),
            "candidate_mask": np.asarray([False, True, True]),
            "edge_index": np.asarray([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=np.int64),
            "edge_attr": np.asarray([[2., 1.], [-2., -1.], [-1., 3.], [1., -3.]], dtype=np.float32),
            "target_idx_gt": np.asarray(1), "target_index": np.asarray(1),
            "lang_feat": np.asarray([1., 0.], dtype=np.float32)}


class FreezeChecks(unittest.TestCase):
    def test_query_flip_and_node_order_do_not_create_a_new_graph(self):
        original = graph()
        flipped = {k: v.copy() for k, v in original.items()}
        order = np.asarray([2, 0, 1])
        inverse = np.argsort(order)
        for key in ("node_feats", "candidate_mask"):
            flipped[key] = flipped[key][order]
        flipped["edge_index"] = inverse[flipped["edge_index"]][:, ::-1]
        flipped["edge_attr"] = flipped["edge_attr"][::-1]
        flipped["lang_feat"] = np.asarray([0., 1.], dtype=np.float32)
        flipped["target_index"] = flipped["target_idx_gt"] = np.asarray(inverse[2])
        self.assertEqual(canonical_graph(original)[0], canonical_graph(flipped)[0])
        self.assertNotEqual(canonical_graph(original)[1], canonical_graph(flipped)[1])
        changed = graph()
        changed["node_feats"][1, 0] += .25
        self.assertNotEqual(canonical_graph(original)[0], canonical_graph(changed)[0])

    def test_test_wins_and_validation_wins_without_query_based_grouping(self):
        records = [dict(original_split=s, canonical_graph_sha256=g, canonical_base_scene_sha256=b)
                   for s, g, b in [("train", "a", "A"), ("val", "a", "A"),
                                   ("train", "b", "B"), ("test", "b", "B"),
                                   ("test", "b", "B"), ("train", "c", "C")]]
        assign_groups(records)
        self.assertEqual([r["retained"] for r in records], [False, True, False, True, True, True])
        self.assertEqual(records[3]["group_id"], records[4]["group_id"])
        splits = {s: [r for r in records if r["split"] == s] for s in ("train", "val", "test")}
        self.assertFalse(any(n for pair in intersections(splits).values() for n in pair.values()))

    def test_target_mask_and_duplicate_node_keys_fail_closed(self):
        invalid = graph()
        invalid["target_index"] = invalid["target_idx_gt"] = np.asarray(0)
        with self.assertRaisesRegex(ValueError, "target"):
            canonical_graph(invalid)
        duplicate = graph()
        duplicate["node_feats"][2] = duplicate["node_feats"][1]
        with self.assertRaisesRegex(ValueError, "unique"):
            canonical_graph(duplicate)


if __name__ == "__main__":
    unittest.main()
