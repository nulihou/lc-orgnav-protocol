# Frozen R3 hard_confusable protocol (2026-10-02)

The test task sequence and all NPZ tensors/labels are unchanged from R2. This is
an outcome-blind correction: no fitted predictions or online outcomes are read.
Precedence is test, then validation, then training. A task is removed from a
lower-priority partition if its full canonical graph OR query/label-independent
base scene appears in a higher-priority partition. All queries of a retained
group remain in one partition; within-partition query diversity is preserved.

| Partition | Historical tasks | Frozen tasks | Frozen graph clusters | Removed |
|---|---:|---:|---:|---:|
| train | 3000 | 2932 | 2838 | 68 |
| val | 800 | 798 | 792 | 2 |
| test | 300 | 300 | 299 | 0 |


Use lists/train.txt, lists/val.txt and lists/test.txt with the existing dataset
loader. Each line is an absolute path to an unchanged original trainable NPZ.
The JSONL manifests also retain workspace-relative paths for relocation and
SHA-256 source checks. All original graph/query/target arrays remain readable.
The short train.jsonl/val.jsonl/test.jsonl names are exact manifest aliases.

Canonical G consists of the complete node_feats, candidate_mask, edge_index and
edge_attr tensors. Query vector, targets, labels, IDs and source split do not
enter the G hash. Node rows are ordered by complete feature values and candidate
status; edge rows are sorted after remapping. Nonunique node keys fail closed.
The base-scene key is (town, ego spawn, anchor spawn, sorted candidate spawns),
independent of candidate order, target and relation/policy query. Groups join
either matching G hashes or matching base-scene hashes across all partitions.

Validation alone selects models/hyperparameters. Test is read for final scoring
after selection; test labels must never guide model selection, scene filtering
or retry decisions. No missing sample is replaced and no denominator is padded.
Training seeds do not create independent test scenes. Closed-loop seeds are
repeated measurements nested within group_id; statistical clusters use this ID.
Hard-confusable retains 300 query tasks on 299 independent graph groups, including
the two different queries on one graph. Scale-300 retains 300 graph groups.

This correction removes historical train/validation/test graph overlap. It does
not establish unseen maps, weather, perception, tracking or real-world validity;
all source tasks still originate in the same controlled CARLA protocol. Map,
query, policy, candidate-count and target-slot distributions are in audit.json.
Historical duplicate groups and every removed task are archived next to it.
