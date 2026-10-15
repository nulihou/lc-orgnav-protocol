# Historical R2 split audit and frozen R3 correction

Audit date: 2026-10-02. Source data are unchanged under
`_data_classification_20260531/paper_used_data/runs/`.

The old signature included target/relation/policy fields and sometimes candidate
order, permitting one input graph with different queries in different splits.
This is same-G overlap; it is not evidence that identical (G, query, label)
samples were copied. The audit found no conflicting labels on identical (G,q).

| Track | Train-val matching G | Train-test matching G | Val-test matching G | Test tasks with historical training/validation G |
|---|---:|---:|---:|---:|
| Scale-300 | 26 | 9 | 4 | 13 / 300 |
| Hard-confusable | 46 | 18 | 2 | 20 / 300 |

All comparisons use the complete node, candidate-mask, edge-index and
edge-attribute tensors after a query/target-independent canonical node/edge
ordering. Hashes and complete duplicate groups are in each track's
`historical_duplicate_groups.json`; removed samples and reasons are represented
by original_split, group_owner_split and retained fields in
`removed_historical_samples.json`.

Concrete source examples (NPZ paths relative to the source run directory):

- Scale train `protocol_graphs/train/scen_2264.npz`, nodes `[7,125,8]`, front
  query, target spawn 8; test `protocol_graphs/test/scen_134.npz`, nodes
  `[7,8,125]`, behind query, target spawn 125. Same G after permutation.
- Hard train `protocol_graphs/train/scen_2026.npz`, nodes
  `[16,78,25,24,129]`, left/farthest query, target spawn 25; test
  `protocol_graphs/test/scen_191.npz`, nodes `[16,78,24,25,129]`,
  left/leftmost query, target spawn 24. Same G after permutation.
- Hard test tasks 27 and 160 share one G with different policy queries. They
  remain two tasks and form one statistical cluster in the corrected protocol.

| Track | Frozen train tasks / G groups | Frozen validation tasks / G groups | Frozen test tasks / G groups | Removed train / validation / test |
|---|---:|---:|---:|---:|
| Scale-300 | 2965 / 2915 | 796 / 793 | 300 / 300 | 35 / 4 / 0 |
| Hard-confusable | 2932 / 2838 | 798 / 792 | 300 / 299 | 68 / 2 / 0 |

All corrected cross-split intersections are zero for canonical-G hashes,
query/target-independent base-scene hashes and group IDs. Test labels, order,
queries and all tensors are preserved. Test and validation precedence is fixed
without reading model outputs, so no scene was filtered by an observed score.
There is no attempt to pad the corrected partitions to 3000/800.

The complete label, query-family, candidate-count, target-slot, map and weather
statistics appear in each `audit.json`. Scale's full G hash also merges one
within-training pair with different base-spawn signatures but equal relative
geometry; grouping is therefore stronger than checking spawn tuples alone.

Validation evidence on 2026-10-02: three regression checks passed; independent
verification loaded all 8091 retained NPZ samples, checked the SHA-256 hashes of
all 8200 original trainable NPZs and 8200 original protocol NPZs, and verified
all 26 per-track frozen artifact hashes. See `INDEPENDENT_VERIFICATION.json`.

Existing R2 trained checkpoints remain historical. Corrected-protocol results
require retraining. This split correction does not prove real-world driving,
cross-map transfer, perception/tracking robustness or sustained following.
