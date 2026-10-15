# R3 graph-disjoint protocol correction

Run from the repository root (the directory that contains `code/`) with NumPy available:

```powershell
python -B -m unittest discover -s code/r3_protocol -p "test_*.py" -v
python -B code/r3_protocol/freeze_graph_disjoint.py --workspace . --output ./protocol_freeze
python -B code/r3_protocol/verify_frozen_protocol.py --workspace . --protocol-root ./protocol_freeze
```

The builder reads only original scenarios and protocol/trainable NPZs, never
predictions, checkpoints or closed-loop logs. Original files are not modified.
Test wins over validation, which wins over training. Canonical G excludes query
and target; the base-scene key also excludes query/target and candidate order.
Matching either key joins the tasks into one group before split precedence.
All queries of a retained group remain together. Repeated tasks within a split
are retained; tasks removed from lower-priority splits are explicitly archived.

The regression tests reproduce the original query-sensitive grouping failure,
check permutation invariance, verify split precedence, and fail invalid target
masks and nonunique canonical node keys. The independent verifier checks all
original NPZ hashes, the persisted artifact hashes, manifest/list equivalence,
every retained NPZ label/query/mask, all cross-split graph/group/base keys, and
unchanged original test task labels/order. It does not import the builder.

Training interfaces per track in `02_protocol/{scale300,hard_confusable}/`:

- `lists/{train,val,test}.txt`: absolute unchanged trainable NPZ paths.
- `{train,val,test}_manifest.jsonl`: complete frozen record/label/strata hashes.
- `{train,val,test}.jsonl`: byte-identical aliases for the baseline loader.
- `group_id`: statistical cluster; hard-confusable test has 299 groups for 300 tasks.
- `npz_path_relative_workspace`: portable source reference; `npz_path` is absolute.
- `SHA256SUMS.json`: hashes for frozen artifacts and both original NPZ versions.

The correction changes training and validation membership, so all learned
models must be retrained. Old checkpoint scores are historical and must not be
reported as corrected-protocol scores. Test labels are available for final
scoring only after validation-based model/hyperparameter selection is frozen.
