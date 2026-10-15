# Static deployment audit and separately registered identity-binding follow-up

The formal audit registration and source hashes are frozen in each
`04_results/deployment/{track}/REGISTRATION.json` before test inference. Four
learned models use fixed training seed 42; checkpoints are selected on corrected
validation data. The two deterministic references are also included. Every
corrupted graph is identical across methods; source/input/checkpoint hashes and
all-case records retain adverse outcomes. Inference exceptions are not retried.

```powershell
& .venv -B -m unittest discover -s code/r3_deployment -p 'test_*.py' -v
& .venv -B code/r3_deployment/run_audit.py --root <repo-relative-path> --track hard_confusable --methods graph_relation_gate matched_geometry_mlp query_geometry_attention drivetransformer_task_adapted coarse_rule fine_rank_rule
```

Use `--methods` for the newly ready model only when training completes in stages.
An existing RUN_STARTED.json refuses a second formal execution. A separate
registration/source change requires an explicitly dated protocol amendment;
existing formal failures/results must be preserved. CPU single-thread batching
does not contend with GPU training; validation preflight verifies the public
single-sample predictor against batched choices and maximum softmax scores.

Each track has 300 tasks and 20,100 registered static payload cases (67 per
task), including clean, four position-noise levels with ten replicates, ten
distractor misses, one true-target miss, empty candidates, two unsupported-query
cases, ten node permutations, and valid/stale synthetic timestamp metadata.
The coordinate frame is recovered from original bearing fields. Corrupted node
geometry and all pairwise edge fields are recomputed from the same positions;
ego and anchor positions are fixed. Zero sigma retains exact original tensors.
The development check initially found a 1.75e-9 bearing roundoff from recovered
float32 yaw; fixed-node *positions* remain exact and geometric reconstruction
is checked at documented tolerance. No formal prediction was retried for this.

Maximum softmax is a raw neural ranking score. All 0.5/0.7/0.9 coverage/error
points are reported. An auxiliary threshold is selected only from clean
validation to maximize coverage subject to >=95% accepted accuracy; if no
threshold qualifies, 0.9 is used with explicit validation failure. No absent-
target validation/training or learned null head is introduced. Rules have no
softmax score, so threshold wrappers do not apply. Missing-target truth is
available only to corruption generation and retrospective scoring, never to
the validator/model's rejection decision.

Same-G query diagnostics generate the first distinct unique alternate target
using the declared deterministic geometry task definition. Ineligible source
tasks and reasons are disclosed. Hard uses same-relation policy flips; Scale
uses relation flips. These are additional diagnostics, distinct from the old
same-lane/left-lane Query-Flip benchmark. All original graph tensors are fixed.
No-Query intentionally bypasses schema validation for raw ranker diagnostics.
No-query task accuracy has an upper bound of 50%, and can be lower if the model
selects a third candidate. It is not necessarily exactly 50%.

Postprocess completed runs without new inference:

```powershell
& .venv -B code/r3_deployment/analysis/paired_summary.py --root <repo-relative-path> --track hard_confusable
```

The separately dated `02_protocol/SELECTED_IDENTITY_DISAPPEARANCE_FOLLOWUP_20261002.md`
registers a static two-payload contract. It holds only the clean selector's
predicted stable identity, returns TARGET_LOST/null if that ID disappears from
the observed candidate payload, and never auto-substitutes a new ID. Initial
wrong semantic identities can persist and are reported. Held-ID current
softmax thresholds use the same clean-validation selection and fixed points.
This reuses the original physical deletion records without model retraining or
new inference and leaves the formal target-missing errors unchanged.

```powershell
& .venv -B -m unittest discover -s code/r3_deployment/followup -p 'test_*.py' -v
& .venv -B code/r3_deployment/followup/selected_identity_contract.py --root <repo-relative-path> --track hard_confusable --methods graph_relation_gate matched_geometry_mlp query_geometry_attention drivetransformer_task_adapted coarse_rule fine_rank_rule
```

Static permutation is not temporal tracking. Declared timestamp tests are not
measured timing robustness. The identity binding contract is not a real tracker,
loss/reacquisition or safety experiment. Input-corruption results cannot claim
actual perception accuracy, sustained following, planning safety or real-world
deployment. The planner retains all surrounding obstacles and owns feasibility
and authorization for a new target.
