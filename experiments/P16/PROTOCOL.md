# P16: matched-deletion audit of frozen remedies

Written 2026-10-15 after the v43 review, before P16 model prediction. This is a review-motivated, post hoc diagnostic on reused test tasks, not a fresh holdout or public preregistration. No training, calibration, model selection or original result replacement is permitted.

## Inputs and matrix

Use the original corrected Scale-300 and Hard-confusable test manifests (300 tasks each, 300/299 groups), four selectors, seeds 7/21/42, and arms B, P14a/b/c, P15 M1/M12/FULL. Read original checkpoints and p15_thresholds.json; preserve their hashes. Archived NPZ paths may be resolved by the documented directory move; original file SHA256 must match the manifest. No input tensors are edited in place.

## Conditions

- present: original graph.
- deleted: remove the original target; fixed-original-referent label is null.
- matching: remove the first non-target relation-matching candidate in original node order, exactly as the older presence audit. Original target remains; remap its index.
- other: remove the first non-target nonmatching candidate, same rule as the older presence audit. Original target remains; remap its index.
- Report absent control availability, never substitute a different control. Scale has no matching candidate by construction. Do not discard any available condition after predictions.
- All removed conditions reduce K by one. For Hard, deleted and matching must have equal remaining K and equal remaining relation-match counts, checked within each source task before inference.
- Separately compute the public fine-rule target on the post-deletion graph. Re-resolution is eligible only if at least one relation-compatible candidate remains and the rule winner is unique (one match, or margin > 1e-5). Record raw ranking agreement under this alternate semantics; do not change the fixed-referent CAR score.

## Frozen operating points and outputs

Use B/b/M1 G3noK, a/c cal, M12 exist, FULL ltt from the saved P15 thresholds. No threshold refit. Primary outputs per condition: rejection rate, raw ranking accuracy where target is present, and accepted-correct rate using the full condition denominator. On deleted, report fixed-referent CAR; on matching/other/present, rejection is FRR. Keep the original 90% CAR/2% clean FRR pass flag distinct from the added control-aware check requiring all available target-present condition FRRs <=2% on both tracks. The latter is a new diagnostic, not a retrospective change to the original registered gate.

## Analysis and validation

All four selectors and all seven arms are reported. Average the three seed indicators per task, then use paired source-group bootstrap (5000 draws, RNG 20261015) for 95% descriptive intervals and for CAR(deleted)-FRR(matching) on Hard. Intervals condition on these three fitted models. No new superiority or multiplicity-adjusted significance claim; no post hoc winner selection. Report per-seed rates and full available denominators. Analyze the K-only control using the TRAIN present K support, without tuning on TEST. All current model inputs omit the original label, intervention condition, deleted identity and history; these fields are scoring metadata only.

Prediction parity: compare re-inferred present/deleted raw identities and primary decisions with the saved P14/P15 predictions; any mismatch is retained and diagnosed before interpretation. Hash source code/protocol/checkpoints/thresholds/data before P16 inference, and hash every prediction file before scoring. The output directory is new and all old artifacts remain read-only.
