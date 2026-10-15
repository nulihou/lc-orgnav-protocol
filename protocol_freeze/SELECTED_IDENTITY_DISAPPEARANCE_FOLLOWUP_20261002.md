# Separately registered engineering follow-up: previous selected identity disappears

Registration date: 2026-10-02. Trigger: the registered static target-missing
audit issued a non-null wrong target in every raw forced-choice case for the
initial completed Scale-300 methods. These adverse formal results are retained.
This follow-up changes an interface binding rule, not the trained selector.

Use the unchanged source NPZs and already registered physical-node deletion
cases, including all ten distractor-removal replicates and the single target
removal per task. Do not regenerate, replace or filter cases by any score.
Checkpoints and initial clean predictions are the same fixed seed-42 versions.

Synthetic t0: store the selector's original clean predicted persistent track ID
as the previously selected ID, even if the selector selected a wrong semantic
target. No ground-truth identity is available to this decision rule.

Synthetic t1: derive live candidate IDs from the original observed mask and the
registered removed ID. If the previous selected ID is absent, return null with
TARGET_LOST. Do not substitute another track; a new target requires the behavior
planner's explicit authorization. If it remains, keep the existing identity
binding. Structural validation still applies. Where a neural score exists,
calculate the held identity's current softmax probability from the persisted
candidate logits. A validation-selected threshold or each fixed 0.5/0.7/0.9
threshold may abstain with AMBIGUOUS; raw score is not calibrated uncertainty.
The raw binding-only point imposes no confidence cutoff. Deterministic rules
have no probabilistic score; threshold wrappers are inapplicable to them.

Report all-case coverage, TARGET_LOST/null counts, held-ID retention, original
clean correctness, semantic wrong non-null issuance and accepted error. Preserve
initial wrong selection: its wrong ID can remain live after the true target is
deleted. True identity is read only for retrospective semantic scoring, never
for the TARGET_LOST or AMBIGUOUS decision. Do not present this rule as learned
absence recognition, tracker performance, dynamic target-loss/reacquisition
validation, driving safety or a deployment demonstration.

This is a static two-payload contract test. It verifies fail-closed handling
when a previously selected persistent ID is explicitly absent from a later
payload. It cannot detect a silently reassigned ID, an unseen intended target,
or an incorrect initial semantic identity. Real detector/tracker and planner
authorization integration remains unexecuted.

Implementation and results are separate from the original static audit under
03_code/r3_deployment/followup and 04_results/deployment/identity_contract.
Existing case records and formal summary files are never overwritten.
