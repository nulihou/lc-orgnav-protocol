# Analysis and implementation disclosure (does not edit the registered protocol)

The frozen `BASELINE_PROTOCOL.md` SHA256 is
`0b0c3c82618c03a31fefff4b55c2b47160d2577e7fb7c2a0cea2610706f83834`.
This explanatory appendix does not alter training, test tasks or checkpoint
selection. The core training/model source hashes are saved per run.

## Capacity

The original graph state dictionary retains 50,311 dormant parameters in its
relation and geometry auxiliary heads. The shared CE+ranking objective never
calls these heads. Therefore Graph has 575,496 total parameters and 525,185
active selector parameters. The MLP, geometric attention and DriveTransformer
task adapter have 452,481, 598,765 and 566,641 active/total parameters,
respectively. The revised report exposes both counts rather than implying
that all original auxiliary parameters participate in selection.

## Published adapter limitations

The DriveTransformer official source is pinned and hash-recorded in
`references/drivetransformer_official/PROVENANCE.json`. The static adapter
retains the `AttentionLayer` and `SwiGLULayer` primitives and direct task-token
interaction. It shares the FFN between static agent/task tokens rather than
retaining every full-stack task-specific branch. The geometry/query input
adapter and target-ID scoring head are task conversions. It does not evaluate
the original sensor/temporal streams or original trajectory metrics and must
not be advertised as an exact/full-stack SOTA reproduction.

## Statistical families

These inferential reporting details were added after formal training began,
with the Scale-300 results already available. They are a transparent
supplemental analysis, not an originally preregistered statistical plan. The
four models, training seeds, data, objective, budget and validation checkpoint
selection were frozen before formal training and are unchanged.

For each track and fixed training seed, the three graph-versus-baseline
contrasts form one Holm family. Primary p-values use connected graph/base-scene
group wins/losses; task McNemar p-values are secondary and independently Holm
adjusted within that same three-contrast family. A separate three-contrast
family is reported for each track after averaging query tasks inside each
group, then averaging the three fixed training seeds. The bootstrap resamples
test groups (10,000 replicates, random seed 20261002); it does not treat training
seeds or repeated query tasks as independent datasets.

Training-seed sample standard deviation describes training variability. The
group bootstrap CI describes uncertainty over test groups conditional on those
three seeds; these two quantities answer different questions. All comparisons
are reported, regardless of sign or adjusted p-value. No test-based seed
selection is performed. These reporting additions do not change checkpoint
selection or trigger additional fits.

## Timing and deployment scope

Per-run timings are warm model-forward measurements on normalized/prepared
tensors. They exclude input loading, graph reconstruction, normalization,
perception, tracking and control. The workstation is shared with other
authorized CARLA/analysis work; exclusive GPU conditions were not guaranteed.
The optional common-input benchmark interleaves all four frozen seed-42 models
in one process and is run separately. Neither timing artifact supports a
full-stack FPS, sensor-to-actuation deadline or real-world safety claim.

## Cohort provenance

These are new training cohorts and the same historical R2 test tasks, not a
new external dataset. Canonical graph/base-scene grouping removes exact group
overlap, but does not establish transfer to an unseen town, road family,
natural-language instruction, tracking condition or real sensor domain.
Frozen CUDA predictions remain authoritative for the shared-controller
execution cohort; CPU replay checks are a numerical integrity check and never
select a checkpoint or alter a target.
