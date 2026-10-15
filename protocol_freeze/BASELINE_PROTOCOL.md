# R3 baseline protocol — frozen before formal training

Registered 2026-10-02. This is a new cohort; it does not replace or correct the
historical R2 tables. Test labels/tasks and their order remain the R2 frozen
300-task test sets; train/validation remove connected graph/base-scene groups
assigned to later splits. All four models use these identical frozen manifests.

- Scale-300: 2965 train / 796 validation / 300 test tasks.
- Hard-confusable: 2932 train / 798 validation / 300 test tasks (299 test graph
  groups; uncertainty must cluster by group, not pretend all query tasks are
  independent graphs).
- Training seeds: 7, 21, 42. Every model/track/seed is reported, including failures.
- Four comparisons: original Graph+Relation-Gate, matched geometry MLP,
  query-conditioned geometric attention, DriveTransformer task-block adapted.
- No prior test result, test loss, closed-loop success or controller failure is
  used to select a model, seed, epoch or hyperparameter. No winner is promised.

## Inputs, preprocessing and objective

All models receive identical 32-d node features, all existing 8-d pairwise edge
geometry, the identical 32-d query, candidate mask and valid node mask. The MLP
uses incoming/outgoing edge mean/max pooling; it no longer discards edge input.
Graph uses the original FiLM message passing and original neural relation gate.
Geometric attention uses learned per-head pairwise edge biases and semantic
query-to-node cross attention. The DriveTransformer adapter encodes the same
edge summaries in agent tokens, adds positional encodings to attention Q/K,
and lets agent and semantic task queries interact directly at every block.

Node and edge means/standard deviations are estimated exclusively from the
retained training arrays, frozen in each checkpoint, and applied identically
to all models. Constant feature standard deviations become 1. Query one-hot
fields and masks are preserved. No target feature, target slot, ground-truth
label or dataset identifier is supplied to a forward pass.

Every model trains with candidate-masked cross entropy plus 0.2 times mean
negative hinge ranking loss, margin 1.0. No model receives an auxiliary relation
or geometry loss. All use AdamW, lr 3e-4, weight decay 1e-5, 80 epochs, batch
64, gradient clipping 5, cosine lr down to 3e-5, float32, TF32 disabled.
There is no test-driven hyperparameter sweep. Checkpoints maximize validation
ProtoAcc@1; ties select lower validation CE, with exact ties keeping the earlier
epoch. Each frozen checkpoint is evaluated on test once after training.

Capacity was chosen by parameter count before test scoring: Graph hidden 128
575496 total parameters; MLP hidden 160 452481; geometric attention hidden 96
598765; DriveTransformer adapter hidden 112 566641. Three layers/four heads
where applicable. Graph retains original dormant auxiliary heads in the
architecture/state dict, although the shared selector loss does not train them.
Parameter counts are comparable rather than identical and will be disclosed.

## Published-method lineage, adapter scope

DriveTransformer (Jia et al., ICLR 2025), official repository:
https://github.com/Thinklab-SJTU/DriveTransformer
Pinned commit: 19702015eb700e804bafb53e63b04571c7b8604a.
Official reference files are saved in `references/drivetransformer_official/`.

The adapter reimplements, in plain PyTorch, the official `AttentionLayer`
independent LayerNorm→Q/K/V projections, positional additions to Q/K,
scaled-dot-product attention and residual output; and `SwiGLULayer`
pre-normalization, SiLU product FFN, 2/3 hidden adjustment and residual output.
Agent queries and a semantic task query interact in each repeated block.
Sensor cross attention and temporal cross attention have no counterpart in
the static NPZ protocol and are removed. Map/detection/trajectory task heads
are replaced by a masked candidate-ID ranking head and common input adapter.
The implementation is a **DriveTransformer task-block adapted selector**;
it is not the complete DriveTransformer, not its pretrained driving model,
and not a reproduction of its Bench2Drive/nuScenes SOTA score.

The separate geometric attention reference is purpose-built and is not
misattributed to any published method. FocalAD/SGDrive/DriveGATr remain
related-work and deployment-context references unless their exact public
implementations can be reproduced on a compatible task.

## Outputs and downstream interface

`04_results/offline/{track}/{model}/seed_{seed}/` contains checkpoint.pt,
run_config.json, history.jsonl, validation_predictions.jsonl,
test_predictions.jsonl and summary.json. Predictions use whole-node index,
complete N-node logits, masked probabilities, target index, scene_id,
canonical group/hash and seed. Source/checkpoint/manifest SHA256 and timing
are recorded. Model timing excludes graph construction, perception and control.

`load_checkpoint(path, device)` returns a model/normalization bundle;
`predict_arrays(bundle,node_feats,edge_index,edge_attr,lang_feat,candidate_mask)`
returns pred_index, logits and probabilities; `predict_npz(bundle,path)` wraps
NPZ loading. The closed-loop cohort is independently frozen before inspecting
these outputs; it uses fixed shared control and retains invalid/failure cases.

Implementation checks use training samples only: original graph arithmetic
equivalence, padding invariance and node-order permutation. Smoke output is
separate and does not score the frozen test set. If a implementation defect
requires a code change, disclose it and keep preceding failure artifacts;
do not silently count a superseded run as a formal result.
