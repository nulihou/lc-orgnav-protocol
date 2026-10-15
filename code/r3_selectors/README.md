# R3 selectors

Use `.venv (PyTorch 2.11, CUDA 12.8).
Set PYTHONPATH to the absolute `R3_revision_20261002/03_code` path.
Only the new R3 folder is written. Source NPZs are hash-verified and read-only.

```powershell
python -m r3_selectors.validate_implementation --root R3_revision_20261002
python -m r3_selectors.pipeline --root R3_revision_20261002 --track scale300
python -m r3_selectors.pipeline --root R3_revision_20261002 --track hard_confusable
python -m r3_selectors.summarize_results --root R3_revision_20261002
python -m r3_selectors.audit_training_freeze --root R3_revision_20261002
python -m r3_selectors.check_frozen_predictions --root R3_revision_20261002
```

## Inference

```python
from r3_selectors import load_checkpoint, predict_arrays, predict_npz
bundle = load_checkpoint(checkpoint_path, device='cpu')
prediction = predict_arrays(bundle, node_feats, edge_index, edge_attr,
                            lang_feat, candidate_mask)
# prediction: pred_index / chosen_idx whole-node ID, logits [N], probabilities [N]
```

For batched static stress checks use
`r3_selectors.pipeline.assemble_batch(samples,bundle['normalization'],device)`
and `forward_logits(bundle['model'],batch)`. Samples require nodes, edge_index,
edge_attr, query, candidates, and a dummy target selected from the candidate
mask (assembly validates it; forward does not receive targets). Mask invalid
candidates before argmax/softmax and slice each result to its original node count.
Padding/permutation invariance and original graph arithmetic equivalence were
checked on training inputs; results are in `07_audit/selector_implementation_checks.json`.

The exact training/preselection contract and model lineage are in
`02_protocol/BASELINE_PROTOCOL.md`. Do not call the DriveTransformer adapter
a reproduction of the full model or its driving SOTA result. Its official
reference code is pinned at commit 19702015eb700e804bafb53e63b04571c7b8604a.

`BASELINE_ANALYSIS_APPENDIX.md` records supplementary statistics, active versus
dormant parameter counts and deployment limitations without editing the original
registered protocol. Statistical reporting details were added during the
training campaign; only model/training/selection choices were originally frozen.
The three seeds use the same test cohort and are not independent datasets.

`benchmark_frozen.py` measures warm batch-1 model-forward latency with a common
training input, interleaved frozen seed-42 models and a fixed repetition count.
Run it after other GPU workloads finish; record shared-machine limitations.
It excludes perception, graph construction, normalization and controller time.
