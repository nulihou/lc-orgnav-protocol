# LC-ORGNav Target-Selection Evaluation Protocol

Companion software for an IEEE Transactions on Intelligent Transportation Systems (T-ITS) revision manuscript.

**Repository:** https://github.com/nulihou/lc-orgnav-protocol  
**Zenodo DOI:** *to be minted on first public deposit* (`https://doi.org/10.5281/zenodo.XXXXXXX`)  
**License:** MIT (see `LICENSE`)  
**Author:** Mingyang Hou (Dalian Neusoft University of Information)

T-ITS review is single-anonymous. This release is named (not anonymized). Replace the Zenodo placeholder only after Publish; do not invent a DOI.

## What this is

An **auditable evaluation protocol** for query-conditioned semantic target selection:

- matched inputs and persistent candidate IDs
- shared-command execution
- freeze-before-test, with **hash-before-scoring**
- dual-track abstention gates on Town10HD:
  - correct-rejection rate (CAR) >= 90%
  - false-rejection rate (FRR) <= 2%
  - tracks: `t10_hard_confusable_test` and `t10_scale300_test`

The revision's primary claim is the **protocol** (honest gates and disclosure), not a single architecture win. Under the frozen P15 scorecard, the structured remedy arm **M12|graph** passes both Town10HD tracks; **FULL** is reported as a calibration / structure trade-off and is **not** declared the winner. Transfer sets (Town05, nuScenes layouts, Talk2Car keyword parses) are reported for stress testing; they are outside the dual-track gate claim.

Numbers in this README are pointers to frozen JSON only. Do not treat prose as a second scorecard.

## Repository layout

```text
code/                 # r3_protocol, r3_selectors, r3_deployment, r3_closedloop,
                      # lc_org_nav, configs, scripts
protocol_freeze/      # dual-track freeze manifests, lists, SHA256SUMS, PROTOCOL.md
experiments/P14/      # remedy scripts, freeze manifest, thresholds, results JSON
experiments/P15/      # structured remedy + yaw-leak diagnostic; thresholds, results
results_meta/         # deployment / offline registration and SHA256 verification JSON
LICENSE               # MIT
CITATION.cff
requirements.txt
```

Heavy binaries are **not** in git (see below).

## What is not in this git tree

Intentionally omitted (large / binary):

- checkpoints under `runs/`
- prediction JSONL under `preds/`
- local `smoke/` / `dev/` caches

These are referenced by SHA256 in:

- `protocol_freeze/*/SHA256SUMS.json`
- `experiments/P14/PREDICTION_HASHES.json`
- `experiments/P15/PREDICTION_HASHES_P15.json`
- `protocol_freeze/CODE_AND_PROTOCOL_SHA256.json`

Ship them later as Zenodo / GitHub Release assets **after** hash verification. Do not retrain or reseat thresholds to make a local rescore match.

## Reproduce frozen gates (high level)

1. Use Python 3.10+ and install `requirements.txt` (CPU is enough to read scores; GPU only if you retrain).
2. Read the freeze protocols before running anything:
   - `protocol_freeze/hard_confusable/PROTOCOL.md`
   - `protocol_freeze/scale300/PROTOCOL.md`
   - `experiments/P14/PROTOCOL.md`
   - `experiments/P15/PROTOCOL.md`
3. Verify listed SHA256 sums against any local checkpoint / prediction files **before** scoring.
4. Score with the frozen entrypoints under `experiments/P14` and `experiments/P15` against the frozen candidate lists.
5. Compare dual-track CAR / FRR to the frozen result files:
   - `experiments/P14/p14_results.json`
   - `experiments/P15/p15_results.json`
6. If a local rescore disagrees, **stop and diff hashes**. Do not edit thresholds post hoc.

### Reading P15 quickly

- Primary operating points are recorded in `p15_results.json` under `primary_op` (M12 uses `exist`; FULL uses `ltt`).
- **M12|graph**: dual-track pass on Town10HD (structured existence head; yaw leak closed). See also `experiments/P15/DIAGNOSTIC_yaw_leak_20261014.md`.
- **FULL**: Hard-track CAR drop under stricter LTT / K-range structure is a disclosed trade-off, not a claim that removing the yaw leak alone "unmasked" false rejections.

Exact means and seed triples live only in the JSON files above.

## Data statement

| Source | Status in this deposit |
| --- | --- |
| CARLA-derived protocol graphs | Self-generated; redistributable as listed hashes |
| nuScenes | Terms of Use (non-commercial); obtain from the original provider; this repo keeps metadata / layout references only |
| Talk2Car | CC BY-NC-SA 4.0; obtain commands from the original provider |

Do not redistribute third-party raw datasets from this repository.

## Citation

Cite the T-ITS manuscript and, once minted, the Zenodo DOI. Machine-readable metadata is in `CITATION.cff`.

## Contact

Mingyang Hou  
Department of Artificial Intelligence  
Dalian Neusoft University of Information, Dalian, China  
Email: houmingyang@neusoft.edu.cn  
GitHub: https://github.com/nulihou/lc-orgnav-protocol

## Status

Public GitHub: `nulihou/lc-orgnav-protocol` (named release; T-ITS is single-anonymous).

Release plan:
1. **Skeleton commit first** (this README, LICENSE, CITATION, layout stubs).
2. **Protocol / code commit next**, after the author explicitly approves push of the full companion tree.
3. **Zenodo Publish** is a separate approval; replace `XXXXXXX` only after the DOI is minted.

Until those approvals, do not treat this mirror as complete. Do not invent a DOI.
