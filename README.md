# LC-ORGNav Target-Selection Evaluation Protocol

Companion software for the IEEE Transactions on Intelligent Transportation Systems (T-ITS) revision (round 3) of the LC-ORGNav manuscript.

**Repository:** https://github.com/nulihou/lc-orgnav-protocol
**Release:** `v1.0.1` (code identical to `v1.0-tits-r3`; metadata update)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23395287.svg)](https://doi.org/10.5281/zenodo.23395287)
**Zenodo DOI:** [10.5281/zenodo.23395287](https://doi.org/10.5281/zenodo.23395287)
**License:** MIT (see `LICENSE`)
**Author:** Mingyang Hou (Dalian Neusoft University of Information)

## What this is

An auditable evaluation protocol for query-conditioned semantic target selection:

- matched inputs and persistent candidate IDs
- shared-command execution
- freeze-before-test, with hash-before-scoring
- dual-track abstention gates on Town10HD: correct-rejection rate (CAR) >= 90% and false-rejection rate (FRR) <= 2%, on `t10_hard_confusable_test` and `t10_scale300_test`

The primary claim is the protocol, not a single architecture win. Under the frozen P15 scorecard, the structured remedy arm **M12|graph** passes both Town10HD tracks; **FULL** is reported as a calibration/structure trade-off and is not declared the winner. Transfer sets (Town05, nuScenes layouts, Talk2Car keyword parses) are stress tests outside the dual-track gate claim.

Numbers below are pointers to frozen JSON files. The JSON files are the authoritative scorecard.

## Layout

```text
code/                 r3_protocol, r3_selectors, r3_deployment, r3_closedloop, lc_org_nav, configs, scripts
protocol_freeze/      dual-track freeze manifests, candidate lists, SHA256SUMS, PROTOCOL.md
experiments/P3/       ten-seed equivalence (TOST) analysis
experiments/P14/      remedy scripts, freeze manifest, thresholds, results
experiments/P15/      structured remedy + yaw-leak diagnostic; thresholds, results
experiments/P16/      matched-deletion audit of frozen remedies; scoring and diagnostics
results_meta/         deployment / offline registration and SHA256 verification
```

## Reproduction paths (aligned with the R3 manuscript)

Python 3.10+; `pip install -r requirements.txt` (plus `scikit-learn` for the P16 diagnostics). CPU is enough for scoring.

| Manuscript item | Frozen result | Script |
| --- | --- | --- |
| Dual-track gates, P14 remedies | `experiments/P14/p14_results.json` | `experiments/P14/` scoring entrypoint |
| Dual-track gates, P15 (M12 `exist`, FULL `ltt` in `primary_op`) | `experiments/P15/p15_results.json`, `p15_thresholds.json` | `experiments/P15/` scoring entrypoint |
| Matched-deletion audit (P16: 168 prediction files, 4 selectors x 7 arms x 3 seeds x 2 tracks) | `experiments/P16/results.json`, `PREDICTION_HASHES.json` | `experiments/P16/score_p16_v2.py`, `recount_portable.py` |
| Marginal-feature AUCs, tie-aware (midrank): candidate count 0.500, score margin 0.443, range 0.44-0.51 | `experiments/P16/diagnostics/auc_check.json` (second value of each pair) | `python experiments/P16/diagnostics/verify_auc.py <out_dir>` |
| Linear separability of the two interventions, scene-grouped 5-fold CV: AUC 0.530 | `experiments/P16/diagnostics/cv_check.json` (`groupkfold_group_id`) | `python experiments/P16/diagnostics/verify_cv.py <out_dir>` |
| Ten-seed equivalence (P3) | `experiments/P3/p3_results.json` | `experiments/P3/p3_tost.py` (set `LCORGNAV_R3_ROOT`) |

`auc_check.json` also keeps the earlier non-tie-aware values (first value of each pair), and `cv_check.json` keeps the earlier ungrouped split (0.437), so the correction stays auditable. The diagnostics read the bundled feature file `experiments/P16/diagnostics/features/hard_confusable_test.npz`.

Procedure:

1. Read the freeze protocols first: `protocol_freeze/*/PROTOCOL.md`, `experiments/P14/PROTOCOL.md`, `experiments/P15/PROTOCOL.md`, `experiments/P16/PROTOCOL.md`.
2. Verify SHA256 sums of any local checkpoint or prediction file before scoring.
3. Score with the frozen entrypoints against the frozen candidate lists.
4. Compare CAR/FRR to the frozen result JSON. If a rescore disagrees, stop and diff hashes. Do not refit thresholds.

## Not included

- checkpoints (`runs/`), prediction JSONL (`preds/`, P16 `predictions/`), local caches. They are referenced by SHA256 in `protocol_freeze/*/SHA256SUMS.json`, `experiments/P14/PREDICTION_HASHES.json`, `experiments/P15/PREDICTION_HASHES_P15.json`, `experiments/P16/PREDICTION_HASHES.json`, `protocol_freeze/CODE_AND_PROTOCOL_SHA256.json`.
- CARLA trainable graph tensors (`.npz`) beyond the P16 diagnostic feature file.
- Third-party raw data (nuScenes, Talk2Car).

## Path sanitization

Before release, local absolute path prefixes were removed from 24 manifest/list files. Their bytes therefore differ from the frozen originals; nothing else changed (no score, threshold, candidate ID or ordering). `protocol_freeze/RELEASE_PATH_SANITIZATION.json` lists, for each file, the frozen `original_sha256` (the value in the frozen manifests) and the `released_sha256`. Some frozen records (for example `experiments/P3/PROTOCOL.md`) are kept byte-identical and still mention the original machine's paths or contain Chinese text; they are retained unchanged as historical records. Some legacy source files under `code/` carry Chinese comments. One P3 planning draft listed in `experiments/P3/FREEZE_MANIFEST.json` is not shipped.

## Data statement

| Source | Status |
| --- | --- |
| CARLA-derived protocol graphs | Self-generated; manifests and lists redistributed here |
| nuScenes | Terms of Use (non-commercial); obtain from the original provider; only metadata references here |
| Talk2Car | CC BY-NC-SA 4.0; obtain commands from the original provider |

## Citation

Cite the T-ITS manuscript and the Zenodo DOI of this release. Machine-readable metadata: `CITATION.cff`.

## Contact

Mingyang Hou, Department of Artificial Intelligence, Dalian Neusoft University of Information, Dalian, China. houmingyang@neusoft.edu.cn
