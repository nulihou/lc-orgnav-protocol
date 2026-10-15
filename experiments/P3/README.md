# P3 — ten-seed TOST / Holm (Hard + Scale)

Frozen companion for the manuscript's ten-seed equivalence and Holm contrasts.

- `p3_results.json` — per-seed ProtoAcc@1 means/SDs and graph-vs-* paired TOST / Holm.
- `PREDICTION_HASHES.json` — SHA-256 of frozen prediction artifacts (verify before rescoring).
- `PROTOCOL.md` — freeze-before-test protocol (Chinese draft notes retained as historical).
- `p3_tost.py` — scoring entry used to produce `p3_results.json` from hashed predictions.

The main-text Hard range **95.00--96.56%** is the **three-seed** (7/21/42) descriptive row in
`results_meta/offline/SUMMARY.md` / `matched_hard_row.json`, not the ten-seed means in this folder.
Ten-seed TOST (5/6 contrasts within 2 pp; Holm not significant) is reported separately.

Do not retrain or edit thresholds to force a local rescore to match.
