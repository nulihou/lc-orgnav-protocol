# Heavy artifacts not staged

Source tree: P15 structured remedy.

Not copied into this git companion (too large / binary):

- `runs/` (checkpoints)
- `preds/` (prediction JSONL)
- `smoke/` / `dev/`

Ship via Zenodo or a GitHub Release only after verifying SHA256 against `PREDICTION_HASHES_P15.json` and the protocol freeze manifests. Do not retrain or reseat thresholds. See also `DIAGNOSTIC_yaw_leak_20261014.md`.
