# Heavy artifacts not staged

Source tree: P14 remedies.

Not copied into this git companion (too large / binary):

- `runs/` (checkpoints)
- `preds/` (prediction JSONL)
- `smoke/` / `dev/`

Ship via Zenodo or a GitHub Release only after verifying SHA256 against `PREDICTION_HASHES.json` and the protocol freeze manifests. Do not retrain or reseat thresholds.
