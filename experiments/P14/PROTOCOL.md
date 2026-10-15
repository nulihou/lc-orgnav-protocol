# P14 remedies: FROZEN PROTOCOL (2026-10-14; written and hashed before any P14 training or evaluation)

Purpose: test two remedies for weaknesses the audit exposed. Results are reported whatever their direction.
- Forced binding is addressed by a null option (P14a).
- The rotation and candidate-count shortcuts are addressed by augmentation (P14b).
- P14c combines both.

## Training (p14_train.py)
- **Selectors and hyperparameters:** the 4 selectors (graph_relation_gate, matched_geometry_mlp, query_geometry_attention, drivetransformer_task_adapted), same hidden sizes as the frozen Town10HD models.
  - 80 epochs, batch 64, AdamW lr 3e-4 wd 1e-5, cosine to 3e-5, gradient clip 5, rank weight 0.2 / margin 1.0.
  - Seeds 7/21/42; deterministic algorithms.
- **Data:** tracks scale300 and hard_confusable. Fitting uses TRAIN; checkpoint selection uses VAL (best accuracy, ties broken by lower CE). TEST is refused by a guard.
- **Normalization:** identical to the frozen baseline (train-manifest statistics, asserted equal).
- **P14a (null option):**
  - The model gets one extra "no target" logit. It is computed by an MLP on (mean, max) pooling over candidate nodes of the input to the selector's own final scoring head, and trained end to end.
  - For every train/val graph a deleted-target copy is added (transforms.delete_node, as in P9), labelled NULL. Copies whose candidate set becomes empty are skipped (they are trivially abstentions).
  - Loss: CE over candidates + null. The rank loss is applied on present graphs only.
  - VAL selection uses accuracy over present and deleted graphs (target or NULL).
- **P14b (augmentation):**
  - Each train/val graph gets 3 augmented copies, seeded by sha256('20261014|P14b|track|split|sample_id|copy').
  - Each copy applies a global rotation U(0, 2*pi), computed as in P12d.
  - With probability 0.5, each non-target candidate other than ego/anchor is then dropped with probability q ~ U(0.2, 0.6). The target is kept, and the labels are invariant because the relation and extreme-order policies are monotone under removing other candidates.
  - VAL selection uses the original plus augmented VAL graphs.
- **P14c:** P14b copies plus deleted-target copies of all of them, with the P14a model and loss.
- **Disclosed differences:** same epochs but more optimizer steps per epoch for the enlarged sets; P14a/c have one extra head.

## Evaluation (p14_predict.py; only after all training; then PREDICTION_HASHES.json before any scoring)
- **Models:** baseline frozen checkpoints (B, recomputed on the same inputs) and the P14 variants, seeds 7/21/42.
- **Sets:**
  - Town10HD test, present and deleted, both tracks.
  - Town10HD Hard test rotated by 90/180/270 degrees.
  - Town05 Hard test (P6), present and deleted.
  - nuScenes 2,628 layouts (P11), present and deleted, Hard checkpoints.
  - Talk2Car P13 keyword parses, present and deleted (P13 routing: ordinal → Hard, otherwise Scale).
  - Train/val outputs are written for gate fitting only.

## Scoring (p14_score.py)
- **Operating points:**
  - argmax: null is the argmax.
  - cal: abstain iff 1-P(null) < t, with t chosen on VAL present by the P9 rule for FRR <= 2%.
  - G3noK: P12a head refit on TRAIN present vs deleted, threshold on VAL by the same rule. Applies to every model.
  - An empty candidate set counts as an abstention.
- **Preset criterion (P9, unchanged):** CAR >= 90% at FRR <= 2% on both Town10HD tracks. The primary op is cal for null models and G3noK for the others.
- **Statistics:** item values averaged over 3 seeds; group/scene-cluster bootstrap 2000, seed 20261014. Paired difference = variant minus baseline; Holm within each family of 16 contrasts. Bootstrap-floor p-values are reported as p<0.001 (unadjusted) or with their Holm bound.
- **Families:**
  - P14a: per selector, {CAR Hard (cal vs B G3noK), CAR Scale, forced accuracy Hard, forced accuracy Scale}.
  - P14b: per selector, {rotated pooled accuracy, Town05 accuracy, nuScenes accuracy, Hard clean accuracy}.
  - P14c: per selector, {CAR Hard, CAR Scale, rotated pooled accuracy, nuScenes accuracy}.
  - Everything else (Talk2Car, Town05/nuScenes CAR/FRR, per-angle accuracy, G3noK for b) is descriptive.
- **Deviations:** go to DEVIATION_*.md with timestamps. The runner writes P14_DONE.txt only after training, prediction and hashing all succeed. Scoring is run separately afterwards.
