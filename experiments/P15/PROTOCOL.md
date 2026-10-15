# P15 structured remedy: FROZEN PROTOCOL (2026-10-14; written and hashed before any real P15 training or evaluation)

Purpose: test a structured remedy for the three audited weaknesses (forced binding, rotation shortcut, transfer/large-K gaps), against the frozen baselines (B) and the P14 remedies (a: null option, b: rotation + subsampling augmentation, c: a+b). Results are reported whatever their direction. Only CPU smoke runs on TRAIN data preceded this freeze (dev/, smoke/).

## Components
- **M1 ego-relative SE(2)-invariant encoding** (`p15_common.ego_encode`, applied to every graph at training and inference):
  - World-axis offsets (node cols 0:2 ego offset, 5:7 anchor offset; edge cols 0:2 delta, 3:5 direction) are re-expressed in the ego heading frame (frame recovered by `transforms.recover_frame`).
  - Smoke finding (disclosed): in the source schema the zero-distance self-bearing of the ego (cols 3:5) and of the anchor (cols 8:10) equals (cos(-yaw), sin(-yaw)), i.e. the absolute world heading leaks into the features. M1 replaces these by the canonical (1,0).
  - Verified on TRAIN: encoded features are identical (max abs diff 7.6e-6) under global rotations of 37/90/180/270 deg; smoke models give identical predictions on rotated TRAIN graphs (960/960).
  - Input normalization is recomputed on ego-encoded TRAIN originals (the encoding changes the feature distribution) and stored in each checkpoint.
- **M2 separate existence head** (`ExistWrapper`): a sigmoid logit from (mean, max) pooling over candidates of the input to the selector's own final scoring head (contains the query); no candidate count feature. Loss = CE + 0.2 rank (present graphs only) + 1.0 BCE(existence) on all graphs. Negatives, built from TRAIN (fitting) and VAL (selection) only:
  1. target deleted (`transforms.delete_node`, as P9/P14a);
  2. counterfactual query: relation bit replaced by another relation (1..4) that no candidate satisfies under the frozen relation definition (policy bits kept);
  3. near-miss: the distractor nearest to the target is moved half-way to the target position, then the target is deleted; kept only if that distractor still fails the query relation.
  - TRAIN counts (smoke, full TRAIN): Scale 2965 deleted / 2963 counterfactual / 1925 near-miss; Hard 2932 / 2917 / 143.
- **M3 calibration with finite-sample guarantee** (scoring side, `ltt_threshold`, `crc_threshold`): abstain iff P(exist) < lambda.
  - LTT (fixed-sequence testing over increasing lambda, exact binomial p-values): FRR <= 2% with probability >= 0.9.
  - CRC: E[FRR] <= 2%.
  - Calibration data for FULL = the calibration half of VAL (VAL split by scene group with sha256('P15|valsplit|group')), never used for training or checkpoint selection. For M12 these are reported descriptively on full VAL.
  - The guarantee holds only in distribution (exchangeable calibration and test data). It does not hold for Town05, nuScenes or Talk2Car, and is not claimed there.
  - Smoke Monte Carlo (400 trials, n=400): LTT violation rate 8.75% (bound 10%); CRC mean FRR 2.00%.
- **M4 K-range training** (`k_augment`, FULL only): one extra copy per TRAIN graph.
  - p=0.5 'up': add 1-4 synthetic distractors (attribute row copied from a random non-target candidate, anchor flag cleared) at ego-frame range U(5,60) m, bearing U(-pi,pi).
  - Otherwise 'down': drop non-target candidates with q~U(0.2,0.6).
  - Label = frozen label generator (`rule_prediction`, fine policy on Hard) on the modified graph; copies with generator margin <= 1e-3 are discarded. TRAIN: Hard 6 discarded, 120 labels changed by added distractors; Scale 0 / 227.
  - Training K range becomes Hard 1-12, Scale 1-9 (test K is 3/5/8).
  - Graph uses mean (instead of sum) aggregation of incoming messages (`MeanAggGraph`). The other selectors already use masked mean/max or softmax-attention aggregation and are unchanged.

## Arms (4 selectors x seeds 7/21/42 x tracks scale300, hard_confusable)
| Arm | M1 | M2 | M3 | M4 | VAL use |
|---|---|---|---|---|---|
| M1 | yes | - | - (G3noK as B) | - | all VAL for selection |
| M12 | yes | yes | P9 rule on VAL (LTT/CRC descriptive) | - | all VAL for selection and threshold |
| FULL | yes | yes | LTT (primary), CRC | yes | selection half / calibration half |

- **Hyperparameters:** identical to the frozen selectors and P14: 80 epochs, batch 64, AdamW lr 3e-4 wd 1e-5, cosine to 3e-5, clip 5, rank 0.2 / margin 1.0, same hidden sizes, deterministic algorithms. The existence BCE weight of 1.0 is new.
- **Checkpoint selection on VAL:**
  - M1: accuracy.
  - M12 and FULL: mean of (present forced accuracy, existence balanced accuracy at 0.5).
  - Ties are broken by lower VAL loss.
- **Disclosed differences:** more optimizer steps per epoch for the enlarged sets; extra head; recomputed normalization; Graph aggregation change (FULL only).

## Evaluation (p15_predict.py, only after all training; PREDICTION_HASHES_P15.json before any scoring)
- **Sets:** item sets and ids are identical to P14:
  - Town10HD train/val/test, present and deleted, both tracks;
  - Hard test rotated 90/180/270;
  - Town05 Hard test (P6), present and deleted;
  - nuScenes P11 layouts with Hard checkpoints, present and deleted;
  - Talk2Car P13 keyword parses, present and deleted.
- **Baseline predictions:** B and P14 a/b/c are taken from the hash-checked P14 prediction files, not recomputed.
- **Strata stored per item:** K, relation, policy, target anchor distance.

## Scoring (p15_score.py)
- **Operating points:**
  - G3noK (P12a head, all models).
  - cal (P14a/c).
  - exist: P9 rule (threshold = 2% quantile of VAL present scores); FULL uses the calibration half.
  - ltt and crc (M12/FULL).
  - An empty candidate set counts as an abstention.
  - Primary op: B/P14b/M1 G3noK; P14a/c cal; M12 exist; FULL ltt.
- **Preset criterion (P9, unchanged):** CAR >= 90% and FRR <= 2% (point estimates) on BOTH Town10HD test tracks at the primary op.
- **Statistics:** item values averaged over seeds; scene/group-cluster bootstrap 2000, seed 20261014; paired difference = arm minus reference; Holm within each family (16 contrasts = 4 selectors x 4 metrics). Bootstrap-floor p reported as p<0.001.
- **Families:**
  - **F1 (primary, confirmatory):** FULL vs B: {CAR Hard, CAR Scale, rotated pooled accuracy, nuScenes accuracy}.
  - **F2:** FULL vs P14c, same four metrics.
  - **F3:** M1 vs B: {rotated pooled accuracy, Town05 accuracy, nuScenes accuracy, Hard clean accuracy}.
  - **F4:** M12 vs P14a: {CAR Hard, CAR Scale, FRR Hard, FRR Scale}.
  - **F5 (ablation):** FULL vs M12: {CAR Hard, CAR Scale, nuScenes accuracy, Town05 accuracy}.
- **Non-inferiority of clean forced accuracy** vs B (each P15 arm, each selector, both tracks): one-sided 95% bootstrap lower bound of the paired difference > -2 pp (primary) and > -1 pp (sensitivity).
- **Descriptive:**
  - per-angle rotation accuracy;
  - rotation prediction identity (sanity: P15 predictions should be unchanged under rotation);
  - K-stratified acc/FRR/CAR on all sets;
  - Town05 and nuScenes weak strata: K=8, relation 3, relation 4, farthest policy, anchor distance >= 35 m;
  - Talk2Car acc/CAR/FRR;
  - Town05 and nuScenes CAR/FRR (no guarantee claimed);
  - all non-primary ops.
- **Deviations:** go to DEVIATION_*.md with timestamps. Automatic resumes after a crash are appended by the runner to DEVIATION_runtime_auto.md. The runner writes P15_DONE.txt only after training, prediction and hashing all succeed. Scoring is run separately after P14_DONE.txt and P15_DONE.txt.

## Literature basis (verified)
- Satorras, Hoogeboom, Welling, "E(n) Equivariant Graph Neural Networks", ICML 2021 (PMLR 139), https://arxiv.org/abs/2102.09844 (invariant inputs: relative geometry only).
- He, Ding, Liu, Jiang, "GREC: Generalized Referring Expression Comprehension", arXiv 2023, https://arxiv.org/abs/2308.16182; Liu, Ding, Jiang, "GRES: Generalized Referring Expression Segmentation", CVPR 2023 (explicit no-target outputs).
- Angelopoulos, Bates, Fisch, Lei, Schuster, "Conformal Risk Control", ICLR 2024, https://arxiv.org/abs/2208.02814; Angelopoulos, Bates, Candes, Jordan, Lei, "Learn then Test", arXiv 2110.01052.
- Yehudai, Fetaya, Meirom, Chechik, Maron, "From Local Structures to Size Generalization in Graph Neural Networks", ICML 2021.