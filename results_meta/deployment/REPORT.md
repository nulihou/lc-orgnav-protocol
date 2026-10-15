# Completed static-input audit and identity contract follow-up

All six methods were evaluated once on every registered payload for each track. There are 241,200 per-method static case records across the two tracks; repeated corruptions are clustered within the 300 Scale and 299 Hard base graph groups. No inference/adapter failure was resampled. These are synthetic static observation inputs.

| Track / method | Clean correct / 300 | sigma2m correct / 3000 | Raw target-missing wrong / 300 | Identity-contract held wrong / 300 |
|---|---:|---:|---:|---:|
| scale300 / graph_relation_gate | 300 | 2929 | 300 | 0 |
| scale300 / matched_geometry_mlp | 296 | 2919 | 300 | 4 |
| scale300 / query_geometry_attention | 295 | 2908 | 300 | 5 |
| scale300 / drivetransformer_task_adapted | 293 | 2896 | 300 | 7 |
| scale300 / coarse_rule | 300 | 2644 | 300 | 0 |
| scale300 / fine_rank_rule | 300 | 2644 | 300 | 0 |
| hard_confusable / graph_relation_gate | 290 | 2543 | 300 | 10 |
| hard_confusable / matched_geometry_mlp | 290 | 2572 | 300 | 10 |
| hard_confusable / query_geometry_attention | 289 | 2561 | 300 | 11 |
| hard_confusable / drivetransformer_task_adapted | 288 | 2629 | 300 | 12 |
| hard_confusable / coarse_rule | 88 | 1042 | 300 | 212 |
| hard_confusable / fine_rank_rule | 300 | 2248 | 300 | 0 |

The forced-choice target-missing failures remain in the formal audit. Fixed score thresholds and a clean-validation selected threshold do not establish reliable absent-target rejection. The separately registered identity contract only rejects when the *previous predicted ID* is absent from the live candidate payload; an initially wrong ID that stays live can remain wrong. Semantic truth is used for scoring only. This is a static two-payload binding check, not tracker loss/reacquisition or driving safety validation.

Empty candidates, unsupported queries and stale declared metadata produce null outputs via explicit schema logic, not learned recognition. Static permutations preserve original-ID choices but do not test temporal ID switches. Hard same-G policy flips use 296 of 300 source tasks; Scale relation flips use 131, with all exclusions disclosed independently of model scores. Paired scene-cluster intervals and all score/error/coverage points are saved beside this report.
