# R3 matched baseline results

All rows use newly graph-disjoint train/validation manifests; historical R2 tables are a separate cohort.
DriveTransformer task-block adapted is a static protocol adapter, not a full-stack SOTA reproduction.
No seed is selected by test accuracy. Group bootstrap and group sign tests are primary; task McNemar is secondary.

| Track | Model | Seed | Validation % | Test correct/N | Test % | Best epoch | Parameters |
|---|---|---:|---:|---:|---:|---:|---:|
|scale300|graph_relation_gate|7|100.00|297/300|99.00|11|575496|
|scale300|graph_relation_gate|21|99.87|299/300|99.67|15|575496|
|scale300|graph_relation_gate|42|100.00|300/300|100.00|80|575496|
|scale300|matched_geometry_mlp|7|99.87|298/300|99.33|45|452481|
|scale300|matched_geometry_mlp|21|100.00|296/300|98.67|40|452481|
|scale300|matched_geometry_mlp|42|100.00|296/300|98.67|60|452481|
|scale300|query_geometry_attention|7|99.87|297/300|99.00|58|598765|
|scale300|query_geometry_attention|21|99.75|297/300|99.00|52|598765|
|scale300|query_geometry_attention|42|100.00|295/300|98.33|26|598765|
|scale300|drivetransformer_task_adapted|7|99.62|296/300|98.67|57|566641|
|scale300|drivetransformer_task_adapted|21|99.75|295/300|98.33|42|566641|
|scale300|drivetransformer_task_adapted|42|99.75|293/300|97.67|40|566641|
|hard_confusable|graph_relation_gate|7|96.62|291/300|97.00|31|575496|
|hard_confusable|graph_relation_gate|21|95.74|288/300|96.00|30|575496|
|hard_confusable|graph_relation_gate|42|96.24|290/300|96.67|39|575496|
|hard_confusable|matched_geometry_mlp|7|94.74|282/300|94.00|46|452481|
|hard_confusable|matched_geometry_mlp|21|94.74|283/300|94.33|52|452481|
|hard_confusable|matched_geometry_mlp|42|94.24|290/300|96.67|60|452481|
|hard_confusable|query_geometry_attention|7|93.98|284/300|94.67|58|598765|
|hard_confusable|query_geometry_attention|21|94.24|288/300|96.00|60|598765|
|hard_confusable|query_geometry_attention|42|95.11|289/300|96.33|37|598765|
|hard_confusable|drivetransformer_task_adapted|7|94.11|286/300|95.33|53|566641|
|hard_confusable|drivetransformer_task_adapted|21|94.24|284/300|94.67|54|566641|
|hard_confusable|drivetransformer_task_adapted|42|94.49|288/300|96.00|52|566641|

## Three-seed descriptive summaries

| Track/model | Mean test % | Sample standard deviation (percentage points) |
|---|---:|---:|
|scale300/graph_relation_gate|99.56|0.51|
|scale300/matched_geometry_mlp|98.89|0.38|
|scale300/query_geometry_attention|98.78|0.38|
|scale300/drivetransformer_task_adapted|98.22|0.51|
|hard_confusable/graph_relation_gate|96.56|0.51|
|hard_confusable/matched_geometry_mlp|95.00|1.45|
|hard_confusable/query_geometry_attention|95.67|0.88|
|hard_confusable/drivetransformer_task_adapted|95.33|0.67|

The three training seeds share one test cohort; the standard deviation measures training variability, not three independent datasets.

## Seed-averaged group comparisons

Positive differences favor Graph+Relation-Gate. Query tasks are averaged inside each graph/base-scene group, then the three training seeds are averaged before test-group resampling. Holm families contain the three graph-versus-baseline contrasts separately for each track. These inferential reporting details supplement the frozen training protocol; they were not originally preregistered.

| Comparison | Groups | Delta pp | Group bootstrap 95% CI pp | Group sign Holm p |
|---|---:|---:|---:|---:|
|scale300/graph_vs_matched_geometry_mlp/seed_averaged|300|+0.67|[-0.22, +1.78]|0.90625|
|scale300/graph_vs_query_geometry_attention/seed_averaged|300|+0.78|[-0.11, +1.89]|0.90625|
|scale300/graph_vs_drivetransformer_task_adapted/seed_averaged|300|+1.33|[+0.22, +2.67]|0.32812|
|hard_confusable/graph_vs_matched_geometry_mlp/seed_averaged|299|+1.56|[-0.33, +3.68]|1|
|hard_confusable/graph_vs_query_geometry_attention/seed_averaged|299|+0.89|[-0.56, +2.45]|1|
|hard_confusable/graph_vs_drivetransformer_task_adapted/seed_averaged|299|+1.23|[-0.67, +3.23]|1|

Incomplete runs: 0.
