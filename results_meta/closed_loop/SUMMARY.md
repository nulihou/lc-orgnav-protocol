# R3 shared-controller execution audit

Each identical scene, seed and selected actor is physically executed once. Reused outcomes are paired observations, not additional runs.

| Track | Method | Planned ProtoAcc | Sem8 count/N | Sem8 % [cluster CI] | Sem4 % | Wrong reach8 % | Valid |
|---|---|---:|---:|---:|---:|---:|---:|
|scale300|graph_relation_gate|180/180|119/180|66.1 [54.4,76.7]|30.0|0.0|180/180|
|scale300|matched_geometry_mlp|174/180|116/180|64.4 [52.8,75.0]|28.3|0.6|175/180|
|scale300|query_geometry_attention|171/180|113/180|62.8 [51.1,73.3]|27.8|1.1|173/180|
|scale300|drivetransformer_task_adapted|171/180|113/180|62.8 [51.1,73.3]|27.8|1.1|173/180|
|scale300|oracle_rule_reference|180/180|119/180|66.1 [54.4,76.7]|30.0|0.0|180/180|

scale300: 189 unique command execution attempts (including actor-association rejections); 180 paired observations per method; 60 independent groups.

|hard_confusable|graph_relation_gate|168/180|120/180|66.7 [55.6,77.2]|19.4|1.7|171/180|
|hard_confusable|matched_geometry_mlp|174/180|124/180|68.9 [58.3,78.9]|20.0|0.0|174/180|
|hard_confusable|query_geometry_attention|171/180|121/180|67.2 [56.1,77.2]|19.4|1.7|174/180|
|hard_confusable|drivetransformer_task_adapted|168/180|118/180|65.6 [54.4,76.1]|19.4|0.0|168/180|
|hard_confusable|oracle_rule_reference|180/180|127/180|70.6 [60.0,80.6]|21.1|0.0|180/180|

hard_confusable: 204 unique command execution attempts (including actor-association rejections); 180 paired observations per method; 60 independent groups.
