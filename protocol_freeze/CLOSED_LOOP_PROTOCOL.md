# R3 frozen-input shared-target execution protocol

Registration is fixed before new learned test predictions are read by the execution process.

Each track uses 60 distinct base graphs, selected using RNG20261002 as five tasks per relation x candidate-count cell. Three original paired rollout seeds are retained. Training seed42 is fixed for every learned selector; epoch is selected on validation only.

The controller input is the selected actor. Each unique (scene, rollout seed, selected actor) tuple is physically executed once under improved_v20, 0.05s simulation step and 1000-step horizon. Selectors issuing the same target command share that realization. This design attributes target-selection differences under a common execution realization. It does not test online perception, runtime feedback or concurrent model inference.

For each selector, report 180 paired acquisition outcomes, and separately report unique physically executed episodes and 60 independent graph clusters. Reused outcomes are not additional physical runs. Invalid and failed executions remain failures in every selector that issued the corresponding command. Do not retry failed conditions.

Oracle and the exact protocol rule have identical target commands on all 300 test tasks and share one execution-reference artifact. No controller parameters or scenes are chosen using the new model test results.

Primary metric: SemTargetAcq@8m. Secondary: ProtoAcc@1, @4m, wrong-target acquisition, validity, execution/selection failure decomposition. Uncertainty and contrasts use base-graph clusters with paired differences, bootstrap CI and Holm correction.
