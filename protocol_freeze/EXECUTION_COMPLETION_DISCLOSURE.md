# Execution completion disclosure

The original `CLOSED_LOOP_PROTOCOL.md` is preserved. It intended one physical execution per unique (scene, rollout seed, chosen actor) command. The completed actor-association stage rejected 25 of 393 unique command attempts before a valid controller trajectory could be produced. The completion report therefore distinguishes attempts from valid controller trajectories: 393 attempted, 368 valid, 25 invalid. A rejected command remains an invalid observation for each selector issuing it and is never retried or replaced.

The 25 rejections comprise seven of nine additional Scale commands and 18 of 24 additional Hard commands. Reference commands produced 180 valid trajectories per track. Equal-command reuse remains exact and is not counted as a new physical run. Low wrong-target reach rates partly reflect rejected associations and cannot demonstrate safe avoidance. The original registration is not rewritten to conceal these outcomes.
