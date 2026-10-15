"""Shared-command execution reuse library for protocol-driven CARLA evaluation.

Contract (shared-command attribution)
-------------------------------------
Given a frozen world layout, a frozen rollout seed, and a frozen controller
configuration, the physical outcome of an episode is a function of the issued
command, i.e. of the *selected protocol identity*:

    outcome = F(scene, rollout_seed, selected_identity; controller_config)

Two selectors that issue the identical command therefore describe the *same*
physical realization. Spending a second CARLA episode on that command does not
produce new information about either selector; it produces a new realization,
which is exactly the confound this library removes.

Usage
-----
    cache = ExecutionCache(path, readonly=False)
    key = cache.make_key(scen_id=0, carla_seed=7, chosen_idx_graph=2,
                         controller_profile="...", planner_mode="")
    hit = cache.get(key)             # -> payload dict or None
    cache.put(key, nav_result=..., actor=...)   # only on a real execution

`readonly=True` makes lookups reuse-only: a miss is reported and nothing is
written, so a verification pass cannot silently extend the library.

Every stored entry keeps the resolved actor signature and the source selector,
so a reviewer can confirm that a reused command genuinely denotes the same
physical actor rather than merely the same protocol index.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

CACHE_VERSION = 2

# Outcome fields required to reconstruct a result record without re-execution.
NAV_RESULT_FIELDS = (
    "start_dist",
    "min_dist",
    "final_dist",
    "steps",
    "success_4m",
    "success_8m",
    "stuck",
)


def _normalize(value: Any) -> str:
    return "" if value is None else str(value)


class ExecutionCache:
    """Append-only JSONL library of physical realizations keyed by command."""

    def __init__(self, path: str, readonly: bool = False) -> None:
        self.path = os.path.abspath(path)
        self.readonly = bool(readonly)
        self._entries: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if os.path.exists(self.path):
            self._load()

    # ------------------------------------------------------------------ keys
    @staticmethod
    def make_key(
        scen_id: int,
        carla_seed: Any,
        chosen_idx_graph: int,
        controller_profile: Any = "",
        planner_mode: Any = "",
    ) -> str:
        """Identity of a physical command under a fixed controller configuration."""
        return "|".join(
            [
                f"v{CACHE_VERSION}",
                f"scen={int(scen_id)}",
                f"seed={_normalize(carla_seed)}",
                f"idx={int(chosen_idx_graph)}",
                f"ctrl={_normalize(controller_profile)}",
                f"plan={_normalize(planner_mode)}",
            ]
        )

    # --------------------------------------------------------------- storage
    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = record.get("key")
                if key:
                    self._entries[key] = record

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return entry

    def put(
        self,
        key: str,
        nav_result: Dict[str, Any],
        actor: Optional[Dict[str, Any]] = None,
        source: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store one physical realization. Refuses to overwrite an existing entry."""
        if self.readonly:
            raise RuntimeError(
                "ExecutionCache is read-only; refusing to write a new realization"
            )
        if key in self._entries:
            return
        payload = {field: nav_result.get(field) for field in NAV_RESULT_FIELDS}
        # Safety telemetry is part of the physical realization and must survive reuse:
        # a reused command has to report the same contact/lane evidence as the original.
        payload["safety"] = nav_result.get("safety", {}) or {}
        record = {
            "cache_version": CACHE_VERSION,
            "key": key,
            "nav_result": payload,
            "actor": actor or {},
            "source": source or {},
        }
        self._entries[key] = record
        self.writes += 1
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    # ----------------------------------------------------------------- audit
    def stats(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "readonly": self.readonly,
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
        }

    def write_stats(self, out_path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = self.stats()
        if extra:
            payload.update(extra)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
