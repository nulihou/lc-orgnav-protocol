"""Shared JSONL safety-field normalization for protocol-driven CARLA runs."""

from __future__ import annotations

from typing import Any, Dict


SAFETY_DEFAULTS: Dict[str, Any] = {
    "safety_logger_version": None,
    "collision": None,
    "collision_count": None,
    "collision_actor_ids": [],
    "collision_actor_type_ids": [],
    "collision_events": [],
    "collision_first_frame": None,
    "collision_max_normal_impulse": None,
    "dynamic_actor_collision": None,
    "dynamic_actor_collision_count": None,
    "static_only_collision": None,
    "severe_collision": None,
    "severe_collision_count": None,
    "severe_impulse_threshold": None,
    "lane_invasion": None,
    "lane_invasion_count": None,
    "lane_marking_types": [],
    "min_ttc_s": None,
    "min_ttc_actor_id": None,
    "min_ttc_distance_m": None,
    "min_distance_to_vehicle_m": None,
    "safety_sensor_status": {},
}

NAV_DEFAULTS: Dict[str, Any] = {
    "success_4m": False,
    "success_8m": False,
}


def _copy_default(value: Any) -> Any:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def ensure_safety_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """Mutate and return a JSONL record with a uniform safety schema."""
    record.setdefault("valid", True)
    for key, default_value in NAV_DEFAULTS.items():
        record.setdefault(key, default_value)

    safety = record.get("safety")
    if isinstance(safety, dict):
        for key in SAFETY_DEFAULTS:
            if key in safety:
                record[key] = safety[key]

    if record.get("safety_logger_version") is None:
        for key, default_value in SAFETY_DEFAULTS.items():
            record[key] = _copy_default(default_value)
    else:
        for key, default_value in SAFETY_DEFAULTS.items():
            record.setdefault(key, _copy_default(default_value))
    return record
