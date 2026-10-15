# Diagnostic: absolute-yaw leakage in self-bearing columns (2026-10-14 22:42+08)

Reconfirmed on TRAIN scale300 first 40 graphs with zero-distance ego rows.

Finding (matches frozen PROTOCOL.md smoke note):
- Source schema stores ego self-bearing (node cols 3:5) and anchor self-bearing (cols 8:10) as (cos(-yaw), sin(-yaw)) when distance is ~0.
- That injects absolute world heading into features.
- M1 ego_encode replaces those bearings by canonical (1,0).

Counts: n_checked=40, match_abs_yaw_leak=40, after_canonical=40, max_err=2.788e-08.
Prior smoke: encoding invariance max abs diff 7.6e-6 on 2400 pairs; rotation prediction identity 10800/10800 for M1/M12/FULL.
