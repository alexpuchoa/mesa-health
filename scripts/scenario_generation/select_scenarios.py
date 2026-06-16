#!/usr/bin/env python3
"""Select the benchmark core and derived pools from a file-based scenario feature table.

This is a transparency script. It expects a precomputed scenario feature CSV, typically
produced by `generate_scenarios.py` plus any extra eligibility flags required by a
particular pool design.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
import sys

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _benchmark_lib.io_utils import read_csv_rows, write_csv

BUNDLE_ROOT = SCRIPT_DIR.parents[1]


@dataclass(frozen=True)
class ScenarioFeature:
    scenario_id: int
    scenario_code: str
    conflict_level: str
    borda_winner_id: int
    is_divergent: bool
    pareto_set_size: int
    monotonicity_eligible: bool
    manipulation_eligible: bool
    targeted_exaggeration_eligible: bool


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raw = str(value or "").strip().lower()
    return raw in {"1", "true", "t", "yes", "y", "on"}


def _load_features(path: Path) -> List[ScenarioFeature]:
    rows = read_csv_rows(path)
    out: List[ScenarioFeature] = []
    for row in rows:
        out.append(
            ScenarioFeature(
                scenario_id=int(row["scenario_id"]),
                scenario_code=str(row.get("scenario_code") or f"scenario_{row['scenario_id']}"),
                conflict_level=str(row["conflict_level"]),
                borda_winner_id=int(row["borda_winner_id"]),
                is_divergent=_parse_bool(row.get("is_divergent")),
                pareto_set_size=int(row.get("pareto_set_size") or 0),
                monotonicity_eligible=_parse_bool(row.get("monotonicity_eligible")),
                manipulation_eligible=_parse_bool(row.get("manipulation_eligible")),
                targeted_exaggeration_eligible=_parse_bool(row.get("targeted_exaggeration_eligible")),
            )
        )
    return out


def _select_core_exact(rows: Sequence[ScenarioFeature], *, size: int, conflict_bins: Dict[str, int]) -> List[ScenarioFeature]:
    try:
        from pulp import LpBinary, LpMinimize, LpProblem, LpStatusOptimal, LpVariable, PULP_CBC_CMD, lpSum  # type: ignore
    except Exception as exc:
        raise SystemExit("Missing dependency: pulp. Install with: pip install pulp") from exc

    if size % 4 != 0:
        raise SystemExit("Core size must be divisible by 4 for balanced Borda-winner quotas")
    winner_quota = size // 4
    rows_sorted = sorted(rows, key=lambda item: item.scenario_id)
    x = {row.scenario_id: LpVariable(f"x_{row.scenario_id}", 0, 1, LpBinary) for row in rows_sorted}
    prob = LpProblem("benchmark_core_selection", LpMinimize)
    prob += 0
    prob += lpSum(x[row.scenario_id] for row in rows_sorted) == int(size)
    for level, quota in conflict_bins.items():
        level_rows = [row for row in rows_sorted if row.conflict_level == level]
        prob += lpSum(x[row.scenario_id] for row in level_rows) == int(quota)
    for winner_id in (1, 2, 3, 4):
        winner_rows = [row for row in rows_sorted if row.borda_winner_id == winner_id]
        prob += lpSum(x[row.scenario_id] for row in winner_rows) == int(winner_quota)
    status = prob.solve(PULP_CBC_CMD(msg=False))
    if status != LpStatusOptimal:
        raise SystemExit("Exact core selection was infeasible for the supplied feature table")
    return [row for row in rows_sorted if int(round(x[row.scenario_id].value())) == 1]


def _balanced_take(rows: Sequence[ScenarioFeature], *, size: int) -> List[ScenarioFeature]:
    buckets: Dict[int, List[ScenarioFeature]] = {1: [], 2: [], 3: [], 4: []}
    for row in sorted(rows, key=lambda item: item.scenario_id):
        buckets[int(row.borda_winner_id)].append(row)
    target = size // 4 if size % 4 == 0 else None
    picked: List[ScenarioFeature] = []
    if target is not None and all(len(bucket) >= target for bucket in buckets.values()):
        for winner_id in (1, 2, 3, 4):
            picked.extend(buckets[winner_id][:target])
        return sorted(picked, key=lambda item: item.scenario_id)
    merged = sorted(rows, key=lambda item: (item.borda_winner_id, item.scenario_id))
    return merged[:size]


def _write_selection(path: Path, rows: Sequence[ScenarioFeature], *, selection_name: str) -> None:
    out_rows = [
        {
            "selection_name": selection_name,
            "scenario_id": row.scenario_id,
            "scenario_code": row.scenario_code,
            "conflict_level": row.conflict_level,
            "borda_winner_id": row.borda_winner_id,
        }
        for row in rows
    ]
    write_csv(path, out_rows, fieldnames=["selection_name", "scenario_id", "scenario_code", "conflict_level", "borda_winner_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Select the benchmark core and derived pools from scenario features.")
    parser.add_argument("--scenario-features-csv", required=True)
    parser.add_argument("--design-yaml", default=str(BUNDLE_ROOT / "config" / "test_set_design.yaml"))
    parser.add_argument("--output-dir", default=str(BUNDLE_ROOT / "results" / "selection_sets"))
    args = parser.parse_args()

    features = _load_features(Path(args.scenario_features_csv))
    design = yaml.safe_load(Path(args.design_yaml).read_text(encoding="utf-8")) or {}
    core_cfg = dict(design.get("core") or {})
    core_size = int(core_cfg.get("size", 160))
    conflict_bins = {str(k): int(v) for k, v in (core_cfg.get("conflict_bins") or {}).items()}
    if set(conflict_bins.keys()) != {"low", "medium", "high"}:
        raise SystemExit("core.conflict_bins must define low/medium/high quotas")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    core_rows = _select_core_exact(features, size=core_size, conflict_bins=conflict_bins)
    _write_selection(out_dir / "core_160.csv", core_rows, selection_name="core_160")

    pools_cfg = dict(design.get("pools") or {})
    core_by_id = {row.scenario_id: row for row in core_rows}

    derived_specs = {
        "divergent": [row for row in core_rows if row.is_divergent],
        "pareto_ge2": [row for row in core_rows if row.pareto_set_size >= 2],
        "monotonicity": [row for row in core_rows if row.monotonicity_eligible],
        "manipulation": [row for row in core_rows if row.manipulation_eligible],
        "targeted_exaggeration": [row for row in core_rows if row.targeted_exaggeration_eligible],
        "all_scenarios": list(core_rows),
    }

    for pool_name, eligible_rows in derived_specs.items():
        pool_cfg = pools_cfg.get(pool_name)
        if not isinstance(pool_cfg, dict):
            continue
        size = int(pool_cfg.get("size") or 0)
        if size <= 0:
            continue
        if len(eligible_rows) < size:
            print(f"skip {pool_name}: only {len(eligible_rows)} eligible core scenarios for requested size={size}")
            continue
        picked = _balanced_take(eligible_rows, size=size)
        _write_selection(out_dir / f"{pool_name}.csv", picked, selection_name=pool_name)

    print(f"ok wrote core + derived pool CSVs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
