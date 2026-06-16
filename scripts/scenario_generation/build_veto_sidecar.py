#!/usr/bin/env python3
"""Build the published veto sidecar from the benchmark veto selection CSV.

This transparency script converts the human-readable veto pool assignment CSV
into the machine-readable sidecar consumed by:

* ``scripts/generate_prompts.py``
* ``scripts/process_runs.py``

The source CSV already exists in the public bundle as
``data/selection_sets/veto_60.csv`` and contains the scenario-specific veto
assignment chosen during benchmark construction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _benchmark_lib.io_utils import read_csv_rows, write_yaml

BUNDLE_ROOT = SCRIPT_DIR.parents[1]


def _normalize_rows(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Convert veto selection CSV rows into published sidecar rows."""

    out: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        scenario_id = int(row["scenario_id"])
        role_raw = row.get("assigned_role_id", row.get("role_id"))
        if role_raw in (None, ""):
            raise ValueError(
                f"Veto CSV row for scenario_id={scenario_id} is missing assigned_role_id/role_id"
            )
        dimension_raw = row.get("dimension_id")
        if dimension_raw in (None, ""):
            raise ValueError(f"Veto CSV row for scenario_id={scenario_id} is missing dimension_id")
        stakeholder_raw = row.get(
            "assigned_scenario_stakeholder_id",
            row.get("scenario_stakeholder_id"),
        )
        if scenario_id in seen:
            raise ValueError(f"Duplicate scenario_id in veto CSV: {scenario_id}")
        seen.add(scenario_id)
        payload: Dict[str, Any] = {
            "scenario_id": scenario_id,
            "role_id": int(role_raw),
            "dimension_id": int(dimension_raw),
        }
        if stakeholder_raw not in (None, ""):
            payload["scenario_stakeholder_id"] = int(stakeholder_raw)
        out.append(payload)
    out.sort(key=lambda row: int(row["scenario_id"]))
    return out


def main() -> int:
    """CLI entry point for publishing the scenario-specific veto sidecar."""
    parser = argparse.ArgumentParser(
        description="Build the published scenario_vetoes.yaml sidecar from veto_60.csv."
    )
    parser.add_argument(
        "--veto-selection-csv",
        default=str(BUNDLE_ROOT / "data" / "selection_sets" / "veto_60.csv"),
        help="CSV with scenario-specific veto assignments.",
    )
    parser.add_argument(
        "--out-yaml",
        default=str(BUNDLE_ROOT / "data" / "scenario_vetoes.yaml"),
        help="Output YAML sidecar path.",
    )
    args = parser.parse_args()

    rows = read_csv_rows(Path(args.veto_selection_csv))
    payload = {"scenario_vetoes": _normalize_rows(rows)}
    write_yaml(Path(args.out_yaml), payload)
    print(f"ok wrote {len(payload['scenario_vetoes'])} veto rows to {args.out_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
