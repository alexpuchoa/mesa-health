#!/usr/bin/env python3
"""Generate the full scenario universe and transparency-side feature tables."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _benchmark_lib.io_utils import write_csv, write_json, write_yaml
from _benchmark_lib.metrics import scenario_properties
from scenario_generation._scenario_generation_lib import (
    build_scenarios,
    load_archetypes,
    load_dimensions,
    load_options,
)


BUNDLE_ROOT = SCRIPT_DIR.parents[1]


def _parse_formats(raw: str) -> List[str]:
    """Validate the requested output-format list for scenario generation artifacts."""
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    allowed = {"yaml", "json", "csv"}
    unknown = [v for v in values if v not in allowed]
    if unknown:
        raise ValueError(f"Unknown formats: {unknown}. Allowed: {sorted(allowed)}")
    if not values:
        raise ValueError("At least one output format is required")
    return values


def main() -> int:
    """CLI entry point for regenerating the transparency-side scenario universe."""
    parser = argparse.ArgumentParser(description="Generate deterministic scenarios and transparency-side feature tables.")
    parser.add_argument("--dimensions", default=str(BUNDLE_ROOT / "config" / "dimensions.yaml"))
    parser.add_argument("--archetypes", default=str(BUNDLE_ROOT / "config" / "archetypes.yaml"))
    parser.add_argument("--options", default=str(BUNDLE_ROOT / "config" / "options.yaml"))
    parser.add_argument("--output-dir", default=str(BUNDLE_ROOT / "results" / "scenario_generation"))
    parser.add_argument("--output-prefix", default="scenarios_master")
    parser.add_argument("--formats", default="yaml,json,csv")
    parser.add_argument("--scenario-id-start", type=int, default=1)
    args = parser.parse_args()

    dims_path = Path(args.dimensions)
    archetypes_path = Path(args.archetypes)
    options_path = Path(args.options)
    out_dir = Path(args.output_dir)
    formats = _parse_formats(args.formats)

    dimensions = load_dimensions(dims_path)
    archetypes = load_archetypes(archetypes_path, dimensions=dimensions)
    options = load_options(options_path, dimensions=dimensions)
    scenarios, utilities, rankings, scenarios_flat = build_scenarios(
        archetypes=archetypes,
        options=options,
        dimensions=dimensions,
        scenario_id_start=args.scenario_id_start,
    )
    scenario_feature_rows = [scenario_properties(scenario) for scenario in scenarios]

    payload = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scenario_count": len(scenarios),
            "dimensions": dimensions,
            "archetypes_source": str(archetypes_path),
            "options_source": str(options_path),
            "dimensions_source": str(dims_path),
            "generator": "scripts/scenario_generation/generate_scenarios.py",
        },
        "scenarios": scenarios,
    }

    base = out_dir / args.output_prefix
    if "yaml" in formats:
        write_yaml(base.with_suffix(".yaml"), payload)
    if "json" in formats:
        write_json(base.with_suffix(".json"), payload)
    if "csv" in formats:
        write_csv(
            base.with_name(f"{args.output_prefix}_scenarios.csv"),
            scenarios_flat,
            fieldnames=list(scenarios_flat[0].keys()) if scenarios_flat else ["scenario_id", "scenario_code"],
        )
        write_csv(
            base.with_name(f"{args.output_prefix}_utilities.csv"),
            utilities,
            fieldnames=["scenario_id", "scenario_code", "role_id", "archetype_id", "option_id", "utility"],
        )
        write_csv(
            base.with_name(f"{args.output_prefix}_rankings.csv"),
            rankings,
            fieldnames=["scenario_id", "scenario_code", "role_id", "archetype_id", "option_id", "utility", "rank"],
        )
        feature_fields = [
            "scenario_id",
            "scenario_code",
            "borda_winner_id",
            "borda_tie_size",
            "borda_winner_set",
            "utilitarian_winner_id",
            "utilitarian_tie_size",
            "utilitarian_winner_set",
            "maximin_winner_id",
            "maximin_tie_size",
            "maximin_winner_set",
            "maximin_ordinal_winner_id",
            "maximin_ordinal_tie_size",
            "maximin_ordinal_winner_set",
            "pareto_optimal_set",
            "pareto_set_size",
            "borda_winner_is_pareto",
            "is_divergent",
            "conflict_level",
            "conflict_score",
            "conflict_kendall_tau_avg",
            "conflict_top_choice_distinct_count",
            "conflict_rules5_entropy",
        ]
        feature_rows = []
        for row in scenario_feature_rows:
            feature_rows.append(
                {
                    key: ("|".join(str(v) for v in row[key]) if isinstance(row.get(key), list) else row.get(key))
                    for key in feature_fields
                }
            )
        write_csv(base.with_name(f"{args.output_prefix}_scenario_properties.csv"), feature_rows, fieldnames=feature_fields)

    print(f"ok generated {len(scenarios)} scenarios")
    print(f"ok outputs at {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
