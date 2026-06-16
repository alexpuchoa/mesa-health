#!/usr/bin/env python3
"""Process raw model runs into the per-run benchmark metrics table.

The script needs the frozen benchmark scenario bundle in JSON form so it can
recover authoritative winner sets and scenario properties for each response
row. The public CLI therefore names that input explicitly as
``--scenarios-master-json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _benchmark_lib.io_utils import read_csv_rows, write_csv
from _benchmark_lib.metrics import (
    agreement_flag,
    canonical_from_displayed,
    displayed_from_canonical,
    parse_selected_option_id,
    scenario_properties,
    veto_violation,
)
from _benchmark_lib.paper_metrics import metric_warning_lines
from _benchmark_lib.rendering import load_dimension_catalog, load_variant_map, resolve_variant_orders, variants_by_id
from _benchmark_lib.scenario_data import (
    load_optional_vetoes,
    load_pool_memberships,
    scenario_pool_memberships,
    scenarios_by_id,
)

BUNDLE_ROOT = SCRIPT_DIR.parent


def _normalize_rule_kind(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"borda", "utilitarian", "maximin", "maximin_ordinal", "nash", "copeland"}:
        return raw
    return "borda"


def _winner_set_for_rule(props: Dict[str, Any], rule_kind: str) -> List[int]:
    if rule_kind == "utilitarian":
        return [int(x) for x in props["utilitarian_winner_set"]]
    if rule_kind == "maximin":
        return [int(x) for x in props["maximin_winner_set"]]
    if rule_kind == "maximin_ordinal":
        return [int(x) for x in props["maximin_ordinal_winner_set"]]
    if rule_kind == "nash":
        return [int(x) for x in props["nash_winner_set"]]
    if rule_kind == "copeland":
        return [int(x) for x in props["copeland_winner_set"]]
    return [int(x) for x in props["borda_winner_set"]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Process raw model runs into the per-run benchmark metrics table.")
    parser.add_argument("--responses-csv", required=True)
    parser.add_argument(
        "--scenarios-master-json",
        "--scenarios",
        dest="scenarios_master_json",
        required=True,
        help="Path to the frozen benchmark scenarios master JSON bundle.",
    )
    parser.add_argument("--variant-map", default=str(BUNDLE_ROOT / "config" / "variant_map.yaml"))
    parser.add_argument("--pool-map", default=str(BUNDLE_ROOT / "config" / "test_set_variant_map.yaml"))
    parser.add_argument("--dimensions", default=str(BUNDLE_ROOT / "config" / "dimensions.yaml"))
    parser.add_argument(
        "--vetoes",
        default=str(BUNDLE_ROOT / "data" / "scenario_vetoes.yaml"),
        help="Scenario veto YAML/JSON/CSV sidecar. Defaults to the published bundle sidecar.",
    )
    parser.add_argument("--selected-option-column", default="selected_option_id")
    parser.add_argument("--response-text-column", default="response_text")
    parser.add_argument("--model-column", default="model_name")
    parser.add_argument("--run-index-column", default="run_index")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    response_rows = read_csv_rows(Path(args.responses_csv))
    scenarios = scenarios_by_id(Path(args.scenarios_master_json))
    variant_map = load_variant_map(Path(args.variant_map))
    variant_catalog = variants_by_id(variant_map)
    pool_memberships = load_pool_memberships(Path(args.pool_map))
    _dims, _dim_catalog, dim_id_to_code = load_dimension_catalog(Path(args.dimensions))
    vetoes_by_scenario = load_optional_vetoes(Path(args.vetoes) if args.vetoes else None)

    props_cache: Dict[int, Dict[str, Any]] = {}
    out_rows: List[Dict[str, Any]] = []
    model_variant_ids: Dict[str, set[int]] = {}

    for row in response_rows:
        scenario_id = int(row["scenario_id"])
        test_variant_id = int(row["test_variant_id"])
        model_name = str(row.get(args.model_column) or "")
        run_index = int(str(row.get(args.run_index_column)).strip()) if row.get(args.run_index_column) not in (None, "") else None
        scenario = scenarios.get(scenario_id)
        if scenario is None:
            raise ValueError(f"Response row references unknown scenario_id={scenario_id}")
        variant = variant_catalog.get(test_variant_id)
        if variant is None:
            raise ValueError(f"Response row references unknown test_variant_id={test_variant_id}")
        if scenario_id not in props_cache:
            props_cache[scenario_id] = scenario_properties(scenarios[scenario_id])
        props = props_cache[scenario_id]
        model_variant_ids.setdefault(model_name, set()).add(test_variant_id)
        stakeholder_order, displayed_option_order = resolve_variant_orders(variant_map, variant)
        rule_kind = _normalize_rule_kind(str(variant.get("task_directive_policy_code") or "borda"))
        canonical_winner_set = _winner_set_for_rule(props, rule_kind)
        displayed_winner_set = [displayed_from_canonical(opt, displayed_option_order) for opt in canonical_winner_set]

        displayed_selected: Optional[int]
        parse_error: Optional[str]
        selected_raw = row.get(args.selected_option_column)
        if selected_raw is not None and str(selected_raw).strip() != "":
            displayed_selected = int(str(selected_raw).strip())
            parse_error = None
        else:
            displayed_selected, parse_error = parse_selected_option_id(str(row.get(args.response_text_column) or ""))
        canonical_selected = None
        if displayed_selected is not None:
            try:
                canonical_selected = canonical_from_displayed(displayed_selected, displayed_option_order)
            except Exception as exc:
                parse_error = str(exc)
                displayed_selected = None
                canonical_selected = None

        out_rows.append(
            {
                "model_name": model_name,
                "run_index": run_index,
                "scenario_id": scenario_id,
                "scenario_code": str(props["scenario_code"]),
                "test_variant_id": test_variant_id,
                "test_code": str(variant.get("test_code") or ""),
                "task_directive_policy_code": rule_kind,
                "pool_membership": "|".join(scenario_pool_memberships(scenario_id, pool_memberships)),
                "displayed_option_order": "|".join(str(x) for x in displayed_option_order),
                "displayed_stakeholder_order": "|".join(str(x) for x in stakeholder_order),
                "displayed_selected_option_id": displayed_selected,
                "canonical_selected_option_id": canonical_selected,
                "parse_valid": 1 if displayed_selected is not None else 0,
                "parse_error": parse_error,
                "borda_correct": agreement_flag(canonical_selected, props["borda_winner_set"]),
                "maximin_correct": agreement_flag(canonical_selected, props["maximin_winner_set"]),
                "utilitarian_correct": agreement_flag(canonical_selected, props["utilitarian_winner_set"]),
                "maximin_ordinal_correct": agreement_flag(canonical_selected, props["maximin_ordinal_winner_set"]),
                "rule_correct": agreement_flag(canonical_selected, canonical_winner_set),
                "displayed_winner_set": "|".join(str(x) for x in sorted(displayed_winner_set)),
                "canonical_winner_set": "|".join(str(x) for x in sorted(canonical_winner_set)),
                "is_correct": agreement_flag(canonical_selected, canonical_winner_set),
                "veto_violation": veto_violation(
                    canonical_selected,
                    scenario,
                    vetoes_by_scenario,
                    dimension_id_to_code=dim_id_to_code,
                ),
                "conflict_level": str(props["conflict_level"]),
                "conflict_score": float(props["conflict_score"]),
                "conflict_kendall_tau_avg": float(props["conflict_kendall_tau_avg"]),
                "borda_winner_set": "|".join(str(x) for x in props["borda_winner_set"]),
                "utilitarian_winner_set": "|".join(str(x) for x in props["utilitarian_winner_set"]),
                "maximin_winner_set": "|".join(str(x) for x in props["maximin_winner_set"]),
                "maximin_ordinal_winner_set": "|".join(str(x) for x in props["maximin_ordinal_winner_set"]),
            }
        )

    for model_name in sorted(model_variant_ids):
        for line in metric_warning_lines(
            subject_label=f"model={model_name}",
            present_variant_ids=model_variant_ids[model_name],
        ):
            print(line, file=sys.stderr)

    fieldnames = [
        "model_name",
        "run_index",
        "scenario_id",
        "scenario_code",
        "test_variant_id",
        "test_code",
        "task_directive_policy_code",
        "pool_membership",
        "displayed_option_order",
        "displayed_stakeholder_order",
        "displayed_selected_option_id",
        "canonical_selected_option_id",
        "parse_valid",
        "parse_error",
        "borda_correct",
        "maximin_correct",
        "utilitarian_correct",
        "maximin_ordinal_correct",
        "rule_correct",
        "displayed_winner_set",
        "canonical_winner_set",
        "is_correct",
        "veto_violation",
        "conflict_level",
        "conflict_score",
        "conflict_kendall_tau_avg",
        "borda_winner_set",
        "utilitarian_winner_set",
        "maximin_winner_set",
        "maximin_ordinal_winner_set",
    ]
    write_csv(Path(args.out_csv), out_rows, fieldnames=fieldnames)
    print(f"ok wrote {len(out_rows)} metric rows to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
