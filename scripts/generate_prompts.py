#!/usr/bin/env python3
"""Render public benchmark prompts from fixed scenario data and variant config."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _benchmark_lib.io_utils import write_csv
from _benchmark_lib.rendering import (
    load_dimension_catalog,
    load_role_catalog,
    load_task_directives,
    load_variant_map,
    render_prompt_for_scenario,
    variants_by_id,
)
from _benchmark_lib.scenario_data import load_optional_vetoes, load_scenarios, load_selection_set_ids

BUNDLE_ROOT = SCRIPT_DIR.parent


def _parse_ids(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    out: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or None


def main() -> int:
    parser = argparse.ArgumentParser(description="Render prompts from fixed scenario data and public variant config.")
    parser.add_argument("--scenarios", required=True, help="Scenario bundle JSON/YAML from the published data artifact.")
    parser.add_argument("--variant-map", default=str(BUNDLE_ROOT / "config" / "variant_map.yaml"))
    parser.add_argument("--task-directives", default=str(BUNDLE_ROOT / "config" / "task_directives.yaml"))
    parser.add_argument("--dimensions", default=str(BUNDLE_ROOT / "config" / "dimensions.yaml"))
    parser.add_argument("--stakeholder-roles", default=str(BUNDLE_ROOT / "config" / "stakeholder_roles.yaml"))
    parser.add_argument("--selection-set-csv", default=None, help="Optional scenario_id filter CSV.")
    parser.add_argument("--scenario-ids", default=None, help="Optional comma-separated scenario_ids filter.")
    parser.add_argument("--test-variant-ids", default=None, help="Optional comma-separated test_variant_ids filter.")
    parser.add_argument(
        "--vetoes",
        default=str(BUNDLE_ROOT / "data" / "scenario_vetoes.yaml"),
        help="Scenario veto YAML/JSON/CSV sidecar. Defaults to the published bundle sidecar.",
    )
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    scenarios = load_scenarios(Path(args.scenarios))
    variant_map = load_variant_map(Path(args.variant_map))
    variant_catalog = variants_by_id(variant_map)
    task_directives = load_task_directives(Path(args.task_directives))
    _dims, dimension_catalog, dim_id_to_code = load_dimension_catalog(Path(args.dimensions))
    role_catalog = load_role_catalog(Path(args.stakeholder_roles))
    vetoes_by_scenario = load_optional_vetoes(Path(args.vetoes) if args.vetoes else None)

    scenario_ids = set(_parse_ids(args.scenario_ids) or [])
    if args.selection_set_csv:
        scenario_ids.update(load_selection_set_ids(Path(args.selection_set_csv)))
    test_variant_ids = _parse_ids(args.test_variant_ids)
    variant_ids = test_variant_ids if test_variant_ids is not None else sorted(variant_catalog.keys())

    rows: List[Dict[str, object]] = []
    for scenario in scenarios:
        scenario_id = int(scenario["scenario_id"])
        if scenario_ids and scenario_id not in scenario_ids:
            continue
        for test_variant_id in variant_ids:
            variant = variant_catalog.get(int(test_variant_id))
            if variant is None:
                raise ValueError(f"Unknown test_variant_id={test_variant_id} in variant map")
            rendered = render_prompt_for_scenario(
                scenario=scenario,
                variant=variant,
                variant_map=variant_map,
                config_dir=Path(args.variant_map).resolve().parent,
                task_directives=task_directives,
                dimension_catalog=dimension_catalog,
                dim_id_to_code=dim_id_to_code,
                role_catalog=role_catalog,
                vetoes_by_scenario=vetoes_by_scenario,
            )
            prompt_text = str(rendered["prompt_text"])
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_code": str(scenario.get("scenario_code") or ""),
                    "test_variant_id": int(test_variant_id),
                    "test_code": str(rendered["test_code"]),
                    "variant_name": str(rendered["variant_name"]),
                    "task_directive_policy_code": str(rendered["task_directive_policy_code"]),
                    "task_directive": str(rendered["task_directive"]),
                    "displayed_option_order": "|".join(str(x) for x in rendered["displayed_option_order"]),
                    "displayed_stakeholder_order": "|".join(str(x) for x in rendered["displayed_stakeholder_order"]),
                    "prompt_yaml": str(rendered["prompt_yaml"]),
                    "render_config_yaml": str(rendered["render_config_yaml"]),
                    "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                    "prompt_text": prompt_text,
                }
            )

    rows.sort(key=lambda row: (int(row["scenario_id"]), int(row["test_variant_id"])))
    write_csv(
        Path(args.out_csv),
        rows,
        fieldnames=[
            "scenario_id",
            "scenario_code",
            "test_variant_id",
            "test_code",
            "variant_name",
            "task_directive_policy_code",
            "task_directive",
            "displayed_option_order",
            "displayed_stakeholder_order",
            "prompt_yaml",
            "render_config_yaml",
            "prompt_sha256",
            "prompt_text",
        ],
    )
    print(f"ok wrote {len(rows)} prompts to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
