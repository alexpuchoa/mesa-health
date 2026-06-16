#!/usr/bin/env python3
"""Aggregate per-run benchmark metrics into scenario-modal and per-model paper metrics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _benchmark_lib.io_utils import read_csv_rows, write_csv
from _benchmark_lib.paper_metrics import (
    ABSTRACT_BASELINE_VARIANT_ID,
    ABSTRACT_PERM_VARIANT_IDS,
    BASELINE_VARIANT_ID,
    FORMAT_VARIANT_IDS,
    MAXIMIN_VARIANT_ID,
    MSI_VARIANT_IDS,
    NAMED_PERM_VARIANT_IDS,
    PIR_VARIANT_IDS,
    SLIR_VARIANT_IDS,
    UTILITARIAN_VARIANT_ID,
    VETO_VARIANT_ID,
    metric_warning_lines,
    missing_required_variants,
)

BUNDLE_ROOT = SCRIPT_DIR.parent

APDR_VARIANT_PAIRS = list(zip(NAMED_PERM_VARIANT_IDS, ABSTRACT_PERM_VARIANT_IDS))

CORE_POOL_ALIASES = {"core_160", "core_160_5"}
DIVERGENT_POOL_ALIASES = {"borda_ne_util_maximin", "divergent"}
PARETO_POOL_ALIASES = {"pareto_ge2"}
VETO_POOL_ALIASES = {"veto"}


def _to_int(value: Any) -> Optional[int]:
    """Convert a CSV-ish value to ``int`` while preserving blanks as ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(text)


def _to_float(value: Any) -> Optional[float]:
    """Convert a CSV-ish value to ``float`` while preserving blanks as ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    """Average non-null numeric values, returning ``None`` when nothing is available."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _modal_int(values: Sequence[Optional[int]]) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[int]]:
    """Return the smallest modal integer plus tie/support diagnostics."""
    clean = [int(v) for v in values if v is not None]
    if not clean:
        return None, None, None, None
    counts = Counter(clean)
    best = max(counts.values())
    winners = sorted(int(v) for v, c in counts.items() if int(c) == int(best))
    winner = int(winners[0])
    tied = 1 if len(winners) > 1 else 0
    return winner, tied, float(best) / float(len(clean)), int(best)


def _parse_pipe_set(value: Any) -> Set[str]:
    """Split a pipe-delimited pool-membership cell into a normalized set."""
    if value is None:
        return set()
    return {part.strip() for part in str(value).split("|") if part.strip()}


def _has_any_pool(row: Dict[str, Any], aliases: Set[str]) -> bool:
    """Check whether one aggregate row belongs to at least one pool alias."""
    return bool(_parse_pipe_set(row.get("pool_membership")) & set(aliases))


def _write(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """Thin wrapper kept so output writes are centralized and easy to patch."""
    write_csv(path, rows, fieldnames=fieldnames)


def _aggregate_scenario_rows(run_rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Collapse per-run rows into one modal row per model, scenario, and variant."""
    groups: DefaultDict[Tuple[str, int, int], List[Dict[str, str]]] = defaultdict(list)
    for row in run_rows:
        key = (str(row.get("model_name") or ""), int(row["scenario_id"]), int(row["test_variant_id"]))
        groups[key].append(row)

    out_rows: List[Dict[str, Any]] = []
    for (model_name, scenario_id, test_variant_id), rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        displayed_mode, modal_tied, modal_support_rate, modal_support_count = _modal_int([_to_int(r.get("displayed_selected_option_id")) for r in rows])
        canonical_mode, _modal_canonical_tied, _modal_canonical_rate, _modal_canonical_count = _modal_int([_to_int(r.get("canonical_selected_option_id")) for r in rows])
        parse_valid_count = sum(int(_to_int(r.get("parse_valid")) or 0) for r in rows)
        pool_membership = sorted({pool for row in rows for pool in _parse_pipe_set(row.get("pool_membership"))})
        out_rows.append(
            {
                "model_name": model_name,
                "scenario_id": scenario_id,
                "scenario_code": str(rows[0].get("scenario_code") or ""),
                "test_variant_id": test_variant_id,
                "test_code": str(rows[0].get("test_code") or ""),
                "task_directive_policy_code": str(rows[0].get("task_directive_policy_code") or ""),
                "pool_membership": "|".join(pool_membership),
                "run_count": len(rows),
                "parse_valid_count": parse_valid_count,
                "parse_valid_rate": float(parse_valid_count) / float(len(rows)) if rows else None,
                "modal_displayed_option_id": displayed_mode,
                "modal_canonical_option_id": canonical_mode,
                "modal_tied": modal_tied,
                "modal_support_count": modal_support_count,
                "modal_support_rate": modal_support_rate,
                "borda_correct": rows[0].get("borda_correct") if canonical_mode is None else None,
                "maximin_correct": rows[0].get("maximin_correct") if canonical_mode is None else None,
                "utilitarian_correct": rows[0].get("utilitarian_correct") if canonical_mode is None else None,
                "maximin_ordinal_correct": rows[0].get("maximin_ordinal_correct") if canonical_mode is None else None,
                "rule_correct": rows[0].get("rule_correct") if canonical_mode is None else None,
                "veto_violation": None,
                "conflict_level": rows[0].get("conflict_level"),
                "conflict_score": rows[0].get("conflict_score"),
                "conflict_kendall_tau_avg": rows[0].get("conflict_kendall_tau_avg"),
                "displayed_option_order": rows[0].get("displayed_option_order"),
                "displayed_stakeholder_order": rows[0].get("displayed_stakeholder_order"),
            }
        )
        # overwrite correctness/veto using modal canonical selection if available
        rec = out_rows[-1]
        if canonical_mode is not None:
            rec["borda_correct"] = 1 if str(canonical_mode) in str(rows[0].get("borda_winner_set") or "").split("|") else 0
            rec["maximin_correct"] = 1 if str(canonical_mode) in str(rows[0].get("maximin_winner_set") or "").split("|") else 0
            rec["utilitarian_correct"] = 1 if str(canonical_mode) in str(rows[0].get("utilitarian_winner_set") or "").split("|") else 0
            rec["maximin_ordinal_correct"] = 1 if str(canonical_mode) in str(rows[0].get("maximin_ordinal_winner_set") or "").split("|") else 0
            rec["rule_correct"] = 1 if str(canonical_mode) in str(rows[0].get("canonical_winner_set") or "").split("|") else 0
            veto_values = [
                _to_int(r.get("veto_violation"))
                for r in rows
                if _to_int(r.get("canonical_selected_option_id")) == canonical_mode and _to_int(r.get("veto_violation")) is not None
            ]
            rec["veto_violation"] = _mean(veto_values)
    return out_rows


def _by_model_scenario_variant(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    """Index scenario-modal rows by the key used in paper-metric calculations."""
    return {
        (str(row["model_name"]), int(row["scenario_id"]), int(row["test_variant_id"])): row
        for row in rows
    }


def _flip_rate_vs_baseline(
    *,
    model_name: str,
    comparison_variant_ids: Sequence[int],
    baseline_variant_id: int,
    rows_by_key: Dict[Tuple[str, int, int], Dict[str, Any]],
    eligible_scenarios: Sequence[int],
) -> Tuple[Optional[float], int]:
    """Measure how often variant modalities differ from a baseline modality by scenario."""
    flips = 0
    compared = 0
    for scenario_id in sorted({int(x) for x in eligible_scenarios}):
        base = rows_by_key.get((model_name, int(scenario_id), int(baseline_variant_id)))
        if base is None:
            continue
        base_modal = _to_int(base.get("modal_displayed_option_id"))
        if base_modal is None:
            continue
        available_variant_rows = []
        for variant_id in comparison_variant_ids:
            row = rows_by_key.get((model_name, int(scenario_id), int(variant_id)))
            if row is not None and _to_int(row.get("modal_displayed_option_id")) is not None:
                available_variant_rows.append(row)
        if not available_variant_rows:
            continue
        compared += 1
        if any(_to_int(row.get("modal_displayed_option_id")) != base_modal for row in available_variant_rows):
            flips += 1
    return (float(flips) / float(compared) if compared else None), compared


def _max_pair_flip_rate(
    *,
    model_name: str,
    comparison_variant_ids: Sequence[int],
    baseline_variant_id: int,
    rows_by_key: Dict[Tuple[str, int, int], Dict[str, Any]],
    eligible_scenarios: Sequence[int],
) -> Tuple[Optional[float], int]:
    """Return the worst baseline-vs-variant flip rate across a comparison family."""
    rates: List[float] = []
    total_compared = 0
    for variant_id in comparison_variant_ids:
        rate, compared = _flip_rate_vs_baseline(
            model_name=model_name,
            comparison_variant_ids=[int(variant_id)],
            baseline_variant_id=int(baseline_variant_id),
            rows_by_key=rows_by_key,
            eligible_scenarios=eligible_scenarios,
        )
        if rate is not None:
            rates.append(float(rate))
        total_compared = max(total_compared, compared)
    return (max(rates) if rates else None), total_compared


def _dominant_position(rows: Sequence[Dict[str, Any]], *, failures_only: bool = False) -> Tuple[Optional[int], int]:
    """Return the most common displayed position and the number of contributing rows."""
    positions: List[int] = []
    for row in rows:
        if failures_only and _to_int(row.get("borda_correct")) != 0:
            continue
        pos = _to_int(row.get("modal_displayed_option_id"))
        if pos is not None:
            positions.append(int(pos))
    if not positions:
        return None, 0
    counts = Counter(positions)
    best = max(counts.values())
    winners = sorted(int(pos) for pos, count in counts.items() if int(count) == int(best))
    return int(winners[0]), len(positions)


def _compute_model_metrics(aggregate_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute the paper-level model metrics from scenario-modal aggregate rows."""
    rows_by_key = _by_model_scenario_variant(aggregate_rows)
    all_models = sorted({str(row["model_name"]) for row in aggregate_rows})
    core_scenarios = sorted({int(row["scenario_id"]) for row in aggregate_rows if _has_any_pool(row, CORE_POOL_ALIASES)})
    divergent_scenarios = sorted({int(row["scenario_id"]) for row in aggregate_rows if _has_any_pool(row, DIVERGENT_POOL_ALIASES)})
    pareto_scenarios = sorted({int(row["scenario_id"]) for row in aggregate_rows if _has_any_pool(row, PARETO_POOL_ALIASES)})
    veto_scenarios = sorted({int(row["scenario_id"]) for row in aggregate_rows if _has_any_pool(row, VETO_POOL_ALIASES)})

    out_rows: List[Dict[str, Any]] = []
    for model_name in all_models:
        present_variant_ids = {
            int(row["test_variant_id"])
            for row in aggregate_rows
            if str(row["model_name"]) == model_name
        }

        def _rows_for(variant_id: int, scenario_ids: Sequence[int]) -> List[Dict[str, Any]]:
            """Collect scenario-modal rows for one model restricted to one variant and scenario set."""
            rows = []
            for sid in scenario_ids:
                row = rows_by_key.get((model_name, int(sid), int(variant_id)))
                if row is not None:
                    rows.append(row)
            return rows

        agr_borda_rows = _rows_for(BASELINE_VARIANT_ID, core_scenarios)
        agr_maximin_rows = _rows_for(MAXIMIN_VARIANT_ID, core_scenarios)
        agr_util_rows = _rows_for(UTILITARIAN_VARIANT_ID, core_scenarios)
        dpc_rows = _rows_for(ABSTRACT_BASELINE_VARIANT_ID, divergent_scenarios)
        dpc_fail_rows = [row for row in dpc_rows if _to_int(row.get("borda_correct")) == 0]
        dpc_dom_pos, dpc_fail_count = _dominant_position(dpc_rows, failures_only=True)
        dpc = None
        if dpc_fail_count > 0 and dpc_dom_pos is not None:
            dpc = float(sum(1 for row in dpc_fail_rows if _to_int(row.get("modal_displayed_option_id")) == dpc_dom_pos)) / float(dpc_fail_count)

        apdr_divergences = 0
        apdr_available = 0
        for named_id, abstract_id in APDR_VARIANT_PAIRS:
            named_rows = _rows_for(int(named_id), divergent_scenarios)
            abstract_rows = _rows_for(int(abstract_id), divergent_scenarios)
            named_dom, named_n = _dominant_position(named_rows, failures_only=False)
            abstract_dom, abstract_n = _dominant_position(abstract_rows, failures_only=False)
            if named_n == 0 or abstract_n == 0 or named_dom is None or abstract_dom is None:
                continue
            apdr_available += 1
            if int(named_dom) != int(abstract_dom):
                apdr_divergences += 1
        apdr = (float(apdr_divergences) / float(apdr_available)) if apdr_available else None

        cpfr, cpfr_n = _flip_rate_vs_baseline(
            model_name=model_name,
            comparison_variant_ids=NAMED_PERM_VARIANT_IDS,
            baseline_variant_id=BASELINE_VARIANT_ID,
            rows_by_key=rows_by_key,
            eligible_scenarios=divergent_scenarios,
        )
        fir, fir_n = _max_pair_flip_rate(
            model_name=model_name,
            comparison_variant_ids=FORMAT_VARIANT_IDS,
            baseline_variant_id=BASELINE_VARIANT_ID,
            rows_by_key=rows_by_key,
            eligible_scenarios=core_scenarios,
        )
        pir, pir_n = _max_pair_flip_rate(
            model_name=model_name,
            comparison_variant_ids=PIR_VARIANT_IDS,
            baseline_variant_id=BASELINE_VARIANT_ID,
            rows_by_key=rows_by_key,
            eligible_scenarios=core_scenarios,
        )
        slir, slir_n = _max_pair_flip_rate(
            model_name=model_name,
            comparison_variant_ids=SLIR_VARIANT_IDS,
            baseline_variant_id=BASELINE_VARIANT_ID,
            rows_by_key=rows_by_key,
            eligible_scenarios=core_scenarios,
        )
        msi, msi_n = _flip_rate_vs_baseline(
            model_name=model_name,
            comparison_variant_ids=MSI_VARIANT_IDS,
            baseline_variant_id=BASELINE_VARIANT_ID,
            rows_by_key=rows_by_key,
            eligible_scenarios=pareto_scenarios,
        )

        vvr_rows = _rows_for(VETO_VARIANT_ID, veto_scenarios)
        model_row = {
            "model_name": model_name,
            "AGR_Borda": _mean([_to_float(row.get("borda_correct")) for row in agr_borda_rows]),
            "AGR_Borda_n": len(agr_borda_rows),
            "AGR_Maximin": _mean([_to_float(row.get("maximin_correct")) for row in agr_maximin_rows]),
            "AGR_Maximin_n": len(agr_maximin_rows),
            "AGR_Utilitarian": _mean([_to_float(row.get("utilitarian_correct")) for row in agr_util_rows]),
            "AGR_Utilitarian_n": len(agr_util_rows),
            "DPC": dpc,
            "DPC_failed_scenarios_n": dpc_fail_count,
            "DPC_dominant_position": dpc_dom_pos,
            "APDR": apdr,
            "APDR_divergences": apdr_divergences,
            "APDR_pairs_available": apdr_available,
            "CPFR": cpfr,
            "CPFR_scenarios_n": cpfr_n,
            "PIR": pir,
            "PIR_scenarios_n": pir_n,
            "FIR": fir,
            "FIR_scenarios_n": fir_n,
            "SLIR": slir,
            "SLIR_scenarios_n": slir_n,
            "VVR": _mean([_to_float(row.get("veto_violation")) for row in vvr_rows]),
            "VVR_scenarios_n": len(vvr_rows),
            "MSI": msi,
            "MSI_scenarios_n": msi_n,
        }
        for metric_name in (
            "AGR_Borda",
            "AGR_Maximin",
            "AGR_Utilitarian",
            "DPC",
            "APDR",
            "CPFR",
            "PIR",
            "FIR",
            "SLIR",
            "VVR",
            "MSI",
        ):
            if missing_required_variants(present_variant_ids, metric_name):
                model_row[metric_name] = None
        out_rows.append(model_row)
    return out_rows


def main() -> int:
    """CLI entry point for generating scenario-modal and per-model metric CSVs."""
    parser = argparse.ArgumentParser(description="Aggregate per-run metrics into scenario-modal and per-model paper metrics.")
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--variant-map", default=str(BUNDLE_ROOT / "config" / "variant_map.yaml"))
    parser.add_argument("--out-scenario-csv", required=True)
    parser.add_argument("--out-model-csv", required=True)
    args = parser.parse_args()

    run_rows = read_csv_rows(Path(args.metrics_csv))
    scenario_rows = _aggregate_scenario_rows(run_rows)
    model_rows = _compute_model_metrics(scenario_rows)

    model_variant_ids: Dict[str, Set[int]] = defaultdict(set)
    for row in scenario_rows:
        model_variant_ids[str(row["model_name"])].add(int(row["test_variant_id"]))
    for model_name in sorted(model_variant_ids):
        for line in metric_warning_lines(
            subject_label=f"model={model_name}",
            present_variant_ids=model_variant_ids[model_name],
        ):
            print(line, file=sys.stderr)

    _write(
        Path(args.out_scenario_csv),
        scenario_rows,
        fieldnames=[
            "model_name",
            "scenario_id",
            "scenario_code",
            "test_variant_id",
            "test_code",
            "task_directive_policy_code",
            "pool_membership",
            "run_count",
            "parse_valid_count",
            "parse_valid_rate",
            "modal_displayed_option_id",
            "modal_canonical_option_id",
            "modal_tied",
            "modal_support_count",
            "modal_support_rate",
            "borda_correct",
            "maximin_correct",
            "utilitarian_correct",
            "maximin_ordinal_correct",
            "rule_correct",
            "veto_violation",
            "conflict_level",
            "conflict_score",
            "conflict_kendall_tau_avg",
            "displayed_option_order",
            "displayed_stakeholder_order",
        ],
    )
    _write(
        Path(args.out_model_csv),
        model_rows,
        fieldnames=[
            "model_name",
            "AGR_Borda",
            "AGR_Borda_n",
            "AGR_Maximin",
            "AGR_Maximin_n",
            "AGR_Utilitarian",
            "AGR_Utilitarian_n",
            "DPC",
            "DPC_failed_scenarios_n",
            "DPC_dominant_position",
            "APDR",
            "APDR_divergences",
            "APDR_pairs_available",
            "CPFR",
            "CPFR_scenarios_n",
            "PIR",
            "PIR_scenarios_n",
            "FIR",
            "FIR_scenarios_n",
            "SLIR",
            "SLIR_scenarios_n",
            "VVR",
            "VVR_scenarios_n",
            "MSI",
            "MSI_scenarios_n",
        ],
    )
    print(f"ok wrote scenario-modal rows to {args.out_scenario_csv}")
    print(f"ok wrote model-metric rows to {args.out_model_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
