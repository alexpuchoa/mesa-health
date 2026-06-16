#!/usr/bin/env python3
"""Transparency-only ordinal rank-sum design optimizer.

This script is adapted from the internal ordinal design optimizer and is included
for methodological inspection only. It searches over dimension-order permutations
for:
  - 4 options
  - 16 archetypes (4 roles x 4 archetypes)

Each row is constrained to the fixed ordinal rank-sum weights:
  rank 1 -> 0.4
  rank 2 -> 0.3
  rank 3 -> 0.2
  rank 4 -> 0.1

The optimizer evaluates the full 256-scenario grid induced by the 16 archetypes.
It does not generate benchmark responses or metrics; it only helps explain how the
published archetype/option profiles could be designed before the benchmark was frozen.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import sys

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

BUNDLE_ROOT = SCRIPT_DIR.parents[1]

RANK_SUM_WEIGHTS = np.array([0.4, 0.3, 0.2, 0.1], dtype=float)
SCENARIO_ROLE_ARCH_IDX = np.array(list(itertools.product(range(4), repeat=4)), dtype=int)


@dataclass(frozen=True)
class InputContent:
    dim_codes: List[str]
    archetypes: List[dict]
    options: List[dict]
    role_ids: List[int]
    role_to_archetype_indices: List[List[int]]
    archetypes_doc: dict
    options_doc: dict
    archetypes_path: Path
    options_path: Path


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_inputs(content_dir: Path) -> InputContent:
    dimensions_path = content_dir / "dimensions.yaml"
    archetypes_path = content_dir / "archetypes.yaml"
    options_path = content_dir / "options.yaml"

    dimensions = _load_yaml(dimensions_path)
    archetypes_doc = _load_yaml(archetypes_path)
    options_doc = _load_yaml(options_path)

    dim_codes = [d["code"] for d in dimensions.get("dimensions", [])]
    if len(dim_codes) != 4:
        raise ValueError(f"Expected exactly 4 dimensions in {dimensions_path}, found {len(dim_codes)}")

    archetypes = list(archetypes_doc.get("archetypes", []))
    if len(archetypes) != 16:
        raise ValueError(f"Expected exactly 16 archetypes in {archetypes_path}, found {len(archetypes)}")

    options = list(options_doc.get("options", []))
    if len(options) != 4:
        raise ValueError(f"Expected exactly 4 options in {options_path}, found {len(options)}")

    role_ids = sorted({int(a["role_id"]) for a in archetypes})
    if len(role_ids) != 4:
        raise ValueError(f"Expected exactly 4 distinct role_id values in {archetypes_path}, found {role_ids}")

    role_to_archetype_indices: List[List[int]] = []
    for rid in role_ids:
        idx = [i for i, a in enumerate(archetypes) if int(a["role_id"]) == rid]
        if len(idx) != 4:
            raise ValueError(f"Role {rid} must have exactly 4 archetypes, found {len(idx)}")
        idx = sorted(idx, key=lambda i: int(archetypes[i]["archetype_id"]))
        role_to_archetype_indices.append(idx)

    return InputContent(
        dim_codes=dim_codes,
        archetypes=archetypes,
        options=options,
        role_ids=role_ids,
        role_to_archetype_indices=role_to_archetype_indices,
        archetypes_doc=archetypes_doc,
        options_doc=options_doc,
        archetypes_path=archetypes_path,
        options_path=options_path,
    )


def _weights_row_from_mapping(mapping: Dict[str, float], dim_codes: List[str]) -> np.ndarray:
    row = np.zeros(len(dim_codes), dtype=float)
    for j, d in enumerate(dim_codes):
        if d not in mapping:
            raise ValueError(f"Missing dimension '{d}' in dimension_weights")
        row[j] = float(mapping[d])
    return row


def _ordered_dim_indices(row: np.ndarray) -> Tuple[int, ...]:
    if row.shape != (4,):
        raise ValueError("row must be shape (4,)")
    return tuple(int(i) for i in np.argsort(-row, kind="mergesort").tolist())


def _perm_to_row(perm: Sequence[int]) -> np.ndarray:
    out = np.zeros(4, dtype=float)
    for rank_pos, dim_idx in enumerate(perm):
        out[int(dim_idx)] = float(RANK_SUM_WEIGHTS[int(rank_pos)])
    return out


def _candidate_to_matrices(
    *,
    candidate_options_perm_idx: List[int],
    candidate_arch_perm_idx_by_role: List[List[int]],
    all_perms: List[Tuple[int, ...]],
) -> Tuple[np.ndarray, np.ndarray]:
    options = np.zeros((4, 4), dtype=float)
    for i in range(4):
        options[i, :] = _perm_to_row(all_perms[int(candidate_options_perm_idx[i])])

    archetypes_by_role = np.zeros((4, 4, 4), dtype=float)
    for role_i in range(4):
        for k in range(4):
            perm = all_perms[int(candidate_arch_perm_idx_by_role[role_i][k])]
            archetypes_by_role[role_i, k, :] = _perm_to_row(perm)
    return archetypes_by_role, options


def _credit_with_ties(scores_2d: np.ndarray, *, atol: float) -> Tuple[np.ndarray, np.ndarray]:
    top = np.max(scores_2d, axis=1, keepdims=True)
    tied = np.isclose(scores_2d, top, atol=atol)
    tie_size = np.sum(tied, axis=1, keepdims=True).astype(float)
    credit = tied.astype(float) / tie_size
    return credit, tie_size.reshape(-1).astype(int)


def _evaluate_scenario_set(
    *,
    archetypes_by_role: np.ndarray,
    options: np.ndarray,
    atol: float,
) -> Dict[str, object]:
    idx = SCENARIO_ROLE_ARCH_IDX
    stakeholder_weights = np.empty((256, 4, 4), dtype=float)
    stakeholder_weights[:, 0, :] = archetypes_by_role[0, idx[:, 0], :]
    stakeholder_weights[:, 1, :] = archetypes_by_role[1, idx[:, 1], :]
    stakeholder_weights[:, 2, :] = archetypes_by_role[2, idx[:, 2], :]
    stakeholder_weights[:, 3, :] = archetypes_by_role[3, idx[:, 3], :]

    utilities = stakeholder_weights @ options.T
    util_totals = utilities.sum(axis=1)

    util_credit, util_tie_size = _credit_with_ties(util_totals, atol=atol)
    util_wins = util_credit.sum(axis=0)

    ranks = np.argsort(-utilities, axis=2, kind="mergesort")
    borda_scores = np.zeros((256, 4), dtype=float)
    scenario_rep = np.repeat(np.arange(256), 4)
    for rank_pos, pts in enumerate((3.0, 2.0, 1.0, 0.0)):
        opt_idx = ranks[:, :, rank_pos].reshape(-1)
        np.add.at(borda_scores, (scenario_rep, opt_idx), float(pts))
    borda_credit, borda_tie_size = _credit_with_ties(borda_scores, atol=atol)
    borda_wins = borda_credit.sum(axis=0)

    borda_top_mask = borda_credit > 0.0
    util_top_mask = util_credit > 0.0
    rule_diff_rate = float(np.mean(~np.all(borda_top_mask == util_top_mask, axis=1)))

    conflict = np.var(stakeholder_weights, axis=1).mean(axis=1)

    scenario_rows: List[dict] = []
    for scen_i in range(256):
        s1, s2, s3, s4 = [int(x) for x in idx[scen_i].tolist()]
        scenario_rows.append(
            {
                "scenario_index": int(scen_i),
                "s1": s1,
                "s2": s2,
                "s3": s3,
                "s4": s4,
                "borda_tie_for_first": int(borda_tie_size[scen_i] > 1),
                "borda_credit_o1": float(borda_credit[scen_i, 0]),
                "borda_credit_o2": float(borda_credit[scen_i, 1]),
                "borda_credit_o3": float(borda_credit[scen_i, 2]),
                "borda_credit_o4": float(borda_credit[scen_i, 3]),
                "utilitarian_tie_for_first": int(util_tie_size[scen_i] > 1),
                "util_credit_o1": float(util_credit[scen_i, 0]),
                "util_credit_o2": float(util_credit[scen_i, 1]),
                "util_credit_o3": float(util_credit[scen_i, 2]),
                "util_credit_o4": float(util_credit[scen_i, 3]),
                "rule_differs_borda_vs_utilitarian": int(not np.all(borda_top_mask[scen_i] == util_top_mask[scen_i])),
                "conflict": float(conflict[scen_i]),
            }
        )

    return {
        "borda_wins": borda_wins,
        "util_wins": util_wins,
        "borda_tie_count": int(np.sum(borda_tie_size > 1)),
        "util_tie_count": int(np.sum(util_tie_size > 1)),
        "borda_tie_rate": float(np.mean(borda_tie_size > 1)),
        "util_tie_rate": float(np.mean(util_tie_size > 1)),
        "rule_diff_rate": rule_diff_rate,
        "scenario_rows": scenario_rows,
    }


def _loss_from_metrics(
    *,
    metrics: Dict[str, object],
    alpha: float,
    beta: float,
    tie_weight: float,
    target_diff_rate: float | None,
    diff_weight: float,
) -> float:
    borda_wins = np.array(metrics["borda_wins"], dtype=float)
    util_wins = np.array(metrics["util_wins"], dtype=float)
    target = 64.0
    loss = float(alpha) * float(np.sum((borda_wins - target) ** 2))
    loss += float(beta) * float(np.sum((util_wins - target) ** 2))
    if tie_weight > 0:
        loss += float(tie_weight) * float(metrics["borda_tie_rate"] ** 2)
    if target_diff_rate is not None and diff_weight > 0:
        loss += float(diff_weight) * float((float(metrics["rule_diff_rate"]) - float(target_diff_rate)) ** 2)
    return float(loss)


def _check_candidate_constraints(
    *,
    option_perm_idx: List[int],
    arch_perm_idx_by_role: List[List[int]],
    all_perms: List[Tuple[int, ...]],
    require_unique_option_top_dim: bool,
    require_unique_archetypes_within_role: bool,
) -> bool:
    if len(option_perm_idx) != 4:
        return False
    if len(set(int(x) for x in option_perm_idx)) != 4:
        return False
    if require_unique_option_top_dim:
        tops = [int(all_perms[int(p)][0]) for p in option_perm_idx]
        if len(set(tops)) != 4:
            return False
    if len(arch_perm_idx_by_role) != 4:
        return False
    for role_rows in arch_perm_idx_by_role:
        if len(role_rows) != 4:
            return False
        if require_unique_archetypes_within_role and len(set(int(x) for x in role_rows)) != 4:
            return False
    return True


def _sample_candidate(
    *,
    rng: np.random.Generator,
    all_perms: List[Tuple[int, ...]],
    require_unique_option_top_dim: bool,
    require_unique_archetypes_within_role: bool,
) -> Tuple[List[int], List[List[int]]]:
    n_perms = len(all_perms)
    option_perm_idx: List[int]
    if require_unique_option_top_dim:
        by_top: Dict[int, List[int]] = {0: [], 1: [], 2: [], 3: []}
        for i, perm in enumerate(all_perms):
            by_top[int(perm[0])].append(int(i))
        option_perm_idx = [int(rng.choice(by_top[top])) for top in range(4)]
        rng.shuffle(option_perm_idx)
    else:
        option_perm_idx = [int(x) for x in rng.choice(np.arange(n_perms), size=4, replace=False).tolist()]

    arch_perm_idx_by_role: List[List[int]] = []
    for _role_i in range(4):
        if require_unique_archetypes_within_role:
            vals = [int(x) for x in rng.choice(np.arange(n_perms), size=4, replace=False).tolist()]
        else:
            vals = [int(x) for x in rng.choice(np.arange(n_perms), size=4, replace=True).tolist()]
        arch_perm_idx_by_role.append(vals)
    return option_perm_idx, arch_perm_idx_by_role


def _mutate_candidate(
    *,
    rng: np.random.Generator,
    option_perm_idx: List[int],
    arch_perm_idx_by_role: List[List[int]],
    all_perms: List[Tuple[int, ...]],
    require_unique_option_top_dim: bool,
    require_unique_archetypes_within_role: bool,
    max_attempts: int = 200,
) -> Tuple[List[int], List[List[int]]]:
    n_perms = len(all_perms)
    for _ in range(max_attempts):
        new_options = [int(x) for x in option_perm_idx]
        new_arch = [[int(x) for x in row] for row in arch_perm_idx_by_role]

        if rng.random() < 0.35:
            oi = int(rng.integers(0, 4))
            new_options[oi] = int(rng.integers(0, n_perms))
        else:
            ri = int(rng.integers(0, 4))
            ai = int(rng.integers(0, 4))
            new_arch[ri][ai] = int(rng.integers(0, n_perms))

        if _check_candidate_constraints(
            option_perm_idx=new_options,
            arch_perm_idx_by_role=new_arch,
            all_perms=all_perms,
            require_unique_option_top_dim=require_unique_option_top_dim,
            require_unique_archetypes_within_role=require_unique_archetypes_within_role,
        ):
            return new_options, new_arch
    return option_perm_idx, arch_perm_idx_by_role


def _candidate_from_yaml(
    *,
    inputs: InputContent,
    all_perms: List[Tuple[int, ...]],
) -> Tuple[List[int], List[List[int]]]:
    perm_to_idx = {tuple(p): i for i, p in enumerate(all_perms)}
    option_perm_idx: List[int] = []
    for option in inputs.options:
        row = _weights_row_from_mapping(option["dimension_weights"], inputs.dim_codes)
        order = _ordered_dim_indices(row)
        option_perm_idx.append(int(perm_to_idx[order]))

    arch_perm_idx_by_role: List[List[int]] = []
    for idxs in inputs.role_to_archetype_indices:
        role_rows: List[int] = []
        for ai in idxs:
            row = _weights_row_from_mapping(inputs.archetypes[ai]["dimension_weights"], inputs.dim_codes)
            order = _ordered_dim_indices(row)
            role_rows.append(int(perm_to_idx[order]))
        arch_perm_idx_by_role.append(role_rows)
    return option_perm_idx, arch_perm_idx_by_role


def _resolve_yaml_out_path(*, source_path: Path, suffix: str, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path).resolve()
    return source_path.with_name(f"{source_path.stem}{suffix}{source_path.suffix}").resolve()


def _write_yaml_from_design(
    *,
    inputs: InputContent,
    archetypes_by_role: np.ndarray,
    options: np.ndarray,
    out_archetypes_yaml: Path,
    out_options_yaml: Path,
    decimals: int,
) -> None:
    if archetypes_by_role.shape != (4, 4, 4):
        raise ValueError("archetypes_by_role must be (4,4,4)")
    if options.shape != (4, 4):
        raise ValueError("options must be (4,4)")

    arche_doc = dict(inputs.archetypes_doc)
    opt_doc = dict(inputs.options_doc)
    arche_list = list(arche_doc.get("archetypes", []))
    opt_list = list(opt_doc.get("options", []))

    matrix_archetypes = np.zeros((16, 4), dtype=float)
    for role_i, idxs in enumerate(inputs.role_to_archetype_indices):
        for k, ai in enumerate(idxs):
            matrix_archetypes[int(ai), :] = archetypes_by_role[role_i, k, :]

    for i, entry in enumerate(arche_list):
        mapping = {}
        for j, dim_code in enumerate(inputs.dim_codes):
            mapping[str(dim_code)] = round(float(matrix_archetypes[i, j]), int(decimals))
        entry["dimension_weights"] = mapping

    for i, entry in enumerate(opt_list):
        mapping = {}
        for j, dim_code in enumerate(inputs.dim_codes):
            mapping[str(dim_code)] = round(float(options[i, j]), int(decimals))
        entry["dimension_weights"] = mapping

    out_archetypes_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_options_yaml.parent.mkdir(parents=True, exist_ok=True)
    with out_archetypes_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(arche_doc, f, sort_keys=False, allow_unicode=False, width=120)
    with out_options_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(opt_doc, f, sort_keys=False, allow_unicode=False, width=120)


def _write_scenario_outcomes(path: Path, rows: List[dict]) -> None:
    if not rows:
        raise ValueError("scenario rows empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Transparency-only optimizer for ordinal rank-sum archetype/option design.")
    parser.add_argument("--content-dir", default=str(BUNDLE_ROOT / "config"))
    parser.add_argument("--out-dir", default=str(BUNDLE_ROOT / "results" / "scenario_generation" / "ordinal_ranksum_opt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-starts", type=int, default=80)
    parser.add_argument("--iters-per-start", type=int, default=1000)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--tie-weight", type=float, default=0.0)
    parser.add_argument("--target-diff-rate", type=float, default=None)
    parser.add_argument("--diff-weight", type=float, default=0.0)
    parser.add_argument("--require-unique-option-top-dim", action="store_true")
    parser.add_argument("--require-unique-archetypes-within-role", action="store_true")
    parser.add_argument("--init-from-yaml", action="store_true")
    parser.add_argument("--yaml-suffix", default="_optimized_ordinal")
    parser.add_argument("--out-archetypes-yaml", default=None)
    parser.add_argument("--out-options-yaml", default=None)
    parser.add_argument("--yaml-decimals", type=int, default=6)
    args = parser.parse_args()

    content_dir = Path(args.content_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = _load_inputs(content_dir)
    all_perms = list(itertools.permutations(range(4)))
    rng = np.random.default_rng(int(args.seed))

    out_archetypes_yaml = _resolve_yaml_out_path(
        source_path=inputs.archetypes_path,
        suffix=str(args.yaml_suffix),
        explicit_path=args.out_archetypes_yaml,
    )
    out_options_yaml = _resolve_yaml_out_path(
        source_path=inputs.options_path,
        suffix=str(args.yaml_suffix),
        explicit_path=args.out_options_yaml,
    )

    best_loss = float("inf")
    best_data: Dict[str, object] = {}

    def evaluate_candidate(opt_idx: List[int], arch_idx_by_role: List[List[int]]) -> Tuple[float, Dict[str, object]]:
        archetypes_by_role, options = _candidate_to_matrices(
            candidate_options_perm_idx=opt_idx,
            candidate_arch_perm_idx_by_role=arch_idx_by_role,
            all_perms=all_perms,
        )
        metrics = _evaluate_scenario_set(archetypes_by_role=archetypes_by_role, options=options, atol=float(args.atol))
        loss = _loss_from_metrics(
            metrics=metrics,
            alpha=float(args.alpha),
            beta=float(args.beta),
            tie_weight=float(args.tie_weight),
            target_diff_rate=args.target_diff_rate,
            diff_weight=float(args.diff_weight),
        )
        payload = {
            "loss": float(loss),
            "option_perm_idx": [int(x) for x in opt_idx],
            "arch_perm_idx_by_role": [[int(x) for x in row] for row in arch_idx_by_role],
            "metrics": metrics,
            "archetypes_by_role": archetypes_by_role,
            "options": options,
        }
        return float(loss), payload

    if args.init_from_yaml:
        opt_idx, arch_idx = _candidate_from_yaml(inputs=inputs, all_perms=all_perms)
        if _check_candidate_constraints(
            option_perm_idx=opt_idx,
            arch_perm_idx_by_role=arch_idx,
            all_perms=all_perms,
            require_unique_option_top_dim=bool(args.require_unique_option_top_dim),
            require_unique_archetypes_within_role=bool(args.require_unique_archetypes_within_role),
        ):
            loss, payload = evaluate_candidate(opt_idx, arch_idx)
            best_loss = float(loss)
            best_data = payload
            print(f"init_from_yaml loss={loss:.6f}")
        else:
            print("init_from_yaml skipped: YAML ordering does not satisfy active hard constraints")

    for start in range(int(args.n_starts)):
        opt_idx, arch_idx = _sample_candidate(
            rng=rng,
            all_perms=all_perms,
            require_unique_option_top_dim=bool(args.require_unique_option_top_dim),
            require_unique_archetypes_within_role=bool(args.require_unique_archetypes_within_role),
        )
        cur_loss, cur_payload = evaluate_candidate(opt_idx, arch_idx)
        for _iter in range(int(args.iters_per_start)):
            new_opt_idx, new_arch_idx = _mutate_candidate(
                rng=rng,
                option_perm_idx=opt_idx,
                arch_perm_idx_by_role=arch_idx,
                all_perms=all_perms,
                require_unique_option_top_dim=bool(args.require_unique_option_top_dim),
                require_unique_archetypes_within_role=bool(args.require_unique_archetypes_within_role),
            )
            if new_opt_idx == opt_idx and new_arch_idx == arch_idx:
                continue
            new_loss, new_payload = evaluate_candidate(new_opt_idx, new_arch_idx)
            if new_loss < cur_loss:
                opt_idx, arch_idx = new_opt_idx, new_arch_idx
                cur_loss, cur_payload = new_loss, new_payload
        if cur_loss < best_loss:
            best_loss = float(cur_loss)
            best_data = cur_payload
        if (start + 1) % max(1, int(args.n_starts) // 10) == 0:
            print(f"progress {start + 1}/{int(args.n_starts)} best_loss={best_loss:.6f}")

    if not best_data:
        raise SystemExit("No valid candidate found.")

    best_archetypes = np.array(best_data["archetypes_by_role"], dtype=float)
    best_options = np.array(best_data["options"], dtype=float)
    best_metrics = dict(best_data["metrics"])

    _write_yaml_from_design(
        inputs=inputs,
        archetypes_by_role=best_archetypes,
        options=best_options,
        out_archetypes_yaml=out_archetypes_yaml,
        out_options_yaml=out_options_yaml,
        decimals=int(args.yaml_decimals),
    )

    scenario_outcomes_path = out_dir / "scenario_outcomes.csv"
    _write_scenario_outcomes(scenario_outcomes_path, best_metrics["scenario_rows"])

    result = {
        "mode": "ordinal_ranksum_discrete_search",
        "weights": [float(x) for x in RANK_SUM_WEIGHTS.tolist()],
        "seed": int(args.seed),
        "search": {
            "n_starts": int(args.n_starts),
            "iters_per_start": int(args.iters_per_start),
            "require_unique_option_top_dim": bool(args.require_unique_option_top_dim),
            "require_unique_archetypes_within_role": bool(args.require_unique_archetypes_within_role),
            "init_from_yaml": bool(args.init_from_yaml),
        },
        "objective": {
            "alpha": float(args.alpha),
            "beta": float(args.beta),
            "tie_weight": float(args.tie_weight),
            "target_diff_rate": (float(args.target_diff_rate) if args.target_diff_rate is not None else None),
            "diff_weight": float(args.diff_weight),
            "atol": float(args.atol),
            "best_loss": float(best_loss),
        },
        "best_metrics": {
            "borda_wins": [float(x) for x in np.array(best_metrics["borda_wins"], dtype=float).tolist()],
            "util_wins": [float(x) for x in np.array(best_metrics["util_wins"], dtype=float).tolist()],
            "borda_tie_count": int(best_metrics["borda_tie_count"]),
            "util_tie_count": int(best_metrics["util_tie_count"]),
            "borda_tie_rate": float(best_metrics["borda_tie_rate"]),
            "util_tie_rate": float(best_metrics["util_tie_rate"]),
            "rule_diff_rate": float(best_metrics["rule_diff_rate"]),
        },
        "outputs": {
            "archetypes_yaml": str(out_archetypes_yaml),
            "options_yaml": str(out_options_yaml),
            "scenario_outcomes_csv": str(scenario_outcomes_path),
        },
    }
    result_path = out_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print("ok ordinal discrete optimization")
    print(f"- best_loss: {best_loss:.6f}")
    print(f"- borda_wins: {result['best_metrics']['borda_wins']}")
    print(f"- util_wins: {result['best_metrics']['util_wins']}")
    print(f"- borda_tie_rate: {result['best_metrics']['borda_tie_rate']:.4f}")
    print(f"- util_tie_rate: {result['best_metrics']['util_tie_rate']:.4f}")
    print(f"- rule_diff_rate: {result['best_metrics']['rule_diff_rate']:.4f}")
    print(f"- wrote: {result_path}")
    print(f"- wrote: {scenario_outcomes_path}")
    print(f"- wrote: {out_archetypes_yaml}")
    print(f"- wrote: {out_options_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
