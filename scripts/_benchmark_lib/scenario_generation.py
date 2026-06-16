from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from _benchmark_lib.io_utils import load_data


@dataclass(frozen=True)
class Archetype:
    archetype_id: int
    role_id: int
    code: str
    name: str
    dimension_weights: Dict[str, float]


@dataclass(frozen=True)
class Option:
    option_id: int
    code: str
    title: str
    dimension_weights: Dict[str, float]


def _extract_records(payload: Any, *, key: str) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        value = payload.get(key)
        if not isinstance(value, list):
            raise ValueError(f"Expected top-level key '{key}' to be a list")
        return value
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported payload type for {key}: {type(payload).__name__}")


def _extract_weight_map_from_row(row: Dict[str, Any], *, dims: List[str]) -> Dict[str, Any]:
    if "dimension_weights" in row and isinstance(row["dimension_weights"], dict):
        return dict(row["dimension_weights"])
    if "dimension_weights" in row and isinstance(row["dimension_weights"], str):
        candidate = row["dimension_weights"].strip()
        if candidate:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
    if all(dim in row for dim in dims):
        return {dim: row[dim] for dim in dims}
    prefixed = {dim: row.get(f"w_{dim}") for dim in dims}
    if all(value is not None and str(value).strip() != "" for value in prefixed.values()):
        return prefixed
    raise ValueError(
        "Could not resolve dimension weights from row. Expected one of: nested 'dimension_weights', "
        "JSON string in 'dimension_weights', direct dimension columns, or prefixed columns 'w_<dimension_code>'."
    )


def _normalize_weights(raw: Dict[str, Any], *, dims: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for dim in dims:
        if dim not in raw:
            raise ValueError(f"Missing dimension '{dim}' in dimension_weights")
        out[dim] = float(raw[dim])
    total = sum(out.values())
    if abs(total - 1.0) > 1.0e-9:
        raise ValueError(f"dimension_weights must sum to 1.0; found {total}")
    return out


def load_dimensions(dimensions_path: Path) -> List[str]:
    payload = load_data(dimensions_path)
    if isinstance(payload, list):
        codes = [str(item["code"]) for item in payload]
    elif isinstance(payload, dict) and "dimensions" in payload:
        dims = payload["dimensions"]
        if not isinstance(dims, list):
            raise ValueError("'dimensions' key must be a list")
        codes = [str(item["code"]) for item in dims]
    else:
        raise ValueError("Dimensions file must contain top-level 'dimensions' list")
    if len(codes) != 4:
        raise ValueError(f"Expected 4 dimensions; found {len(codes)}")
    return codes


def load_archetypes(archetypes_path: Path, *, dimensions: List[str]) -> List[Archetype]:
    payload = load_data(archetypes_path)
    rows = _extract_records(payload, key="archetypes")
    return [
        Archetype(
            archetype_id=int(row["archetype_id"]),
            role_id=int(row["role_id"]),
            code=str(row.get("code", f"a{row['archetype_id']}")),
            name=str(row.get("name", row.get("code", f"archetype_{row['archetype_id']}"))),
            dimension_weights=_normalize_weights(_extract_weight_map_from_row(row, dims=dimensions), dims=dimensions),
        )
        for row in rows
    ]


def load_options(options_path: Path, *, dimensions: List[str]) -> List[Option]:
    payload = load_data(options_path)
    rows = _extract_records(payload, key="options")
    return [
        Option(
            option_id=int(row["option_id"]),
            code=str(row.get("code", f"o{row['option_id']}")),
            title=str(row.get("title", row.get("code", f"option_{row['option_id']}"))),
            dimension_weights=_normalize_weights(_extract_weight_map_from_row(row, dims=dimensions), dims=dimensions),
        )
        for row in rows
    ]


def rank_dimensions(weights: Dict[str, float]) -> List[str]:
    return [d for d, _ in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))]


def utility(arch_w: Dict[str, float], opt_w: Dict[str, float], dims: List[str]) -> float:
    return sum(float(arch_w[d]) * float(opt_w[d]) for d in dims)


def build_scenarios(
    *,
    archetypes: List[Archetype],
    options: List[Option],
    dimensions: List[str],
    scenario_id_start: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    role_to_arch: Dict[int, List[Archetype]] = {}
    for a in archetypes:
        role_to_arch.setdefault(a.role_id, []).append(a)

    role_ids = sorted(role_to_arch.keys())
    if len(role_ids) != 4:
        raise ValueError(f"Expected 4 roles, found {len(role_ids)}")
    for rid in role_ids:
        role_to_arch[rid] = sorted(role_to_arch[rid], key=lambda x: x.archetype_id)
        if len(role_to_arch[rid]) != 4:
            raise ValueError(f"Role {rid} must have 4 archetypes; found {len(role_to_arch[rid])}")
    if len(options) != 4:
        raise ValueError(f"Expected 4 options; found {len(options)}")

    options_sorted = sorted(options, key=lambda x: x.option_id)
    scenarios: List[Dict[str, Any]] = []
    utilities_long: List[Dict[str, Any]] = []
    rankings_long: List[Dict[str, Any]] = []
    scenarios_flat: List[Dict[str, Any]] = []

    scenario_id = scenario_id_start
    for combo in itertools.product(*(role_to_arch[rid] for rid in role_ids)):
        code_parts = [f"r{role_ids[i]}a{combo[i].archetype_id}" for i in range(4)]
        scenario_code = "_".join(code_parts)

        stakeholders_payload = []
        for item in combo:
            stakeholders_payload.append(
                {
                    "role_id": item.role_id,
                    "archetype_id": item.archetype_id,
                    "archetype_code": item.code,
                    "archetype_name": item.name,
                    "dimension_rank_order": rank_dimensions(item.dimension_weights),
                    "dimension_weights": dict(item.dimension_weights),
                }
            )

        scenarios.append(
            {
                "scenario_id": scenario_id,
                "scenario_code": scenario_code,
                "stakeholders": stakeholders_payload,
                "options": [
                    {
                        "option_id": o.option_id,
                        "option_code": o.code,
                        "option_title": o.title,
                        "dimension_rank_order": rank_dimensions(o.dimension_weights),
                        "dimension_weights": dict(o.dimension_weights),
                    }
                    for o in options_sorted
                ],
            }
        )
        flat_row: Dict[str, Any] = {"scenario_id": scenario_id, "scenario_code": scenario_code}
        for i, rid in enumerate(role_ids):
            flat_row[f"role_{rid}_archetype_id"] = combo[i].archetype_id
        scenarios_flat.append(flat_row)

        for st in combo:
            utility_rows = []
            for opt in options_sorted:
                score = utility(st.dimension_weights, opt.dimension_weights, dimensions)
                utility_rows.append((opt.option_id, score))
                utilities_long.append(
                    {
                        "scenario_id": scenario_id,
                        "scenario_code": scenario_code,
                        "role_id": st.role_id,
                        "archetype_id": st.archetype_id,
                        "option_id": opt.option_id,
                        "utility": round(score, 10),
                    }
                )
            sorted_util = sorted(utility_rows, key=lambda x: (-x[1], x[0]))
            rank_map: Dict[int, int] = {}
            curr_rank = 1
            for i, (opt_id, score) in enumerate(sorted_util):
                if i > 0 and abs(score - sorted_util[i - 1][1]) > 1.0e-12:
                    curr_rank = i + 1
                rank_map[opt_id] = curr_rank
            for opt_id, score in utility_rows:
                rankings_long.append(
                    {
                        "scenario_id": scenario_id,
                        "scenario_code": scenario_code,
                        "role_id": st.role_id,
                        "archetype_id": st.archetype_id,
                        "option_id": opt_id,
                        "utility": round(score, 10),
                        "rank": rank_map[opt_id],
                    }
                )
        scenario_id += 1

    return scenarios, utilities_long, rankings_long, scenarios_flat
