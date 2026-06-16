from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from _benchmark_lib.io_utils import load_data, read_csv_rows


def load_scenarios(path: Path) -> List[Dict[str, Any]]:
    """Load the frozen scenario bundle from either a top-level list or ``scenarios`` mapping."""
    payload = load_data(path)
    if isinstance(payload, dict) and isinstance(payload.get("scenarios"), list):
        scenarios = payload["scenarios"]
    elif isinstance(payload, list):
        scenarios = payload
    else:
        raise ValueError("Scenario file must be a list or a mapping with top-level 'scenarios'")
    out: List[Dict[str, Any]] = []
    for row in scenarios:
        if not isinstance(row, dict):
            raise ValueError("Every scenario entry must be a mapping")
        out.append(dict(row))
    return out


def scenarios_by_id(path: Path) -> Dict[int, Dict[str, Any]]:
    """Index the published scenarios by ``scenario_id`` and fail on duplicates."""
    rows = load_scenarios(path)
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        sid = int(row["scenario_id"])
        if sid in out:
            raise ValueError(f"Duplicate scenario_id={sid} in {path}")
        out[sid] = row
    return out


def load_selection_set_ids(path: Path) -> List[int]:
    """Read a one-column-or-more selection-set CSV and extract ``scenario_id`` values."""
    rows = read_csv_rows(path)
    out: List[int] = []
    for row in rows:
        if "scenario_id" not in row:
            raise ValueError(f"Selection set CSV missing scenario_id column: {path}")
        out.append(int(row["scenario_id"]))
    return out


def load_pool_memberships(map_yaml: Path) -> Dict[str, List[int]]:
    """Resolve the published pool map into explicit scenario-id memberships per pool."""
    doc = yaml.safe_load(map_yaml.read_text(encoding="utf-8")) or {}
    pools = doc.get("pools")
    if not isinstance(pools, dict):
        raise ValueError(f"Pool map must define a top-level pools mapping: {map_yaml}")
    out: Dict[str, List[int]] = {}
    for pool_name, cfg in pools.items():
        if not isinstance(cfg, dict):
            continue
        csv_path_raw = cfg.get("csv")
        if not csv_path_raw:
            continue
        csv_path = Path(str(csv_path_raw))
        if not csv_path.is_absolute():
            csv_path = (map_yaml.parent.parent / csv_path).resolve()
        if not csv_path.exists():
            raise ValueError(f"Pool CSV not found for pool={pool_name}: {csv_path}")
        out[str(pool_name)] = load_selection_set_ids(csv_path)
    return out


def scenario_pool_memberships(scenario_id: int, pool_memberships: Dict[str, List[int]]) -> List[str]:
    """Return the sorted list of published pool names containing one scenario."""
    sid = int(scenario_id)
    return sorted(pool_name for pool_name, scenario_ids in pool_memberships.items() if sid in {int(x) for x in scenario_ids})


def load_optional_vetoes(path: Optional[Path]) -> Dict[int, Dict[str, int]]:
    """Load published veto assignments keyed by ``scenario_id``.

    Accepted inputs:
    - YAML/JSON with top-level ``scenario_vetoes``
    - plain YAML/JSON list of veto rows
    - CSV rows such as the published ``veto_60.csv``

    Supported row keys:
    - ``role_id`` or ``assigned_role_id``
    - ``dimension_id``
    - optional ``scenario_stakeholder_id`` or ``assigned_scenario_stakeholder_id``

    The benchmark scoring/rendering logic currently needs only ``role_id`` and
    ``dimension_id``. We preserve ``scenario_stakeholder_id`` when available so
    the sidecar remains semantically richer and forward-compatible.
    """
    if path is None:
        return {}
    payload = load_data(path)
    rows: List[Dict[str, Any]]
    if isinstance(payload, dict) and isinstance(payload.get("scenario_vetoes"), list):
        rows = [dict(x) for x in payload["scenario_vetoes"]]
    elif isinstance(payload, list):
        rows = [dict(x) for x in payload]
    else:
        raise ValueError("Veto file must be a list or a mapping with top-level 'scenario_vetoes'")
    out: Dict[int, Dict[str, int]] = {}
    for row in rows:
        sid = int(row["scenario_id"])
        role_raw = row.get("role_id", row.get("assigned_role_id"))
        if role_raw in (None, ""):
            raise ValueError(f"Veto row for scenario_id={sid} is missing role_id/assigned_role_id")
        stakeholder_raw = row.get(
            "scenario_stakeholder_id",
            row.get("assigned_scenario_stakeholder_id"),
        )
        out[sid] = {
            "role_id": int(role_raw),
            "dimension_id": int(row["dimension_id"]),
        }
        if stakeholder_raw not in (None, ""):
            out[sid]["scenario_stakeholder_id"] = int(stakeholder_raw)
    return out


def filter_scenarios(
    scenarios: Sequence[Dict[str, Any]],
    *,
    scenario_ids: Optional[Iterable[int]] = None,
) -> List[Dict[str, Any]]:
    """Apply an optional ``scenario_id`` filter while preserving row structure."""
    if scenario_ids is None:
        return [dict(s) for s in scenarios]
    wanted = {int(x) for x in scenario_ids}
    return [dict(s) for s in scenarios if int(s["scenario_id"]) in wanted]
