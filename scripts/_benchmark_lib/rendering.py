from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from _benchmark_lib.io_utils import load_data
from _benchmark_lib.metrics import sorted_dimension_codes


def _load_yaml(path: Path) -> Dict[str, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"Expected mapping YAML at {path}")
    return doc


def load_variant_map(path: Path) -> Dict[str, Any]:
    doc = _load_yaml(path)
    if not isinstance(doc.get("variants"), list):
        raise ValueError(f"variant_map must define a variants list: {path}")
    return doc


def variants_by_id(doc: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for item in doc.get("variants", []):
        row = dict(item)
        out[int(row["test_variant_id"])] = row
    return out


def load_task_directives(path: Path) -> Dict[str, Dict[str, str]]:
    doc = _load_yaml(path)
    task_directives = doc.get("task_directives")
    if not isinstance(task_directives, dict):
        raise ValueError(f"task_directives missing in {path}")
    out: Dict[str, Dict[str, str]] = {}
    for policy, variants in task_directives.items():
        if not isinstance(variants, dict):
            raise ValueError(f"task_directives[{policy}] must be a mapping")
        out[str(policy)] = {str(k): str(v) for k, v in variants.items()}
    return out


def load_dimension_catalog(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[int, str]]:
    doc = _load_yaml(path)
    rows = doc.get("dimensions")
    if not isinstance(rows, list):
        raise ValueError(f"dimensions list missing in {path}")
    by_code: Dict[str, Dict[str, Any]] = {}
    by_id: Dict[int, str] = {}
    ordered: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = str(row["code"])
        ordered.append(dict(row))
        by_code[code] = dict(row)
        by_id[int(row["dimension_id"])] = code
    return ordered, by_code, by_id


def load_role_catalog(path: Path) -> Dict[int, Dict[str, Any]]:
    doc = _load_yaml(path)
    rows = doc.get("stakeholder_roles")
    if not isinstance(rows, list):
        raise ValueError(f"stakeholder_roles list missing in {path}")
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        out[int(row["role_id"])] = dict(row)
    return out


def _resolve_relative(base_dir: Path, raw_path: str) -> Path:
    p = Path(str(raw_path))
    return p if p.is_absolute() else (base_dir / p)


def _format_prefix(template: str, *, index: int, role_name: str) -> str:
    return str(template).format(index=int(index), role_name=str(role_name))


def _dimension_label(
    *,
    dim_code: str,
    render_labels: Dict[str, str],
    dim_meta: Dict[str, Dict[str, Any]],
    variant_code: str,
) -> str:
    base = str(render_labels.get(dim_code) or dim_meta[dim_code].get("name") or dim_code)
    if variant_code and variant_code != "default":
        override = ((dim_meta[dim_code].get("label_variants") or {}).get(variant_code))
        if override:
            return str(override)
    return base


def _level_map(weight_map: Dict[str, Any]) -> Dict[str, int]:
    ordered = sorted(((str(k), float(v)) for k, v in weight_map.items()), key=lambda item: (-item[1], item[0]))
    return {dim_code: 4 - idx for idx, (dim_code, _w) in enumerate(ordered)}


def _adjective(
    *,
    dim_code: str,
    level: int,
    dim_meta: Dict[str, Dict[str, Any]],
    scale_variant: str,
    target: str,
) -> str:
    key = "scale_adjectives_options" if target == "option" else "scale_adjectives_preferences"
    variants = dim_meta[dim_code].get(key) or {}
    variant_map = variants.get(scale_variant) or variants.get("default") or {}
    value = variant_map.get(int(level))
    if value is None:
        raise ValueError(f"Missing adjective for dimension={dim_code} level={level} variant={scale_variant} target={target}")
    return str(value).replace("_", " ")


def _render_weight_items(
    *,
    ordered_dims: Sequence[str],
    weight_map: Dict[str, Any],
    dim_meta: Dict[str, Dict[str, Any]],
    render_labels: Dict[str, str],
    label_variant: str,
    scale_variant: str,
    target: str,
    adjective_format: str,
    include_scale_adjectives: bool,
) -> List[str]:
    levels = _level_map(weight_map)
    out: List[str] = []
    for dim_code in ordered_dims:
        label = _dimension_label(
            dim_code=str(dim_code),
            render_labels=render_labels,
            dim_meta=dim_meta,
            variant_code=label_variant,
        )
        if include_scale_adjectives:
            adjective = _adjective(
                dim_code=str(dim_code),
                level=int(levels[str(dim_code)]),
                dim_meta=dim_meta,
                scale_variant=scale_variant,
                target=target,
            )
            out.append(str(adjective_format).format(adjective=str(adjective), label=str(label)))
        else:
            out.append(str(label))
    return out


def _render_block(
    *,
    rows: Sequence[Dict[str, Any]],
    block_cfg: Dict[str, Any],
    target: str,
    dim_meta: Dict[str, Dict[str, Any]],
    render_labels: Dict[str, str],
    label_variant: str,
    scale_variant: str,
    role_names: Optional[Sequence[str]] = None,
) -> str:
    style = str(block_cfg.get("style") or "order")
    line_prefix = str(block_cfg.get("line_prefix") or "")
    separator = str(block_cfg.get("order_separator") or " > ")
    item_separator = str(block_cfg.get("item_separator") or ", ")
    adjective_format = str(block_cfg.get("adjective_format") or "{adjective} {label}")
    include_scale_adjectives = bool(block_cfg.get("include_scale_adjectives"))

    rendered: List[str] = []
    for idx, row in enumerate(rows, start=1):
        role_name = role_names[idx - 1] if role_names is not None else f"Stakeholder {idx}"
        weight_map = {str(k): float(v) for k, v in row["dimension_weights"].items()}
        ordered_dims = list(row.get("dimension_rank_order") or sorted_dimension_codes(weight_map))
        prefix = _format_prefix(line_prefix, index=idx, role_name=role_name)
        if style == "order":
            labels = [
                _dimension_label(
                    dim_code=str(dim_code),
                    render_labels=render_labels,
                    dim_meta=dim_meta,
                    variant_code=label_variant,
                )
                for dim_code in ordered_dims
            ]
            rendered.append(prefix + separator.join(labels))
            continue
        items = _render_weight_items(
            ordered_dims=ordered_dims,
            weight_map=weight_map,
            dim_meta=dim_meta,
            render_labels=render_labels,
            label_variant=label_variant,
            scale_variant=scale_variant,
            target=target,
            adjective_format=adjective_format,
            include_scale_adjectives=include_scale_adjectives,
        )
        if style == "weights":
            rendered.append(prefix + item_separator.join(items))
            continue
        if style == "template":
            template = str(block_cfg.get("template") or "{dim1}, {dim2}, {dim3}, {dim4}")
            values = {f"dim{i+1}": items[i] for i in range(len(items))}
            values.update({"index": idx, "role_name": role_name})
            rendered.append(prefix + template.format(**values))
            continue
        raise ValueError(f"Unsupported block style: {style}")
    return "\n".join(rendered).rstrip() + "\n"


def _render_veto_block(
    *,
    scenario_id: int,
    scenario_veto: Optional[Dict[str, int]],
    block_cfg: Dict[str, Any],
    stakeholder_order: Sequence[int],
    role_names_by_role_id: Dict[int, str],
    dim_meta: Dict[str, Dict[str, Any]],
    dim_id_to_code: Dict[int, str],
    render_labels: Dict[str, str],
    label_variant: str,
) -> str:
    style = str(block_cfg.get("style") or "none_literal")
    none_literal = str(block_cfg.get("none_literal") or "None")
    if style == "none_literal":
        return none_literal.rstrip() + "\n"
    if style != "scenario_veto_phrase":
        raise ValueError(f"Unsupported stakeholder_veto style: {style}")
    if not scenario_veto:
        return none_literal.rstrip() + "\n"
    role_id = int(scenario_veto["role_id"])
    dim_id = int(scenario_veto["dimension_id"])
    if role_id not in stakeholder_order:
        raise ValueError(f"Scenario {scenario_id}: veto role_id={role_id} not in stakeholder order")
    dim_code = str(dim_id_to_code[dim_id])
    dim_label = _dimension_label(
        dim_code=dim_code,
        render_labels=render_labels,
        dim_meta=dim_meta,
        variant_code=label_variant,
    )
    idx = list(stakeholder_order).index(role_id) + 1
    prefix = _format_prefix(str(block_cfg.get("line_prefix") or "Stakeholder {index}: "), index=idx, role_name=role_names_by_role_id[role_id])
    item_format = str(block_cfg.get("item_format") or "{dimension_label}")
    return (prefix + item_format.format(dimension_label=dim_label, dimension_code=dim_code)).rstrip() + "\n"


def resolve_variant_orders(variant_map: Dict[str, Any], variant: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    defaults = variant_map.get("defaults") or {}
    base_stakeholders = [int(item["role_id"]) for item in defaults.get("prompt_stakeholder_order", [])]
    base_options = [int(item["option_id"]) for item in defaults.get("prompt_option_order", [])]
    overrides = variant.get("overrides") or {}
    stakeholder_order = list(base_stakeholders)
    option_order = list(base_options)
    if overrides.get("prompt_stakeholder_order"):
        stakeholder_order = [int(item["role_id"]) for item in overrides["prompt_stakeholder_order"]]
    if overrides.get("prompt_option_order"):
        option_order = [int(item["option_id"]) for item in overrides["prompt_option_order"]]
    if len(stakeholder_order) != 4 or sorted(stakeholder_order) != [1, 2, 3, 4]:
        raise ValueError(f"Invalid stakeholder order for variant {variant.get('test_variant_id')}")
    if len(option_order) != 4 or sorted(option_order) != [1, 2, 3, 4]:
        raise ValueError(f"Invalid option order for variant {variant.get('test_variant_id')}")
    return stakeholder_order, option_order


def render_prompt_for_scenario(
    *,
    scenario: Dict[str, Any],
    variant: Dict[str, Any],
    variant_map: Dict[str, Any],
    config_dir: Path,
    task_directives: Dict[str, Dict[str, str]],
    dimension_catalog: Dict[str, Dict[str, Any]],
    dim_id_to_code: Dict[int, str],
    role_catalog: Dict[int, Dict[str, Any]],
    vetoes_by_scenario: Optional[Dict[int, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    prompt_path = _resolve_relative(config_dir, str(variant["prompt_yaml"]))
    render_config_path = _resolve_relative(config_dir, str(variant["render_config_yaml"]))
    prompt_doc = _load_yaml(prompt_path)
    render_cfg = _load_yaml(render_config_path)

    stakeholder_order, option_order = resolve_variant_orders(variant_map, variant)
    stakeholders_by_role = {int(item["role_id"]): dict(item) for item in scenario["stakeholders"]}
    options_by_id = {int(item["option_id"]): dict(item) for item in scenario["options"]}

    role_variant = str(variant.get("stakeholder_role_variant_code") or "default")
    label_variant = str(variant.get("dimension_label_variant_code") or "default")
    scale_variant = str(variant.get("dimension_scale_variant_code") or "default")

    ordered_stakeholders = [stakeholders_by_role[role_id] for role_id in stakeholder_order]
    ordered_options = [options_by_id[option_id] for option_id in option_order]

    role_names_by_role_id: Dict[int, str] = {}
    stakeholder_role_names: List[str] = []
    for role_id in stakeholder_order:
        role_meta = role_catalog[int(role_id)]
        label_variants = role_meta.get("label_variants") or {}
        role_name = str(label_variants.get(role_variant) or label_variants.get("default") or role_meta.get("name") or role_meta.get("code") or role_id)
        role_names_by_role_id[int(role_id)] = role_name
        stakeholder_role_names.append(role_name)

    render_labels = {str(k): str(v) for k, v in (render_cfg.get("dimension_labels") or {}).items()}

    blocks = render_cfg.get("blocks") or {}
    options_block = _render_block(
        rows=ordered_options,
        block_cfg=dict(blocks.get("options") or {}),
        target="option",
        dim_meta=dimension_catalog,
        render_labels=render_labels,
        label_variant=label_variant,
        scale_variant=scale_variant,
    )
    preferences_block = _render_block(
        rows=ordered_stakeholders,
        block_cfg=dict(blocks.get("stakeholder_preferences") or {}),
        target="stakeholder",
        dim_meta=dimension_catalog,
        render_labels=render_labels,
        label_variant=label_variant,
        scale_variant=scale_variant,
        role_names=stakeholder_role_names,
    )
    scenario_veto = None
    if vetoes_by_scenario:
        scenario_veto = vetoes_by_scenario.get(int(scenario["scenario_id"]))
    if scenario_veto is None:
        scenario_veto = scenario.get("scenario_veto")
    veto_block = _render_veto_block(
        scenario_id=int(scenario["scenario_id"]),
        scenario_veto=scenario_veto,
        block_cfg=dict(blocks.get("stakeholder_veto") or {}),
        stakeholder_order=stakeholder_order,
        role_names_by_role_id=role_names_by_role_id,
        dim_meta=dimension_catalog,
        dim_id_to_code=dim_id_to_code,
        render_labels=render_labels,
        label_variant=label_variant,
    )

    policy = str(variant.get("task_directive_policy_code") or "").strip()
    variant_code = str(variant.get("task_directive_variant_code") or "default").strip()
    try:
        task_directive = str(task_directives[policy][variant_code])
    except KeyError as exc:
        raise ValueError(f"Unknown task directive policy/variant: {policy}/{variant_code}") from exc

    prompt_text = str(prompt_doc["template"]).format(
        task_directive=task_directive,
        options=options_block.rstrip(),
        stakeholder_preferences=preferences_block.rstrip(),
        stakeholder_veto=veto_block.rstrip(),
    )
    return {
        "scenario_id": int(scenario["scenario_id"]),
        "test_variant_id": int(variant["test_variant_id"]),
        "test_code": str(variant.get("test_code") or ""),
        "variant_name": str(variant.get("name") or ""),
        "prompt_text": prompt_text,
        "displayed_option_order": [int(x) for x in option_order],
        "displayed_stakeholder_order": [int(x) for x in stakeholder_order],
        "task_directive": task_directive,
        "task_directive_policy_code": policy,
        "prompt_yaml": str(prompt_path),
        "render_config_yaml": str(render_config_path),
    }
