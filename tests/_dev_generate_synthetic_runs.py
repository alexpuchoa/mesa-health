#!/usr/bin/env python3
"""Generate fake response CSVs for end-to-end benchmark pipeline testing."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _benchmark_lib.io_utils import read_csv_rows, write_csv


def _parse_csv_list(raw: str) -> List[str]:
    """Parse a non-empty comma-separated CLI list."""
    values = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def _stable_model_offset(model_name: str) -> int:
    """Derive a deterministic small offset so fake models do not behave identically."""
    return sum(ord(ch) for ch in str(model_name)) % 4


def _make_valid_response(selected_option_id: int, *, include_reasoning: bool) -> str:
    """Build a schema-conforming fake response payload."""
    payload: Dict[str, object] = {"selected_option_id": int(selected_option_id)}
    if include_reasoning:
        payload["reasoning"] = (
            f"Option {selected_option_id} was selected by the fake response generator for pipeline testing."
        )
    return json.dumps(payload, ensure_ascii=True)


def _make_invalid_response(selected_option_id: int) -> str:
    """Build a deliberately unparsable response for pipeline robustness tests."""
    return (
        "I choose option "
        + str(int(selected_option_id))
        + " because it looks best for this fake pipeline test."
    )


def main() -> int:
    """CLI entry point for creating synthetic response CSVs for smoke testing."""
    parser = argparse.ArgumentParser(description="Generate fake model responses from a prompt CSV.")
    parser.add_argument("--prompts-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--models", default="fake_gpt,fake_claude,fake_gemini")
    parser.add_argument("--runs-per-prompt", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--invalid-rate", type=float, default=0.0)
    parser.add_argument(
        "--selected-option-source",
        choices=("column_only", "response_only", "both"),
        default="both",
        help="Whether to emit selected_option_id, response_text, or both.",
    )
    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        help="Include a fake reasoning field inside valid JSON responses.",
    )
    args = parser.parse_args()

    if args.runs_per_prompt <= 0:
        raise ValueError("--runs-per-prompt must be > 0")
    if not (0.0 <= float(args.invalid_rate) <= 1.0):
        raise ValueError("--invalid-rate must be between 0 and 1")

    rng = random.Random(int(args.seed))
    prompt_rows = read_csv_rows(Path(args.prompts_csv))
    model_names = _parse_csv_list(args.models)

    out_rows: List[Dict[str, object]] = []
    for prompt_row in prompt_rows:
        scenario_id = int(prompt_row["scenario_id"])
        test_variant_id = int(prompt_row["test_variant_id"])
        for model_name in model_names:
            model_offset = _stable_model_offset(model_name)
            for run_index in range(1, int(args.runs_per_prompt) + 1):
                sampled = ((scenario_id + test_variant_id + model_offset + run_index + rng.randint(0, 3)) % 4) + 1
                is_invalid = rng.random() < float(args.invalid_rate)
                response_text = (
                    _make_invalid_response(sampled)
                    if is_invalid
                    else _make_valid_response(sampled, include_reasoning=bool(args.include_reasoning))
                )
                selected_option_id = sampled if args.selected_option_source in {"column_only", "both"} else ""
                response_payload = response_text if args.selected_option_source in {"response_only", "both"} else ""
                out_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "test_variant_id": test_variant_id,
                        "model_name": model_name,
                        "run_index": run_index,
                        "selected_option_id": selected_option_id,
                        "response_text": response_payload,
                    }
                )

    out_rows.sort(key=lambda row: (int(row["scenario_id"]), int(row["test_variant_id"]), str(row["model_name"]), int(row["run_index"])))
    write_csv(
        Path(args.out_csv),
        out_rows,
        fieldnames=[
            "scenario_id",
            "test_variant_id",
            "model_name",
            "run_index",
            "selected_option_id",
            "response_text",
        ],
    )
    print(f"ok wrote {len(out_rows)} fake responses to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
