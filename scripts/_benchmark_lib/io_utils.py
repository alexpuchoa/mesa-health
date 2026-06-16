from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import yaml

_SUPPORTED_EXTENSIONS = {".yaml", ".yml", ".json", ".csv"}


def ensure_supported(path: Path) -> None:
    """Reject file paths whose extension is outside the small public bundle IO set."""
    if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension for {path}. Supported: {sorted(_SUPPORTED_EXTENSIONS)}")


def load_data(path: Path) -> Any:
    """Load YAML, JSON, or CSV data using a predictable Python representation."""
    ensure_supported(path)
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_yaml(path: Path, payload: Any) -> None:
    """Write a YAML file after creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def write_json(path: Path, payload: Any) -> None:
    """Write pretty-printed JSON with a trailing newline for stable diffs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write dictionaries to CSV using the provided field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({str(k): row.get(k) for k in fieldnames})


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV file into a list of string-keyed row dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]
