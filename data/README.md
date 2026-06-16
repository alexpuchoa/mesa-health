# Benchmark Data

This folder is a landing point for the published benchmark data release.

Recommended contents from the external dataset release:
- `scenarios_master.json` or `scenarios_master.yaml`: fixed benchmark scenario bundle.
- `scenario_vetoes.yaml`: published sidecar for veto scenarios.
- `selection_sets/*.csv`: fixed scenario subsets with a `scenario_id` column.

Recommended raw response schema for `scripts/process_runs.py`:
- `scenario_id` (required)
- `test_variant_id` (required)
- `model_name` (required)
- `run_index` (recommended)
- `selected_option_id` (optional if `response_text` is provided)
- `response_text` (optional if `selected_option_id` is provided)

`process_runs.py` produces the individual-run benchmark CSV.
`aggregate_metrics.py` consumes that file and emits:
- `scenario_aggregates.csv`
- `model_metrics.csv`

The public scripts do not require a database.
