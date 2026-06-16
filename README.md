# Benchmark Publication Bundle

This folder is the standalone code-and-config bundle intended for public release alongside the benchmark paper.

## Scope

User-facing workflow:
- `scripts/generate_prompts.py`: render benchmark prompts from fixed scenario data and published variants.
- `scripts/process_runs.py`: score raw model responses into the run-level benchmark output.
- `scripts/aggregate_metrics.py`: produce
  1. per-scenario / per-model / per-test-variant modal aggregates, and
  2. the final per-model paper metrics.

Transparency-only workflow:
- `scripts/scenario_generation/generate_scenarios.py`: regenerate the full scenario universe from archetypes, options, and dimensions.
- `scripts/scenario_generation/optimize_ordinal_ranksum_design.py`: transparency-only search over ordinal rank-sum archetype/option profiles before the benchmark was frozen.
- `scripts/scenario_generation/select_scenarios.py`: select the benchmark core and derived pools from a scenario feature table.
- `scripts/scenario_generation/build_veto_sidecar.py`: convert the frozen veto selection CSV into the published `scenario_vetoes.yaml` sidecar.

## Output Layers

The public evaluation flow is designed around three CSV outputs:

1. Individual runs
- one row per raw model response
- produced by `scripts/process_runs.py`

2. Scenario aggregates
- one row per `(model_name, scenario_id, test_variant_id)`
- modal selected option and correctness against the relevant rule(s)
- produced by `scripts/aggregate_metrics.py --out-scenario-csv ...`

3. Final model metrics
- one row per model
- paper metrics: `AGR_Borda`, `AGR_Maximin`, `AGR_Utilitarian`, `DPC`, `APDR`, `CPFR`, `PIR`, `FIR`, `SLIR`, `VVR`, `MSI`
- produced by `scripts/aggregate_metrics.py --out-model-csv ...`

## Layout

- `config/variant_map.yaml`: published test-variant definitions.
- `config/task_directives.yaml`: task directive wording.
- `config/rendering/`: prompt templates and block render configs used by published variants.
- `config/test_set_variant_map.yaml`: pool-to-variant mapping used by the public metrics script.
- `data/`: placeholder for downloaded benchmark data; see `data/README.md`.
- `results/responses/`: recommended location for raw model outputs.
- `results/metrics/`: recommended location for computed metrics.
- `tests/`: smoke tests for the standalone scripts.

## Expected published data

The ready benchmark scenarios are data artifacts and should be downloaded from the benchmark dataset release.

The scripts are designed to consume at minimum:
- a scenario bundle JSON/YAML with top-level `scenarios`
- selection-set CSVs with `scenario_id` under `data/selection_sets/`
- the published veto sidecar `data/scenario_vetoes.yaml` for veto variants

## Minimal user flow

Render prompts:

```bash
python scripts/generate_prompts.py \
  --scenarios data/scenarios_master.json \
  --test-variant-ids 0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,23,24,31,33,44,111,112,113 \
  --out-csv results/prompts.csv
```

Compute run-level metrics:

```bash
python scripts/process_runs.py \
  --responses-csv results/responses/model_outputs.csv \
  --scenarios-master-json data/scenarios_master.json \
  --pool-map config/test_set_variant_map.yaml \
  --out-csv results/metrics/individual_runs.csv
```

Aggregate to scenario-modal and final model metrics:

```bash
python scripts/aggregate_metrics.py \
  --metrics-csv results/metrics/individual_runs.csv \
  --out-scenario-csv results/metrics/scenario_aggregates.csv \
  --out-model-csv results/metrics/model_metrics.csv
```

## Transparency notes

The generation and selection scripts are included for methodological inspection.
They are not intended to replace the fixed, citable benchmark data artifact.
