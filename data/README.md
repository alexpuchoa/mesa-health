# Benchmark Data

Download the frozen scenario bundle from HuggingFace and place it here:

> **https://huggingface.co/datasets/aluchoa/mesa-health-v1**

## Expected contents

| File | Source | Description |
|---|---|---|
| `scenarios_master.json` | Download from HuggingFace | Frozen scenario bundle (required by all scripts) |
| `scenario_vetoes.yaml` | Included in this repo | Veto assignments for the 60 veto-pool scenarios |
| `selection_sets/*.csv` | Included in this repo | Scenario subsets: core_160, divergent_80, pareto_ge2_120, veto_60 |

## Response schema

Place your model response CSVs in `results/responses/`. Required columns:

- `scenario_id`, `test_variant_id`, `model_name` (required)
- `run_index` (recommended, for multi-run stability analysis)
- `selected_option_id` or `response_text` (at least one)

See root README for the full evaluation pipeline.
