# MESA-Health

**Multi-stakeholder Evaluation Scenarios for Assessing Health-governance mediation by LLMs**

MESA-Health is a benchmark for evaluating whether large language models can reliably mediate multi-stakeholder decision problems with computable ground truth. It covers 160 core scenarios, 21 test variants, and four scenario pools designed to probe specific failure modes: positional anchoring, label sensitivity, format invariance, veto compliance, aggregation-rule switching, and manipulation sensitivity.

> **Paper:** *MESA-Health - Evaluating LLMs in Multi-Stakeholder Collective Decision-Making* — [link forthcoming]
>
> **Dataset:** The frozen scenario data artifact is hosted on HuggingFace — (https://huggingface.co/datasets/aluchoa/mesa-health-v1)
>
> **License:** Code: MIT · Data: CC-BY 4.0

---

## What this repository contains

This repo provides the code and configuration needed to (1) render benchmark prompts for any LLM, (2) score model responses, and (3) compute the paper's metrics. The frozen scenario data is hosted separately and downloaded into `data/`.

**User-facing pipeline** (run these):

| Script | Input | Output |
|---|---|---|
| `scripts/generate_prompts.py` | scenarios + variant configs | prompt CSV |
| `scripts/process_runs.py` | model response CSV + scenarios | per-run metrics CSV |
| `scripts/aggregate_metrics.py` | per-run metrics CSV | scenario aggregates + model-level paper metrics |

**Transparency-only** (inspect, don't run):

| Script | Purpose |
|---|---|
| `scripts/scenario_generation/optimize_ordinal_ranksum_design.py` | Weight-profile optimization before benchmark was frozen |
| `scripts/scenario_generation/generate_scenarios.py` | Scenario universe construction from archetype combinations |
| `scripts/scenario_generation/select_scenarios.py` | MILP selection of core 160 and derived pools |
| `scripts/scenario_generation/build_veto_sidecar.py` | Veto assignment generation |

These scripts produced the frozen benchmark artifact. They are provided for methodological transparency. Users of the benchmark do not need to run them.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download benchmark data

Download the scenario bundle from HuggingFace (see link above) and place it at:

```
data/scenarios_master.json
```

The selection-set CSVs and veto sidecar are already included in this repo under `data/`.

### 3. Render prompts

```bash
python scripts/generate_prompts.py \
  --scenarios data/scenarios_master.json \
  --out-csv results/prompts.csv
```

To render only specific variants or scenarios, use `--test-variant-ids` and `--scenario-ids` (comma-separated) or `--selection-set-csv`. See `config/variant_map.yaml` for the full variant definitions.

### 4. Run your model

Submit each prompt to your model and collect responses into a CSV with at minimum these columns:

| Column | Required | Description |
|---|---|---|
| `scenario_id` | yes | Integer scenario identifier |
| `test_variant_id` | yes | Integer variant identifier |
| `model_name` | yes | Model identifier string |
| `run_index` | recommended | Run number (for multi-run stability) |
| `selected_option_id` | yes* | Integer option selected (1–4) |
| `response_text` | yes* | Raw model output (if `selected_option_id` is not pre-extracted) |

\* Provide at least one of `selected_option_id` or `response_text`.

### 5. Score responses

```bash
python scripts/process_runs.py \
  --responses-csv results/responses/model_outputs.csv \
  --scenarios-master-json data/scenarios_master.json \
  --out-csv results/metrics/individual_runs.csv
```

### 6. Aggregate metrics

```bash
python scripts/aggregate_metrics.py \
  --metrics-csv results/metrics/individual_runs.csv \
  --out-scenario-csv results/metrics/scenario_aggregates.csv \
  --out-model-csv results/metrics/model_metrics.csv
```

---

## Repository layout

```
mesa-health/
├── config/
│   ├── archetypes.yaml            # Stakeholder archetype profiles (frozen)
│   ├── dimensions.yaml            # Evaluation dimensions and labels
│   ├── options.yaml               # Option profiles (frozen)
│   ├── stakeholder_roles.yaml     # Stakeholder role definitions
│   ├── task_directives.yaml       # Aggregation-rule directive wording
│   ├── variant_map.yaml           # Full test-variant definitions (21 variants)
│   ├── test_set_variant_map.yaml  # Pool-to-variant assignment
│   ├── test_set_design.yaml       # Pool size and balance parameters
│   └── rendering/                 # Per-variant prompt template and block configs
├── data/
│   ├── scenarios_master.json      # ← download from HuggingFace
│   ├── scenario_vetoes.yaml       # Veto assignments per scenario
│   └── selection_sets/            # Scenario subsets (core_160, divergent, pareto, veto)
├── scripts/
│   ├── generate_prompts.py        # Prompt rendering
│   ├── process_runs.py            # Per-run scoring
│   ├── aggregate_metrics.py       # Scenario and model-level aggregation
│   ├── _benchmark_lib/            # Shared library modules
│   └── scenario_generation/       # Transparency-only design scripts
├── design/
│   └── README.md                  # Design transparency notes
├── tests/
│   └── test_public_bundle.py      # Smoke tests
├── LICENSE                        # MIT
├── requirements.txt
└── README.md
```

---

## Benchmark design

### Scenarios

Each scenario is a four-stakeholder, four-option decision problem in a healthcare governance domain. Stakeholders are drawn from four roles (patient, clinician, administrator, policymaker), each instantiated as one of four archetypes with distinct dimension-weight profiles over four evaluation dimensions (clinical benefit, safety, affordability, convenience). Weights are constrained to the ordinal rank-sum set {0.4, 0.3, 0.2, 0.1}.

The full scenario universe is 4⁴ = 256 combinations. The core benchmark uses 160 scenarios selected via MILP optimization for balanced conflict distribution, Borda-winner coverage, and manipulation-susceptible scenario inclusion.

### Scenario pools

| Pool | Size | Purpose |
|---|---|---|
| Core 160 | 160 | Primary evaluation set (Borda baseline, 10 runs) |
| Divergent 80 | 80 | Scenarios where Borda, Utilitarian, and Maximin winners disagree |
| Pareto ≥ 2 | 120 | Scenarios with Pareto-dominant sets of size ≥ 2 |
| Veto 60 | 60 | Scenarios with assigned veto constraints |

### Test variants

The 21 test variants probe distinct failure modes by varying prompt structure while preserving semantic equivalence. See `config/variant_map.yaml` for full definitions. Key categories:

- **Baseline** (variant 0): ordinal rankings, anonymous stakeholders, Borda directive
- **Aggregation-rule switching** (variants 3, 4): Maximin and Utilitarian directives
- **Format invariance** (variants 2, 5): adjective and sentence-based preference expression
- **Positional anchoring** (variants 11–13, 14–16): option and stakeholder permutations with named/abstract labels
- **Label sensitivity** (variants 7, 8, 9, 10): paraphrasing, synonym substitution, exaggeration
- **Veto compliance** (variant 1): hard constraint that must override aggregate preference
- **Manipulation sensitivity** (variants 23, 24, etc.): redundant-option and IIA probes
- **Context framing** (variant 17): neutral context without governance framing

### Metrics

| Metric | What it measures |
|---|---|
| AGR (Borda / Maximin / Utilitarian) | Agreement with ground-truth winner under each rule |
| DPC | Dominant Position Capture — positional anchoring bias |
| APDR | Abstract-to-Named Position Divergence Rate |
| CPFR | Cross-Permutation Flip Rate |
| PIR | Paraphrase Instability Rate |
| FIR | Format Instability Rate |
| SLIR | Semantic-Label Instability Rate |
| VVR | Veto Violation Rate |
| MSI | Manipulation Susceptibility Index |

---

## Citing this work

```bibtex
@article{puchoa2026mesa,
  title   = {[title forthcoming]},
  author  = {Puchoa, Alexandre},
  year    = {2026},
  journal = {[forthcoming]},
  note    = {Benchmark code: https://github.com/alexpuchoa/mesa-health}
}
```

---

## License

Code in this repository is released under the [MIT License](LICENSE).

The benchmark data artifact (scenarios, selection sets, veto assignments) is released under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). If you use the benchmark, please cite the paper.
