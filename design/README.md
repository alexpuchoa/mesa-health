# Design Transparency

These scripts produced the frozen benchmark artifact. They are provided for methodological transparency. Users of the benchmark do not need to run them.

Transparency-side scripts:
- `scripts/scenario_generation/optimize_ordinal_ranksum_design.py`
- `scripts/scenario_generation/generate_scenarios.py`
- `scripts/scenario_generation/select_scenarios.py`
- `scripts/scenario_generation/build_veto_sidecar.py`

Expected use:
- reviewers and maintainers may inspect or rerun these scripts to understand how the benchmark was constructed
- benchmark users should work from the published fixed data artifact and the user-facing evaluation scripts
