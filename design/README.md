# Design Artifacts

These files are design-time inputs for the transparency scripts under
`scripts/scenario_generation/`.

They produced the frozen benchmark artifact and are included so reviewers can
inspect the construction process. Users of the benchmark do not need these
files to render prompts, process runs, or aggregate metrics.

- `test_set_design.yaml`: MILP selection targets such as pool sizes, conflict
  bins, and minimum manipulation-eligible scenarios.
