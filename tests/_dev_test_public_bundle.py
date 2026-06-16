from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import yaml

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = BUNDLE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _benchmark_lib.metrics import scenario_properties
from _benchmark_lib.scenario_generation import build_scenarios, load_archetypes, load_dimensions, load_options


class PublicBundleSmokeTest(unittest.TestCase):
    """Smoke tests that exercise the published benchmark bundle end to end."""

    def _build_one_scenario_bundle(self) -> tuple[Path, Path]:
        """Create a tiny temporary bundle that still matches the public file layout."""
        dimensions = load_dimensions(BUNDLE_ROOT / "config" / "dimensions.yaml")
        archetypes = load_archetypes(BUNDLE_ROOT / "config" / "archetypes.yaml", dimensions=dimensions)
        options = load_options(BUNDLE_ROOT / "config" / "options.yaml", dimensions=dimensions)
        scenarios, _utilities, _rankings, _flat = build_scenarios(
            archetypes=archetypes,
            options=options,
            dimensions=dimensions,
            scenario_id_start=1,
        )
        payload = {"scenarios": [scenarios[0]]}
        tmpdir = Path(tempfile.mkdtemp(prefix="benchmark_bundle_test_"))
        scenario_path = tmpdir / "scenarios.json"
        scenario_path.write_text(json.dumps(payload), encoding="utf-8")
        selection_dir = tmpdir / "data" / "selection_sets"
        selection_dir.mkdir(parents=True, exist_ok=True)
        for name in ("core_160", "core_160_5", "divergent", "pareto_ge2", "veto"):
            with (selection_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["scenario_id"])
                writer.writeheader()
                writer.writerow({"scenario_id": 1})
        pool_map = {
            "pools": {
                "core_160": {"csv": "data/selection_sets/core_160.csv", "test_variant_ids": [0]},
                "core_160_5": {"csv": "data/selection_sets/core_160_5.csv", "test_variant_ids": [2, 3, 4, 5, 7, 10, 17, 23, 24, 31, 33, 44]},
                "borda_ne_util_maximin": {"csv": "data/selection_sets/divergent.csv", "test_variant_ids": [11, 12, 13, 111, 112, 113]},
                "pareto_ge2": {"csv": "data/selection_sets/pareto_ge2.csv", "test_variant_ids": [8, 9]},
                "veto": {"csv": "data/selection_sets/veto.csv", "test_variant_ids": [1]},
            }
        }
        pool_map_path = tmpdir / "config" / "test_set_variant_map.yaml"
        pool_map_path.parent.mkdir(parents=True, exist_ok=True)
        pool_map_path.write_text(yaml.safe_dump(pool_map, sort_keys=False), encoding="utf-8")
        self.addCleanup(lambda: shutil.rmtree(tmpdir, ignore_errors=True))
        return scenario_path, pool_map_path

    def test_generate_prompts_baseline_and_named_variant(self) -> None:
        """Prompt generation should render both baseline and permuted variants coherently."""
        scenario_path, _pool_map_path = self._build_one_scenario_bundle()
        out_csv = scenario_path.parent / "prompts.csv"
        cmd = [
            sys.executable,
            str(BUNDLE_ROOT / "scripts" / "generate_prompts.py"),
            "--scenarios",
            str(scenario_path),
            "--test-variant-ids",
            "0,10",
            "--out-csv",
            str(out_csv),
        ]
        subprocess.run(cmd, check=True)
        with out_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        baseline = next(row for row in rows if row["test_variant_id"] == "0")
        named = next(row for row in rows if row["test_variant_id"] == "10")
        self.assertIn("Option 1: Patient convenience > Safety > Affordability > Clinical benefit", baseline["prompt_text"])
        self.assertIn("Patient priorities:", named["prompt_text"])

    def test_process_runs_maps_permuted_displayed_choice_back_to_canonical(self) -> None:
        """Run processing should undo displayed-option permutations before scoring."""
        scenario_path, pool_map_path = self._build_one_scenario_bundle()
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))["scenarios"][0]
        props = scenario_properties(scenario)
        canonical_borda = int(props["borda_winner_set"][0])
        self.assertEqual(canonical_borda, 3)

        responses_csv = scenario_path.parent / "responses.csv"
        with responses_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["scenario_id", "test_variant_id", "model_name", "selected_option_id"])
            writer.writeheader()
            writer.writerow({"scenario_id": 1, "test_variant_id": 0, "model_name": "demo", "selected_option_id": 3})
            writer.writerow({"scenario_id": 1, "test_variant_id": 11, "model_name": "demo", "selected_option_id": 1})

        out_csv = scenario_path.parent / "metrics.csv"
        cmd = [
            sys.executable,
            str(BUNDLE_ROOT / "scripts" / "process_runs.py"),
            "--responses-csv",
            str(responses_csv),
            "--scenarios-master-json",
            str(scenario_path),
            "--pool-map",
            str(pool_map_path),
            "--out-csv",
            str(out_csv),
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        with out_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        baseline = next(row for row in rows if row["test_variant_id"] == "0")
        permuted = next(row for row in rows if row["test_variant_id"] == "11")
        self.assertEqual(baseline["canonical_selected_option_id"], "3")
        self.assertEqual(permuted["canonical_selected_option_id"], "3")
        self.assertEqual(baseline["is_correct"], "1")
        self.assertEqual(permuted["is_correct"], "1")
        self.assertIn("warning: model=demo missing required test variant(s) for APDR", result.stderr)

    def test_aggregate_metrics_outputs_scenario_and_model_layers(self) -> None:
        """Aggregation should emit both scenario-modal rows and paper-metric rows."""
        tmpdir = Path(tempfile.mkdtemp(prefix="benchmark_bundle_aggregate_"))
        self.addCleanup(lambda: shutil.rmtree(tmpdir, ignore_errors=True))
        metrics_csv = tmpdir / "metrics.csv"
        with metrics_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model_name",
                    "run_index",
                    "scenario_id",
                    "scenario_code",
                    "test_variant_id",
                    "test_code",
                    "task_directive_policy_code",
                    "pool_membership",
                    "displayed_option_order",
                    "displayed_stakeholder_order",
                    "displayed_selected_option_id",
                    "canonical_selected_option_id",
                    "parse_valid",
                    "parse_error",
                    "borda_correct",
                    "maximin_correct",
                    "utilitarian_correct",
                    "maximin_ordinal_correct",
                    "rule_correct",
                    "displayed_winner_set",
                    "canonical_winner_set",
                    "is_correct",
                    "veto_violation",
                    "conflict_level",
                    "conflict_score",
                    "conflict_kendall_tau_avg",
                    "borda_winner_set",
                    "utilitarian_winner_set",
                    "maximin_winner_set",
                    "maximin_ordinal_winner_set",
                ],
            )
            writer.writeheader()
            writer.writerow({"model_name": "demo", "run_index": 1, "scenario_id": 1, "scenario_code": "s1", "test_variant_id": 0, "test_code": "borda", "task_directive_policy_code": "borda", "pool_membership": "core_160|core_160_5|borda_ne_util_maximin|pareto_ge2", "displayed_option_order": "1|2|3|4", "displayed_stakeholder_order": "1|2|3|4", "displayed_selected_option_id": 3, "canonical_selected_option_id": 3, "parse_valid": 1, "borda_correct": 1, "maximin_correct": 0, "utilitarian_correct": 1, "maximin_ordinal_correct": 1, "rule_correct": 1, "displayed_winner_set": "3", "canonical_winner_set": "3", "is_correct": 1, "veto_violation": "", "conflict_level": "medium", "conflict_score": 0.5, "conflict_kendall_tau_avg": 0.5, "borda_winner_set": "3", "utilitarian_winner_set": "3", "maximin_winner_set": "4", "maximin_ordinal_winner_set": "3"})
            writer.writerow({"model_name": "demo", "run_index": 1, "scenario_id": 1, "scenario_code": "s1", "test_variant_id": 3, "test_code": "maximin", "task_directive_policy_code": "maximin", "pool_membership": "core_160|core_160_5", "displayed_option_order": "1|2|3|4", "displayed_stakeholder_order": "1|2|3|4", "displayed_selected_option_id": 4, "canonical_selected_option_id": 4, "parse_valid": 1, "borda_correct": 0, "maximin_correct": 1, "utilitarian_correct": 0, "maximin_ordinal_correct": 0, "rule_correct": 1, "displayed_winner_set": "4", "canonical_winner_set": "4", "is_correct": 1, "veto_violation": "", "conflict_level": "medium", "conflict_score": 0.5, "conflict_kendall_tau_avg": 0.5, "borda_winner_set": "3", "utilitarian_winner_set": "3", "maximin_winner_set": "4", "maximin_ordinal_winner_set": "3"})
            writer.writerow({"model_name": "demo", "run_index": 1, "scenario_id": 1, "scenario_code": "s1", "test_variant_id": 4, "test_code": "utilitarian", "task_directive_policy_code": "utilitarian", "pool_membership": "core_160|core_160_5", "displayed_option_order": "1|2|3|4", "displayed_stakeholder_order": "1|2|3|4", "displayed_selected_option_id": 3, "canonical_selected_option_id": 3, "parse_valid": 1, "borda_correct": 1, "maximin_correct": 0, "utilitarian_correct": 1, "maximin_ordinal_correct": 1, "rule_correct": 1, "displayed_winner_set": "3", "canonical_winner_set": "3", "is_correct": 1, "veto_violation": "", "conflict_level": "medium", "conflict_score": 0.5, "conflict_kendall_tau_avg": 0.5, "borda_winner_set": "3", "utilitarian_winner_set": "3", "maximin_winner_set": "4", "maximin_ordinal_winner_set": "3"})
            writer.writerow({"model_name": "demo", "run_index": 1, "scenario_id": 1, "scenario_code": "s1", "test_variant_id": 11, "test_code": "permulation_options_1", "task_directive_policy_code": "borda", "pool_membership": "borda_ne_util_maximin", "displayed_option_order": "3|1|4|2", "displayed_stakeholder_order": "1|2|3|4", "displayed_selected_option_id": 1, "canonical_selected_option_id": 3, "parse_valid": 1, "borda_correct": 1, "maximin_correct": 0, "utilitarian_correct": 1, "maximin_ordinal_correct": 1, "rule_correct": 1, "displayed_winner_set": "1", "canonical_winner_set": "3", "is_correct": 1, "veto_violation": "", "conflict_level": "medium", "conflict_score": 0.5, "conflict_kendall_tau_avg": 0.5, "borda_winner_set": "3", "utilitarian_winner_set": "3", "maximin_winner_set": "4", "maximin_ordinal_winner_set": "3"})
            writer.writerow({"model_name": "demo", "run_index": 1, "scenario_id": 1, "scenario_code": "s1", "test_variant_id": 12, "test_code": "permulation_options_2", "task_directive_policy_code": "borda", "pool_membership": "borda_ne_util_maximin", "displayed_option_order": "4|3|1|2", "displayed_stakeholder_order": "1|2|3|4", "displayed_selected_option_id": 2, "canonical_selected_option_id": 3, "parse_valid": 1, "borda_correct": 1, "maximin_correct": 0, "utilitarian_correct": 1, "maximin_ordinal_correct": 1, "rule_correct": 1, "displayed_winner_set": "2", "canonical_winner_set": "3", "is_correct": 1, "veto_violation": "", "conflict_level": "medium", "conflict_score": 0.5, "conflict_kendall_tau_avg": 0.5, "borda_winner_set": "3", "utilitarian_winner_set": "3", "maximin_winner_set": "4", "maximin_ordinal_winner_set": "3"})
            writer.writerow({"model_name": "demo", "run_index": 1, "scenario_id": 1, "scenario_code": "s1", "test_variant_id": 13, "test_code": "permulation_options_3", "task_directive_policy_code": "borda", "pool_membership": "borda_ne_util_maximin", "displayed_option_order": "2|4|3|1", "displayed_stakeholder_order": "1|2|3|4", "displayed_selected_option_id": 3, "canonical_selected_option_id": 3, "parse_valid": 1, "borda_correct": 1, "maximin_correct": 0, "utilitarian_correct": 1, "maximin_ordinal_correct": 1, "rule_correct": 1, "displayed_winner_set": "3", "canonical_winner_set": "3", "is_correct": 1, "veto_violation": "", "conflict_level": "medium", "conflict_score": 0.5, "conflict_kendall_tau_avg": 0.5, "borda_winner_set": "3", "utilitarian_winner_set": "3", "maximin_winner_set": "4", "maximin_ordinal_winner_set": "3"})
            writer.writerow({"model_name": "demo", "run_index": 1, "scenario_id": 1, "scenario_code": "s1", "test_variant_id": 44, "test_code": "abstract_denominations", "task_directive_policy_code": "borda", "pool_membership": "core_160|core_160_5|borda_ne_util_maximin", "displayed_option_order": "1|2|3|4", "displayed_stakeholder_order": "1|2|3|4", "displayed_selected_option_id": 2, "canonical_selected_option_id": 2, "parse_valid": 1, "borda_correct": 0, "maximin_correct": 0, "utilitarian_correct": 0, "maximin_ordinal_correct": 0, "rule_correct": 0, "displayed_winner_set": "3", "canonical_winner_set": "3", "is_correct": 0, "veto_violation": "", "conflict_level": "medium", "conflict_score": 0.5, "conflict_kendall_tau_avg": 0.5, "borda_winner_set": "3", "utilitarian_winner_set": "3", "maximin_winner_set": "4", "maximin_ordinal_winner_set": "3"})
            writer.writerow({"model_name": "demo", "run_index": 1, "scenario_id": 1, "scenario_code": "s1", "test_variant_id": 1, "test_code": "borda_veto", "task_directive_policy_code": "borda", "pool_membership": "veto", "displayed_option_order": "1|2|3|4", "displayed_stakeholder_order": "1|2|3|4", "displayed_selected_option_id": 2, "canonical_selected_option_id": 2, "parse_valid": 1, "borda_correct": 0, "maximin_correct": 0, "utilitarian_correct": 0, "maximin_ordinal_correct": 0, "rule_correct": 0, "displayed_winner_set": "3", "canonical_winner_set": "3", "is_correct": 0, "veto_violation": 1, "conflict_level": "medium", "conflict_score": 0.5, "conflict_kendall_tau_avg": 0.5, "borda_winner_set": "3", "utilitarian_winner_set": "3", "maximin_winner_set": "4", "maximin_ordinal_winner_set": "3"})
            writer.writerow({"model_name": "demo", "run_index": 1, "scenario_id": 1, "scenario_code": "s1", "test_variant_id": 8, "test_code": "emphasis_pos", "task_directive_policy_code": "borda", "pool_membership": "pareto_ge2", "displayed_option_order": "1|2|3|4", "displayed_stakeholder_order": "1|2|3|4", "displayed_selected_option_id": 1, "canonical_selected_option_id": 1, "parse_valid": 1, "borda_correct": 0, "maximin_correct": 0, "utilitarian_correct": 0, "maximin_ordinal_correct": 0, "rule_correct": 0, "displayed_winner_set": "3", "canonical_winner_set": "3", "is_correct": 0, "veto_violation": "", "conflict_level": "medium", "conflict_score": 0.5, "conflict_kendall_tau_avg": 0.5, "borda_winner_set": "3", "utilitarian_winner_set": "3", "maximin_winner_set": "4", "maximin_ordinal_winner_set": "3"})
            writer.writerow({"model_name": "demo", "run_index": 1, "scenario_id": 1, "scenario_code": "s1", "test_variant_id": 9, "test_code": "emphasis_pos_2", "task_directive_policy_code": "borda", "pool_membership": "pareto_ge2", "displayed_option_order": "1|2|3|4", "displayed_stakeholder_order": "1|2|3|4", "displayed_selected_option_id": 3, "canonical_selected_option_id": 3, "parse_valid": 1, "borda_correct": 1, "maximin_correct": 0, "utilitarian_correct": 1, "maximin_ordinal_correct": 1, "rule_correct": 1, "displayed_winner_set": "3", "canonical_winner_set": "3", "is_correct": 1, "veto_violation": "", "conflict_level": "medium", "conflict_score": 0.5, "conflict_kendall_tau_avg": 0.5, "borda_winner_set": "3", "utilitarian_winner_set": "3", "maximin_winner_set": "4", "maximin_ordinal_winner_set": "3"})

        out_scenario = tmpdir / "scenario.csv"
        out_model = tmpdir / "model.csv"
        cmd = [
            sys.executable,
            str(BUNDLE_ROOT / "scripts" / "aggregate_metrics.py"),
            "--metrics-csv",
            str(metrics_csv),
            "--out-scenario-csv",
            str(out_scenario),
            "--out-model-csv",
            str(out_model),
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        with out_scenario.open("r", encoding="utf-8", newline="") as f:
            scenario_rows = list(csv.DictReader(f))
        with out_model.open("r", encoding="utf-8", newline="") as f:
            model_rows = list(csv.DictReader(f))
        self.assertEqual(len(scenario_rows), 10)
        self.assertEqual(len(model_rows), 1)
        model = model_rows[0]
        self.assertEqual(model["AGR_Borda"], "1.0")
        self.assertEqual(model["AGR_Maximin"], "1.0")
        self.assertEqual(model["AGR_Utilitarian"], "1.0")
        self.assertEqual(model["DPC"], "1.0")
        self.assertEqual(model["APDR"], "")
        self.assertEqual(model["CPFR"], "1.0")
        self.assertEqual(model["VVR"], "1.0")
        self.assertEqual(model["MSI"], "1.0")
        self.assertEqual(model["FIR"], "")
        self.assertIn("warning: model=demo missing required test variant(s) for APDR", result.stderr)
        self.assertIn("warning: model=demo missing required test variant(s) for FIR", result.stderr)

    def test_optimize_ordinal_ranksum_design_smoke(self) -> None:
        """The transparency optimizer should produce its expected artifact set."""
        tmpdir = Path(tempfile.mkdtemp(prefix="benchmark_bundle_opt_"))
        self.addCleanup(lambda: shutil.rmtree(tmpdir, ignore_errors=True))
        out_dir = tmpdir / "opt"
        out_archetypes = tmpdir / "archetypes_opt.yaml"
        out_options = tmpdir / "options_opt.yaml"
        cmd = [
            sys.executable,
            str(BUNDLE_ROOT / "scripts" / "scenario_generation" / "optimize_ordinal_ranksum_design.py"),
            "--content-dir",
            str(BUNDLE_ROOT / "config"),
            "--out-dir",
            str(out_dir),
            "--n-starts",
            "1",
            "--iters-per-start",
            "2",
            "--seed",
            "7",
            "--init-from-yaml",
            "--out-archetypes-yaml",
            str(out_archetypes),
            "--out-options-yaml",
            str(out_options),
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        self.assertTrue((out_dir / "result.json").exists())
        self.assertTrue((out_dir / "scenario_outcomes.csv").exists())
        self.assertTrue(out_archetypes.exists())
        self.assertTrue(out_options.exists())
        self.assertIn("ok ordinal discrete optimization", result.stdout)


if __name__ == "__main__":
    unittest.main()
