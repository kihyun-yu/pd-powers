"""Check fidelity to the restored learner and isolation of the new baseline."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import cmdp_primal_dual_power as experiment


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.reference = json.loads((Path(__file__).parent / "original_pd_powers.json").read_text())
        settings = dict(self.reference["settings"], thetastar=np.array([.01, .01, 1.]),
                        REWARD_MODE="phase-flip", beta=.35, beta1=.45)
        context = patch.multiple(experiment, **settings)
        context.start()
        self.addCleanup(context.stop)

    def test_original_variance_learner_matches_saved_full_state_reference(self):
        for case in self.reference["fixtures"]:
            with self.subTest(mode=case["mode"]):
                experiment.REWARD_MODE = case["mode"]
                diagnostics = {}
                rewards, deficits = experiment.run_primal_dual(
                    seed=self.reference["seed"], diagnostics=diagnostics)
                np.testing.assert_array_equal(rewards, case["rewards"])
                np.testing.assert_array_equal(deficits, case["deficits"])
                for metric in ("policy", "reward_covariance", "utility_covariance"):
                    np.testing.assert_allclose(diagnostics[f"final_{metric}"], case[metric], rtol=1e-10, atol=1e-10)

    def test_novar_has_no_moment_regressions_or_beta1_dependency(self):
        solve = np.linalg.lstsq
        with patch.object(np.linalg, "lstsq", wraps=solve) as calls:
            before = experiment.run_primal_dual(use_variance=False, seed=7)
        # One primary fit and one batch of confidence radii for each critic.
        self.assertEqual(calls.call_count, 4 * experiment.K * experiment.H)
        with patch.object(experiment, "beta1", float("nan")):
            after = experiment.run_primal_dual(use_variance=False, seed=7)
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left, right)

    def test_beta_defaults_and_rng_are_isolated(self):
        for use_variance, other_beta in ((True, "BASELINE_BETA"), (False, "beta")):
            before = experiment.run_primal_dual(use_variance=use_variance, seed=7)
            global_state = np.random.get_state()
            with patch.object(experiment, other_beta, 100.):
                after = experiment.run_primal_dual(use_variance=use_variance, seed=7)
            for left, right in zip(before, after):
                np.testing.assert_array_equal(left, right)
            np.testing.assert_array_equal(np.random.get_state()[1], global_state[1])
            self.assertEqual(np.random.get_state()[2:], global_state[2:])
        yes = experiment.run_primal_dual(use_variance=True, seed=7)
        no = experiment.run_primal_dual(use_variance=False, seed=7)
        np.testing.assert_array_equal(yes[0][:experiment.H], no[0][:experiment.H])

    def test_reporting_does_not_change_learning_and_exact_values_match(self):
        for use_variance in (True, False):
            before = experiment.run_primal_dual(use_variance=use_variance, seed=7)
            diagnostics = {}
            after = experiment.run_primal_dual(use_variance=use_variance, seed=7, diagnostics=diagnostics)
            for left, right in zip(before, after):
                np.testing.assert_array_equal(left, right)
        with patch.multiple(experiment, alpha=0., K=2):
            diagnostics = {}
            experiment.run_primal_dual(seed=7, diagnostics=diagnostics)
            for metric, immediate in (("reward", lambda s, a: experiment.reward(s, a, 0)),
                                      ("utility", experiment.cost)):
                values = np.zeros(experiment.STATE)
                for _ in range(experiment.H):
                    values = np.array([np.mean([
                        immediate(s, a) + sum(experiment.proba(s, a, sp) * values[sp]
                                              for sp in range(experiment.STATE))
                        for a in range(experiment.ACTION)]) for s in range(experiment.STATE)])
                np.testing.assert_allclose(diagnostics[f"expected_{metric}"], values[0])

    def test_illustrative_selection_is_explicit_and_checks_violation(self):
        reference = {"mean_regret": 100., "mean_violation": 20.}
        candidates = [{"beta": .5, "mean_regret": 95., "mean_violation": 21.},
                      {"beta": 1., "mean_regret": 110., "mean_violation": 19.},
                      {"beta": 2., "mean_regret": 112., "mean_violation": 22.}]
        selected = experiment.choose_illustrative_candidate(reference, candidates)
        self.assertEqual(selected["beta"], 2.)
        self.assertTrue(selected["target_met"])
        self.assertEqual(len(candidates), 3)  # Preserve the better baseline too.

    def test_cli_writes_disclosed_comparison_and_genuine_curve_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ["experiment", "--episodes", "4", "--seeds", "7", "8",
                    "--baseline-beta", "1.75", "--output-dir", directory]
            with patch("sys.argv", args), redirect_stdout(StringIO()):
                experiment.main()
            summary = json.loads((Path(directory) / "comparison.json").read_text())
            self.assertIn("illustrative", summary["purpose"])
            self.assertEqual(summary["methods"]["novar"]["beta"], 1.75)
            self.assertEqual(set(summary["methods"]), {"pd_powers", "novar"})
            with np.load(Path(directory) / "curves.npz") as curves:
                for method in ("pd_powers", "novar"):
                    for index, row in enumerate(summary["methods"][method]["runs"]):
                        self.assertAlmostEqual(curves[f"{method}_regret"][index, -1], row["regret"])
                        self.assertAlmostEqual(max(0, curves[f"{method}_deficit"][index, -1]), row["violation"])
            for filename in ("regret_plot.jpg", "violation_plot.jpg"):
                with Image.open(Path(directory) / filename) as image:
                    image.verify()
            self.assertEqual(len(list(Path(directory).iterdir())), 4)


if __name__ == "__main__":
    unittest.main()
