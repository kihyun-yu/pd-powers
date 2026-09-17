"""Validate the linear adapter and efficient learner against full-history fits."""

import unittest
from unittest.mock import patch

import numpy as np

import cmdp_primal_dual_power as e


def dense_paper_reference(seed, coefficient):
    """Literal full-state, full-history Algorithm 1 for a small test instance.

    Unlike production, use all 2H+2 coordinates, store every transition, and
    refit current next-state values directly on all preceding observations.
    """
    rng = np.random.RandomState(seed)
    x = np.array([[e.linear_features(s, a) for a in range(e.ACTION)] for s in range(e.STATE)])
    size = x.shape[-1]
    settings = e.linear_settings(coefficient)
    rate, eta, theta = settings["alpha"], settings["eta"], settings["theta"]
    design = np.repeat(np.eye(size)[None], e.H, axis=0)
    policy = np.full((e.H, e.STATE, e.ACTION), 1 / e.ACTION)
    states = np.zeros((e.K, e.H + 1), dtype=int)
    actions = np.zeros((e.K, e.H), dtype=int)
    past_features = np.zeros((e.K, e.H, size))
    past_costs = np.zeros((e.K, e.H))
    total_reward, total_deficit, dual = 0., 0., 0.
    rewards, deficits, duals, epochs, mixes = [], [], [], [], []
    epoch_det = np.ones(e.H)
    for k in range(e.K):
        determinants = np.linalg.det(design)
        if k == 0 or np.any(determinants >= 2 * epoch_det * (1 - 1e-12)):
            start = k
            epochs.append(k + 1)
            epoch_det = determinants.copy()
            policy.fill(1 / e.ACTION)
            dual = 0.
            contracted, bonus = [], []
            for h in range(e.H):
                inverse = np.linalg.inv(design[h])
                norm = np.sqrt(np.einsum('sad,df,saf->sa', x, inverse, x))
                scale = 1 / (1 + np.exp(settings["beta_w"] * norm - np.log(e.K)))
                contracted.append(x * scale[:, :, None])
                bonus.append(coefficient * norm * scale)
        for h in range(e.H):
            s = states[k, h]
            a = rng.choice(e.ACTION, p=policy[h, s])
            actions[k, h] = a
            past_features[k, h] = x[s, a]
            past_costs[k, h] = 1 - e.cost(s, a)
            total_reward += e.reward(s, a, k)
            rewards.append(total_reward)
            states[k, h + 1] = (e.H + 1 - rng.binomial(1, e.proba(s, a, s + 1)) * (e.H - s)
                                 if s < e.H else s)
        loss = np.array([[1 - e.reward(s, a, k) for a in range(e.ACTION)] for s in range(e.STATE)])
        loss_parameter = np.linalg.lstsq(x.reshape(-1, size), loss.ravel(), rcond=None)[0]
        values = np.zeros((e.H + 1, e.STATE, 2))
        mixing = (k - start) % settings["mixing_period"] == 0
        if mixing:
            mixes.append(k + 1)
        for h in reversed(range(e.H)):
            observations = past_features[:k, h]
            cost_parameter = np.linalg.solve(design[h], observations.T @ past_costs[:k, h])
            qs = []
            for metric, parameter in enumerate((loss_parameter, cost_parameter)):
                targets = values[h + 1, states[:k, h + 1], metric]
                transition_fit = np.linalg.solve(design[h], observations.T @ targets)
                q = contracted[h] @ (parameter + transition_fit) - bonus[h]
                values[h, :, metric] = (policy[h] * q).sum(axis=1)
                qs.append(q)
            base = (1 - theta) * policy[h] + theta / e.ACTION if mixing else policy[h]
            weights = base * np.exp(-rate * (qs[0] + dual * qs[1]))
            policy[h] = weights / weights.sum(axis=1, keepdims=True)
        dual = max(0., (1 - 4 * rate * eta * e.H**3) * dual + eta * (
            values[0, 0, 1] - (e.H - e.B_CONSTR) - 4 * rate * e.H**3 - 4 * theta * e.H**2))
        duals.append(dual)
        for h in range(e.H):
            design[h] += np.outer(past_features[k, h], past_features[k, h])
        total_deficit += past_costs[k].sum() - (e.H - e.B_CONSTR)
        deficits.append(total_deficit)
    return dict(rewards=rewards, deficits=deficits, dual=duals, epochs=epochs, mixes=mixes,
                policy=np.array([policy[h, h] for h in range(e.H)]), design=design)


class LinearCMDPTests(unittest.TestCase):
    def setUp(self):
        settings = dict(dim=3, H=3, STATE=5, ACTION=4, K=40, B_CONSTR=1.8,
                        thetastar=np.array([.01, .01, 1.]), REWARD_MODE="phase-flip")
        context = patch.multiple(e, **settings)
        context.start()
        self.addCleanup(context.stop)

    def test_features_exactly_represent_transitions_losses_and_costs(self):
        x = np.array([e.linear_features(s, a) for s in range(e.STATE) for a in range(e.ACTION)])
        self.assertLessEqual(np.linalg.norm(x, axis=1).max(), 1.)
        transitions = np.array([[e.proba(s, a, sp) for sp in range(e.STATE)]
                                for s in range(e.STATE) for a in range(e.ACTION)])
        measure = np.linalg.lstsq(x, transitions, rcond=None)[0]
        np.testing.assert_allclose(x @ measure, transitions, atol=1e-14)
        self.assertLessEqual(np.linalg.norm(np.abs(measure).sum(axis=1)), np.sqrt(x.shape[1]) + 1e-12)
        for signal in (lambda s, a: 1 - e.cost(s, a),
                       lambda s, a: 1 - e.reward(s, a, 0),
                       lambda s, a: 1 - e.reward(s, a, 10)):
            targets = [signal(s, a) for s in range(e.STATE) for a in range(e.ACTION)]
            parameter = np.linalg.lstsq(x, targets, rcond=None)[0]
            np.testing.assert_allclose(x @ parameter, targets, atol=1e-14)
            self.assertLessEqual(np.linalg.norm(parameter), np.sqrt(x.shape[1]) + 1e-12)

    def test_efficient_learner_matches_full_history_paper_algorithm(self):
        for rate in (.1, .001):
            with self.subTest(alpha=rate), patch.multiple(e, LINEAR_ALPHA=rate, B_CONSTR=2.9):
                diagnostics = {}
                coefficient = .8 if rate == .1 else .05
                rewards, deficits = e.run_linear_cmdp(seed=37, beta_value=coefficient, diagnostics=diagnostics)
                expected = dense_paper_reference(37, coefficient)
                np.testing.assert_array_equal(rewards, expected["rewards"])
                np.testing.assert_array_equal(deficits, expected["deficits"])
                np.testing.assert_allclose(diagnostics["dual"], expected["dual"], atol=1e-12)
                np.testing.assert_allclose(diagnostics["final_policy"], expected["policy"], atol=1e-12)
                self.assertEqual(diagnostics["epoch_starts"], expected["epochs"])
                self.assertEqual(diagnostics["mixing_episodes"], expected["mixes"])
                for h in range(e.H):
                    indices = [2 * h, 2 * h + 1, 2 * e.H + 1]
                    np.testing.assert_allclose(diagnostics["final_design"][h],
                                               expected["design"][h][np.ix_(indices, indices)])
                if rate == .001:
                    self.assertGreater(diagnostics["dual"].max(), 0.)

    def test_first_update_does_not_fit_current_episode_samples(self):
        with patch.object(e, "K", 1):
            first, second = {}, {}
            e.run_linear_cmdp(seed=7, beta_value=1., diagnostics=first)
            e.run_linear_cmdp(seed=13, beta_value=1., diagnostics=second)
        np.testing.assert_array_equal(first["final_policy"], second["final_policy"])
        self.assertFalse(np.array_equal(first["final_design"], second["final_design"]))

    def test_periodic_mixing_inside_an_epoch_matches_paper_reference(self):
        with patch.multiple(e, H=1, STATE=3, K=400, B_CONSTR=.9):
            diagnostics = {}
            e.run_linear_cmdp(seed=37, beta_value=.05, diagnostics=diagnostics)
            expected = dense_paper_reference(37, .05)
        self.assertTrue(set(diagnostics["mixing_episodes"]) - set(diagnostics["epoch_starts"]))
        self.assertEqual(diagnostics["mixing_episodes"], expected["mixes"])
        np.testing.assert_allclose(diagnostics["final_policy"], expected["policy"], atol=1e-12)

    def test_reporting_and_pd_hyperparameters_do_not_change_linear_learner(self):
        before = e.run_linear_cmdp(seed=7, beta_value=.8)
        state = np.random.get_state()
        with patch.multiple(e, beta=float("nan"), beta1=float("nan"),
                            alpha=float("nan"), theta=float("nan"), dual_lr=float("nan"), LAMBDA=99.):
            diagnostics = {}
            after = e.run_linear_cmdp(seed=7, beta_value=.8, diagnostics=diagnostics)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(state[1], np.random.get_state()[1])
        self.assertEqual(state[2:], np.random.get_state()[2:])

    def test_uniform_policy_reporting_matches_random_reference(self):
        with patch.object(e, "LINEAR_ALPHA", 0.):
            diagnostics = {}
            e.run_linear_cmdp(seed=7, diagnostics=diagnostics)
        benchmark = np.zeros(e.K)
        random_curves = e.random_policy_curves(benchmark)
        np.testing.assert_allclose(-np.cumsum(diagnostics["expected_reward"]), random_curves["random_regret"][0])
        np.testing.assert_allclose(np.cumsum(e.B_CONSTR - diagnostics["expected_utility"]),
                                   random_curves["random_deficit"][0])

    def test_near_random_selection_excludes_identical_and_stronger_candidates(self):
        reference = dict(mean_regret=100., mean_violation=200.)
        candidates = [dict(beta=1., mean_regret=50., mean_violation=100.),
                      dict(beta=2., mean_regret=92., mean_violation=184.),
                      dict(beta=3., mean_regret=100., mean_violation=200.),
                      dict(beta=4., mean_regret=92., mean_violation=220.)]
        selected = e.choose_near_random_candidate(reference, candidates)
        self.assertEqual(selected["beta"], 2.)
        self.assertTrue(selected["target_met"])
        self.assertFalse(candidates[0]["target_met"])
        self.assertFalse(candidates[2]["target_met"])
        self.assertFalse(candidates[3]["target_met"])


if __name__ == "__main__":
    unittest.main()
