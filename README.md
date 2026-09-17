# PD-POWERS

Original PD-POWERS with **PD-POWERS without variance** and **Yu et al. 2026** baselines.
The implementation is based on [POWERS (AISTATS 2022)](https://proceedings.mlr.press/v151/he22a.html).
The linear baseline implements Algorithm 1 of
[Yu, Bae, and Lee (ICLR 2026)](https://arxiv.org/abs/2605.11535).

This configuration is an **illustrative hyperparameter comparison**. The
no-variance coefficient is selected to produce a modest performance gap, rather
than to maximize the baseline's performance. This does not establish that
variance weighting beats a well-tuned alternative. The linear baseline's beta
is separately selected to place its curves near Random. It is also not a
best-tuned baseline. The plots and saved metadata disclose these objectives.

## Run

```bash
python3 cmdp_primal_dual_power.py
```

Requires Python 3.9+, NumPy, and Matplotlib. `tqdm` is optional:

```bash
pip install numpy matplotlib tqdm
```

Detailed outputs go into `results/no_variance_baseline/`, relative to the script's
directory even when launched from another working directory. After all runs
finish, a default run also overwrites `regret_plot.jpg` and `violation_plot.jpg`
beside the script with the same updated images. The console prints the absolute
output paths. An explicit `--output-dir` keeps all outputs in the chosen folder
and leaves the root plots alone. The original checked-in logs are retained.
Importing the module does not run experiments. No separate tuning or reproduction
script is needed. Repeating the same settings and seeds reproduces the same
curves; the images need not look different on each run.

## Algorithms

| Setting | PD-POWERS | Without variance |
| --- | --- | --- |
| Primary reward and utility regressions | Inverse-variance weighted ridge | Unweighted ridge |
| Auxiliary mean and second-moment regressions | Reward and utility | Absent |
| Exploration coefficient | `beta = 0.35` | `BASELINE_BETA = 2.0` |
| Moment confidence coefficient | `beta1 = 0.45` | Absent |
| Actor objective | Reward Q + dual × utility Q | Same |
| Dual update | Original | Original |

PD-POWERS retains the restored version's update order, action encoding,
`H²/d` variance floor, Q bounds, and penalized dual update. This change does
not incorporate the previous adaptive-variance or dual-update modifications.
In particular, the original action encoding contains duplicate action vectors,
and the original penalty keeps the dual multiplier inactive with these defaults.
Those limitations apply to both methods.

Fixed features/rewards are cached, identical confidence-radius solves are
batched, and unreachable state updates are omitted. Regression fixtures from
commit `344b4b0` verify that the original learner's sampled returns, deficits,
critic matrices, and reachable policies are preserved on small instances in
all three reward modes. Each method has independent state and uses paired seeds.

The problem size remains `dim=5`, `H=10`, 16 action indices, 12 states, and
2,000 episodes. Evaluation uses 20 paired runs. The required episode utility
remains 6.

## Linear CMDP baseline

`Yu et al. 2026` is this project's legend label for the paper's Algorithm 1.
It uses state-action features, distinct from the transition features used by
PD-POWERS. On this chain, let `u(a)` be the fraction of positive action bits.
Give each chain state its own feature block `(1-u(a), u(a))`, and give each
absorbing state an indicator coordinate. The feature dimension is **22**
(`2H+2`), compared with the original mixture dimension 5. These known features
do not use the unknown transition probabilities. Both transition probabilities
and signals are exactly linear in them for the current equal-coefficient chain.

The adapter converts the project's conventions to the paper's conventions:
`loss = 1 - reward`, `paper_cost = 1 - utility`, and `paper_budget = H - B_CONSTR`.
This preserves reward regret and cumulative utility deficit exactly. The same
environment, episodes, seeds, and fixed-action comparator are used for every
method. This is a new experiment on this chain, not a reproduction of the
paper's job-scheduling experiment or its unrestricted optimal-policy comparator.

The implementation retains determinant-doubling epoch resets, epoch-frozen
sigmoid feature contraction and confidence bonuses, regressions using only
preceding episodes, unclipped Q estimates, periodic mixing before the policy
step, and the regularized dual update with its factors of 4. Following the
paper's Appendix L experimental convention, it uses `alpha=0.1`,
`beta_w=beta_b*log(K)`, `eta=H^-2 K^-3/4`, and `theta=1/K`. The mixing period
is `ceil(K^3/4)` (300 at K=2,000), and ridge regularization is 1.
These settings are independent of PD-POWERS' parameters.

Only three feature coordinates are reachable at each stage. Solving those
blocks and keeping sums of observed features grouped by successor gives the
same regression as refitting all prior observations, without repeatedly
scanning the history. A separate full-state, full-history test implementation
checks this equivalence, including a case with an active dual multiplier.

## Explicit illustrative calibration

```bash
python3 cmdp_primal_dual_power.py --calibrate-demo
python3 cmdp_primal_dual_power.py --calibrate-linear-demo

# Inspect any alternative coefficient directly.
python3 cmdp_primal_dual_power.py --baseline-beta 0.35 --output-dir results/sensitivity
python3 cmdp_primal_dual_power.py --linear-beta 4 --output-dir results/linear_sensitivity
```

Calibration fixes PD-POWERS and evaluates no-var coefficients
`[0.35, 0.75, 1.25, 1.75, 1.9, 2.0, 2.1, 2.25, 2.75]` on seeds **100–104**. The declared target
is **5–15% higher mean regret** and no lower final cumulative violation for the
illustrative baseline. Among eligible candidates, selection targets a 10% regret
gap. Every candidate, including stronger settings, is retained in
`calibration.json`. If none meets the target, calibration stops before evaluation.
The three values between 1.75 and 2.25 were added after the initial grid
bracketed the target; this refinement used only the calibration seeds.

The selected settings are frozen and evaluated on separate seeds
**4000–4019**. The earlier no-var evaluation was extended from five to 20 runs
with beta fixed at 2.0.
There is no guarantee that the ordering holds on every run or in another
environment. `--beta-grid`, `--seeds`, `--episodes`, and `--output-dir`
allow explicit alternative configurations. Calibration rejects overlapping
calibration/evaluation seeds. Ordinary runs use the preset constants;
after recalibrating, set `BASELINE_BETA` to the newly selected value if desired.
The original PD-POWERS betas stay fixed.

For the linear baseline, `--calibrate-linear-demo` searches
`[0.35, 1, 2, 4, K^0.25, 8, 16, 32]` on seeds **100–104**. The objective is
**2–15% lower final regret AND violation than Random**, targeting 8% on both.
All trials, including stronger settings and the paper's `K^0.25` coefficient,
are saved in `linear_calibration.json`. The selection uses final metrics;
closeness throughout the curves is inspected separately. No curve is offset,
rescaled, interpolated with Random, or generated by adding artificial noise.

The selected **`LINEAR_BETA=16`** was frozen before evaluation on seeds
**4000–4019**. Calibration means were 1,844.12 regret and 3,525.30 violation.
For comparison, `beta_b=4` achieved lower values, 1,189.29 and 2,164.22,
respectively; the chosen coefficient deliberately does not maximize performance.
Use `--linear-beta` for an explicit coefficient, or `--linear-beta-grid` with
`--calibrate-linear-demo` for a different search. Both calibration flags can
be combined. An ordinary run uses the constants in the script; calibration
does not rewrite them. The near-Random ordering is not guaranteed on other seeds
or settings.

## Measured result for this preset

PD-POWERS uses its original `beta=0.35`, `beta1=0.45`. The illustrative no-var
preset uses `beta=2.0`, and the linear baseline uses `beta_b=16`.
At 2,000 episodes, averaged over seeds 4000–4019:

| Expected metric | PD-POWERS | Without variance | Yu et al. 2026 | Random |
| --- | ---: | ---: | ---: | ---: |
| Final regret | 855.60 | 929.80 | 1,825.71 | 1,985.59 |
| Final positive cumulative utility deficit | 1,233.48 | 1,406.35 | 3,487.98 | 3,974.74 |

The linear baseline has 8.1% lower regret and 12.2% lower violation than Random.
Its curves nearly overlap Random early and separate later as its contracted
features become more informative. The linear baseline's dual multiplier stays
zero under this empirical preset, so these results do not demonstrate a dual
update benefit. All methods have positive final cumulative violation here.
The existing PD-POWERS, no-var, and Random data were preserved when adding this
baseline; only the new method required additional evaluation runs.

Relative to PD-POWERS, no-var has 8.7% higher regret and 14.0% higher violation.
Against no-var, PD-POWERS has lower regret and lower violation on 16/20 seeds.
The paired no-var-minus-PD regret difference is 74.20, with approximate 95%
interval [30.06, 118.34]; the violation difference is 172.87, with interval
[64.38, 281.36]. Both methods still have nonzero constraint violation.

This is deliberately a parameter-sensitive illustration. On the calibration
seeds, no-var at `beta=0.75` had **lower** regret (766.40) than PD-POWERS
(916.19), and lower violation. That stronger setting remains in the calibration
record. The chosen `beta=2.0` therefore does not represent a best-tuned baseline.
PD-POWERS also hits its `H²/d` variance floor in every measured nonterminal
update here; the observed gap does not demonstrate a benefit from adapting
weights to changing variance.

## Metrics and files

All learners train on sampled transitions. The linear baseline fits loss and
transformed cost, while PD-POWERS and no-var fit reward and utility.
Scoring uses exact expected reward and utility of each deployed policy, calculated only for
reporting. This reduces rollout noise; it does not alter learning or add an
offset to either curve. The benchmark remains the best feasible deterministic
fixed action, selected by enumeration, not an LP over all policies.

Violation is `max(0, sum_k(B_CONSTR - expected_episode_utility))`, calculated
per run before averaging. It allows earlier utility surplus to offset later
shortfalls. Sampled rewards, sampled signed deficits, and expected per-episode
values are retained for inspection. Confidence bands are approximate 95%
intervals across independent runs.

- `comparison.json`: all settings, per-seed results, paired differences, dual
  activity, variance-floor diagnostics, and the linear baseline's contraction
  and epoch counts. Method keys are `pd_powers`, `novar`, and `linear_cmdp`.
- `curves.npz`: raw sampled histories and exact expected-value curves.
- `regret_plot.jpg` and `violation_plot.jpg`: the original 8-by-5-inch,
  single-panel style with 20-point axis labels, 14-point legends, confidence bands,
  and no grid or large title. Legend order is `Random`, `Yu et al. 2026`,
  `PD-POWERS w/o Var`, and `PD-POWERS (Ours)`; beta values remain in the saved metadata.
  A small footer identifies the illustrative comparison.
- `calibration.json`: all development trials and the explicit gap-selection
  objective; created with `--calibrate-demo`.
- `linear_calibration.json`: all linear-baseline trials and the explicit
  proximity-to-Random objective; created with `--calibrate-linear-demo`.

## Checks

```bash
python3 -m unittest discover -s tests -v
```

Tests cover original-learner fidelity, removal of moment regressions for no-var,
isolated parameters/RNGs, expected-value evaluation, explicit illustrative
selection, and agreement between saved curves, metrics, and plots.
Linear-baseline tests additionally check feature realizability and norm bounds,
equivalence to full-history Algorithm 1, pre-episode feedback ordering, and
independence from PD-POWERS hyperparameters.
