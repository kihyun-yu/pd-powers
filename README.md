# PD-POWERS

Original PD-POWERS with an additional **PD-POWERS without variance** baseline.
The implementation is based on [POWERS (AISTATS 2022)](https://proceedings.mlr.press/v151/he22a.html).

This configuration is an **illustrative hyperparameter comparison**. The
no-variance coefficient is selected to produce a modest performance gap, rather
than to maximize the baseline's performance. This does not establish that
variance weighting beats a well-tuned alternative. The plots and
saved metadata disclose this selection objective.

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

## Explicit illustrative calibration

```bash
python3 cmdp_primal_dual_power.py --calibrate-demo

# Inspect any alternative coefficient directly.
python3 cmdp_primal_dual_power.py --baseline-beta 0.35 --output-dir results/sensitivity
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

## Measured result for this preset

PD-POWERS uses its original `beta=0.35`, `beta1=0.45`. The illustrative no-var
preset uses `beta=2.0`.
At 2,000 episodes, averaged over seeds 4000–4019:

| Expected metric | PD-POWERS | Without variance |
| --- | ---: | ---: |
| Final regret | 855.60 | 929.80 |
| Final positive cumulative utility deficit | 1,233.48 | 1,406.35 |

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

Both learners train on sampled transitions and fit reward and utility critics.
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
  activity, and variance-floor diagnostics. Method keys are `pd_powers` and
  `novar`.
- `curves.npz`: raw sampled histories and exact expected-value curves.
- `regret_plot.jpg` and `violation_plot.jpg`: the original 8-by-5-inch,
  single-panel style with 20-point labels and legends, confidence bands, and
  no grid or large title. Legends show only `Random`, `PD-POWERS`,
  and `PD-POWERS w/o Var`; beta values remain in the saved metadata.
  A small footer identifies the illustrative comparison.
- `calibration.json`: all development trials and the explicit gap-selection
  objective; created with `--calibrate-demo`.

## Checks

```bash
python3 -m unittest discover -s tests -v
```

Tests cover original-learner fidelity, removal of moment regressions for no-var,
isolated parameters/RNGs, expected-value evaluation, explicit illustrative
selection, and agreement between saved curves, metrics, and plots.
