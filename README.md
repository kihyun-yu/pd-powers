# PD-POWERS

PD-POWERS is a Python project for comparing reinforcement learning algorithms
in a constrained Markov decision process (CMDP). It compares PD-POWERS,
PD-POWERS without variance weighting, the Yu et al. (2026) linear CMDP baseline,
and a random policy using illustrative hyperparameter settings.

## How to run

Requires Python 3.9+, NumPy, and Matplotlib. `tqdm` is optional for progress bars.

From the project directory, install dependencies and run:

```bash
python3 -m pip install numpy matplotlib tqdm
python3 cmdp_primal_dual_power.py
```

To view available run options:

```bash
python3 cmdp_primal_dual_power.py --help
```

## Plot location

Both scripts save all figures in the project's **`plots/`** folder:

- `comparison_regret.jpg`
- `comparison_violation.jpg`
- `sensitivity_regret.jpg`
- `sensitivity_violation.jpg`

Raw data and summaries are saved under each experiment's `results/` directory.
`--output-dir` changes the raw data location; `--plot-dir` optionally overrides
the shared plot folder. Re-running an experiment updates its two named figures.

## Coupled-size sensitivity (PD-POWERS only)

Run the reviewer experiment with **H, S, A, and d changing together**, keeping
the original relationships `S = H + 2` and `A = 2**(d - 1)`:

```bash
python3 sweep_pd_powers.py --workers 4
```

Each metric has one plot with a curve for each complete configuration:

| H | S | A | d |
|---|---|---|---|
| 5 | 7 | 4 | 3 |
| 8 | 10 | 8 | 4 |
| 10 | 12 | 16 | 5 |

`--horizons` and `--dimensions` are paired by position. Both lists must have
the same length and each must contain unique values, so every configuration
changes all four sizes together.

Only PD-POWERS runs. Each setting uses 2,000 episodes and the same 20 seeds
(4000–4019). All learner
hyperparameters stay fixed. The utility lower bound scales as `b = 0.6H`
(recovering `b = 6` at `H = 10`), since keeping `b = 6` at `H = 5` is infeasible.
The transition parameter remains `theta_i = 0.01`, with its length set by `d`.

Plots are saved as `plots/sensitivity_regret.jpg` and
`plots/sensitivity_violation.jpg`. `results/pd_powers_sensitivity/` contains
`sensitivity.json` (full settings and per-seed summaries) and `curves.npz`
(per-seed histories). Bands show approximate pointwise 95%
confidence intervals across seeds. Regret uses each setting's best feasible
deterministic fixed action as its comparator and can be negative. Constraint
violation is the positive part of cumulative expected utility deficit, taken
per seed before averaging. Raw metrics are in episode-return units, so changing
H also changes their scale.

These plots measure the **combined effect of H, S, A, and d**. Changing d
also changes the feature dimension and available
transition probabilities. There is no hyperparameter tuning or selection of
settings based on their observed performance.

Custom paired settings and shorter runs are available, for example:

```bash
python3 sweep_pd_powers.py --horizons 4 6 8 10 --dimensions 3 4 5 6 \
  --episodes 200 --seeds 4000 4001 4002 --workers 4 \
  --output-dir results/sensitivity_preview
```

With the original transition coefficients, valid dimensions are 2–6. The runner
rejects invalid transition probabilities and settings without a feasible
fixed-action comparator before starting simulations. For example, `H = 15`
is infeasible at `d = 5, b/H = 0.6`; use a smaller horizon or explicitly choose
a lower `--budget-ratio` for the whole study.
