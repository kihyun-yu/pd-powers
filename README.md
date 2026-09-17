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
