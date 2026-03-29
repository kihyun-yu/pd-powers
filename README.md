# PD-POWERS

Implementation of **PD-POWERS** — a primal-dual policy optimization algorithm for linear mixture CMDPs with adversarial rewards and a fixed constraint.


---

## Overview

This repository provides:

- Finite-horizon CMDP with linear transition features  
- Support for non-stationary rewards  
- Methods:
  - Random baseline  
  - PD-POWERS (primal–dual algorithm)

---

## Reference

Based on:
- POWERS (AISTATS 2022): https://proceedings.mlr.press/v151/he22a.html

---

## Features

### Environment
- Finite horizon
- Linear transition model
- Constraint: `avg_utility >= B_CONSTR`

### Reward Modes (`REWARD_MODE`)
- `stationary`
- `gradual`
- `phase-flip`

### Outputs
- Logs saved in `logs/`
- Plots:
  - `regret_plot.jpg`
  - `violation_plot.jpg`

---

## Requirements

- Python 3.9+
- numpy
- matplotlib
- tqdm (optional)

```bash
pip install numpy matplotlib tqdm