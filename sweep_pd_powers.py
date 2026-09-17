#!/usr/bin/env python3
"""PD-POWERS sensitivity with the original S=H+2 and A=2**(d-1) couplings."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import cmdp_primal_dual_power as experiment


@contextmanager
def configured_problem(horizon, dimension, episodes, budget_ratio):
    """Set every coupled environment quantity, restoring the original on exit."""
    settings = dict(H=horizon, STATE=horizon + 2, dim=dimension,
                    ACTION=2 ** (dimension - 1), K=episodes,
                    thetastar=np.r_[np.full(dimension - 1, .01), 1.],
                    B_CONSTR=budget_ratio * horizon)
    original = {name: getattr(experiment, name) for name in settings}
    try:
        for name, value in settings.items():
            setattr(experiment, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(experiment, name, value)


def setting_id(horizon, dimension):
    return f"h{horizon}_d{dimension}"


def validate_setting(horizon, dimension, episodes, budget_ratio):
    """Reject invalid linear probabilities and infeasible comparison instances."""
    if horizon < 1 or dimension < 2 or episodes < 1:
        raise ValueError("Horizons/episodes must be positive and dimensions at least 2")
    if not np.isfinite(budget_ratio) or not 0 < budget_ratio <= 1:
        raise ValueError("The utility-threshold ratio must be in (0, 1]")
    # Do not silently rely on proba() clipping: that changes the linear model.
    radius = .01 * (dimension - 1)
    if radius > min(experiment.delta, 1 - experiment.delta) + 1e-12:
        raise ValueError(f"d={dimension} gives invalid transition probabilities with "
                         f"delta={experiment.delta:g} and theta_i=0.01; use d=2,...,6")
    with configured_problem(horizon, dimension, episodes, budget_ratio):
        maximum = max(experiment.expected_episode_values_for_action(a, 0)[1]
                      for a in range(experiment.ACTION))
        if maximum < experiment.B_CONSTR:
            raise ValueError(f"H={horizon}, d={dimension}: no feasible fixed-action comparator "
                             f"(maximum utility {maximum:.4g} < b={experiment.B_CONSTR:g}). "
                             "Use a smaller horizon or utility-threshold ratio.")


def run_setting(horizon, dimension, episodes, budget_ratio, seeds):
    """Reuse the unmodified variance-weighted learner and its existing metrics."""
    with configured_problem(horizon, dimension, episodes, budget_ratio):
        action = experiment.select_constrained_optimal_action()
        benchmark = np.asarray([experiment.expected_episode_values_for_action(action, k)[0]
                                for k in range(episodes)])
        # Keep worker output together; the parent reports completed settings.
        with redirect_stdout(StringIO()):
            summary, histories = experiment.evaluate_method(
                True, experiment.beta, seeds, benchmark)
        settings = {name: getattr(experiment, name) for name in (
            "H", "STATE", "ACTION", "dim", "K", "B_CONSTR", "delta", "LAMBDA",
            "beta", "beta1", "dual_lr", "theta", "alpha", "TERMINAL_REWARD",
            "SHAPING_WEIGHT", "REWARD_MODE", "REWARD_BLEND_START", "REWARD_BLEND_END",
            "REWARD_PHASE_LEN", "MAX_REWARD_PER_STEP")}
        settings["thetastar"] = experiment.thetastar.tolist()
        summary.update(settings=settings,
                       benchmark={"type": "best feasible deterministic fixed action",
                                  "action": action})
        return summary, dict(benchmark_reward=benchmark, **histories)


def plot_sensitivity(summary, curves, output_dir):
    episodes = np.arange(1, summary["episodes"] + 1)
    panels = summary["panels"]
    for metric, ylabel in (("regret", "Regret"), ("deficit", "Constraint violation")):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for ax, panel in zip(axes, panels):
            for key in panel["settings"]:
                settings = summary["settings"][key]["settings"]
                label = (f"H = {settings['H']}, S = {settings['STATE']}"
                         if panel["axis"] == "horizon"
                         else f"A = {settings['ACTION']}, d = {settings['dim']}")
                values = curves[f"{key}_{metric}"]
                if metric == "deficit":
                    # Positive part per run, BEFORE averaging across seeds.
                    values = np.maximum(0, values)
                mean = values.mean(axis=0)
                line, = ax.plot(episodes, mean, label=label, linewidth=1.8)
                if len(values) > 1:
                    half_width = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(len(values))
                    ax.fill_between(episodes, mean - half_width, mean + half_width,
                                    color=line.get_color(), alpha=.18, linewidth=0)
            ax.set(title=panel["title"], xlabel="Episode", ylabel=ylabel)
            ax.legend(frameon=False, fontsize=10)
            ax.grid(alpha=.18)
            ax.set_xlim(1, max(2, summary["episodes"]))
            if metric == "deficit":
                ax.set_ylim(bottom=0)
        fig.suptitle("PD-POWERS sensitivity", fontsize=15)
        band_note = ("Approximate pointwise 95% confidence bands"
                     if len(summary["evaluation_seeds"]) > 1 else "One seed; no confidence band")
        fig.text(.5, .025,
                 f"{len(summary['evaluation_seeds'])} seeds · {band_note} · "
                 f"Utility threshold b = {summary['budget_ratio']:g}H",
                 ha="center", fontsize=9, color=".35")
        fig.tight_layout(rect=(0, .065, 1, .95))
        filename = "sensitivity_regret.jpg" if metric == "regret" else "sensitivity_violation.jpg"
        try:
            fig.savefig(output_dir / filename, dpi=180)
        finally:
            plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 8, 10],
                        help="H values at fixed d; S=H+2 (default: 5 8 10)")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[3, 4, 5],
                        help="d values at fixed H; A=2**(d-1) (default: 3 4 5)")
    parser.add_argument("--reference-horizon", type=int, default=experiment.H,
                        help="Fixed H for the A/d sweep (default: 10)")
    parser.add_argument("--reference-dimension", type=int, default=experiment.dim,
                        help="Fixed d for the H/S sweep (default: 5)")
    parser.add_argument("--budget-ratio", type=float,
                        default=experiment.B_CONSTR / experiment.H,
                        help="Utility lower bound b/H, held fixed (default: 0.6)")
    parser.add_argument("--episodes", type=int, default=experiment.K)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(experiment.BASE_SEED,
                                           experiment.BASE_SEED + experiment.repeat)))
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent settings to run concurrently (default: 1)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results/pd_powers_sensitivity",
                        help="Raw data directory (default: results/pd_powers_sensitivity)")
    parser.add_argument("--plot-dir", type=Path, default=experiment.PLOT_DIR,
                        help="Shared plot directory (default: the project's plots folder)")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("Workers must be positive")
    if len(set(args.seeds)) != len(args.seeds) or not all(0 <= s < 2**32 for s in args.seeds):
        parser.error("Provide unique seeds in [0, 2**32)")
    if len(set(args.horizons)) != len(args.horizons) or len(set(args.dimensions)) != len(args.dimensions):
        parser.error("Horizon and dimension lists must each contain unique values")
    horizon_settings = [(h, args.reference_dimension) for h in args.horizons]
    action_settings = [(args.reference_horizon, d) for d in args.dimensions]
    # The shared reference is evaluated once and appears in both panels.
    configurations = list(dict.fromkeys(horizon_settings + action_settings))
    try:
        for horizon, dimension in configurations:
            validate_setting(horizon, dimension, args.episodes, args.budget_ratio)
    except ValueError as error:
        parser.error(str(error))
    output_dir = args.output_dir.expanduser().resolve()
    plot_dir = args.plot_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "purpose": "PD-POWERS sensitivity under coupled environment sizes",
        "method": "pd_powers", "episodes": args.episodes,
        "evaluation_seeds": args.seeds, "budget_ratio": args.budget_ratio,
        "couplings": {"S": "H + 2", "A": "2 ** (d - 1)"},
        "interpretation": "Joint H/S and A/d sensitivity, not independent H, S, A effects. "
                          "Hyperparameters are held fixed; b scales with H. "
                          "Each setting has its own feasible fixed-action comparator.",
        "metric": "Exact expected deployed-policy values; regret against the best feasible "
                  "deterministic fixed action (may be negative); violation is the positive "
                  "part of cumulative utility deficit, taken separately per seed.",
        "uncertainty": "Mean +/- 1.96 times sample standard error across seeds, pointwise",
        "panels": [
            {"axis": "horizon", "title": f"Vary H and S (A = {2 ** (args.reference_dimension - 1)}, "
                                         f"d = {args.reference_dimension})",
             "settings": [setting_id(*pair) for pair in horizon_settings]},
            {"axis": "action", "title": f"Vary A and d (H = {args.reference_horizon}, "
                                        f"S = {args.reference_horizon + 2})",
             "settings": [setting_id(*pair) for pair in action_settings]},
        ],
        "settings": {},
    }
    curves = {}

    def collect(pair, result):
        key = setting_id(*pair)
        row, histories = result
        summary["settings"][key] = row
        curves.update({f"{key}_{name}": value for name, value in histories.items()})
        print(f"Finished H={pair[0]}, S={pair[0] + 2}, A={2 ** (pair[1] - 1)}, d={pair[1]}: "
              f"regret={row['mean_regret']:.2f}, violation={row['mean_violation']:.2f}", flush=True)

    print(f"Running {len(configurations)} PD-POWERS settings, {len(args.seeds)} seeds each, "
          f"{args.episodes} episodes.\nResults directory: {output_dir}\n"
          f"Plot directory: {plot_dir}", flush=True)
    if args.workers == 1:
        for pair in configurations:
            collect(pair, run_setting(*pair, args.episodes, args.budget_ratio, args.seeds))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(configurations))) as executor:
            futures = {executor.submit(run_setting, *pair, args.episodes,
                                       args.budget_ratio, args.seeds): pair
                       for pair in configurations}
            for future in as_completed(futures):
                collect(futures[future], future.result())
    # Stable serialization regardless of worker completion order.
    summary["settings"] = {setting_id(*pair): summary["settings"][setting_id(*pair)]
                           for pair in configurations}
    experiment.save_json(output_dir / "sensitivity.json", summary)
    np.savez_compressed(output_dir / "curves.npz", **curves)
    plot_sensitivity(summary, curves, plot_dir)
    print(f"Saved raw per-seed curves to {output_dir}\n"
          f"Saved sensitivity plots to {plot_dir}", flush=True)


if __name__ == "__main__":
    main()
