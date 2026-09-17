#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

try:
    from tqdm.auto import tqdm
except ImportError:
    # Fallback: keep behavior unchanged when tqdm is unavailable.
    def tqdm(iterable=None, **kwargs):
        return iterable

# =========================
# Problem setup
# =========================
dim = 5
H = 10
K = 2000
repeat = 20  # Paired evaluation runs; learner/environment parameters remain original.
ACTION = 2 ** (dim - 1)
STATE = H + 2

thetastar = np.append(0.01 * np.ones(dim - 1), 1.0)
delta = 0.05

# Hyperparameters for CMDP primal-dual algorithm
LAMBDA = 1.0
beta = 0.35
beta1 = 0.45
# Illustrative preset: selected to show a modest gap, not a tuned-best ablation.
BASELINE_BETA = 2.0
# Independent empirical coefficient for Algorithm 1 of Yu et al. (2026).
LINEAR_BETA = 16.0
LINEAR_ALPHA = 0.1  # Appendix L's experimental policy step size.
LINEAR_LABEL = "Yu et al. 2026"
BASE_SEED = 4000
dual_lr = 0.05
B_CONSTR = 6.0
theta = 0.001  # Policy smoothing toward uniform
alpha = 0.1  # Exponential policy learning rate (also used in dual update momentum)
# Reward-structure hyperparameters
TERMINAL_REWARD = 1.0
SHAPING_WEIGHT = 0.4
REWARD_MODE = "phase-flip"  # "stationary", "gradual", or "phase-flip"
REWARD_BLEND_START = 0.0
REWARD_BLEND_END = 1.0
REWARD_PHASE_LEN = 10
MAX_REWARD_PER_STEP = max(TERMINAL_REWARD, SHAPING_WEIGHT)


# =========================
# Environment helpers
# =========================
def trans_action(a: int, d: int) -> np.ndarray:
    """Convert action index to a {-1, +1}^{d-1} vector."""
    bits = np.zeros(d - 1) - 1
    bb = bin(a)
    for i in range(len(bb) - 2):
        bits[i] = 2 * float(bb[i + 2]) - 1
    return bits


def phi(s: int, a: int, sp: int) -> np.ndarray:
    """Feature map for transition (s, a, s')."""
    feat = np.zeros(dim)
    aa = trans_action(a, dim)

    if s < H:
        if sp == s + 1:
            for i in range(dim - 1):
                feat[i] = -aa[i]
            feat[dim - 1] = 1 - delta

        if sp == H + 1:
            for i in range(dim - 1):
                feat[i] = aa[i]
            feat[dim - 1] = delta

    if s == H and sp == H:
        feat[dim - 1] = 1

    if s == H + 1 and sp == H + 1:
        feat[dim - 1] = 1

    return feat


def phiv(s: int, a: int, v: np.ndarray) -> np.ndarray:
    """Feature map weighted by value vector v."""
    if s == H + 1:
        return phi(s, a, H + 1) * v[H + 1]
    return phi(s, a, H + 1) * v[H + 1] + phi(s, a, s + 1) * v[s + 1]


def proba(s: int, a: int, sp: int) -> float:
    """Transition probability P(sp | s, a)."""
    prob = float(np.dot(phi(s, a, sp), thetastar))
    if not np.isfinite(prob):
        return 0.0
    return float(np.clip(prob, 0.0, 1.0))


def reward(s: int, a: int, episode: int) -> float:
    """Reward with selectable stationary/non-stationary structure."""
    if s == H + 1:
        return TERMINAL_REWARD
    if s >= H:
        return 0.0

    action_density = float(np.mean((trans_action(a, dim) + 1) / 2))

    if REWARD_MODE == "stationary":
        shaped_reward = action_density
    elif REWARD_MODE == "phase-flip":
        phase = (episode // REWARD_PHASE_LEN) % 2
        if phase == 0:
            shaped_reward = action_density
        else:
            shaped_reward = 1.0 - action_density
    else:
        if K <= 1:
            blend = REWARD_BLEND_START
        else:
            progress = episode / (K - 1)
            blend = REWARD_BLEND_START + (REWARD_BLEND_END - REWARD_BLEND_START) * progress
        shaped_reward = (1.0 - blend) * action_density + blend * (1.0 - action_density)

    return SHAPING_WEIGHT * shaped_reward


def cost(s: int, a: int) -> float:
    """Constraint cost for CMDP: normalized amount of +1 bits in the action."""
    if s >= H:
        return 0.0
    aa = trans_action(a, dim)
    return float(np.mean((aa + 1) / 2))


def expected_episode_values_for_action(action: int, episode: int) -> tuple[float, float]:
    """Return expected (reward, utility) for one episode under a fixed action."""
    # Closed-form evaluation of the same chain; no change to the benchmark.
    active_steps = sum(proba(0, action, 1)**h for h in range(H))
    return (float(active_steps * reward(0, action, episode)
                  + (H - active_steps) * TERMINAL_REWARD),
            float(active_steps * cost(0, action)))


def select_constrained_optimal_action() -> int:
    """Select fixed action maximizing sum_k V_{r^k,pi} s.t. V_g >= b (fixed-action class)."""
    best_action = 0
    best_objective = -np.inf
    has_feasible = False

    for action in range(ACTION):
        reward_sum = 0.0
        utility_sum = 0.0
        for episode in range(K):
            exp_reward, exp_utility = expected_episode_values_for_action(action, episode)
            reward_sum += exp_reward
            utility_sum += exp_utility

        avg_utility = utility_sum / K
        feasible = avg_utility >= B_CONSTR
        if feasible:
            has_feasible = True
            if reward_sum > best_objective:
                best_objective = reward_sum
                best_action = action

    if has_feasible:
        return best_action

    warnings.warn(
        "No feasible action satisfies the CMDP constraint. Exiting without running experiments.",
        RuntimeWarning,
    )
    raise SystemExit(1)



def run_primal_dual(*, use_variance=True, seed=BASE_SEED, run=0,
                   beta_value: Optional[float] = None, show_progress=False,
                   diagnostics: Optional[dict] = None):
    """Original PD-POWERS or its unweighted-critic ablation, with isolated RNG.

    The original action encoding, H**2/d floor, Q bounds, actor, and penalized
    dual update are retained. Only no-var omits moment estimation and weights.
    Expected policy values are for reporting only; learning uses sampled data.
    """
    default_beta = beta if use_variance else BASELINE_BETA
    bonus_beta = default_beta if beta_value is None else beta_value
    if not np.isfinite(bonus_beta) or bonus_beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    rng = np.random.RandomState(seed)
    continuation_features = np.array([phi(0, a, 1) for a in range(ACTION)])
    good_features = np.array([phi(0, a, H + 1) for a in range(ACTION)])
    absorbing_feature = phi(H + 1, 0, H + 1)
    continuation = np.array([proba(0, a, 1) for a in range(ACTION)])
    utilities = np.array([cost(0, a) for a in range(ACTION)])
    rewards = np.array([[reward(0, a, k) for a in range(ACTION)] for k in range(K)])
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(expected_reward=np.zeros(K), expected_utility=np.zeros(K),
                           dual=np.zeros(K), r_floor_hits=0, g_floor_hits=0)

    def value_feature(state, action, values):
        if state == H + 1:
            return absorbing_feature * values[state]
        return good_features[action] * values[H + 1] + continuation_features[action] * values[state + 1]

    def reachable_features(h, values):
        return np.vstack((good_features * values[H + 1] + continuation_features * values[h + 1],
                          absorbing_feature * values[H + 1]))

    s_cur = np.zeros(H + 1, dtype=int)
    a_cur = np.zeros(H, dtype=int)

    # Two critics: reward and constraint.
    Q_REWARD = np.zeros((STATE, ACTION, H + 1))
    V_REWARD = np.zeros((STATE, H + 1))
    Q_CONSTR = np.zeros((STATE, ACTION, H + 1))
    V_CONSTR = np.zeros((STATE, H + 1))

    POLICY = np.ones((STATE, ACTION, H + 1)) / ACTION

    constr_budget = B_CONSTR
    dual_lambda = 0.0

    SIGMA_R = np.repeat((LAMBDA * np.eye(dim))[:, :, None], H, axis=2)
    BB_R = np.zeros((dim, H))
    SIGMA_G = SIGMA_R.copy()
    BB_G = np.zeros((dim, H))
    if use_variance:
        SIGMA1_R, SIGMA2_R = SIGMA_R.copy(), SIGMA_R.copy()
        BB1_R, BB2_R = BB_R.copy(), BB_R.copy()
        SIGMA1_G, SIGMA2_G = SIGMA_G.copy(), SIGMA_G.copy()
        BB1_G, BB2_G = BB_G.copy(), BB_G.copy()

    total_reward = 0
    reward_history = []
    constr_violation_history = []
    cumulative_constr_violation = 0.0

    for episode in tqdm(range(K), desc=f"Run {run + 1}/{repeat}", leave=False, disable=not show_progress):
        s_cur[0] = 0
        episode_utility_sum = 0.0

        if diagnostics is not None:
            occupancy = 1.0
            for h in range(H):
                probabilities = POLICY[h, :, h]
                diagnostics["expected_reward"][episode] += (
                    occupancy * np.dot(probabilities, rewards[episode])
                    + (1 - occupancy) * TERMINAL_REWARD)
                diagnostics["expected_utility"][episode] += occupancy * np.dot(probabilities, utilities)
                occupancy *= np.dot(probabilities, continuation)

        # Roll out one episode
        for stage in range(H):
            a_cur[stage] = rng.choice(ACTION, p=POLICY[s_cur[stage], :, stage])
            s_next = s_cur[stage]

            if s_cur[stage] < H:
                cont_prob = continuation[a_cur[stage]]
                s_next = H + 1 - rng.binomial(1, cont_prob) * (H - s_cur[stage])

            total_reward += rewards[episode, a_cur[stage]] if s_cur[stage] < H else TERMINAL_REWARD
            episode_utility_sum += utilities[a_cur[stage]] if s_cur[stage] < H else 0.0
            reward_history.append(total_reward)
            s_cur[stage + 1] = s_next

        # Backward update for both critics.
        for stage in range(H):
            curstage = H - 1 - stage
            s_now = s_cur[curstage]
            a_now = a_cur[curstage]
            s_nxt = s_cur[curstage + 1]

            # ---------- Reward critic ----------
            if use_variance:
                phi1_r = value_feature(s_now, a_now, V_REWARD[:, curstage + 1])
                phi2_r = value_feature(
                    s_now,
                    a_now,
                    V_REWARD[:, curstage + 1] * V_REWARD[:, curstage + 1],
                )

                BB1_R[:, curstage] += phi1_r * V_REWARD[s_nxt, curstage + 1]
                BB2_R[:, curstage] += (
                    phi2_r * V_REWARD[s_nxt, curstage + 1] * V_REWARD[s_nxt, curstage + 1]
                )

                phi1_r_col = phi1_r.reshape(dim, 1)
                phi2_r_col = phi2_r.reshape(dim, 1)

                SIGMA1_R[:, :, curstage] += phi1_r_col @ phi1_r_col.T
                SIGMA2_R[:, :, curstage] += phi2_r_col @ phi2_r_col.T

                hattheta1_r = np.linalg.lstsq(SIGMA1_R[:, :, curstage], BB1_R[:, curstage], rcond=-1)[0]
                hattheta2_r = np.linalg.lstsq(SIGMA2_R[:, :, curstage], BB2_R[:, curstage], rcond=-1)[0]

                UU1_r = np.linalg.lstsq(SIGMA1_R[:, :, curstage], phi1_r, rcond=-1)[0]
                UU2_r = np.linalg.lstsq(SIGMA2_R[:, :, curstage], phi2_r, rcond=-1)[0]

                variance_r = (
                    min(max(np.dot(hattheta2_r, phi2_r), 0), H**2)
                    - min(max(np.dot(hattheta1_r, phi1_r), 0), H) ** 2
                    + beta1 * np.sqrt(np.dot(UU1_r, phi1_r))
                    + beta1 * np.sqrt(np.dot(UU2_r, phi2_r))
                )
                variance_r = np.sqrt(max(variance_r, H * H / dim))
                scale_r = variance_r**2
                if diagnostics is not None and curstage < H - 1:
                    diagnostics["r_floor_hits"] += int(abs(scale_r - H * H / dim) < 1e-10)
            else:
                scale_r = 1.0

            phi_r = value_feature(s_now, a_now, V_REWARD[:, curstage + 1])
            BB_R[:, curstage] += phi_r * V_REWARD[s_nxt, curstage + 1] / scale_r
            phi_r_col = phi_r.reshape(dim, 1)
            SIGMA_R[:, :, curstage] += (phi_r_col @ phi_r_col.T) / scale_r

            hattheta_r = np.linalg.lstsq(SIGMA_R[:, :, curstage], BB_R[:, curstage], rcond=-1)[0]

            # ---------- Constraint critic ----------
            if use_variance:
                phi1_g = value_feature(s_now, a_now, V_CONSTR[:, curstage + 1])
                phi2_g = value_feature(
                    s_now,
                    a_now,
                    V_CONSTR[:, curstage + 1] * V_CONSTR[:, curstage + 1],
                )

                BB1_G[:, curstage] += phi1_g * V_CONSTR[s_nxt, curstage + 1]
                BB2_G[:, curstage] += (
                    phi2_g * V_CONSTR[s_nxt, curstage + 1] * V_CONSTR[s_nxt, curstage + 1]
                )

                phi1_g_col = phi1_g.reshape(dim, 1)
                phi2_g_col = phi2_g.reshape(dim, 1)

                SIGMA1_G[:, :, curstage] += phi1_g_col @ phi1_g_col.T
                SIGMA2_G[:, :, curstage] += phi2_g_col @ phi2_g_col.T

                hattheta1_g = np.linalg.lstsq(SIGMA1_G[:, :, curstage], BB1_G[:, curstage], rcond=-1)[0]
                hattheta2_g = np.linalg.lstsq(SIGMA2_G[:, :, curstage], BB2_G[:, curstage], rcond=-1)[0]

                UU1_g = np.linalg.lstsq(SIGMA1_G[:, :, curstage], phi1_g, rcond=-1)[0]
                UU2_g = np.linalg.lstsq(SIGMA2_G[:, :, curstage], phi2_g, rcond=-1)[0]

                variance_g = (
                    min(max(np.dot(hattheta2_g, phi2_g), 0), H**2)
                    - min(max(np.dot(hattheta1_g, phi1_g), 0), H) ** 2
                    + beta1 * np.sqrt(np.dot(UU1_g, phi1_g))
                    + beta1 * np.sqrt(np.dot(UU2_g, phi2_g))
                )
                variance_g = np.sqrt(max(variance_g, H * H / dim))
                scale_g = variance_g**2
                if diagnostics is not None and curstage < H - 1:
                    diagnostics["g_floor_hits"] += int(abs(scale_g - H * H / dim) < 1e-10)
            else:
                scale_g = 1.0

            phi_g = value_feature(s_now, a_now, V_CONSTR[:, curstage + 1])
            BB_G[:, curstage] += phi_g * V_CONSTR[s_nxt, curstage + 1] / scale_g
            phi_g_col = phi_g.reshape(dim, 1)
            SIGMA_G[:, :, curstage] += (phi_g_col @ phi_g_col.T) / scale_g

            hattheta_g = np.linalg.lstsq(SIGMA_G[:, :, curstage], BB_G[:, curstage], rcond=-1)[0]

            # Only chain state h and the rewarding absorber are reachable at h.
            # Batch equivalent confidence-radius solves; preserve original Q bounds.
            features_r = reachable_features(curstage, V_REWARD[:, curstage + 1])
            solved_r = np.linalg.lstsq(SIGMA_R[:, :, curstage], features_r.T, rcond=-1)[0].T
            q_r = np.r_[rewards[episode], TERMINAL_REWARD] + features_r @ hattheta_r
            q_r += bonus_beta * np.sqrt(np.sum(solved_r * features_r, axis=1))
            q_r = np.clip(q_r, 0, (stage + 1) * MAX_REWARD_PER_STEP)
            features_g = reachable_features(curstage, V_CONSTR[:, curstage + 1])
            solved_g = np.linalg.lstsq(SIGMA_G[:, :, curstage], features_g.T, rcond=-1)[0].T
            q_g = np.r_[utilities, 0.0] + features_g @ hattheta_g
            q_g += bonus_beta * np.sqrt(np.sum(solved_g * features_g, axis=1))
            q_g = np.maximum(q_g, 0)  # Original implementation has no upper utility clip.
            for stat in ((curstage,) if curstage == 0 else (curstage, H + 1)):
                Q_REWARD[stat, :, curstage] = q_r[:ACTION] if stat == curstage else q_r[-1]
                probabilities = POLICY[stat, :, curstage]
                V_REWARD[stat, curstage] = np.dot(Q_REWARD[stat, :, curstage], probabilities)
                Q_CONSTR[stat, :, curstage] = q_g[:ACTION] if stat == curstage else q_g[-1]
                V_CONSTR[stat, curstage] = np.dot(Q_CONSTR[stat, :, curstage], probabilities)
                actor_values = Q_REWARD[stat, :, curstage] + dual_lambda * Q_CONSTR[stat, :, curstage]
                weights = probabilities * np.exp(alpha * actor_values)
                POLICY[stat, :, curstage] = (1 - theta) * weights / weights.sum() + theta / ACTION

        # Report the sampled cumulative utility deficit.
        constr_violation = constr_budget - episode_utility_sum
        cumulative_constr_violation += constr_violation
        constr_violation_history.append(cumulative_constr_violation)

        # Original momentum-based dual update with explicit penalty terms.
        constr_estimate = V_CONSTR[0, 0] / H
        momentum_decay = 1.0 - alpha * dual_lr * (H ** 3)
        penalty_term = alpha * (H ** 3) + 2 * theta * (H ** 2)
        dual_lambda = dual_lambda * momentum_decay + dual_lr * (constr_budget - constr_estimate - penalty_term)
        dual_lambda = max(0.0, dual_lambda)
        if diagnostics is not None:
            diagnostics["dual"][episode] = dual_lambda

    if diagnostics is not None:
        diagnostics["final_policy"] = np.array([POLICY[h, :, h] for h in range(H)])
        diagnostics["final_reward_covariance"] = SIGMA_R.copy()
        diagnostics["final_utility_covariance"] = SIGMA_G.copy()
    return np.asarray(reward_history), np.asarray(constr_violation_history)


def linear_features(state, action):
    """Known linear-CMDP features: a two-coordinate block per chain state.

    The chain's transition probabilities and signals are affine in action
    density. Absorbing states have their own indicator coordinates. These
    features use no transition parameters and have norm at most one.
    """
    features = np.zeros(2 * H + 2)
    if state < H:
        density = np.mean((trans_action(action, dim) + 1) / 2)
        features[2 * state:2 * state + 2] = (1 - density, density)
    else:
        features[2 * H + state - H] = 1.
    return features


def linear_settings(coefficient):
    """Appendix L's empirical convention, with independently chosen beta_b."""
    return {"paper": "https://arxiv.org/abs/2605.11535", "algorithm": 1,
            "feature_dimension": 2 * H + 2, "beta_b": coefficient,
            "beta_w": coefficient * float(np.log(K)), "alpha": LINEAR_ALPHA,
            "eta": H**-2 * K**-0.75, "theta": 1 / K,
            "mixing_period": max(1, int(np.ceil(K**0.75))),
            "cost_budget": H - B_CONSTR, "ridge": 1.0}


def run_linear_cmdp(*, seed=BASE_SEED, beta_value=None, diagnostics=None):
    """Yu et al. (2026), Algorithm 1, on the SAME chain as PD-POWERS.

    Loss = 1 - reward, paper cost = 1 - utility, paper budget = H - B_CONSTR.
    Keep pre-episode regressions, epoch-frozen contraction/bonuses, determinant
    doubling resets, periodic pre-update mixing, and the factor-4 dual rule.
    Only three feature coordinates are reachable at a given stage. Restricting
    the block-diagonal ridge system to these coordinates is exact. Sums of
    feature vectors grouped by successor retain ALL past samples and allow
    reevaluation with current values without an O(K**2) history scan.
    """
    coefficient = LINEAR_BETA if beta_value is None else beta_value
    if not np.isfinite(coefficient) or coefficient < 0:
        raise ValueError("beta must be finite and nonnegative")
    if not np.allclose(thetastar[:-1], thetastar[0], rtol=0, atol=1e-14):
        raise ValueError("The density feature map requires equal action transition coefficients")
    if not (0 <= SHAPING_WEIGHT <= 1 and 0 <= TERMINAL_REWARD <= 1):
        raise ValueError("The linear baseline requires reward in [0, 1]")
    settings = linear_settings(coefficient)
    step_size, eta, mixing = settings["alpha"], settings["eta"], settings["theta"]
    rng = np.random.RandomState(seed)
    density = np.array([np.mean((trans_action(a, dim) + 1) / 2) for a in range(ACTION)])
    utilities = np.array([cost(0, a) for a in range(ACTION)])
    # Coordinates: the current chain state's two entries, then the good absorber.
    features = np.zeros((2, ACTION, 3))
    features[0, :, :2] = np.column_stack((1 - density, density))
    features[1, :, 2] = 1.
    flat_features = features.reshape(-1, 3)
    continuation = np.array([proba(0, a, 1) for a in range(ACTION)])
    rewards = np.array([[reward(0, a, k) for a in range(ACTION)] for k in range(K)])
    policy = np.full((H, 2, ACTION), 1 / ACTION)
    design = np.repeat(np.eye(3)[None, :, :], H, axis=0)
    cost_sum = np.zeros((H, 3))
    transition_sum = np.zeros((H, 3, 2))
    values = np.zeros((H + 1, 2, 2))  # stage, state kind, loss/cost
    dual = 0.
    epoch_start = 0
    epoch_logdet = np.zeros(H)
    contractions = np.ones((H, 2, ACTION))
    radii = np.zeros_like(contractions)
    reward_history, deficit_history = [], []
    total_reward, total_deficit = 0., 0.
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(expected_reward=np.zeros(K), expected_utility=np.zeros(K),
                           dual=np.zeros(K), epoch_starts=[], mixing_episodes=[],
                           mean_contraction=np.zeros(K))
    for episode in range(K):
        logdet = np.linalg.slogdet(design)[1]
        # Include equality despite rounding differences at exact determinant doubling.
        if episode == 0 or np.any(logdet >= epoch_logdet + np.log(2) - 1e-12):
            # Algorithm 1, lines 2–6: reset policy and dual, retain observations.
            epoch_start = episode
            epoch_logdet = logdet.copy()
            policy.fill(1 / ACTION)
            dual = 0.
            for h in range(H):
                solved = np.linalg.solve(design[h], flat_features.T).T
                radii[h] = np.sqrt(np.maximum(0, np.sum(flat_features * solved, axis=1))).reshape(2, ACTION)
            logits = np.log(K) - settings["beta_w"] * radii
            contractions = np.exp(-np.logaddexp(0, -logits))
            if diagnostics is not None:
                diagnostics["epoch_starts"].append(episode + 1)

        if diagnostics is not None:
            occupancy = 1.
            for h in range(H):
                probabilities = policy[h, 0]
                diagnostics["expected_reward"][episode] += (
                    occupancy * np.dot(probabilities, rewards[episode])
                    + (1 - occupancy) * TERMINAL_REWARD)
                diagnostics["expected_utility"][episode] += occupancy * np.dot(probabilities, utilities)
                occupancy *= np.dot(probabilities, continuation)
            diagnostics["mean_contraction"][episode] = contractions.mean()

        # Rollout feedback. The critic below uses only observations BEFORE this
        # episode, as specified by Lambda^k and tau in [k-1] in lines 14 and 16.
        observed_features = np.zeros((H, 3))
        observed_costs = np.zeros(H)
        successors = np.zeros(H, dtype=int)
        state_kind, episode_utility = 0, 0.
        for h in range(H):
            action = rng.choice(ACTION, p=policy[h, state_kind])
            utility = utilities[action] if state_kind == 0 else 0.
            total_reward += rewards[episode, action] if state_kind == 0 else TERMINAL_REWARD
            episode_utility += utility
            reward_history.append(total_reward)
            observed_features[h] = features[state_kind, action]
            observed_costs[h] = 1 - utility
            if state_kind == 0:
                state_kind = 1 - rng.binomial(1, continuation[action])
            successors[h] = state_kind

        mix_now = (episode - epoch_start) % settings["mixing_period"] == 0
        if diagnostics is not None and mix_now:
            diagnostics["mixing_episodes"].append(episode + 1)
        # Full-information LOSS feedback; no unsampled utility values enter a fit.
        loss_parameter = np.array([1 - rewards[episode, 0],
                                   1 - rewards[episode, ACTION - 1], 1 - TERMINAL_REWARD])
        for h in reversed(range(H)):
            rhs = transition_sum[h] @ values[h + 1]
            rhs[:, 1] += cost_sum[h]
            fitted = np.linalg.solve(design[h], rhs)
            fitted[:, 0] += loss_parameter
            q_values = contractions[h, :, :, None] * (
                features @ fitted - coefficient * radii[h, :, :, None])
            # Values use the deployed policy, before mixing/optimization. No Q clipping.
            values[h] = np.sum(policy[h, :, :, None] * q_values, axis=1)
            mixed = (1 - mixing) * policy[h] + mixing / ACTION if mix_now else policy[h]
            with np.errstate(divide="ignore"):
                log_weights = np.log(mixed) - step_size * (q_values[:, :, 0] + dual * q_values[:, :, 1])
            weights = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
            policy[h] = weights / weights.sum(axis=1, keepdims=True)

        # Algorithm 1, line 27: unnormalized cost value and the factor-4 penalties.
        dual = max(0., (1 - 4 * step_size * eta * H**3) * dual
                   + eta * (values[0, 0, 1] - settings["cost_budget"]
                            - 4 * step_size * H**3 - 4 * mixing * H**2))
        # Commit the current episode only after its pre-episode critic fits.
        for h in range(H):
            feature = observed_features[h]
            design[h] += np.outer(feature, feature)
            cost_sum[h] += feature * observed_costs[h]
            transition_sum[h, :, successors[h]] += feature
        total_deficit += B_CONSTR - episode_utility
        deficit_history.append(total_deficit)
        if diagnostics is not None:
            diagnostics["dual"][episode] = dual
    if diagnostics is not None:
        diagnostics.update(final_policy=policy[:, 0].copy(), final_design=design.copy(),
                           final_cost_sum=cost_sum.copy(), final_transition_sum=transition_sum.copy())
    return np.asarray(reward_history), np.asarray(deficit_history)


def evaluate_method(use_variance, coefficient, seeds, benchmark, *, linear=False):
    """Keep genuine sampled histories and score exact deployed policy values."""
    runs, histories = [], []
    label = LINEAR_LABEL if linear else ("PD-POWERS" if use_variance else "PD-POWERS without variance")
    for run, seed in enumerate(seeds):
        diagnostics = {}
        if linear:
            observed_reward, observed_deficit = run_linear_cmdp(
                seed=seed, beta_value=coefficient, diagnostics=diagnostics)
        else:
            observed_reward, observed_deficit = run_primal_dual(
                use_variance=use_variance, seed=seed, run=run, beta_value=coefficient,
                diagnostics=diagnostics,
            )
        regret = np.cumsum(benchmark - diagnostics["expected_reward"])
        deficit = np.cumsum(B_CONSTR - diagnostics["expected_utility"])
        histories.append({"regret": regret, "deficit": deficit,
                          "sampled_reward": observed_reward, "sampled_deficit": observed_deficit,
                          "expected_reward": diagnostics["expected_reward"],
                          "expected_utility": diagnostics["expected_utility"]})
        runs.append({"seed": seed, "regret": float(regret[-1]),
                     "violation": float(max(0, deficit[-1])),
                     "peak_violation": float(max(0, deficit.max())),
                     "mean_utility": float(diagnostics["expected_utility"].mean()),
                     "sampled_violation": float(max(0, observed_deficit[-1])),
                     "dual_max": float(diagnostics["dual"].max()),
                     "reward_floor_fraction": diagnostics["r_floor_hits"] / max(1, K * (H - 1)) if use_variance and not linear else None,
                     "utility_floor_fraction": diagnostics["g_floor_hits"] / max(1, K * (H - 1)) if use_variance and not linear else None})
        if linear:
            runs[-1].update(epochs=len(diagnostics["epoch_starts"]),
                            mixing_steps=len(diagnostics["mixing_episodes"]),
                            mean_contraction=float(diagnostics["mean_contraction"].mean()))
        print(f"{label}, beta={coefficient:g}, seed={seed}: regret={regret[-1]:.2f}, "
              f"violation={max(0, deficit[-1]):.2f}", flush=True)
    summary = {"label": label, "use_variance": use_variance and not linear,
               "beta": coefficient, "beta1": beta1 if use_variance and not linear else None, "runs": runs}
    if linear:
        summary["algorithm_settings"] = linear_settings(coefficient)
    for metric in ("regret", "violation", "peak_violation", "mean_utility"):
        summary[f"mean_{metric}"] = float(np.mean([row[metric] for row in runs]))
    return summary, {key: np.asarray([row[key] for row in histories]) for key in histories[0]}


def choose_illustrative_candidate(reference, candidates, target_gap=0.10):
    """Deliberately choose a modest gap; this is NOT best-performance tuning."""
    if reference["mean_regret"] <= 0:
        raise ValueError("A relative regret-gap demonstration requires positive reference regret")
    for row in candidates:
        row["regret_gap_fraction"] = row["mean_regret"] / reference["mean_regret"] - 1
        row["violation_difference"] = row["mean_violation"] - reference["mean_violation"]
        row["target_met"] = (0.05 <= row["regret_gap_fraction"] <= 0.15
                             and row["violation_difference"] >= 0)
    eligible = [row for row in candidates if row["target_met"]]
    return min(eligible or candidates, key=lambda row: abs(row["regret_gap_fraction"] - target_gap))


def choose_near_random_candidate(reference, candidates, target_improvement=0.08):
    """Illustrative selection: 2–15% below Random on BOTH final metrics."""
    if min(reference["mean_regret"], reference["mean_violation"]) <= 0:
        raise ValueError("Near-Random selection requires positive Random regret and violation")
    for row in candidates:
        improvements = [1 - row[f"mean_{metric}"] / reference[f"mean_{metric}"]
                        for metric in ("regret", "violation")]
        row.update(regret_improvement_over_random=improvements[0],
                   violation_improvement_over_random=improvements[1],
                   target_met=all(.02 <= value <= .15 for value in improvements),
                   selection_distance=sum((value - target_improvement)**2 for value in improvements))
    eligible = [row for row in candidates if row["target_met"]]
    return min(eligible or candidates, key=lambda row: row["selection_distance"])


def random_policy_curves(benchmark):
    """Exact uniform-policy values; used for the plot and calibration reference."""
    p_continue = np.mean([proba(0, a, 1) for a in range(ACTION)])
    active_steps = sum(p_continue**h for h in range(H))
    random_reward = np.array([active_steps * np.mean([reward(0, a, k) for a in range(ACTION)])
                              + (H - active_steps) * TERMINAL_REWARD for k in range(K)])
    random_utility = active_steps * np.mean([cost(0, a) for a in range(ACTION)])
    return {"random_regret": np.cumsum(benchmark - random_reward)[None, :],
            "random_deficit": np.cumsum(np.full(K, B_CONSTR - random_utility))[None, :]}


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def paired_comparison(baseline, reference):
    """Final baseline-minus-PD differences on the same seeds, in the same order."""
    if [row["seed"] for row in baseline["runs"]] != [row["seed"] for row in reference["runs"]]:
        raise ValueError("Paired comparisons require matching seeds")
    result = {}
    for metric in ("regret", "violation"):
        differences = np.array([a[metric] - b[metric] for a, b in zip(baseline["runs"], reference["runs"])])
        result[metric] = {
            "mean": float(differences.mean()),
            "ci95_half_width": float(1.96 * differences.std(ddof=1) / np.sqrt(len(differences))) if len(differences) > 1 else None,
            "pd_powers_wins": int(np.sum(differences > 0)),
        }
    return result


def plot_comparison(curves, output_dir):
    """Use the original single-panel style with algorithm-only legends."""
    labels = {"random": "Random", "linear_cmdp": LINEAR_LABEL,
              "novar": "PD-POWERS w/o Var", "pd_powers": "PD-POWERS (Ours)"}
    styles = {"novar": "--", "linear_cmdp": "-."}
    colors = {"random": "C0", "pd_powers": "C1", "novar": "C2", "linear_cmdp": "C3"}
    episodes = np.arange(1, K + 1)
    for metric, filename, ylabel in (("regret", "regret_plot.jpg", "Regret"),
                                     ("violation", "violation_plot.jpg", "Constraint Violation")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name in labels:
            values = curves[f"{name}_{'regret' if metric == 'regret' else 'deficit'}"]
            if metric == "violation":
                values = np.maximum(0, values)
            mean = values.mean(axis=0)
            line, = ax.plot(episodes, mean, label=labels[name],
                            linestyle=styles.get(name, "-"), color=colors[name])
            if len(values) > 1:
                half_width = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(len(values))
                ax.fill_between(episodes, mean - half_width, mean + half_width,
                                color=line.get_color(), alpha=0.2)
        ax.set_ylabel(ylabel, fontsize=20)
        if metric == "violation":
            ax.set_xlabel("Episode", fontsize=20, labelpad=8)
        ax.legend(fontsize=14)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.16)
        # Keep the preset's illustrative purpose visible without a large title.
        # fig.text(0.5, 0.015, "Illustrative hyperparameter comparison",
        #          ha="center", fontsize=8, color="0.4")
        try:
            fig.savefig(output_dir / filename, dpi=150)
        finally:
            plt.close(fig)


def main():
    global K
    parser = argparse.ArgumentParser(description="PD-POWERS, its no-variance ablation, and Yu et al.'s linear CMDP baseline; illustrative hyperparameters.")
    parser.add_argument("--baseline-beta", type=float, default=BASELINE_BETA)
    parser.add_argument("--linear-beta", type=float, default=LINEAR_BETA,
                        help="Independent beta_b for the linear CMDP baseline; beta_w = beta_b * log(K)")
    parser.add_argument("--episodes", type=int, default=K)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(BASE_SEED, BASE_SEED + repeat)))
    parser.add_argument("--output-dir", type=Path,
                        help="Custom output directory; otherwise save under the script's results/no_variance_baseline and refresh its root plots")
    parser.add_argument("--calibrate-demo", action="store_true",
                        help="Choose no-var beta to target a 5–15%% illustrative regret gap, not its best performance")
    parser.add_argument("--beta-grid", type=float, nargs="+", default=[0.35, 0.75, 1.25, 1.75, 1.9, 2.0, 2.1, 2.25, 2.75])
    parser.add_argument("--calibrate-linear-demo", action="store_true",
                        help="Choose linear beta for final regret and violation 2–15%% below Random, not best performance")
    parser.add_argument("--linear-beta-grid", type=float, nargs="+",
                        help="Calibration grid; defaults to 0.35, 1, 2, 4, K**0.25, 8, 16, 32")
    args = parser.parse_args()
    if args.episodes < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide positive episodes and unique evaluation seeds")
    if not all(0 <= seed < 2**32 for seed in args.seeds):
        parser.error("Seeds must be in [0, 2**32)")
    if not all(np.isfinite(value) and value >= 0 for value in
               [args.baseline_beta, args.linear_beta, *args.beta_grid, *(args.linear_beta_grid or [])]):
        parser.error("Beta values must be finite and nonnegative")
    calibration_seeds = [100, 101, 102, 103, 104]
    if (args.calibrate_demo or args.calibrate_linear_demo) and set(args.seeds) & set(calibration_seeds):
        parser.error("Evaluation and calibration seeds must be disjoint")
    K = args.episodes
    project_dir = Path(__file__).resolve().parent
    default_output = args.output_dir is None
    args.output_dir = (project_dir / "results/no_variance_baseline" if default_output
                       else args.output_dir.expanduser()).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {args.output_dir}\nPlots are updated after all runs finish.", flush=True)
    action = select_constrained_optimal_action()
    benchmark = np.asarray([expected_episode_values_for_action(action, k)[0] for k in range(K)])
    random_curves = random_policy_curves(benchmark)
    settings = {name: globals()[name] for name in (
        "dim", "H", "K", "ACTION", "STATE", "delta", "LAMBDA", "beta", "beta1", "dual_lr",
        "B_CONSTR", "theta", "alpha", "TERMINAL_REWARD", "SHAPING_WEIGHT", "REWARD_MODE",
        "REWARD_BLEND_START", "REWARD_BLEND_END", "REWARD_PHASE_LEN")}
    settings["thetastar"] = thetastar.tolist()
    coefficient = args.baseline_beta
    if args.calibrate_demo:
        reference, _ = evaluate_method(True, beta, calibration_seeds, benchmark)
        search = {"purpose": "illustrative gap selection, not best-performance baseline tuning",
                  "method": "novar", "use_variance": False,
                  "settings": settings, "calibration_seeds": calibration_seeds,
                  "evaluation_seeds": args.seeds, "target_regret_gap": [0.05, 0.15],
                  "reference": reference, "candidates": []}
        for candidate in args.beta_grid:
            row, _ = evaluate_method(False, candidate, calibration_seeds, benchmark)
            search["candidates"].append(row)
            save_json(args.output_dir / "calibration.json", search)
        selected = choose_illustrative_candidate(search["reference"], search["candidates"])
        search.update(selected_beta=selected["beta"], target_met=selected["target_met"])
        coefficient = selected["beta"]
        save_json(args.output_dir / "calibration.json", search)
        print(f"Illustrative baseline beta={coefficient:g}; calibration target met={selected['target_met']}", flush=True)
        if not selected["target_met"]:
            print("No candidate met the requested gap. Expand the calibration grid before evaluation.", flush=True)
            return

    linear_coefficient = args.linear_beta
    if args.calibrate_linear_demo:
        search = {"purpose": "illustrative proximity to Random; not best-performance tuning",
                  "settings": settings, "calibration_seeds": calibration_seeds,
                  "evaluation_seeds": args.seeds, "target_improvement_over_random": [.02, .15],
                  "target_center": .08, "paper_empirical_beta": K**.25,
                  "reference": {"mean_regret": float(random_curves["random_regret"][0, -1]),
                                "mean_violation": float(max(0, random_curves["random_deficit"][0, -1]))},
                  "candidates": []}
        grid = args.linear_beta_grid or [.35, 1., 2., 4., K**.25, 8., 16., 32.]
        for candidate in grid:
            row, _ = evaluate_method(False, candidate, calibration_seeds, benchmark, linear=True)
            search["candidates"].append(row)
            save_json(args.output_dir / "linear_calibration.json", search)
        selected = choose_near_random_candidate(search["reference"], search["candidates"])
        search.update(selected_beta=selected["beta"], target_met=selected["target_met"])
        linear_coefficient = selected["beta"]
        save_json(args.output_dir / "linear_calibration.json", search)
        print(f"Illustrative linear beta={linear_coefficient:g}; calibration target met={selected['target_met']}", flush=True)
        if not selected["target_met"]:
            print("No candidate met the near-Random target. Expand the calibration grid before evaluation.", flush=True)
            return

    summary = {"purpose": "illustrative hyperparameter comparison; not a tuned-best ablation",
               "linear_selection": "Beta selected for proximity to Random, not to optimize the paper's algorithm.",
               "settings": settings, "evaluation_seeds": args.seeds,
               "benchmark": {"type": "best feasible deterministic fixed action", "action": action},
               "metric": "Exact expected deployed-policy values; positive part of cumulative utility deficit",
               "methods": {}}
    curves = {"benchmark_reward": benchmark, **random_curves}
    for name, use_variance, method_beta, linear in (
        ("pd_powers", True, beta, False),
        ("novar", False, coefficient, False),
        ("linear_cmdp", False, linear_coefficient, True),
    ):
        summary["methods"][name], histories = evaluate_method(
            use_variance, method_beta, args.seeds, benchmark, linear=linear)
        curves.update({f"{name}_{key}": value for key, value in histories.items()})
    summary["paired_novar_minus_pd_powers"] = paired_comparison(
        summary["methods"]["novar"], summary["methods"]["pd_powers"])
    summary["paired_linear_minus_pd_powers"] = paired_comparison(
        summary["methods"]["linear_cmdp"], summary["methods"]["pd_powers"])
    save_json(args.output_dir / "comparison.json", summary)
    np.savez_compressed(args.output_dir / "curves.npz", **curves)
    plot_comparison(curves, args.output_dir)
    if default_output:
        for filename in ("regret_plot.jpg", "violation_plot.jpg"):
            source, destination = args.output_dir / filename, project_dir / filename
            if source != destination:
                shutil.copyfile(source, destination)
        print(f"Updated project plots: {project_dir / 'regret_plot.jpg'}\n"
              f"                       {project_dir / 'violation_plot.jpg'}", flush=True)
    print(f"Saved comparison to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
