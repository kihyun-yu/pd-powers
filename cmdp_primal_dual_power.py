#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
from random import randrange

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
K = 1000
repeat = 10
ACTION = 2 ** (dim - 1)
STATE = H + 2

thetastar = np.append(0.01 * np.ones(dim - 1), 1.0)
delta = 0.1
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Hyperparameters for CMDP primal-dual algorithm
LAMBDA = 1.0
beta = 0.35
beta1 = 0.45
dual_lr = 0.05
B_CONSTR = 6.0
theta = 0.001  # Policy smoothing toward uniform
alpha = 0.5  # Exponential policy learning rate (also used in dual update momentum)
# Reward-structure hyperparameters
TERMINAL_REWARD = 1.0
SHAPING_WEIGHT = 0.4
REWARD_MODE = "phase-flip"  # "stationary", "gradual", or "phase-flip"
REWARD_BLEND_START = 0.0
REWARD_BLEND_END = 1.0
REWARD_PHASE_LEN = 100
MAX_REWARD_PER_STEP = max(TERMINAL_REWARD, SHAPING_WEIGHT)


def log_file(name: str) -> str:
    """Return absolute path for experiment text logs."""
    return os.path.join(LOG_DIR, name)


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
    v_reward = np.zeros((H + 1, STATE))
    v_utility = np.zeros((H + 1, STATE))

    for h in range(H - 1, -1, -1):
        for s in range(STATE):
            r_now = reward(s, action, episode)
            g_now = cost(s, action)

            if s < H:
                p_cont = proba(s, action, s + 1)
                v_reward[h, s] = r_now + p_cont * v_reward[h + 1, s + 1] + (1.0 - p_cont) * v_reward[h + 1, H + 1]
                v_utility[h, s] = g_now + p_cont * v_utility[h + 1, s + 1] + (1.0 - p_cont) * v_utility[h + 1, H + 1]
            else:
                v_reward[h, s] = r_now + v_reward[h + 1, s]
                v_utility[h, s] = g_now + v_utility[h + 1, s]

    return float(v_reward[0, 0]), float(v_utility[0, 0])


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

    # Fallback: if no action satisfies the constraint, use best reward action.
    for action in range(ACTION):
        reward_sum = 0.0
        for episode in range(K):
            exp_reward, _ = expected_episode_values_for_action(action, episode)
            reward_sum += exp_reward
        if reward_sum > best_objective:
            best_objective = reward_sum
            best_action = action

    return best_action


OPTIMAL_FIXED_ACTION = select_constrained_optimal_action()


# =========================
# Baseline 1: fixed optimal action
# =========================
for run in tqdm(range(repeat), desc="Baseline 1 (optimal)"):
    total_reward = 0
    reward_history = []

    for episode in tqdm(range(K), desc=f"Run {run + 1}/{repeat}", leave=False):
        s_cur = 0
        for stage in range(H):
            a_cur = OPTIMAL_FIXED_ACTION  # constrained-optimal fixed action benchmark
            s_next = s_cur

            if s_cur < H:
                cont_prob = proba(s_cur, a_cur, s_cur + 1)
                s_next = H + 1 - np.random.binomial(1, cont_prob) * (H - s_cur)

            total_reward += reward(s_cur, a_cur, episode)
            reward_history.append(total_reward)
            s_cur = s_next

    with open(log_file(f"optimal_lowerbound_0.1_{run}.txt"), "w") as f:
        for val in reward_history:
            f.write(f"{val}\n")


# =========================
# Baseline 2: random policy
# =========================
for run in tqdm(range(repeat), desc="Baseline 2 (random)"):
    total_reward = 0
    reward_history = []
    constr_violation_history = []
    cumulative_constr_violation = 0.0

    for episode in tqdm(range(K), desc=f"Run {run + 1}/{repeat}", leave=False):
        s_cur = 0
        episode_cost_sum = 0.0
        for stage in range(H):
            a_cur = randrange(ACTION)
            s_next = s_cur

            if s_cur < H:
                cont_prob = proba(s_cur, a_cur, s_cur + 1)
                s_next = H + 1 - np.random.binomial(1, cont_prob) * (H - s_cur)

            total_reward += reward(s_cur, a_cur, episode)
            episode_cost_sum += cost(s_cur, a_cur)
            reward_history.append(total_reward)
            s_cur = s_next

        constr_violation = B_CONSTR - episode_cost_sum
        cumulative_constr_violation += constr_violation
        constr_violation_history.append(cumulative_constr_violation)

    with open(log_file(f"random_lowerbound_0.1_{run}.txt"), "w") as f:
        for val in reward_history:
            f.write(f"{val}\n")

    with open(log_file(f"random_constraint_violation_0.1_{run}.txt"), "w") as f:
        for val in constr_violation_history:
            f.write(f"{val}\n")


# =========================
# CMDP primal-dual policy gradient (tabular)
# =========================
for run in tqdm(range(repeat), desc="CMDP primal-dual"):
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

    # Reward critic covariance and moments (OPPObern-style).
    SIGMA_R = np.zeros((dim, dim, H))
    SIGMA1_R = np.zeros((dim, dim, H))
    SIGMA2_R = np.zeros((dim, dim, H))
    BB_R = np.zeros((dim, H))
    BB1_R = np.zeros((dim, H))
    BB2_R = np.zeros((dim, H))

    # Constraint critic covariance and moments (same structure).
    SIGMA_G = np.zeros((dim, dim, H))
    SIGMA1_G = np.zeros((dim, dim, H))
    SIGMA2_G = np.zeros((dim, dim, H))
    BB_G = np.zeros((dim, H))
    BB1_G = np.zeros((dim, H))
    BB2_G = np.zeros((dim, H))

    for stage in range(H):
        SIGMA_R[:, :, stage] = LAMBDA * np.eye(dim)
        SIGMA1_R[:, :, stage] = LAMBDA * np.eye(dim)
        SIGMA2_R[:, :, stage] = LAMBDA * np.eye(dim)

        SIGMA_G[:, :, stage] = LAMBDA * np.eye(dim)
        SIGMA1_G[:, :, stage] = LAMBDA * np.eye(dim)
        SIGMA2_G[:, :, stage] = LAMBDA * np.eye(dim)

    total_reward = 0
    reward_history = []
    constr_violation_history = []
    cumulative_constr_violation = 0.0

    for episode in tqdm(range(K), desc=f"Run {run + 1}/{repeat}", leave=False):
        s_cur[0] = 0
        episode_utility_sum = 0.0

        # Roll out one episode
        for stage in range(H):
            a_cur[stage] = np.random.choice(ACTION, p=POLICY[s_cur[stage], :, stage])
            s_next = s_cur[stage]

            if s_cur[stage] < H:
                cont_prob = proba(s_cur[stage], a_cur[stage], s_cur[stage] + 1)
                s_next = H + 1 - np.random.binomial(1, cont_prob) * (H - s_cur[stage])

            total_reward += reward(s_cur[stage], a_cur[stage], episode)
            episode_utility_sum += cost(s_cur[stage], a_cur[stage])
            reward_history.append(total_reward)
            s_cur[stage + 1] = s_next

        # Backward OPPObern-style update for both critics.
        for stage in range(H):
            curstage = H - 1 - stage
            s_now = s_cur[curstage]
            a_now = a_cur[curstage]
            s_nxt = s_cur[curstage + 1]

            # ---------- Reward critic ----------
            phi1_r = phiv(s_now, a_now, V_REWARD[:, curstage + 1])
            phi2_r = phiv(
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

            phi_r = phiv(s_now, a_now, V_REWARD[:, curstage + 1])
            BB_R[:, curstage] += phi_r * V_REWARD[s_nxt, curstage + 1] / (variance_r**2)
            phi_r_col = phi_r.reshape(dim, 1)
            SIGMA_R[:, :, curstage] += (phi_r_col @ phi_r_col.T) / (variance_r**2)

            hattheta_r = np.linalg.lstsq(SIGMA_R[:, :, curstage], BB_R[:, curstage], rcond=-1)[0]

            # ---------- Constraint critic ----------
            phi1_g = phiv(s_now, a_now, V_CONSTR[:, curstage + 1])
            phi2_g = phiv(
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

            phi_g = phiv(s_now, a_now, V_CONSTR[:, curstage + 1])
            BB_G[:, curstage] += phi_g * V_CONSTR[s_nxt, curstage + 1] / (variance_g**2)
            phi_g_col = phi_g.reshape(dim, 1)
            SIGMA_G[:, :, curstage] += (phi_g_col @ phi_g_col.T) / (variance_g**2)

            hattheta_g = np.linalg.lstsq(SIGMA_G[:, :, curstage], BB_G[:, curstage], rcond=-1)[0]

            # ---------- Q/V/Policy update ----------
            for stat in range(STATE):
                for act in range(ACTION):
                    ph_r = phiv(stat, act, V_REWARD[:, curstage + 1])
                    UU_r = np.linalg.lstsq(SIGMA_R[:, :, curstage], ph_r, rcond=-1)[0]
                    q_r = reward(stat, act, episode) + np.dot(hattheta_r, ph_r) + beta * np.sqrt(np.dot(UU_r, ph_r))
                    q_r = min(max(q_r, 0), (stage + 1) * MAX_REWARD_PER_STEP)
                    Q_REWARD[stat, act, curstage] = q_r

                    ph_g = phiv(stat, act, V_CONSTR[:, curstage + 1])
                    UU_g = np.linalg.lstsq(SIGMA_G[:, :, curstage], ph_g, rcond=-1)[0]
                    q_g = cost(stat, act) + np.dot(hattheta_g, ph_g) + beta * np.sqrt(np.dot(UU_g, ph_g))
                    q_g = max(q_g, 0)
                    Q_CONSTR[stat, act, curstage] = q_g

                V_REWARD[stat, curstage] = np.dot(Q_REWARD[stat, :, curstage], POLICY[stat, :, curstage])
                V_CONSTR[stat, curstage] = np.dot(Q_CONSTR[stat, :, curstage], POLICY[stat, :, curstage])

                # Modified primal update: exponential weighting with blend toward uniform
                policy_sum = 0.0
                for act in range(ACTION):
                    q_lagrangian = alpha * (Q_REWARD[stat, act, curstage] + dual_lambda * Q_CONSTR[stat, act, curstage])
                    POLICY[stat, act, curstage] *= np.exp(q_lagrangian)
                    policy_sum += POLICY[stat, act, curstage]

                for act in range(ACTION):
                    POLICY[stat, act, curstage] /= policy_sum
                
                # Blend toward uniform policy
                uniform_policy = 1.0 / ACTION
                for act in range(ACTION):
                    POLICY[stat, act, curstage] = (1.0 - theta) * POLICY[stat, act, curstage] + theta * uniform_policy

        # Modified dual update: momentum-based with explicit penalty terms
        constr_estimate = V_CONSTR[0, 0] / H
        constr_violation = constr_budget - episode_utility_sum
        cumulative_constr_violation += constr_violation
        constr_violation_history.append(cumulative_constr_violation)
        
        # Momentum decay and penalty terms based on H, alpha, theta
        momentum_decay = 1.0 - alpha * dual_lr * (H ** 3)
        penalty_term = alpha * (H ** 3) + 2 * theta * (H ** 2)
        dual_lambda = dual_lambda * momentum_decay + dual_lr * (constr_budget - constr_estimate - penalty_term)
        dual_lambda = max(0.0, dual_lambda)

    with open(log_file(f"primaldual_lowerbound_0.1_{run}.txt"), "w") as f:
        for val in reward_history:
            f.write(f"{val}\n")

    with open(log_file(f"primaldual_constraint_violation_0.1_{run}.txt"), "w") as f:
        for val in constr_violation_history:
            f.write(f"{val}\n")


# =========================
# Plot regret
# =========================
K1 = H * K
optimal_runs = np.zeros((repeat, K1))
random_runs = np.zeros((repeat, K1))
primaldual_runs = np.zeros((repeat, K1))
violation_runs = np.zeros((repeat, K))
random_violation_runs = np.zeros((repeat, K))

for run in range(repeat):
    with open(log_file(f"optimal_lowerbound_0.1_{run}.txt"), "r") as f:
        for idx, line in enumerate(f):
            optimal_runs[run, idx] = float(line)

    with open(log_file(f"random_lowerbound_0.1_{run}.txt"), "r") as f:
        for idx, line in enumerate(f):
            random_runs[run, idx] = float(line)

    with open(log_file(f"primaldual_lowerbound_0.1_{run}.txt"), "r") as f:
        for idx, line in enumerate(f):
            primaldual_runs[run, idx] = float(line)

    with open(log_file(f"primaldual_constraint_violation_0.1_{run}.txt"), "r") as f:
        for idx, line in enumerate(f):
            violation_runs[run, idx] = float(line)

    with open(log_file(f"random_constraint_violation_0.1_{run}.txt"), "r") as f:
        for idx, line in enumerate(f):
            random_violation_runs[run, idx] = float(line)

# Convert cumulative step rewards to per-episode realized returns.
optimal_episode_returns = np.zeros((repeat, K))
random_episode_returns = np.zeros((repeat, K))
primaldual_episode_returns = np.zeros((repeat, K))
for i in range(K):
    end_idx = (i + 1) * H - 1
    if i == 0:
        optimal_episode_returns[:, i] = optimal_runs[:, end_idx]
        random_episode_returns[:, i] = random_runs[:, end_idx]
        primaldual_episode_returns[:, i] = primaldual_runs[:, end_idx]
    else:
        prev_end_idx = i * H - 1
        optimal_episode_returns[:, i] = optimal_runs[:, end_idx] - optimal_runs[:, prev_end_idx]
        random_episode_returns[:, i] = random_runs[:, end_idx] - random_runs[:, prev_end_idx]
        primaldual_episode_returns[:, i] = primaldual_runs[:, end_idx] - primaldual_runs[:, prev_end_idx]

# Paper definition:
#   R_optimal = average single-episode return under optimal policy
#   Regret(K) = sum_{k=1}^K (R_optimal - R_k^alg)
R_optimal = float(np.mean(optimal_episode_returns))
regret_random_by_episode = np.cumsum(R_optimal - random_episode_returns, axis=1)
regret_primaldual_by_episode = np.cumsum(R_optimal - primaldual_episode_returns, axis=1)


def mean_and_ci95(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and 95% confidence half-width along axis 0."""
    mean = np.mean(values, axis=0)
    if values.shape[0] <= 1:
        return mean, np.zeros_like(mean)
    stderr = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    return mean, 1.96 * stderr


regret_random_mean, regret_random_ci = mean_and_ci95(regret_random_by_episode)
regret_primaldual_mean, regret_primaldual_ci = mean_and_ci95(regret_primaldual_by_episode)
violation_random_mean, violation_random_ci = mean_and_ci95(random_violation_runs)
violation_mean, violation_ci = mean_and_ci95(violation_runs)

# Use episode for x-axis
x_episode = list(range(1, K + 1))

# Regret plot (by episode)
fig_regret = plt.figure(figsize=(8, 5))
plt.plot(x_episode, regret_random_mean, label="Random")
plt.plot(x_episode, regret_primaldual_mean, label="PD-POWERS")
if repeat > 1:
    plt.fill_between(
        x_episode,
        regret_random_mean - regret_random_ci,
        regret_random_mean + regret_random_ci,
        alpha=0.2,
    )
    plt.fill_between(
        x_episode,
        regret_primaldual_mean - regret_primaldual_ci,
        regret_primaldual_mean + regret_primaldual_ci,
        alpha=0.2,
    )
# plt.xlabel("Episode", fontsize=20, labelpad=8)
plt.ylabel("Regret", fontsize=20)
plt.legend(fontsize=20)
plt.tight_layout()
plt.subplots_adjust(bottom=0.16)
fig_regret.savefig("regret_plot.jpg", dpi=150)
plt.close(fig_regret)

# Constraint violation plot
fig_violation = plt.figure(figsize=(8, 5))
plt.plot(x_episode, violation_random_mean, label="Random")
plt.plot(x_episode, violation_mean, label="PD-POWERS")
if repeat > 1:
    plt.fill_between(
        x_episode,
        violation_random_mean - violation_random_ci,
        violation_random_mean + violation_random_ci,
        alpha=0.2,
    )
    plt.fill_between(
        x_episode,
        violation_mean - violation_ci,
        violation_mean + violation_ci,
        alpha=0.2,
    )
plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Episode", fontsize=20, labelpad=8)
plt.ylabel("Constraint Violation", fontsize=20)
plt.legend(fontsize=20)
plt.tight_layout()
plt.subplots_adjust(bottom=0.16)
fig_violation.savefig("violation_plot.jpg", dpi=150)
plt.close(fig_violation)

plt.show()