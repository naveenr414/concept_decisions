"""Internal implementation of the public concept-selection API.

All functions accept a standard Gym/SB3 policy + env and return a list of
selected concept indices. The internal Q-estimation and LP steps are handled
automatically.

Baseline methods
----------------
random   – pick k concepts uniformly at random; requires no env interaction
variance – pick k concepts with the highest marginal bit-variance (closest to
           50/50 split); does not use Q-values
greedy   – pick k concepts that minimise the conditional variance of Q-values;
           uses Q-value estimates but no LP solver
"""

import numpy as np
from concept_abstraction.env_utils import estimate_q_values
from concept_abstraction.training import train_concept_predictor as _train_concept_predictor
from concept_abstraction.selection import (
    drs               as _drs,
    drs_log           as _drs_log,
    variance_selection,
    greedy_selection,
    random_selection,
)


def DRS(
    policy,
    concepts,
    env,
    k,
    coverage_ratio=0.75,
    q_estimation_steps=200_000,
    rollout_steps=10_000,
    seed=0,
):
    """Select k decision-relevant concepts using DRS.

    DRS solves a coverage LP to find the k concepts that best separate
    states with different optimal actions, minimising the state-abstraction
    error and providing a performance guarantee.

    Requires Gurobi (free academic licence available at gurobi.com).

    Args:
        policy: Trained SB3-compatible policy with a `predict(obs)` method.
            Should be trained on the *full* state, not on concepts.
        concepts: List of K concept functions, each with signature
            f(obs) -> int (0 or 1), where obs is a single raw observation.
        env: Gym-compatible vectorised environment. Must expose an
            `info["observation"]` key in each step's info dict containing
            the raw state used by `concepts`.
        k: Number of concepts to select.
        coverage_ratio: Fraction of cross-action state pairs that the
            selected concepts must separate (rho in the paper). Lower values
            make the LP easier to solve. Default 0.75.
        q_estimation_steps: Total environment steps used to estimate Q-values
            via TD learning. More steps → better Q-estimates. Default 200_000.
        rollout_steps: Steps used to collect policy observations for the LP
            coverage constraint. Default 10_000.
        seed: Random seed for reproducibility.

    Returns:
        idx: Sorted list of k integer indices into `concepts`.

    Example:
        >>> policy  = PPO.load("my_policy")
        >>> env     = make_vec_env("CartPole-v1", n_envs=8)
        >>> idx     = ca.DRS(policy, concepts, env, k=5)
        >>> selected = [concepts[i] for i in idx]
    """
    np.random.seed(seed)
    q_estimates = estimate_q_values(
        policy, env, concepts, total_timesteps=q_estimation_steps
    )
    _, idx = _drs(
        ground_truth_gym_env=env,
        concept_list=concepts,
        num_concepts=k,
        groundtruth_model=policy,
        q_estimates=q_estimates,
        rollout_steps=rollout_steps,
        coverage_ratio=coverage_ratio,
    )
    return sorted(idx)


def DRS_log(
    policy,
    concepts,
    env,
    k,
    acc_list,
    coverage_ratio=0.75,
    q_estimation_steps=200_000,
    rollout_steps=10_000,
    seed=0,
):
    """Select k concepts using DRS-log for imperfect (learned) predictors.

    DRS-log extends DRS by accounting for per-concept prediction noise.
    Instead of requiring hard separation, it maximises the *expected*
    coverage under the concept accuracy distribution, making it more
    robust when concepts are predicted from raw observations by a CNN.

    Requires Gurobi (free academic licence available at gurobi.com).

    Args:
        policy: Trained SB3-compatible policy with a `predict(obs)` method.
        concepts: List of K concept functions f(obs) -> int (0 or 1).
        env: Gym-compatible vectorised environment.
        k: Number of concepts to select.
        acc_list: List of K floats in [0, 1] — per-concept prediction
            accuracy of your trained concept predictor. Obtain via
            `ca.train_concept_predictor(...)` or supply your own estimates.
        coverage_ratio: Expected coverage fraction (rho). Default 0.75.
        q_estimation_steps: TD learning steps for Q-value estimation.
        rollout_steps: Policy rollout steps for the LP coverage constraint.
        seed: Random seed.

    Returns:
        idx: Sorted list of k integer indices into `concepts`.

    Example:
        >>> _, acc_list = ca.train_concept_predictor(env, policy, concepts)
        >>> idx = ca.DRS_log(policy, concepts, env, k=5, acc_list=acc_list)
    """
    if len(acc_list) != len(concepts):
        raise ValueError(
            f"acc_list has {len(acc_list)} entries but concepts has {len(concepts)}. "
            "Provide one accuracy value per concept."
        )

    np.random.seed(seed)
    q_estimates = estimate_q_values(
        policy, env, concepts, total_timesteps=q_estimation_steps
    )
    _, idx = _drs_log(
        ground_truth_gym_env=env,
        concept_list=concepts,
        num_concepts=k,
        groundtruth_model=policy,
        q_estimates=q_estimates,
        acc_list=acc_list,
        rollout_steps=rollout_steps,
        coverage_ratio=coverage_ratio,
    )
    return sorted(idx)


def variance(policy, concepts, env, k, q_estimation_steps=200_000, seed=0):
    """Select k concepts by highest marginal bit-variance.

    Fast baseline that requires no LP solver and does not use Q-values.
    Picks the k concepts whose empirical distribution across collected
    observations is closest to 50/50, maximising p*(1-p) for each binary
    concept bit.

    Args:
        policy: Trained SB3-compatible policy.
        concepts: List of K concept functions f(obs) -> int (0 or 1).
        env: Gym-compatible vectorised environment.
        k: Number of concepts to select.
        q_estimation_steps: Environment steps used to collect concept
            observations (Q-values themselves are not used).
        seed: Random seed.

    Returns:
        idx: Sorted list of k integer indices into `concepts`.
    """
    np.random.seed(seed)
    q_estimates = estimate_q_values(
        policy, env, concepts, total_timesteps=q_estimation_steps
    )
    _, idx = variance_selection(concepts, k, q_estimates)
    return sorted(idx)


def greedy(policy, concepts, env, k, q_estimation_steps=200_000, seed=0):
    """Select k concepts by lowest conditional variance of Q-values.

    Fast baseline that requires no LP solver. For each concept, computes
    the total variance of Q-values when the concept is 0 vs 1 (weighted by
    group size), then selects the k concepts with the lowest total variance —
    i.e. those that most cleanly partition states by their Q-values.

    Args:
        policy: Trained SB3-compatible policy.
        concepts: List of K concept functions f(obs) -> int (0 or 1).
        env: Gym-compatible vectorised environment.
        k: Number of concepts to select.
        q_estimation_steps: TD learning steps for Q-value estimation.
        seed: Random seed.

    Returns:
        idx: Sorted list of k integer indices into `concepts`.
    """
    np.random.seed(seed)
    q_estimates = estimate_q_values(
        policy, env, concepts, total_timesteps=q_estimation_steps
    )
    _, idx = greedy_selection(concepts, k, q_estimates)
    return sorted(idx)


def random(concepts, k, seed=0):
    """Randomly select k concepts.

    Useful as a lower-bound baseline. Requires no policy or environment.

    Args:
        concepts: List of K concept functions.
        k: Number of concepts to select.
        seed: Random seed.

    Returns:
        idx: Sorted list of k integer indices into `concepts`.
    """
    np.random.seed(seed)
    _, idx = random_selection(concepts, k)
    return sorted(idx)


def train_concept_predictor(env, policy, concepts, concept_idx=None, environment_string=None, **kwargs):
    """Train a CNN to predict binary concept values from pixel observations.

    Convenience wrapper around `training.train_concept_predictor` for use
    with DRS-log. Collects rollout data, trains a NatureCNN-based classifier,
    and returns per-concept accuracy for use as `acc_list` in `DRS_log`.

    Args:
        env: GymnasiumWrapper environment (pixel observations).
        policy: SB3-compatible policy for data collection.
        concepts: Full list of concept functions.
        concept_idx: Indices of concepts to predict. Defaults to all.
        environment_string: Environment name string (affects CNN input shape).
            If None, inferred from env.observation_space shape.
        **kwargs: Passed through to `training.train_concept_predictor`
            (e.g. epochs=25, num_episodes=5).

    Returns:
        predictor: Trained ConceptPredictorCNN (nn.Module)
        acc_list: np.ndarray of per-concept accuracy values in [0, 1]
    """
    if concept_idx is None:
        concept_idx = list(range(len(concepts)))

    if environment_string is None:
        shape = env.observation_space.shape
        if len(shape) == 1 and shape[0] == 4:
            environment_string = "cart_pole"
        elif len(shape) == 3 and shape[1:] == (84, 84):
            environment_string = "pong"
        else:
            raise ValueError(
                "Could not infer environment_string from observation space. "
                "Please pass it explicitly, e.g. environment_string='cart_pole'."
            )

    return _train_concept_predictor(
        env, policy, concepts, concept_idx, environment_string, **kwargs
    )