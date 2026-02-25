"""test_api.py -- run from repo root: python test_api.py

Tests the public concept_abstraction API end-to-end on CartPole,
which is the fastest environment (no Atari, no Gurobi needed for
the variance/random baselines).

Expected output:
    random idx:   [0, 3, 5, 7, 9]        (varies by seed)
    variance idx: [2, 4, 6, 8, 11]       (varies by seed/steps)
    DRS idx:      [1, 3, 5, 8, 10]       (varies by seed/steps)
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import concept_abstraction as ca
from concept_abstraction.concept_bank import get_concepts
from concept_abstraction.environment_wrappers import ConceptWrapper, GymnasiumWrapper


# ── 1. Build a simple wrapped CartPole env ────────────────────────────────────

def make_cartpole():
    """CartPole wrapped so that info['observation'] contains the raw state."""
    def _make():
        env = gym.make("CartPole-v1")
        env = Monitor(env)
        env = ConceptWrapper(
            env,
            observation_space=gym.spaces.Box(low=-10, high=10, shape=(4,), dtype=float),
            get_raw_state=lambda env, obs, info: obs,
        )
        return env
    return DummyVecEnv([_make] * 4)


vec_env = make_cartpole()
gym_env = GymnasiumWrapper(vec_env)

# ── 2. Get concept functions ──────────────────────────────────────────────────

concepts, parsed = get_concepts("cart_pole")
print(f"Total concepts: {len(concepts)}")   # should be ~12


# ── 3. Train a quick policy (or load one if you have it) ─────────────────────

print("\nTraining a quick CartPole policy (~30s)...")
policy = PPO("MlpPolicy", vec_env, verbose=0)
policy.learn(total_timesteps=50_000)
print("Done.")


# ── 4. Test random (no policy/env needed) ────────────────────────────────────

idx = ca.random(concepts, k=5, seed=0)
print(f"\nrandom idx:   {idx}")


# ── 5. Test variance (needs policy + env, no Gurobi) ─────────────────────────
# Use fewer q_estimation_steps than default to keep the test fast

idx = ca.variance(policy, concepts, gym_env, k=5, q_estimation_steps=20_000, seed=0)
print(f"variance idx: {idx}")


# ── 6. Test DRS (needs Gurobi) ────────────────────────────────────────────────
# Comment this out if you don't have Gurobi installed yet.

idx = ca.DRS(policy, concepts, gym_env, k=5, q_estimation_steps=20_000, seed=0)
print(f"DRS idx:      {idx}")


# ── 7. Sanity check: selected concepts are callable ──────────────────────────

selected = [concepts[i] for i in idx]
dummy_obs = np.array([0.1, -0.2, 0.05, 0.3])   # fake CartPole state
outputs = [c(dummy_obs) for c in selected]
print(f"\nConcept outputs on dummy obs {dummy_obs}: {outputs}")
assert all(v in (0, 1) for v in outputs), "Concept functions should return 0 or 1"
print("All checks passed.")