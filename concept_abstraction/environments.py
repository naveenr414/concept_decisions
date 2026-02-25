import gymnasium as gym
from gymnasium import spaces
import numpy as np
import ocatari
import cv2
from gymnasium.wrappers import FrameStackObservation as FrameStack
import random
import torch
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import VecNormalize

from io import StringIO
from contextlib import redirect_stderr
stderr_buffer = StringIO()
with redirect_stderr(stderr_buffer):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import minigrid
from minigrid.core.constants import DIR_TO_VEC
from concept_abstraction.glucose_env import GlucoseEnvironment
from concept_abstraction.environment_wrappers import (
    ConceptWrapper, VecConceptWrapper, FrameSkipWrapper,
    LazyFramesToNumpy, GymnasiumWrapper,
)

cv2.setNumThreads(1)


# ── Raw state extractors ──────────────────────────────────────────────────────

def _get_raw_state_cartpole(_, obs, __):
    return obs


def _get_raw_pixels_cartpole(env, obs=None):
    pixels = env.render()
    small = cv2.resize(pixels, (240, 160), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


def _get_raw_pixels_mini_grid(env, obs=None):
    pixels = env.render()[:160, :160]
    gray = np.dot(pixels[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_NEAREST)


def _can_move(position, direction, grid):
    next_pos = DIR_TO_VEC[direction]
    nx, ny = position[0] + next_pos[0], position[1] + next_pos[1]
    if 1 <= nx < grid.width - 1 and 1 <= ny < grid.height - 1:
        cell = grid.get(nx, ny)
        return cell is None or cell.can_overlap()
    return False


def _get_raw_state_mini_grid(env, obs, info):
    agent_pos = env.unwrapped.agent_pos
    agent_dir = env.unwrapped.agent_dir
    grid = env.unwrapped.grid
    key_pos = (0, 0)
    door_pos = None
    door_open = False

    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None:
                if cell.type == "door":
                    door_pos = (x, y)
                    door_open = cell.is_open
                elif cell.type == "key":
                    key_pos = (x, y)

    movable = [int(_can_move(agent_pos, d, grid)) for d in range(4)]
    return [
        agent_pos[0], agent_pos[1], agent_dir,
        key_pos[0], key_pos[1],
        door_pos[0], door_pos[1], int(door_open),
        *movable,
    ]


def _get_raw_state_atari(env, _, __):
    pixels = env.unwrapped._ale.getScreenGrayscale().astype(np.uint8)
    return cv2.resize(pixels, (84, 84), interpolation=cv2.INTER_NEAREST)


def _get_raw_state_glucose(_, obs, __):
    return obs


# ── Atari helpers ─────────────────────────────────────────────────────────────

def _make_ocenv(env_name, use_concepts, seed=0, num_stack=4, processed_concepts=None):
    env = ocatari.OCAtari(env_name, mode="ram", render_mode=None, frameskip=4)
    env = Monitor(env)
    env.ale = env.unwrapped._ale

    if use_concepts:
        shape = (4, 6) if "pong" in env_name.lower() else (4, 4)
        env = ConceptWrapper(
            env,
            gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8),
            lambda env, obs, info: obs,
        )
    else:
        env = ConceptWrapper(
            env,
            gym.spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8),
            _get_raw_state_atari,
        )
        env = FrameStack(env, num_stack)
        env = LazyFramesToNumpy(env)

    env.reset(seed=seed)
    return env


def _get_n_atari_envs(n_envs, env_name, use_concepts, num_stack=4, processed_concepts=None):
    def safe_make(seed):
        return _make_ocenv(env_name, use_concepts, seed=seed,
                           num_stack=num_stack, processed_concepts=processed_concepts)

    if use_concepts:
        return DummyVecEnv([lambda seed=i: safe_make(seed) for i in range(n_envs)])
    else:
        return SubprocVecEnv([lambda seed=i: safe_make(seed) for i in range(n_envs)])


# ── Public API ────────────────────────────────────────────────────────────────

def get_environment(
    environment_string,
    concept_list,
    seed,
    concept_idx=[],
    use_processed=False,
    processed_concepts=None,
    fast_predictor=None,
    intervention_prob=0.0,
    concept_accuracy=1.0,
):
    """Build a vectorised environment for training or evaluation.

    Args:
        environment_string: One of 'cart_pole', 'mini_grid', 'pong',
                            'boxing', 'glucose'
        concept_list: List of concept functions, or None for pixel obs
        seed: Random seed
        concept_idx: Indices of concepts to use (subset of concept_list)
        use_processed: Use CNN concept predictor instead of ground-truth concepts
        processed_concepts: List of ParsedConcept objects for VecConceptWrapper
        fast_predictor: Trained CNN concept predictor
        intervention_prob: Fraction of concepts to replace with ground truth
        concept_accuracy: Per-concept noise level (1.0 = perfect)

    Returns:
        vec_env: SB3 VecEnv
        gymnasium_env: GymnasiumWrapper around vec_env
    """
    num_envs = 8
    num_stack = 4

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if environment_string == "cart_pole":
        def make_env():
            if concept_list is None or use_processed:
                env = gym.make("CartPole-v1", render_mode="rgb_array")
                env = Monitor(env)
                env = FrameSkipWrapper(env, skip=1, get_pixels_fn=_get_raw_pixels_cartpole)
                env = ConceptWrapper(
                    env,
                    spaces.Box(low=0, high=255, shape=(160, 240), dtype=np.uint8),
                    lambda env, obs, info: obs,
                    use_info_obs=True,
                )
                env = FrameStack(env, num_stack)
                env = LazyFramesToNumpy(env)
            else:
                env = Monitor(gym.make("CartPole-v1"))
                env = ConceptWrapper(
                    env,
                    spaces.Box(low=0, high=255, shape=(4,), dtype=np.uint8),
                    _get_raw_state_cartpole,
                )
            return env

        if concept_list is None or use_processed:
            vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])
        else:
            vec_env = DummyVecEnv([make_env for _ in range(num_envs)])

    elif environment_string == "mini_grid":
        num_stack = 1

        def make_env():
            if concept_list is None or use_processed:
                env = Monitor(gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="rgb_array"))
                env = FrameSkipWrapper(env, skip=1, get_pixels_fn=_get_raw_pixels_mini_grid)
                env = ConceptWrapper(
                    env,
                    spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8),
                    lambda env, obs, info: obs,
                    obs_function=lambda env, obs, info: _get_raw_state_mini_grid(env, info, {"observation": obs}),
                )
                env = FrameStack(env, num_stack)
                env = LazyFramesToNumpy(env)
            else:
                env = Monitor(gym.make("MiniGrid-DoorKey-5x5-v0"))
                env = ConceptWrapper(
                    env,
                    spaces.Box(low=0, high=5, shape=(12,), dtype=np.uint8),
                    lambda env, obs, info: _get_raw_state_mini_grid(env, info, {"observation": obs}),
                    obs_function=lambda env, obs, info: _get_raw_state_mini_grid(env, info, {"observation": obs}),
                )
            return env

        if concept_list is None or use_processed:
            vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])
        else:
            vec_env = DummyVecEnv([make_env for _ in range(num_envs)])

    elif environment_string == "pong":
        use_pixel = concept_list is None or use_processed
        vec_env = _get_n_atari_envs(
            num_envs, "PongNoFrameskip-v4",
            use_concepts=not use_pixel,
            num_stack=num_stack,
            processed_concepts=processed_concepts,
        )

    elif environment_string == "boxing":
        use_pixel = concept_list is None or use_processed
        vec_env = _get_n_atari_envs(
            num_envs, "BoxingNoFrameskip-v4",
            use_concepts=not use_pixel,
            num_stack=num_stack,
            processed_concepts=processed_concepts,
        )

    elif environment_string == "glucose":
        def make_env():
            register(
                id="simglucose/adolescent2-v0",
                entry_point=GlucoseEnvironment,
                max_episode_steps=288,
                kwargs={"patient_name": "adolescent#002"},
            )
            env = gym.make("simglucose/adolescent2-v0", render_mode=None)
            env = Monitor(env)
            env = ConceptWrapper(
                env,
                spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32),
                _get_raw_state_glucose,
                obs_function=_get_raw_state_glucose,
            )
            return env

        vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])

    else:
        raise ValueError(f"Unknown environment: {environment_string}")

    if use_processed:
        vec_env = VecConceptWrapper(
            vec_env, fast_predictor, concept_idx,
            intervention_prob=intervention_prob,
            processed_concepts=processed_concepts,
        )
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    elif concept_list is not None:
        vec_env = VecConceptWrapper(
            vec_env, None, concept_idx,
            intervention_prob=intervention_prob,
            processed_concepts=processed_concepts,
            concept_accuracy=concept_accuracy,
        )
        vec_env.observation_space = spaces.MultiBinary(len(concept_idx))

    gymnasium_env = GymnasiumWrapper(vec_env)
    return vec_env, gymnasium_env