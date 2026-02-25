"""concept_bank.py

Concept definitions for each environment.

Each environment has:
  - Scalar concept functions  f(obs) -> float  used during rollouts
  - A meta map                fn -> dict        used by VecConceptWrapper for
                                                fast batched GPU evaluation
  - A get_concepts() entry    returning (concept_list, parsed_concepts)
"""

import numpy as np
import torch
from dataclasses import dataclass
from copy import deepcopy


# ── Parsed concept dataclass ──────────────────────────────────────────────────

@dataclass
class ParsedConcept:
    name: str
    feature_fn: callable
    threshold: float
    concept_fn: callable
    meta: dict


# ── Concept builders ──────────────────────────────────────────────────────────

def make_threshold_concept(feature_fn, threshold):
    """Return f(obs) -> int: 1 if feature_fn(obs) > threshold."""
    def concept_fn(obs):
        return int(feature_fn(obs) > threshold)
    return concept_fn


def make_equality_concept(feature_fn, target_value):
    """Return f(obs) -> int: 1 if feature_fn(obs) == target_value."""
    def concept_fn(obs):
        return int(feature_fn(obs) == target_value)
    return concept_fn


def _build_threshold_concepts(feature_fns, threshold_lists, meta_map):
    """Build threshold ParsedConcepts for a list of features with per-feature thresholds."""
    parsed = []
    for fn, thresholds in zip(feature_fns, threshold_lists):
        base_meta = meta_map[fn]
        for thr in thresholds:
            thr_val = float(thr)
            meta = deepcopy(base_meta)
            meta["thr"] = thr_val
            parsed.append(ParsedConcept(
                name=f"{fn.__name__} > {thr_val:.3f}",
                feature_fn=fn,
                threshold=thr_val,
                concept_fn=make_threshold_concept(fn, thr_val),
                meta=meta,
            ))
    return parsed


def _build_equality_concepts(feature_fns, value_lists, meta_map):
    """Build equality ParsedConcepts for a list of features with per-feature value ranges."""
    parsed = []
    for base_idx, (fn, values) in enumerate(zip(feature_fns, value_lists)):
        base_meta = meta_map[fn]
        for val in values:
            val_float = float(val)
            meta = deepcopy(base_meta)
            meta["base_idx"] = base_idx
            meta["value"] = val_float
            parsed.append(ParsedConcept(
                name=f"{fn.__name__} == {val_float:.0f}",
                feature_fn=fn,
                threshold=None,
                concept_fn=make_equality_concept(fn, val_float),
                meta=meta,
            ))
    return parsed


# ── CartPole ──────────────────────────────────────────────────────────────────

def cartpole_position(obs):          return obs[0]
def cartpole_velocity(obs):          return obs[1]
def cartpole_angle(obs):             return obs[2]
def cartpole_angular_velocity(obs):  return obs[3]

CARTPOLE_FEATURE_FNS = [
    cartpole_position,
    cartpole_velocity,
    cartpole_angle,
    cartpole_angular_velocity,
]

CARTPOLE_THRESHOLDS = [
    [-0.02, 0.02],
    [-0.2, -0.1, 0.1, 0.2],
    [-0.02, 0.02],
    [-0.3, -0.15, 0.15, 0.3],
]

CARTPOLE_META = {
    cartpole_position:         {"type": "value", "frame": -1, "idx": 0, "scale": 1.0},
    cartpole_velocity:         {"type": "value", "frame": -1, "idx": 1, "scale": 1.0},
    cartpole_angle:            {"type": "value", "frame": -1, "idx": 2, "scale": 1.0},
    cartpole_angular_velocity: {"type": "value", "frame": -1, "idx": 3, "scale": 1.0},
}


# ── MiniGrid ──────────────────────────────────────────────────────────────────

def minigrid_feature_0(obs):  return obs[0]
def minigrid_feature_1(obs):  return obs[1]
def minigrid_feature_2(obs):  return obs[2]
def minigrid_feature_3(obs):  return obs[3]
def minigrid_feature_4(obs):  return obs[4]
def minigrid_feature_5(obs):  return obs[5]
def minigrid_feature_6(obs):  return obs[6]
def minigrid_feature_7(obs):  return obs[7]
def minigrid_feature_8(obs):  return obs[8]
def minigrid_feature_9(obs):  return obs[9]
def minigrid_feature_10(obs): return obs[10]
def minigrid_feature_11(obs): return obs[11]

MINIGRID_FEATURE_FNS = [
    minigrid_feature_0, minigrid_feature_1, minigrid_feature_2,
    minigrid_feature_3, minigrid_feature_4, minigrid_feature_5,
    minigrid_feature_6, minigrid_feature_7, minigrid_feature_8,
    minigrid_feature_9, minigrid_feature_10, minigrid_feature_11,
]

MINIGRID_VALUE_RANGES = [
    [1, 2, 3, 4, 5],  # x pos
    [1, 2, 3, 4, 5],  # y pos
    [1, 2, 3, 4],     # direction
    [1, 2, 3, 4, 5],  # key x
    [1, 2, 3, 4, 5],  # key y
    [1, 2, 3, 4, 5],  # door x
    [1, 2, 3, 4, 5],  # door y
    [0, 1],           # door open
    [0, 1],           # can move right
    [0, 1],           # can move left
    [0, 1],           # can move down
    [0, 1],           # can move up
]

MINIGRID_META = {
    fn: {"type": "value", "frame": -1, "idx": i, "scale": 1.0}
    for i, fn in enumerate(MINIGRID_FEATURE_FNS)
}

MINIGRID_CONCEPT_NAMES = [
    "X Pos", "Y Pos", "Dir",
    "Key X", "Key Y", "Door X", "Door Y", "Door Open",
    "Right", "Left", "Down", "Up",
]


def get_all_mini_grid_names():
    """Return display names for all MiniGrid binary concepts."""
    names = []
    for i, (feat_name, values) in enumerate(zip(MINIGRID_CONCEPT_NAMES, MINIGRID_VALUE_RANGES)):
        for v in values:
            names.append(f"{feat_name}_{v}")
    return names


# ── Pong ──────────────────────────────────────────────────────────────────────

def pong_paddle_y(obs):
    obs = np.array(obs)
    return (obs[-1, 1] - 128) / 255

def pong_ball_x(obs):
    obs = np.array(obs)
    return (obs[-1, 2] - 128) / 255

def pong_ball_y(obs):
    obs = np.array(obs)
    return (obs[-1, 3] - 128) / 255

def pong_ball_v_x(obs):
    obs = np.array(obs)
    return np.clip(obs[-1, 2] - obs[-2, 2], -4, 4) / 4

def pong_ball_v_y(obs):
    obs = np.array(obs)
    return np.clip(obs[-1, 3] - obs[-2, 3], -4, 4) / 4

def pong_enemy_y(obs):
    obs = np.array(obs)
    return (obs[-1, 5] - 128) / 255

def pong_enemy_v_y(obs):
    obs = np.array(obs)
    return np.clip(obs[-2, 5] - obs[-1, 5], -4, 4) / 4

def pong_paddle_x_diff(obs):
    obs = np.array(obs)
    return (obs[-1, 0] - obs[-1, 2]) / 255

def pong_paddle_y_diff(obs):
    obs = np.array(obs)
    return (obs[-1, 1] - obs[-1, 3]) / 255

def pong_enemy_y_diff(obs):
    obs = np.array(obs)
    return (obs[-1, 1] - obs[-1, 5]) / 255

def pong_enemy_ball_x_diff(obs):
    obs = np.array(obs)
    return (obs[-1, 4] - obs[-1, 2]) / 255

def pong_enemy_ball_y_diff(obs):
    obs = np.array(obs)
    return (obs[-1, 5] - obs[-1, 3]) / 255

PONG_FEATURE_FNS = [
    pong_paddle_y, pong_ball_x, pong_ball_y,
    pong_ball_v_x, pong_ball_v_y,
    pong_enemy_y, pong_enemy_v_y,
    pong_paddle_x_diff, pong_paddle_y_diff, pong_enemy_y_diff,
    pong_enemy_ball_x_diff, pong_enemy_ball_y_diff,
]

PONG_THRESHOLDS = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
                   0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

PONG_META = {
    pong_paddle_y:         {"type": "value",    "frame": -1,  "idx": 1, "scale": 1/255.0, "offset": -128},
    pong_ball_x:           {"type": "value",    "frame": -1,  "idx": 2, "scale": 1/255.0, "offset": -128},
    pong_ball_y:           {"type": "value",    "frame": -1,  "idx": 3, "scale": 1/255.0, "offset": -128},
    pong_enemy_y:          {"type": "value",    "frame": -1,  "idx": 5, "scale": 1/255.0, "offset": -128},
    pong_ball_v_x:         {"type": "velocity", "frame1": -1, "idx1": 2, "frame2": -2, "idx2": 2, "clip_min": -4, "clip_max": 4, "scale": 1/4.0},
    pong_ball_v_y:         {"type": "velocity", "frame1": -1, "idx1": 3, "frame2": -2, "idx2": 3, "clip_min": -4, "clip_max": 4, "scale": 1/4.0},
    pong_enemy_v_y:        {"type": "velocity", "frame1": -2, "idx1": 5, "frame2": -1, "idx2": 5, "clip_min": -4, "clip_max": 4, "scale": 1/4.0},
    pong_paddle_x_diff:    {"type": "diff",     "frame1": -1, "idx1": 0, "frame2": -1, "idx2": 2, "scale": 1/255.0},
    pong_paddle_y_diff:    {"type": "diff",     "frame1": -1, "idx1": 1, "frame2": -1, "idx2": 3, "scale": 1/255.0},
    pong_enemy_y_diff:     {"type": "diff",     "frame1": -1, "idx1": 1, "frame2": -1, "idx2": 5, "scale": 1/255.0},
    pong_enemy_ball_x_diff:{"type": "diff",     "frame1": -1, "idx1": 4, "frame2": -1, "idx2": 2, "scale": 1/255.0},
    pong_enemy_ball_y_diff:{"type": "diff",     "frame1": -1, "idx1": 5, "frame2": -1, "idx2": 3, "scale": 1/255.0},
}


# ── Boxing ────────────────────────────────────────────────────────────────────

def boxing_player_x(obs):
    obs = np.array(obs)
    return obs[-1, 0] / 255

def boxing_player_y(obs):
    obs = np.array(obs)
    return obs[-1, 1] / 255

def boxing_enemy_x(obs):
    obs = np.array(obs)
    return obs[-1, 2] / 255

def boxing_enemy_y(obs):
    obs = np.array(obs)
    return obs[-1, 3] / 255

def boxing_player_v_x(obs):
    obs = np.array(obs)
    return obs[-1, 0] - obs[-2, 0]

def boxing_player_v_y(obs):
    obs = np.array(obs)
    return obs[-1, 1] - obs[-2, 1]

def boxing_enemy_v_x(obs):
    obs = np.array(obs)
    return obs[-1, 2] - obs[-2, 2]

def boxing_enemy_v_y(obs):
    obs = np.array(obs)
    return obs[-1, 3] - obs[-2, 3]

def boxing_player_enemy_diff_x(obs):
    obs = np.array(obs)
    return obs[-1, 0] - obs[-1, 2]

def boxing_player_enemy_diff_y(obs):
    obs = np.array(obs)
    return obs[-1, 1] - obs[-1, 3]

BOXING_FEATURE_FNS = [
    boxing_player_x, boxing_player_y,
    boxing_enemy_x, boxing_enemy_y,
    boxing_player_v_x, boxing_player_v_y,
    boxing_enemy_v_x, boxing_enemy_v_y,
    boxing_player_enemy_diff_x, boxing_player_enemy_diff_y,
]

BOXING_THRESHOLDS = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
                     0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

BOXING_META = {
    boxing_player_x:          {"type": "value", "frame": -1, "idx": 0, "scale": 1/255.0},
    boxing_player_y:          {"type": "value", "frame": -1, "idx": 1, "scale": 1/255.0},
    boxing_enemy_x:           {"type": "value", "frame": -1, "idx": 2, "scale": 1/255.0},
    boxing_enemy_y:           {"type": "value", "frame": -1, "idx": 3, "scale": 1/255.0},
    boxing_player_v_x:        {"type": "diff",  "frame1": -1, "idx1": 0, "frame2": -2, "idx2": 0},
    boxing_player_v_y:        {"type": "diff",  "frame1": -1, "idx1": 1, "frame2": -2, "idx2": 1},
    boxing_enemy_v_x:         {"type": "diff",  "frame1": -1, "idx1": 2, "frame2": -2, "idx2": 2},
    boxing_enemy_v_y:         {"type": "diff",  "frame1": -1, "idx1": 3, "frame2": -2, "idx2": 3},
    boxing_player_enemy_diff_x: {"type": "diff","frame1": -1, "idx1": 0, "frame2": -1, "idx2": 2},
    boxing_player_enemy_diff_y: {"type": "diff","frame1": -1, "idx1": 1, "frame2": -1, "idx2": 3},
}


# ── Glucose ───────────────────────────────────────────────────────────────────

def glucose_feature_0(obs): return obs[0]
def glucose_feature_1(obs): return obs[1]
def glucose_feature_2(obs): return obs[2]
def glucose_feature_3(obs): return obs[3]
def glucose_feature_4(obs): return obs[4]
def glucose_feature_5(obs): return obs[5]

GLUCOSE_FEATURE_FNS = [
    glucose_feature_0, glucose_feature_1, glucose_feature_2,
    glucose_feature_3, glucose_feature_4, glucose_feature_5,
]

GLUCOSE_THRESHOLDS = [
    [0.1, 0.3, 0.5, 0.7, 0.75],
    [-0.15, -0.1, -0.075, -0.05, -0.025, 0, 0.05, 0.1, 0.15],
    [-0.001, 0.05, 0.1, 0.15, 0.2],
    [15, 30, 45, 50, 60, 75],
    [-0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.8],
    [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 0.9, 0.95],
]

GLUCOSE_META = {
    fn: {"type": "value", "frame": -1, "idx": i, "scale": 1.0}
    for i, fn in enumerate(GLUCOSE_FEATURE_FNS)
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_concepts(environment_string):
    """Return (concept_list, parsed_concepts) for the given environment.

    Args:
        environment_string: One of 'cart_pole', 'mini_grid', 'pong',
                            'boxing', 'glucose'

    Returns:
        concept_list: List of scalar concept functions f(obs) -> float
        parsed_concepts: List of ParsedConcept objects with metadata
    """
    if environment_string == "cart_pole":
        parsed = _build_threshold_concepts(
            CARTPOLE_FEATURE_FNS, CARTPOLE_THRESHOLDS, CARTPOLE_META
        )

    elif environment_string == "mini_grid":
        parsed = _build_equality_concepts(
            MINIGRID_FEATURE_FNS, MINIGRID_VALUE_RANGES, MINIGRID_META
        )

    elif environment_string == "pong":
        parsed = _build_threshold_concepts(
            PONG_FEATURE_FNS,
            [PONG_THRESHOLDS] * len(PONG_FEATURE_FNS),
            PONG_META,
        )

    elif environment_string == "boxing":
        parsed = _build_threshold_concepts(
            BOXING_FEATURE_FNS,
            [BOXING_THRESHOLDS] * len(BOXING_FEATURE_FNS),
            BOXING_META,
        )

    elif environment_string == "glucose":
        parsed = _build_threshold_concepts(
            GLUCOSE_FEATURE_FNS, GLUCOSE_THRESHOLDS, GLUCOSE_META
        )

    else:
        raise ValueError(f"Unknown environment: {environment_string}")

    concept_list = [p.concept_fn for p in parsed]
    return concept_list, parsed