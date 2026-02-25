import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnvWrapper
import copy


class VecConceptWrapper(VecEnvWrapper):
    """Applies batched GPU concept extraction to all environments.

    Supports ground-truth concepts (no predictor), trained CNN predictors,
    intervention (replacing a subset of predicted concepts with ground truth),
    and concept accuracy noise.
    """

    def __init__(
        self,
        venv,
        fast_predictor,
        concept_idx,
        num_frames=4,
        height=84,
        width=84,
        intervention_prob=0.0,
        processed_concepts=None,
        concept_accuracy=1.0,
    ):
        super().__init__(venv)
        self.fast_predictor   = fast_predictor
        self.concept_idx      = concept_idx
        self.processed_concepts = processed_concepts
        self.num_frames       = num_frames
        self.width            = width
        self.height           = height
        self.intervention_prob = intervention_prob
        self.concept_accuracy = concept_accuracy
        self.training_mode    = True

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(len(concept_idx),), dtype=np.float32
        )

        # Pre-group concepts by type for fast batched GPU evaluation
        value_thr = [p for p in processed_concepts if p.meta["type"] == "value" and "thr"   in p.meta]
        value_eq  = [p for p in processed_concepts if p.meta["type"] == "value" and "value" in p.meta]
        diff      = [p for p in processed_concepts if p.meta["type"] == "diff"]
        velocity  = [p for p in processed_concepts if p.meta["type"] == "velocity"]

        def _t(lst):
            return torch.tensor(lst, device="cuda")

        # Value threshold
        if value_thr:
            self.value_frames     = _t([p.meta["frame"] for p in value_thr])
            self.value_idxs       = _t([p.meta["idx"]   for p in value_thr])
            self.value_scales     = _t([p.meta["scale"] for p in value_thr])
            self.value_thresholds = _t([p.meta["thr"]   for p in value_thr])
            self.value_offsets    = (
                _t([p.meta.get("offset", 0) for p in value_thr])
                if "offset" in value_thr[0].meta else None
            )
        else:
            self.value_frames = self.value_idxs = self.value_scales = self.value_thresholds = _t([])
            self.value_offsets = None
        self.num_value = len(value_thr)

        # Value equality
        if value_eq:
            self.eq_frames = _t([p.meta["frame"] for p in value_eq])
            self.eq_idxs   = _t([p.meta["idx"]   for p in value_eq])
            self.eq_values = _t([p.meta["value"]  for p in value_eq])
            self.eq_scales = _t([p.meta["scale"]  for p in value_eq]) if "scale" in value_eq[0].meta else None
        else:
            self.eq_frames = self.eq_idxs = self.eq_values = _t([])
            self.eq_scales = None
        self.num_equality = len(value_eq)

        # Diff
        if diff:
            self.diff_f1  = _t([p.meta["frame1"] for p in diff])
            self.diff_i1  = _t([p.meta["idx1"]   for p in diff])
            self.diff_f2  = _t([p.meta["frame2"] for p in diff])
            self.diff_i2  = _t([p.meta["idx2"]   for p in diff])
            self.diff_thresholds = _t([p.meta["thr"]   for p in diff])
            self.diff_scales     = _t([p.meta["scale"] for p in diff]) if "scale" in diff[0].meta else None
        else:
            self.diff_f1 = self.diff_i1 = self.diff_f2 = self.diff_i2 = self.diff_thresholds = _t([])
            self.diff_scales = None
        self.num_diff = len(diff)

        # Velocity
        if velocity:
            self.vel_f1        = _t([p.meta["frame1"]   for p in velocity])
            self.vel_i1        = _t([p.meta["idx1"]     for p in velocity])
            self.vel_f2        = _t([p.meta["frame2"]   for p in velocity])
            self.vel_i2        = _t([p.meta["idx2"]     for p in velocity])
            self.vel_scales    = _t([p.meta["scale"]    for p in velocity])
            self.vel_clip_min  = _t([p.meta["clip_min"] for p in velocity])
            self.vel_clip_max  = _t([p.meta["clip_max"] for p in velocity])
            self.vel_thresholds= _t([p.meta["thr"]      for p in velocity])
        else:
            self.vel_f1 = self.vel_i1 = self.vel_f2 = self.vel_i2 = _t([])
            self.vel_scales = self.vel_clip_min = self.vel_clip_max = self.vel_thresholds = None
        self.num_velocity = len(velocity)

        # Intervention mask (which concepts get replaced with ground truth)
        num_concepts = len(concept_idx)
        self.mask = torch.zeros((self.num_envs, num_concepts), device="cuda", dtype=torch.bool)
        self._resample_intervention_mask()

        # GPU buffers (initialised on first observation)
        self._buffers_ready   = False
        self.observations_gpu = None
        self.predictions_gpu  = torch.empty((self.num_envs, num_concepts), device="cuda", dtype=torch.float32)
        self.num_envs         = venv.num_envs

    def _resample_intervention_mask(self):
        num_concepts = self.mask.shape[1]
        k = round(self.intervention_prob * num_concepts)
        idx = torch.randperm(num_concepts, device="cuda")[:k]
        self.mask.zero_()
        self.mask[:, idx] = True

    def _init_buffers(self, infos):
        if self._buffers_ready:
            return
        sample = np.array(infos[0]["observation"])
        shape  = sample.shape
        self.observations_gpu = torch.empty(
            (self.num_envs, *shape) if len(shape) > 1
            else (self.num_envs, shape[0]),
            device="cuda", dtype=torch.float32,
        )
        self._buffers_ready = True

    def reset(self):
        obs   = self.venv.reset()
        infos = self.venv.reset_infos
        self._init_buffers(infos)
        return self._process_batch(obs, infos)

    def set_eval_mode(self):
        self.training_mode = False

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        obs = self._process_batch(obs, infos)
        for info in infos:
            if "terminal_observation" in info:
                info["terminal_observation"] = self._process_batch(
                    np.expand_dims(info["terminal_observation"], 0), [info]
                )[0]
        return obs, rewards, dones, infos

    def _process_batch(self, obs_batch, infos):
        stacked = np.stack([i["observation"] for i in infos], axis=0)
        self.observations_gpu.copy_(torch.from_numpy(stacked), non_blocking=True)
        observations = self.observations_gpu

        if observations.ndim == 2:
            observations = observations.unsqueeze(1)

        out = []

        if self.num_value > 0:
            vals = observations[:, self.value_frames, self.value_idxs]
            if self.value_offsets is not None:
                vals = (vals + self.value_offsets) * self.value_scales
            else:
                vals = vals * self.value_scales
            out.append((vals > self.value_thresholds).float())

        if self.num_diff > 0:
            diffs = observations[:, self.diff_f1, self.diff_i1] - observations[:, self.diff_f2, self.diff_i2]
            if self.diff_scales is not None:
                diffs = diffs * self.diff_scales
            out.append((diffs > self.diff_thresholds).float())

        if self.num_velocity > 0:
            vel = observations[:, self.vel_f1, self.vel_i1] - observations[:, self.vel_f2, self.vel_i2]
            vel = torch.clamp(vel, self.vel_clip_min, self.vel_clip_max) * self.vel_scales
            out.append((vel > self.vel_thresholds).float())

        if self.num_equality > 0:
            eq = observations[:, self.eq_frames, self.eq_idxs]
            if self.eq_scales is not None:
                eq = eq * self.eq_scales
            out.append((eq == self.eq_values).float())

        concept_vals = torch.cat(out, dim=1) if out else torch.empty((len(observations), 0), device="cuda")

        if self.concept_accuracy < 1.0:
            flips = torch.rand_like(concept_vals) > self.concept_accuracy
            concept_vals = torch.where(flips, 1.0 - concept_vals, concept_vals)

        concept_vals = concept_vals[:, self.concept_idx]
        logit_bound  = 5.0

        if self.fast_predictor is not None:
            with torch.no_grad():
                t = (obs_batch if isinstance(obs_batch, torch.Tensor)
                     else torch.from_numpy(obs_batch).to("cuda", dtype=torch.float32, non_blocking=True) / 255.0)
                logits = torch.clamp(
                    self.fast_predictor(t)[:, self.concept_idx].float().detach(),
                    -logit_bound, logit_bound,
                )
                self.predictions_gpu[:] = logits
                for i in range(self.num_envs):
                    self.predictions_gpu[i, self.mask[i]] = (
                        (concept_vals[i, self.mask[i]] * 2 - 1) * logit_bound
                    )
        else:
            self.predictions_gpu[:] = (concept_vals * 2 - 1) * logit_bound

        return self.predictions_gpu.cpu().numpy()


class ConceptWrapper(gym.ObservationWrapper):
    """Wraps a single env to extract concept observations."""

    def __init__(self, env, observation_space, get_raw_state,
                 use_info_obs=False, obs_function=lambda env, obs, info: obs):
        super().__init__(env)
        self.observation_space = observation_space
        self.get_raw_state     = get_raw_state
        self.use_info_obs      = use_info_obs
        self.obs_function      = obs_function

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed = self._process(obs, info)
        obs_out = info["observation"] if self.use_info_obs else obs
        return processed, {"observation": self.obs_function(self, obs_out, info)}

    def observation(self, obs):
        return np.array(self._process(obs))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed = self._process(obs, info)
        info = dict(info)
        if not self.use_info_obs:
            info["observation"] = self.obs_function(self, obs, info)
        return processed, reward, terminated, truncated, info

    def _process(self, obs, info=None):
        return self.get_raw_state(self, obs, info)


class LazyFramesToNumpy(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, obs):
        obs = np.array(obs, copy=False)
        if obs.ndim == 4 and obs.shape[1] == 1:
            obs = obs.squeeze(1)
        return obs


class FrameSkipWrapper(gym.Wrapper):
    """Execute the same action for `skip` frames, return last observation."""

    def __init__(self, env, skip=2, get_pixels_fn=None):
        super().__init__(env)
        self.skip        = skip
        self.get_pixels  = get_pixels_fn

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["observation"] = obs
        return self.get_pixels(self.env, obs), info

    def step(self, action):
        total_reward = 0.0
        for i in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if i == self.skip - 1:
                info["observation"] = obs
                last_obs = self.get_pixels(self.env, obs)
            if terminated or truncated:
                break
        return last_obs, total_reward, terminated, truncated, info


class GymnasiumWrapper:
    """Thin wrapper converting a SB3 VecEnv to the Gymnasium 5-tuple step API."""

    def __init__(self, vec_env):
        self.vec_env          = vec_env
        self.num_envs         = vec_env.num_envs
        self.observation_space = vec_env.observation_space
        self.action_space     = vec_env.action_space

    def reset(self, **kwargs):
        obs   = self.vec_env.reset(**kwargs)
        infos = self.vec_env.reset_infos
        return obs, infos

    def step(self, actions):
        obs, rewards, dones, infos = self.vec_env.step(actions)
        return obs, rewards, dones, dones, infos

    def seed(self, seed):
        return self.vec_env.seed(seed)

    def close(self):
        self.vec_env.close()

    def __getattr__(self, name):
        return getattr(self.vec_env, name)

    def __deepcopy__(self, memo):
        copied = copy.deepcopy(self.vec_env, memo)
        new    = type(self)(copied)
        memo[id(self)] = new
        return new