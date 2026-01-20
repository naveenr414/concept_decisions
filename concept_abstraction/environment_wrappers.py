import gymnasium as gym
import numpy as np
import torch 
from stable_baselines3.common.vec_env import VecEnvWrapper
import copy 

class VecConceptWrapper(VecEnvWrapper):
    """Applies concept prediction to all environments in one batch"""
    
    def __init__(self, venv, fast_predictor, concept_idx, num_frames=4, height=84, width=84, 
                 intervention_prob=0.0, processed_concepts=None,concept_accuracy=1.0):
        super().__init__(venv)
        self.fast_predictor = fast_predictor
        self.concept_idx = concept_idx
        
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(len(concept_idx),), dtype=np.float32
        )
        self.processed_concepts = processed_concepts
        self.num_frames = num_frames
        self.width = width 
        self.height = height 
        self.intervention_prob = intervention_prob
        self.concept_accuracy = concept_accuracy
        self.training_mode = True 
        
        # Group concepts by type (keep your existing code)
        value_threshold_concepts = [p for p in self.processed_concepts 
                                   if p.meta['type'] == 'value' and 'thr' in p.meta]
        value_equality_concepts = [p for p in self.processed_concepts 
                                  if p.meta['type'] == 'value' and 'value' in p.meta]
        diff_concepts = [p for p in self.processed_concepts if p.meta['type'] == 'diff']
        velocity_concepts = [p for p in self.processed_concepts if p.meta['type'] == 'velocity']
        
        # ---- Value-type THRESHOLD concepts (for Pong: value > threshold) ----
        if value_threshold_concepts:
            self.value_frames = torch.tensor([p.meta['frame'] for p in value_threshold_concepts], device='cuda')
            self.value_idxs = torch.tensor([p.meta['idx'] for p in value_threshold_concepts], device='cuda')
            self.value_scales = torch.tensor([p.meta['scale'] for p in value_threshold_concepts], device='cuda')
            self.value_thresholds = torch.tensor([p.meta['thr'] for p in value_threshold_concepts], device='cuda')
            if 'offset' in value_threshold_concepts[0].meta:
                self.value_offsets = torch.tensor([p.meta.get('offset', 0) for p in value_threshold_concepts], device='cuda')
            else:
                self.value_offsets = None
        else:
            self.value_frames = self.value_idxs = self.value_scales = self.value_thresholds = torch.tensor([], device='cuda')
            self.value_offsets = None
        self.num_value = len(value_threshold_concepts)
        
        # ---- Value-type EQUALITY concepts (for MiniGrid: value == target) ----
        if value_equality_concepts:
            self.eq_frames = torch.tensor([p.meta['frame'] for p in value_equality_concepts], device='cuda')
            self.eq_idxs = torch.tensor([p.meta['idx'] for p in value_equality_concepts], device='cuda')
            self.eq_values = torch.tensor([p.meta['value'] for p in value_equality_concepts], device='cuda')
            # Scale is optional for equality concepts
            if 'scale' in value_equality_concepts[0].meta:
                self.eq_scales = torch.tensor([p.meta['scale'] for p in value_equality_concepts], device='cuda')
            else:
                self.eq_scales = None
        else:
            self.eq_frames = self.eq_idxs = self.eq_values = torch.tensor([], device='cuda')
            self.eq_scales = None
        self.num_equality = len(value_equality_concepts)
        
        # ---- Diff-type concepts ----
        if diff_concepts:
            self.diff_f1 = torch.tensor([p.meta['frame1'] for p in diff_concepts], device='cuda')
            self.diff_i1 = torch.tensor([p.meta['idx1'] for p in diff_concepts], device='cuda')
            self.diff_f2 = torch.tensor([p.meta['frame2'] for p in diff_concepts], device='cuda')
            self.diff_i2 = torch.tensor([p.meta['idx2'] for p in diff_concepts], device='cuda')
            self.diff_thresholds = torch.tensor([p.meta['thr'] for p in diff_concepts], device='cuda')
            if 'scale' in diff_concepts[0].meta:
                self.diff_scales = torch.tensor([p.meta['scale'] for p in diff_concepts], device='cuda')
            else:
                self.diff_scales = None
        else:
            self.diff_f1 = self.diff_i1 = self.diff_f2 = self.diff_i2 = self.diff_thresholds = torch.tensor([], device='cuda')
            self.diff_scales = None
        self.num_diff = len(diff_concepts)
        
        # ---- Velocity-type concepts (with clipping) ----
        if velocity_concepts:
            self.vel_f1 = torch.tensor([p.meta['frame1'] for p in velocity_concepts], device='cuda')
            self.vel_i1 = torch.tensor([p.meta['idx1'] for p in velocity_concepts], device='cuda')
            self.vel_f2 = torch.tensor([p.meta['frame2'] for p in velocity_concepts], device='cuda')
            self.vel_i2 = torch.tensor([p.meta['idx2'] for p in velocity_concepts], device='cuda')
            self.vel_scales = torch.tensor([p.meta['scale'] for p in velocity_concepts], device='cuda')
            self.vel_clip_min = torch.tensor([p.meta['clip_min'] for p in velocity_concepts], device='cuda')
            self.vel_clip_max = torch.tensor([p.meta['clip_max'] for p in velocity_concepts], device='cuda')
            self.vel_thresholds = torch.tensor([p.meta['thr'] for p in velocity_concepts], device='cuda')
        else:
            self.vel_f1 = self.vel_i1 = self.vel_f2 = self.vel_i2 = torch.tensor([], device='cuda')
            self.vel_scales = self.vel_clip_min = self.vel_clip_max = self.vel_thresholds = None
        self.num_velocity = len(velocity_concepts)
        self.concept_accuracy = concept_accuracy
        
        num_concepts = len(concept_idx)
        self.mask = torch.zeros(
            (self.num_envs, num_concepts),
            device="cuda",
            dtype=torch.bool
        )

        # PRE-ALLOCATE BUFFERS
        self.num_envs = venv.num_envs
        self._buffers_ready = False
        self.observations_gpu = None
        self.predictions_gpu = None
        self.predictions_cpu = None
        self.obs_batch_gpu = None
        self._resample_intervention_mask(list(range(len(self.mask))))

    def _resample_intervention_mask(self, env_ids):
        print("Resampling Intervention Mask")
        num_concepts = self.mask.shape[1]
        k = round(self.intervention_prob * num_concepts)
        idx = torch.randperm(num_concepts, device="cuda")[:k]
        print(idx)
        self.mask[:,idx] = True 

    def _init_buffers(self,obs,infos):
        """Initialize buffers on first call"""
        if self._buffers_ready:
            return
        
        # Get observation shape from first info
        sample_obs = infos[0]['observation']
        obs_shape = np.array(sample_obs).shape
        
        if len(obs_shape) == 1:
            self.observations_gpu = torch.empty(
                (self.num_envs, obs_shape[0]), 
                device='cuda', dtype=torch.float32
            )
        else:
            self.observations_gpu = torch.empty(
                (self.num_envs, *obs_shape), 
                device='cuda', dtype=torch.float32
            )

        
        
        # Pre-allocate predictions buffer
        self.predictions_gpu = torch.empty(
            (self.num_envs, len(self.concept_idx)), 
            device='cuda', dtype=torch.float32
        )
        
        # Pre-allocate CPU output buffer (pinned memory for fast transfer)
        self.predictions_cpu = torch.empty(
            (self.num_envs, len(self.concept_idx)), 
            dtype=torch.float32,
        )
        
        self._buffers_ready = True

    def reset(self):
        obs = self.venv.reset()
        infos = self.venv.reset_infos
        self._init_buffers(obs, infos)
        return self._process_batch(obs, infos)

    def set_eval_mode(self):
        self.training_mode = False
        # self._resample_intervention_mask(list(range(len(self.mask))))

    def set_train_mode(self):
        self.training_mode = True


    def step_wait(self):
        processed_obs, rewards, dones, infos = self.venv.step_wait()
        if self.training_mode:
            env_ids_done = np.where(dones)[0]
            # if len(env_ids_done) > 0:
            #     self._resample_intervention_mask(env_ids_done)



        processed_obs = self._process_batch(processed_obs,infos)
        
        for idx, info in enumerate(infos):
            if "terminal_observation" in info:
                terminal_pixels = info["terminal_observation"]
                term_obs_batch = np.expand_dims(terminal_pixels, 0)
                info['terminal_observation'] = self._process_batch(term_obs_batch, [info])[0]
        
        return processed_obs, rewards, dones, infos    
        
    def _process_batch(self, obs_batch,infos):
        """Process entire batch with pre-allocated buffers"""

        self.observations_gpu.copy_(torch.from_numpy(np.stack([i['observation'] for i in infos],axis=0)), non_blocking=True)
        observations = self.observations_gpu        
        # Check if observations are 2D or 3D
        is_2d = (observations.ndim == 2)
        if is_2d:
            observations = observations.unsqueeze(1)
        
        # Compute concepts (keep all your existing logic)
        out_list = []
        
        # Value threshold concepts
        if self.num_value > 0:
            vals = observations[:, self.value_frames, self.value_idxs]
            if self.value_offsets is not None:
                vals = (vals + self.value_offsets) * self.value_scales
            else:
                vals = vals * self.value_scales
            vals_binary = (vals > self.value_thresholds).float()
            out_list.append(vals_binary)
        
        # Diff concepts
        if self.num_diff > 0:
            diffs = observations[:, self.diff_f1, self.diff_i1] - observations[:, self.diff_f2, self.diff_i2]
            if self.diff_scales is not None:
                diffs = diffs * self.diff_scales
            diffs_binary = (diffs > self.diff_thresholds).float()
            out_list.append(diffs_binary)
        
        # Velocity concepts
        if self.num_velocity > 0:
            vel_diffs = observations[:, self.vel_f1, self.vel_i1] - observations[:, self.vel_f2, self.vel_i2]
            vel_diffs = torch.clamp(vel_diffs, self.vel_clip_min, self.vel_clip_max)
            vel_diffs = vel_diffs * self.vel_scales
            vel_binary = (vel_diffs > self.vel_thresholds).float()
            out_list.append(vel_binary)
        
        # Equality concepts
        if self.num_equality > 0:
            eq_vals = observations[:, self.eq_frames, self.eq_idxs]
            if self.eq_scales is not None:
                eq_vals = eq_vals * self.eq_scales
            eq_binary = (eq_vals == self.eq_values).float()
            out_list.append(eq_binary)
        
        # Combine concepts
        if out_list:
            concept_vals = torch.cat(out_list, dim=1)
        else:
            concept_vals = torch.empty((len(observations), 0), device='cuda')
        
        # Apply concept accuracy
        if self.concept_accuracy < 1.0:
            flips = torch.rand_like(concept_vals) > self.concept_accuracy
            concept_vals = torch.where(flips, 1.0 - concept_vals, concept_vals)

        concept_vals = concept_vals[:, self.concept_idx]

        # OPTIMIZATION 2: Reuse predictions buffer
        if self.fast_predictor is not None:
            with torch.no_grad():
                if not isinstance(obs_batch, torch.Tensor):
                    obs_tensor = torch.from_numpy(obs_batch).to('cuda', dtype=torch.float32, non_blocking=True) / 255.0
                else:
                    obs_tensor = obs_batch.to('cuda', dtype=torch.float32, non_blocking=True) / 255.0
                
                logits = self.fast_predictor(obs_tensor)[:, self.concept_idx].float()
                logits = logits.detach()

                # 2. CLAMP the logits to a fixed range
                # This prevents any "rogue" high-confidence predictions from 
                # overpowering your intervention.
                logit_bound = 5.0
                logits = torch.clamp(logits, -logit_bound, logit_bound)
                
                self.predictions_gpu[:] = logits
                
                # 3. SET INTERVENTION to the exact bounds
                # If concept is 1, set to +5; if 0, set to -5.
                # This ensures intervention is ALWAYS the max/min possible value.
                for i in range(self.num_envs):
                    self.predictions_gpu[i, self.mask[i]] = (
                        (concept_vals[i, self.mask[i]] * 2 - 1) * logit_bound
                    )

        else:
            base = (concept_vals * 2 - 1)*logit_bound 
            self.predictions_gpu[:] = base

        # Just allocate fresh CPU array - this is actually very cheap
        return self.predictions_gpu.cpu().numpy()
    

    def __setstate__(self, state):
        self.__dict__.update(state)


class ConceptWrapper(gym.ObservationWrapper):
    def __init__(self, env,observation_space,get_raw_state,use_info_obs=False,obs_function=lambda env, obs, info: obs):
        super().__init__(env)
        self.observation_space = observation_space
        self.get_raw_state = get_raw_state
        self.use_info_obs = use_info_obs
        self.obs_function = obs_function

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.use_info_obs:
            return self._process(obs,info), {'observation': self.obs_function(self,info['observation'],info)}
        return self._process(obs,info), {'observation': self.obs_function(self,obs,info)}

    def observation(self, obs):
        processed = np.array(self._process(obs))
        return processed

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed_obs = self._process(obs,info)
        info = dict(info)
        if self.use_info_obs:
            pass
        else:
            info["observation"] = self.obs_function(self,obs,info)
        return processed_obs, reward, terminated, truncated, info

    def _process(self, obs,info):
        return self.get_raw_state(self,obs,info)


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
    """
    Executes the same action for `skip` frames, returns only the last observation.
    """
    def __init__(self, env, skip=2, get_pixels_fn=None):
        super().__init__(env)
        self.skip = skip
        self.get_pixels = get_pixels_fn
        self._last_obs = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['observation'] = obs
        self._last_obs = self.get_pixels(self.env, obs)
        return self._last_obs, info
    
    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        
        # Execute action for `skip` frames
        for i in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            # Update observation on last frame
            if i == self.skip - 1:
                info['observation'] = obs
                self._last_obs = self.get_pixels(self.env, obs)
            
            # Stop early if episode ends
            if terminated or truncated:
                break
        
        return self._last_obs, total_reward, terminated, truncated, info

class GymnasiumWrapper:
    def __init__(self, vec_env):
        self.vec_env = vec_env
        self.num_envs = vec_env.num_envs
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
    
    def reset(self, **kwargs):
        obs = self.vec_env.reset(**kwargs)
        infos = self.vec_env.reset_infos
        return obs, infos
    
    def step(self, actions):
        obs, rewards, dones, infos = self.vec_env.step(actions)
        # Return 5-tuple with truncated = terminated (both equal to dones)
        return obs, rewards, dones, dones, infos
    
    def close(self):
        self.vec_env.close()
    
    def __getattr__(self, name):
        # Delegate any other attributes/methods to the wrapped environment
        return getattr(self.vec_env, name)
    def __deepcopy__(self, memo):
        # Only deepcopy the underlying env
        copied_env = copy.deepcopy(self.vec_env, memo)
        new_wrapper = type(self)(copied_env)
        memo[id(self)] = new_wrapper
        return new_wrapper
