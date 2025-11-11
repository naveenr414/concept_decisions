import gymnasium as gym
import numpy as np
from concept_abstraction.utils import one_hot_state
import torch 
import time 
from stable_baselines3.common.vec_env import VecEnvWrapper

class ConceptEnv(gym.Env):
    """Build a new concept-based environment"""

    def __init__(self,concept_list,observation_space,action_space,rewards,transitions,all_states,max_steps,state_distro=None,done_map = lambda s: False):
        super().__init__()

        self.concept_list = concept_list 
        self.observation_space = observation_space
        self.action_space = action_space
        self.rewards = rewards
        self.transitions = transitions 
        self.max_steps = max_steps 
        self.all_states = all_states 
        self.steps = 0
        self.done_map = done_map 
        if state_distro is None:
            self.state_distro = np.ones(len(self.all_states))/len(all_states)
        else:
            self.state_distro = state_distro

    def get_observation(self):
        if self.concept_list is None:
            return one_hot_state(self.state,len(self.all_states))
        else:
            return np.array([concept(self.state) for concept in self.concept_list])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.choice(self.all_states,p=self.state_distro)
        self.steps = 0
        return self.get_observation(), {'observation': self.state}

    def is_scalar_like(self,x):
        # Python int or NumPy integer scalar
        if isinstance(x, (int, np.integer)):
            return True
        
        # 0-d NumPy array
        if isinstance(x, np.ndarray) and x.shape == ():
            return True
        
        return False


    def step(self, action):
        if not self.is_scalar_like(action):
            action = action[0]

        reward = self.rewards[self.state][action]
        if np.sum(self.transitions[self.state][action]) == 0:
            reward = -1
        else:
            self.state = np.random.choice(self.all_states, p=self.transitions[self.state][action])        
            self.steps += 1
        obs = self.get_observation()

        done = self.steps >= self.max_steps or self.done_map(self.state)
        return obs, reward, done, False, {'observation': self.state}

    def render(self):
        pass 

    def close(self):
        pass

class ConceptWrapper(gym.ObservationWrapper):
    def __init__(self, env,concept_list,observation_space,get_raw_state,use_info_obs=False,obs_function=lambda env, obs, info: obs):
        super().__init__(env)
        self.observation_space = observation_space
        self.concept_list = concept_list 
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
        if self.concept_list is None:
            return self.get_raw_state(self,obs)
        else:
            obs = self.obs_function(self,obs,info)
            obs = np.array(obs, dtype=np.float32)
            vec = [concept(obs) for concept in self.concept_list]
            return vec

class ObservationSubsetWrapper(gym.ObservationWrapper):
    """Wrapper to allow us to select subsets of observations
        from a gymasium environment"""

    def __init__(self, env, indices):
        super().__init__(env)
        self.indices = indices
        original_space = env.observation_space

        low = original_space.low[indices]
        high = original_space.high[indices]
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        return observation[self.indices]


class VecConceptWrapper(VecEnvWrapper):
    """Applies concept prediction to all environments in one batch"""
    
    def __init__(self, venv, fast_predictor, concept_idx):
        super().__init__(venv)
        self.fast_predictor = fast_predictor
        self.concept_idx = concept_idx
        
        # Update observation space to concept space
        self.observation_space = gym.spaces.MultiBinary(len(concept_idx))
        
        # Pre-allocate GPU buffer
        self.obs_buffer = None
        
    def reset(self):
        obs = self.venv.reset()
        return self._process_batch(obs)
    
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        processed_obs = self._process_batch(obs)
        
        # Store original observations in info
        for i, info in enumerate(infos):
            # obs[i] is (4, 84, 84), transpose to (84, 84, 4) for consistency
            info['observation'] = np.transpose(obs[i], (1, 2, 0))
            
        return processed_obs, rewards, dones, infos
    
    def _process_batch(self, obs_batch):
        """Process entire batch in one GPU call"""
        if self.fast_predictor is None:
            return obs_batch
        
        # obs_batch shape is ALREADY (num_envs, 4, 84, 84) - correct format!
        num_envs = obs_batch.shape[0]
        
        # Normalize to [0, 1] like in training
        obs_normalized = obs_batch.astype(np.float32) / 255.0
        
        # NO TRANSPOSE NEEDED - already in correct shape (num_envs, 4, 84, 84)
        
        # Allocate buffer once (or reallocate if num_envs changed)
        if self.obs_buffer is None or self.obs_buffer.shape[0] != num_envs:
            self.obs_buffer = torch.zeros(
                (num_envs, 2, 84, 84), 
                device='cuda', 
                dtype=torch.float32
            )
        # Copy batch to GPU buffer
        self.obs_buffer[:] = torch.as_tensor(obs_normalized, dtype=torch.float32, device='cuda')
        
        # SINGLE batched inference for all environments
        with torch.no_grad():
            logits = self.fast_predictor(self.obs_buffer)[:, self.concept_idx]
            # Apply sigmoid and threshold to binary predictions
            predictions = (torch.sigmoid(logits) > 0.5).float()
        
        # Return as numpy array (num_envs, len(concept_idx))
        return predictions.cpu().numpy()
class BinaryObservationSubsetWrapper(gym.ObservationWrapper):
    """Wrapper that selects a subset of observations from a binary
        observation space"""
    def __init__(self, env, indices,accuracies):
        super().__init__(env)
        self.indices = indices
        self.accuracies = accuracies

        if not isinstance(env.observation_space, gym.spaces.MultiBinary):
            raise ValueError("BinaryObservationSubsetWrapper requires MultiBinary observation space.")

        orig_n = env.observation_space.n
        if max(indices) >= orig_n:
            raise ValueError("Subset indices exceed original observation length.")
        self.observation_space = gym.spaces.MultiBinary(len(indices))

    def observation(self, observation):
        new_obs = observation[self.indices]

        if self.accuracies is not None:
            for idx,i in enumerate(self.indices):
                if np.random.random() > self.accuracies[i]:
                    new_obs[idx] = 1-new_obs[idx]

        return new_obs

class RewardPerturbationWrapper(gym.Wrapper):
    """Wrapper for gymnasium environments that allows for 
        perturbing the reward slightly"""

    def __init__(self, env, noise_std=0.1):
        super().__init__(env)
        self.noise_std = noise_std

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        perturbed_reward = reward + np.random.normal(0, self.noise_std)
        return obs, perturbed_reward, terminated, truncated, info

    def reset(self, **kwargs):
        if "seed" in kwargs or "options" in kwargs:
            return self.env.reset()
        return self.env.reset(**kwargs)

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
