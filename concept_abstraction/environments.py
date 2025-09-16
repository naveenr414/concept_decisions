import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from skimage.transform import resize
import ocatari
import cv2 
from collections import Counter
from sklearn.model_selection import train_test_split
from gymnasium.wrappers import FrameStack  # gym’s own for single envs
from copy import deepcopy
import random
import torch

from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv, VecNormalize
from concept_abstraction.post_hoc import BinaryFeatureEnvironmentWrapper, CartPoleBinaryFeatureExtractor
from concept_abstraction.utils import one_hot_state
from concept_abstraction.mimic import *
from concept_abstraction.concept_bank import clustering_concept_mimic, mimic_concept
import minigrid
from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC, TILE_PIXELS, COLOR_TO_IDX, OBJECT_TO_IDX

class OptimizedConceptWrapper(gym.ObservationWrapper):
    def __init__(self, env, fast_predictor, observation_space, get_raw_state, 
                 use_info_obs=False, obs_function=lambda env, obs, info: obs):
        super().__init__(env)
        self.observation_space = observation_space
        self.fast_predictor = fast_predictor  # The FastGPUPredictor instance
        self.get_raw_state = get_raw_state
        self.use_info_obs = use_info_obs
        self.obs_function = obs_function
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.use_info_obs:
            return self._process(obs, info), {'observation': self.obs_function(self, info['observation'], info)}
        return self._process(obs, info), {'observation': self.obs_function(self, obs, info)}
    
    def observation(self, obs):
        return np.array(self._process(obs))
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed_obs = self._process(obs, info)
        info = dict(info)
        
        if not self.use_info_obs:
            info["observation"] = self.obs_function(self, obs, info)
        
        return processed_obs, reward, terminated, truncated, info
    
    def _process(self, obs, info):
        if self.fast_predictor is None:
            return self.get_raw_state(self, obs)
        
        processed_obs = self.obs_function(self, obs, info)
        obs_array = np.array(processed_obs, dtype=np.float32)
        
        # SINGLE MODEL CALL instead of 24 separate calls
        predictions = self.fast_predictor.predict_all_concepts(obs_array, return_float=False)
        
        # Convert GPU tensor to CPU list
        if hasattr(predictions, 'cpu'):
            return predictions.cpu().tolist()
        else:
            return predictions.tolist()
class FastGPUPredictor:
    def __init__(self, model, device='cuda', cache_size=10000,subset=None):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # Keep tensors on GPU to avoid transfers
        self.cached_predictions = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
        self.subset = subset 
        if self.subset is None:
            self.subset = list(range(24))
        
        # Pre-allocate tensors to avoid repeated allocation
        self.input_tensor = None
        
    def _get_cache_key(self, obs):
        """Fast hash for numpy arrays"""
        if isinstance(obs, np.ndarray):
            return hash(obs.data.tobytes())
        return hash(obs.cpu().numpy().data.tobytes())
    
    def predict_all_concepts(self, obs, threshold=0.5, return_float=False):
        """Single inference for all 24 concepts - MAIN OPTIMIZATION"""
        # Check cache first
        cache_key = self._get_cache_key(obs)
        if cache_key in self.cached_predictions:
            self.cache_hits += 1
            cached_result = self.cached_predictions[cache_key]
            if return_float:
                return cached_result
            else:
                return (cached_result > threshold).float()
        
        self.cache_misses += 1
        
        # Prepare tensor - reuse allocated tensor if possible
        if isinstance(obs, np.ndarray):
            if self.input_tensor is None or self.input_tensor.shape[1:] != obs.shape:
                self.input_tensor = torch.empty((1,) + obs.shape, dtype=torch.float32, device=self.device)
            self.input_tensor[0] = torch.from_numpy(obs)
        else:
            obs_tensor = obs.clone().float()
            if obs_tensor.ndim == 3:
                obs_tensor = obs_tensor.unsqueeze(0)
            obs_tensor = obs_tensor.to(self.device)
            self.input_tensor = obs_tensor
        
        # Single inference for all concepts
        with torch.no_grad():
            raw_output = self.model(self.input_tensor)
            probabilities = torch.sigmoid(raw_output).squeeze(0)  # Keep on GPU
        
        # Cache the GPU tensor (small memory cost, big speed gain)
        if len(self.cached_predictions) < self.cache_size:
            self.cached_predictions[cache_key] = probabilities[self.subset]
        
        if return_float:
            return probabilities[self.subset]
        else:
            return (probabilities > threshold).float()[self.subset]
    
    def get_concept(self, obs, concept_idx, threshold=0.5, return_float=False):
        """Get single concept - uses cached full prediction"""
        all_predictions = self.predict_all_concepts(obs, threshold, return_float)
        return all_predictions[concept_idx].item()
    
    def get_cache_stats(self):
        total = self.cache_hits + self.cache_misses
        if total > 0:
            hit_rate = self.cache_hits / total * 100
            return f"Cache hit rate: {hit_rate:.1f}% ({self.cache_hits}/{total})"
        return "No predictions yet"


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

class DiscretizeObservationWrapper(gym.ObservationWrapper):
    """Wrapper to turn a continuous space into a hard-coded 
        discretized space"""

    def __init__(self, env, bins_per_feature=4):
        super().__init__(env)
        self.bins_per_feature = bins_per_feature
        self.n_features = env.observation_space.shape[0]

        self.bin_edges = [
            np.linspace(-4.8, 4.8, bins_per_feature + 1), 
            np.linspace(-3.0, 3.0, bins_per_feature + 1),
            np.linspace(-0.418, 0.418, bins_per_feature + 1),
            np.linspace(-3.5, 3.5, bins_per_feature + 1)
        ]

        self.observation_space = gym.spaces.MultiBinary(self.n_features * bins_per_feature)

    def observation(self, obs):
        binary_obs = np.zeros(self.n_features * self.bins_per_feature, dtype=np.int8)
        for i in range(self.n_features):
            bin_index = np.digitize(obs[i], self.bin_edges[i]) - 1
            bin_index = np.clip(bin_index, 0, self.bins_per_feature - 1)
            offset = i * self.bins_per_feature
            binary_obs[offset + bin_index] = 1
        return binary_obs


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

class InfoTransformWrapper(gym.Wrapper):
    """
    Wrap an environment and transform the `info` dict
    according to a user-provided function.
    """
    def __init__(self, env, concept_list):
        super().__init__(env)
        self.concept_list = concept_list

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        info['observation'] = [concept(info['observation']) for concept in self.concept_list]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # info['observation'] = [concept(info['observation']) for concept in self.concept_list]
        return obs, reward, terminated, truncated, info


def create_cyclic_env(num_nodes,concept_list):
    """Simple Environment that captures a cyclic structure between states
    0 -> 1 -> 2 -> 3 -> 0
    
    Here, certain states have certain rewards; 
        e.g., 0 and 2 have the same reward
        as do 1 and 3
    Concepts should capture this, so [0,2], and [1,3] should be 
        split by the concept
        
    Example concept splits include
        [0,2],[1,3]
        [0,3],[1,2]
        [0,1,2], [3]"""

    environment_nodes = num_nodes 
    action_space = spaces.Discrete(3)

    if concept_list is None:
        observation_space = spaces.MultiBinary(environment_nodes)
    else:
        observation_space = spaces.MultiBinary(len(concept_list))
    all_states = list(range(environment_nodes))
    max_steps = 20

    rewards = np.zeros((environment_nodes,3)) 
    for i in range(environment_nodes):
        if i%2 == 0:
            rewards[i,0] = rewards[i,1] = 1
        else:
            rewards[i,2] = 1
    rewards = np.array(rewards)

    transitions = []
    for i in range(len(all_states)):
        transitions_by_state = []
        for action in range(3):
            next_probs = [0.0 for i in range(len(all_states))]
            if action == 0:
                next_probs[(i - 1) % environment_nodes] = 1.0
            if action == 1:
                next_probs[(i + 1) % environment_nodes] = 1.0
            if action == 2:
                next_probs[(i) % environment_nodes] = 1.0
            transitions_by_state.append(next_probs)
        transitions.append(transitions_by_state)
    transitions = np.array(transitions)
    return ConceptEnv(concept_list,observation_space,action_space,rewards,transitions,all_states,max_steps)

def create_tree_env(num_nodes,concept_list):
    """Simple Environment that captures a tree structure between states
    0 -> (1,2)
    1 -> (3,4)
    2 -> (5,6)
    3 -> (7,8)
    etc. 
    then 7 -> 0

    The idea is to show that errors can propogate; the ideal path is to 
    go 0->1->3,etc.; this requires playing action LEFT each time
    However, error in the concepts lead to a large loss in the value
        as the agent will instead play RIGHT
            
    Example concept splits include
        Top: [0,1,3,7], all otehrs
        1st Binary Digit: [0,2,4,6,8]...[1,3,5,..]
        2nd Digit: [0,1,4,5],...
        etc."""

    environment_nodes = num_nodes 
    num_layers = int(np.log2(environment_nodes+1))
    all_states = list(range(environment_nodes))
    action_space = spaces.Discrete(2)
    if concept_list is None:
        observation_space = spaces.MultiBinary(environment_nodes)
    else:
        observation_space = spaces.MultiBinary(len(concept_list))
    max_steps = 20

    rewards = np.zeros((environment_nodes,action_space.n))
    for i in range(num_layers):
        rewards[2**i-1][0] = 1
    rewards[:,1] = 0.5

    transitions = np.zeros((len(all_states),
                                action_space.n,
                                len(all_states)))
    for state in range(len(transitions)):
        for action in range(len(transitions[state])):
            if state >= environment_nodes//2:
                if state == environment_nodes//2:
                    transitions[state][0][0] = 1
                    transitions[state][1][2] = 1
                else:
                    transitions[state][action][2] = 1
            else:
                transitions[state][action][2 * (state+1) + action - 1] = 1

    return ConceptEnv(concept_list,observation_space,action_space,rewards,transitions,all_states,max_steps)

def create_mimic_environment(concept_list,seed):
    """Create a MIMIC Environment given a list of concepts, seed
    
    Arguments:
        concept_list: A list of a single concept
            that maps a state (represented by a vector)
            to a single integer number
            It needs to be in this form becuase we need to discretize
            the environment
        seed: Integer, random seed
        
    Returns: ConceptEnv"""

    N_CLUSTERS = 750

    MIMICraw = pd.read_csv("../../data/mimic_github/ai_clinician/data/mimic_model/train/MIMICraw.csv")
    MIMICzs = pd.read_csv("../../data/mimic_github/ai_clinician/data/mimic_model/train/MIMICzs.csv")
    metadata = pd.read_csv("../../data/mimic_github/ai_clinician/data/mimic_model/train/metadata.csv")

    C_ICUSTAYID = "icustayid"
    unique_icu_stays = metadata[C_ICUSTAYID].unique()

    n_action_bins = 5
    n_actions = n_action_bins * n_action_bins # for both vasopressors and fluids

    all_actions, _, _ = fit_action_bins(
        MIMICraw[C_INPUT_STEP],
        MIMICraw[C_MAX_DOSE_VASO],
        n_action_bins=n_action_bins
    )

    train_ids, _ = train_test_split(unique_icu_stays, test_size=0.1,random_state=seed)
    train_indexes = metadata[metadata[C_ICUSTAYID].isin(train_ids)].index

    X_train = MIMICzs.iloc[train_indexes]
    metadata_train = metadata.iloc[train_indexes]
    actions_train = all_actions[train_indexes]

    cluster_concept, centers,clusterer = clustering_concept_mimic(X_train.values,N_CLUSTERS,seed)
    zeros = np.zeros((2, centers.shape[1]))
    centers = np.vstack([centers, zeros])

    states_train = clusterer.predict(X_train.values)

    n_cluster_states = np.max(states_train)+1
    absorbing_states =  [n_cluster_states + 1, n_cluster_states]
    rewards = [15, -15]


    # -------------------------
    # 1. constants / column names - EDIT these if your column names differ
    # -------------------------
    SOFA_COL = "SOFA"        # <- replace with actual SOFA column name in MIMICraw
    LACTATE_COL = "Arterial_lactate"  # <- replace with actual lactate column name in MIMICraw

    C0 = -0.025
    C1 = -0.125
    C2 = -2.0

    # scaling factor in case shaped rewards are too small/large relative to terminal rewards
    SHAPE_SCALE = 1.0  # tune this (e.g., 0.5, 2.0) after you inspect distributions

    # -------------------------
    # 2. prepare aligned arrays / dataframe
    # -------------------------
    # train_indexes already defined earlier in your snippet
    # metadata_train, states_train, actions_train are aligned with train_indexes

    # choose where to pull raw SOFA/lactate values from - I use MIMICraw here.
    sofa_vals = MIMICraw[SOFA_COL].iloc[train_indexes].astype(float).values
    lactate_vals = MIMICraw[LACTATE_COL].iloc[train_indexes].astype(float).values

    # Build a helper dataframe that preserves the exact order used to create states_train/actions_train
    df_steps = pd.DataFrame({
        "icustayid": metadata_train[C_ICUSTAYID].values,
        "state": states_train,
        "action": actions_train,
        "sofa": sofa_vals,
        "lactate": lactate_vals,
    })
    # the index 0..N-1 in df_steps corresponds to the ordering in X_train / states_train
    n_rows = len(df_steps)

    # -------------------------
    # 3. compute shaped reward per non-terminal step
    # -------------------------
    step_rewards = np.zeros(n_rows, dtype=float)

    # We'll compute forward-difference within each icu stay.
    for icu, grp in df_steps.groupby("icustayid"):
        pos = grp.index.values              # positions in df_steps for this ICU stay
        sofa_seq = grp["sofa"].values
        lactate_seq = grp["lactate"].values

        if len(pos) == 1:
            # single-step stay -> no intra-stay transition; keep step reward = 0 and rely on terminal reward
            continue

        # compute offset next values (last step's "next" will be NaN and handled)
        sofa_next = np.append(sofa_seq[1:], np.nan)
        lactate_next = np.append(lactate_seq[1:], np.nan)

        # indicator: same SOFA and SOFA > 0
        indicator = ((sofa_seq == sofa_next) & (sofa_seq > 0)).astype(float)

        # diffs: sofa_t - sofa_{t+1}, lactate_t - lactate_{t+1}
        diff_sofa = sofa_seq - sofa_next
        diff_lactate = lactate_seq - lactate_next

        # replace NaNs (last pos) with 0 for computing shaped contribution — last step will be terminal reward
        diff_sofa = np.nan_to_num(diff_sofa, nan=0.0)
        diff_lactate = np.nan_to_num(diff_lactate, nan=0.0)

        r = C0 * indicator + C1 * diff_sofa + C2 * np.tanh(diff_lactate)
        r = r * SHAPE_SCALE

        step_rewards[pos] = r
    qldata3_shaped = []
    for icu, grp in df_steps.groupby("icustayid"):
        pos = grp.index.values
        for i_idx, p in enumerate(pos):
            s = int(df_steps.loc[p, "state"])
            a = int(df_steps.loc[p, "action"])
            if i_idx < len(pos) - 1:
                next_p = pos[i_idx + 1]
                s_next = int(df_steps.loc[next_p, "state"])
                r = float(step_rewards[p])
                done = False
            else:
                # last step in stay -> treat as terminal transition to absorbing state
                # choose which absorbing-state id you want (your script used two absorbing states: [n_cluster_states + 1, n_cluster_states])
                # Here I set next state to absorbing_states[1] (adjust if your convention differs)
                s_next = absorbing_states[1]
                # terminal reward: keep your previous terminal rewards array (rewards variable)
                # but you must map actual mortality flag to decide +15 or -15. Replace MORT_COL below with your outcome column.
                MORT_COL = "hospital_death"  # <- CHANGE to your column name that flags death (0/1)
                # If you don't have a mortality column in metadata_train, keep using previous default terminal reward:
                try:
                    death_flag = int(metadata_train.iloc[p][MORT_COL])
                    # death_flag==1 => negative terminal reward; death_flag==0 => positive terminal reward
                    if death_flag == 1:
                        r = float(rewards[1])  # -15 (example)
                    else:
                        r = float(rewards[0])  # +15 (example)
                except Exception:
                    # fallback: use the positive terminal reward (change as you see fit)
                    r = float(rewards[0])
                done = True

            # Record format: (icustayid, pos, state, action, reward, next_state, done)
            qldata3_shaped.append((int(df_steps.loc[p, "icustayid"]), int(p), s, a, r, int(s_next), done))
    qldata3 = pd.DataFrame(qldata3_shaped, columns=['icustayid', 'orig_bloc', 'state', 'action', 'reward', 'next_state', 'done'])
    # Group by icustayid and assign bloc starting at 0 for each ICU stay
    qldata3['bloc'] = qldata3.groupby('icustayid').cumcount()+1

    # Create 'outcome' column: 1 if terminal (done=True) else 0
    qldata3['outcome'] = qldata3['done'].astype(int)

    # Keep only the requested columns
    qldata3 = qldata3[['bloc', 'icustayid', 'state', 'action', 'outcome', 'reward']]

    # Reset index
    qldata3.reset_index(drop=True, inplace=True)

    n_states = n_cluster_states + 2
    reward_val = 15
    transition_threshold = 5

    d = Counter(states_train)
    state_distro = np.array([d[i] for i in range(np.max(states_train)+1)])
    state_distro = state_distro / np.sum(state_distro)
    state_distro = np.append(state_distro,0)
    state_distro = np.append(state_distro,0)

    ####### BUILD MODEL ########
    # Initialize counts array: [s', s, a]
    transition_counts = np.zeros((n_states, n_states, n_actions))
    arr = np.array(qldata3)

    # Populate counts from dataframe
    for i,row in enumerate(arr):
        s = int(row[2])
        a = int(row[3])
        outcome = int(row[4])
        reward = row[5]
        if outcome == 1:
            # terminal transition -> to absorbing state
            s_next = absorbing_states[0] if reward > 0 else absorbing_states[1]
        else:
            next_row = arr[i+1]
            if next_row[1] != row[1]:
                s_next = absorbing_states[1] 
            else:
                s_next = int(next_row[2])
        transition_counts[s_next, s, a] += 1

    # # Convert counts to probabilities P(s'|s,a)
    transitionr = np.divide(transition_counts, transition_counts.sum(axis=0, keepdims=True),
                            where=transition_counts.sum(axis=0, keepdims=True) > 0)

    # # Physician policy π_physician(a|s)
    action_counts = transition_counts.sum(axis=0)  # sum over s'
    physpol = np.divide(action_counts, action_counts.sum(axis=1, keepdims=True),
                        where=action_counts.sum(axis=1, keepdims=True) > 0)

    # # Expected reward R[s,a]
    transition_rewards = np.zeros((n_states, n_states, n_actions))
    for i,row in enumerate(arr):
        s = int(row[2])
        a = int(row[3])
        outcome = int(row[4])
        reward = row[5]
        if outcome == 1:
            # terminal transition -> to absorbing state
            s_next = absorbing_states[0] if reward > 0 else absorbing_states[1]
        else:
            next_row = arr[i+1]
            if next_row[1] != row[1]:
                s_next = absorbing_states[1] 
            else:
                s_next = int(next_row[2])
        transition_rewards[s_next,s,a] += reward

    transition_rewards_mean = np.divide(
        transition_rewards, 
        transition_counts, 
        out=np.zeros_like(transition_rewards), 
        where=transition_counts>0
    )

    # Then expected R[s,a] = sum_{s'} P(s'|s,a) * avg_reward[s',s,a]
    R = (transitionr * transition_rewards_mean).sum(axis=0)


    if concept_list is None:
        concept_list = [mimic_concept(i) for i in range(47)]
    modified_concept_list = [lambda s, concept=concept: concept(centers[s]) 
                        for concept in concept_list]

    observation_space = spaces.Box(0,1, shape=(len(concept_list),))
    action_space = spaces.Discrete(25)
    rewards = R
    transitions = transitionr.transpose((1,2,0)) 
    max_steps = 10000
    all_states = list(range(n_states)) 

    done_map = lambda s: s in [n_cluster_states,n_cluster_states+1]
    state_distro = state_distro
    env = ConceptEnv(modified_concept_list,observation_space,action_space,rewards,transitions,all_states,max_steps,state_distro=state_distro,done_map=done_map)
    env = Monitor(env)
    return physpol, env, centers, modified_concept_list, clusterer

def get_raw_pixels_cartpole(env, obs=None):
    pixels = env.render()
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    small_pixels = cv2.resize(gray, (84,84), interpolation=cv2.INTER_NEAREST).astype(float)/255
    return small_pixels

def get_raw_pixels_mini_grid(env,obs=None):
    pixels = env.render()[:160,:160]
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    small_pixels = cv2.resize(gray, (84,84), interpolation=cv2.INTER_NEAREST)
    return small_pixels

def can_move(position, direction, grid):
    """
    Helper function to determine if movement in a specific direction is possible.
    """
    next_pos = DIR_TO_VEC[direction]
    if 1 <= position[0] + next_pos[0] < grid.width-1 and 1 <= position[1] + next_pos[1] < grid.height-1:
        next_cell = grid.get(position[0] + next_pos[0], position[1] + next_pos[1])
        return next_cell is None or next_cell.can_overlap()
    else:
        return False  # Out of bounds

def get_raw_state_mini_grid(env,obs,info):
    agent_pos = env.unwrapped.agent_pos
    agent_dir = env.unwrapped.agent_dir
    grid = env.unwrapped.grid
    key_pos = (0, 0) # default one if not found (carrying)
    door_pos = None
    door_open = False

    # Locate door, key, and goal positions
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None:
                if cell.type == 'door':
                    door_pos = (x, y)
                    door_open = cell.is_open  # Check if the door is open
                elif cell.type == 'key':
                    key_pos = (x, y)

    # Check direction_movable in all four directions
    direction_movable = {
        'right': can_move(agent_pos, 0, grid),
        'down': can_move(agent_pos, 1, grid),
        'left': can_move(agent_pos, 2, grid),
        'up': can_move(agent_pos, 3, grid),
    }

    infos = {
        'agent_position': agent_pos,
        'agent_direction': agent_dir,
        'key_position': key_pos,
        'door_position': door_pos,
        'door_open': door_open,  # Add door_open to infos
        'direction_movable': direction_movable
    }

    vec = [agent_pos[0],agent_pos[1],agent_dir,key_pos[0],key_pos[1],door_pos[0],door_pos[1],int(door_open)]+[int(direction_movable[i]) for i in ['right','down','left','up']]

    return vec 


def get_raw_state_cartpole(env,obs):
    """Get the raw underlying state in a CartPole environment
    Arguments:
        env: CartPole environment
        obs: Current observation, 4-vector
    
    Returns: The same 4-vector observation"""

    return obs 

def get_raw_state_atari(env,obs):
    """Get the raw underlying state in an Atari environment
    
    Arguments:
        env: CartPole environment
        obs: Current observation, 4-vector
    
    Returns: A 1x84x84 numpy array representing the screen"""

    pixels = env.ale.getScreenGrayscale().astype(np.uint8)
    return cv2.resize(pixels, (84,84),interpolation=cv2.INTER_NEAREST)

def make_ocenv(env_name,concept_list,observation_space,seed=0,recordable=False,num_stack=4):
    """Create an OCAtari environment for a given concept list
    
    Arguments:
        env_name: String for the Atari environment
        concept_list: List of functions mapping
            internal Atari states to some number
        observation_space: Type of environment observation
    
    Returns: Gym Environment wrapped with Concepts"""
    if recordable:
        env = ocatari.OCAtari(
            env_name,
            mode="ram",
            render_mode="rgb_array",
            frameskip=1,  # Keep frameskip=1 so consecutive frames actually differ
            difficulty=0  # easiest opponent
        )
    else:
        env = ocatari.OCAtari(
            env_name,
            mode="ram",
            render_mode=None, 
            frameskip=2,  # Keep frameskip=1 so consecutive frames actually differ
            difficulty=0  # easiest opponent
        )
    env.ale = env._ale 
    env = Monitor(env)

    # Apply ConceptWrapper (assuming it returns the pixel observations)
    env = ConceptWrapper(env, concept_list, observation_space, get_raw_state_atari)
    if concept_list is None:
        env = FrameStack(env,num_stack)
        env = LazyFramesToNumpy(env)
    env.reset(seed=seed)
    return env

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
    Only renders pixels every `skip` frames, repeats last observation otherwise.
    """
    def __init__(self, env, skip=2, get_pixels_fn=None):
        super().__init__(env)
        self.skip = skip
        self.get_pixels = get_pixels_fn
        self._last_obs = None
        self._frame_counter = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['observation'] = obs
        self._frame_counter = 0
        self._last_obs = self.get_pixels(self.env, obs)
        return self._last_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['observation'] = obs
        self._frame_counter += 1
        if self._frame_counter % self.skip == 0:
            self._last_obs = self.get_pixels(self.env, obs)
        return self._last_obs, reward, terminated, truncated, info

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

class ActionMaskWrapper(gym.Wrapper):
    """
    Restrict action space to a subset of original actions.
    For Pong: keep only [UP, DOWN] (originally 2 and 3).
    """

    def __init__(self, env, allowed_actions=(2, 3)):
        super().__init__(env)
        self.allowed_actions = allowed_actions
        self.action_space = spaces.Discrete(len(allowed_actions))

    def step(self, action):
        real_action = self.allowed_actions[action]
        ret = self.env.step(real_action)
        return ret

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def get_n_atari_env(n_envs,atari_env_name,concept_list,observation_space,recordable=False,num_stack=4):
    """Create a series of parallel Atari environments 
    
    Arguments:
        n_envs: Integer, number of parallel environments
        atari_env_name: String, which Atari environment we're using
        concept_list: List of concepts which we're using for mapping
        observation_space: Type of environment observation
    
    Returns: SubprocVecEnv with all the environments"""
    def safe_make_env(seed):
        try:
            return make_ocenv(atari_env_name, concept_list, observation_space, seed=seed,num_stack=num_stack)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    if concept_list is None:
        vec_env = DummyVecEnv([
            lambda seed=i: safe_make_env(seed=seed)
            for i in range(n_envs)
        ])#, start_method='spawn')
    else:
        vec_env = SubprocVecEnv([
            lambda seed=i: safe_make_env(seed=seed)
            for i in range(n_envs)
        ], start_method='spawn')
    return vec_env



def get_binary_subset_env(golden_model, env, indices,accuracies=None):
    """
    Minimal function to get an environment that returns binary features at specified indices
    
    Args:
        golden_model: Your trained stable_baselines model
        env: Original CartPole environment  
        indices: List of indices to select from the binary features (e.g., [0, 3, 7, 12])
    
    Returns:
        subset_env: Environment that returns only the selected binary features
    """
    extractor = CartPoleBinaryFeatureExtractor(percentiles=[20, 40, 60, 80])
    extractor.fit_thresholds(golden_model, env, n_samples=5000)
    binary_env = BinaryFeatureEnvironmentWrapper(env, extractor)
    subset_env = BinaryObservationSubsetWrapper(binary_env, indices,accuracies=accuracies)    
    return subset_env

def get_environment(environment_string,concept_list,seed,use_processed=False,fast_predictor=None):
    """Get a specific environment based on a string + concept list
    
    Arguments:
        environment_string: String, mapping to one environment
        concept_list: List of functions mapping state -> concept, or None
    
    Returns: Gymasium environment, and a dictionary of additional information"""

    additional_info = {}
    num_envs = 8
    num_stack = 4

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if "cyclic" in environment_string:
        def make_env():
            num_nodes = int(environment_string.split("_")[-1])
            env = create_cyclic_env(num_nodes,concept_list)
            return env 
        vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])
        gymnasium_env = GymnasiumWrapper(vec_env)
    elif "tree" in environment_string:
        def make_env():
            num_nodes = int(environment_string.split("_")[-1])
            env = create_tree_env(num_nodes,concept_list)
            return env 
        vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])
        gymnasium_env = GymnasiumWrapper(vec_env)
    elif environment_string == "mimic":
        physpol, vec_env, centers,new_concept_list, clusterer = create_mimic_environment(concept_list,seed)
        gymnasium_env = vec_env
        additional_info = {'physpol': physpol, 'centers': centers, 'concept_list': new_concept_list, 'clusterer': clusterer}
    elif environment_string  == "mini_grid":
        def make_env():
            if concept_list is None:
                env = gym.make("MiniGrid-DoorKey-5x5-v0",render_mode="rgb_array")
                env = FrameSkipWrapper(env, skip=4, get_pixels_fn=get_raw_pixels_mini_grid)
                env = ConceptWrapper(env,None,spaces.Box(
                        low=0, high=255,
                        shape=(84,84),  # Height x Width, no color channel
                        dtype=np.uint8
                    ),lambda env, obs: obs, obs_function=lambda e,o,i: get_raw_state_mini_grid(e,o,i))
                env = FrameStack(env,num_stack)
                env = LazyFramesToNumpy(env)

            else:
                env = gym.make("MiniGrid-DoorKey-5x5-v0")
                env = ConceptWrapper(env,concept_list,spaces.MultiBinary(len(concept_list)),lambda env, obs: obs, obs_function=lambda env, obs, info: get_raw_state_mini_grid(env,info,{'observation': obs}))
            return env 

        vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])
        gymnasium_env = GymnasiumWrapper(vec_env)

    elif environment_string == "cart_pole":
        def make_env():
            if concept_list is None or use_processed:
                env = gym.make("CartPole-v1", render_mode="rgb_array")
                env = FrameSkipWrapper(env, skip=4, get_pixels_fn=get_raw_pixels_cartpole)
                env = ConceptWrapper(env,None,spaces.Box(
                        low=0, high=255,
                        shape=(84,84),  # Height x Width, no color channel
                        dtype=np.uint8
                    ),lambda env, obs: obs,use_info_obs=True)
                env = FrameStack(env,1) # TODO: Change this back
                env = LazyFramesToNumpy(env)

                if use_processed:
                    env = OptimizedConceptWrapper(env, fast_predictor, spaces.MultiBinary(len(concept_list)), lambda env, obs: obs, use_info_obs=True)
            else:
                env = gym.make("CartPole-v1")
                env = ConceptWrapper(env,concept_list,spaces.MultiBinary(len(concept_list)),get_raw_state_cartpole)
            return env 

        vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])
        gymnasium_env = GymnasiumWrapper(vec_env)

    elif environment_string == "boxing":
        if concept_list is None:
            vec_env = get_n_atari_env(num_envs,"BoxingNoFrameskip-v4",None,gym.spaces.Box(
                    low=0, high=255,
                    shape=(84,84),  # Height x Width, no color channel
                    dtype=np.uint8
                ),num_stack=num_stack)
        else:
            vec_env = get_n_atari_env(num_envs,"BoxingNoFrameskip-v4",concept_list,spaces.Box(low=-255, high=255, shape=(len(concept_list),), dtype=np.float32))
        gymnasium_env = GymnasiumWrapper(vec_env)

    elif environment_string == "pong":
        if concept_list is None:
            vec_env = get_n_atari_env(num_envs,"PongNoFrameskip-v4",None,gym.spaces.Box(
                    low=0, high=255,
                    shape=(84,84),  # Height x Width, no color channel
                    dtype=np.uint8
                ),num_stack=num_stack)
        else:
            vec_env = get_n_atari_env(num_envs,"PongNoFrameskip-v4",concept_list,spaces.Box(low=-1, high=1, shape=(len(concept_list),), dtype=np.float32))
        # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        gymnasium_env = GymnasiumWrapper(vec_env)
    return vec_env, gymnasium_env, additional_info

def eval_mimic_model(physpol,model,concept_list,clusterer,seed):
    """Evaluate a MIMIC policy via the WIS score
    
    Arguments:
        physpol: Physicians policy, for reference
            Retrieved from compute_physician_policy
        model: Trained stable_baseline model for the environment
                concept_list: A list of a single concept
            that maps a state (represented by a vector)
            to a single integer number
            It needs to be in this form becuase we need to discretize
            the environment
        seed: Integer, random seed
    Returns: 
        Float, WIS score
    """

    MIMICraw = pd.read_csv("../../data/mimic_github/ai_clinician/data/mimic_model/train/MIMICraw.csv")
    metadata = pd.read_csv("../../data/mimic_github/ai_clinician/data/mimic_model/train/metadata.csv")
    MIMICzs = pd.read_csv("../../data/mimic_github/ai_clinician/data/mimic_model/train/MIMICzs.csv")

    C_ICUSTAYID = "icustayid"
    unique_icu_stays = metadata[C_ICUSTAYID].unique()

    n_action_bins = 5
    gamma = 0.99

    all_actions, _, _ = fit_action_bins(
        MIMICraw[C_INPUT_STEP],
        MIMICraw[C_MAX_DOSE_VASO],
        n_action_bins=n_action_bins
    )

    train_ids, val_ids = train_test_split(unique_icu_stays, test_size=0.1,random_state=seed)
    train_indexes = metadata[metadata[C_ICUSTAYID].isin(train_ids)].index
    val_indexes = metadata[metadata[C_ICUSTAYID].isin(val_ids)].index

    X_train = MIMICzs.iloc[train_indexes]
    X_val = MIMICzs.iloc[val_indexes]

    metadata_val = metadata.iloc[val_indexes]
    actions_val = all_actions[val_indexes]

    states_train = clusterer.predict(X_train.values)
    states_val = clusterer.predict(X_val.values)

    phys_probs = compute_physician_probabilities(physpol,np.max(states_train)+1,states=states_val, actions=actions_val)
    model_probs = compute_model_probabilities(model,concept_list,states=states_val, actions=actions_val)
    val_bootwis, _,  _ = evaluate_policy_wis(
        metadata_val,
        phys_probs,
        model_probs,
        [15,-15],
        gamma,
        200
    )

    return np.mean(val_bootwis)

