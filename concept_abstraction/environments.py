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

from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv, VecNormalize
from concept_abstraction.post_hoc import BinaryFeatureEnvironmentWrapper, CartPoleBinaryFeatureExtractor
from concept_abstraction.utils import one_hot_state
from concept_abstraction.mimic import *
from concept_abstraction.concept_bank import clustering_concept_mimic, mimic_concept
import minigrid

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
            reward = -10
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

    cluster_concept, centers = clustering_concept_mimic(X_train.values,N_CLUSTERS,seed)
    zeros = np.zeros((2, centers.shape[1]))
    centers = np.vstack([centers, zeros])

    states_train = np.array([cluster_concept(i) for i in X_train.values])

    n_cluster_states = np.max(states_train)+1
    absorbing_states =  [n_cluster_states + 1, n_cluster_states]
    rewards = [100, -100]
    # Create qldata3
    qldata3 = build_complete_record_sequences(
        metadata_train,
        states_train,
        actions_train,
        absorbing_states,
        rewards
    )
    n_states = n_cluster_states + 2
    reward_val = 100
    transition_threshold = 5

    d = Counter(states_train)
    state_distro = np.array([d[i] for i in range(np.max(states_train)+1)])
    state_distro = state_distro / np.sum(state_distro)
    state_distro = np.append(state_distro,0)
    state_distro = np.append(state_distro,0)

    ####### BUILD MODEL ########
    physpol, transitionr, R = compute_physician_policy(
        qldata3,
        n_states,
        n_actions,
        absorbing_states,
        reward_val=reward_val,
        transition_threshold=transition_threshold
    )

    if concept_list == None:
        concept_list = [mimic_concept(i) for i in range(47)]
        modified_concept_list = [lambda s, concept=concept: concept(centers[s]) 
                            for concept in concept_list]
    else:
        modified_concept_list = concept_list

    observation_space = spaces.Box(0,1, shape=(len(concept_list),))
    action_space = spaces.Discrete(25)
    rewards = R
    transitions = transitionr.transpose((1,2,0)) 
    max_steps = 10000
    all_states = list(range(n_states)) 

    done_map = lambda s: s in [n_cluster_states,n_cluster_states+1]
    state_distro = state_distro
    env = ConceptEnv(modified_concept_list,observation_space,action_space,rewards,transitions,all_states,max_steps,state_distro=state_distro,done_map=done_map)
    return physpol, env, cluster_concept, modified_concept_list

def eval_mimic_model(physpol,model,cluster_concept,concept_list,seed):
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

    states_train = np.array([cluster_concept(i) for i in X_train.values])
    states_val = np.array([cluster_concept(i) for i in X_val.values])

    phys_probs = compute_physician_probabilities(physpol,np.max(states_train)+1,states=states_val, actions=actions_val)
    model_probs = compute_model_probabilities(model,concept_list,states=states_val, actions=actions_val)
    val_bootwis, _,  _ = evaluate_policy_wis(
        metadata_val,
        phys_probs,
        model_probs,
        [100,-100],
        gamma,
        200
    )

    return np.mean(val_bootwis)


small_pixels = np.empty((84,84), dtype=np.uint8)

def get_raw_pixels_cartpole(env, obs=None):
    pixels = env.render()
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    cv2.resize(gray, (84,84), dst=small_pixels[0], interpolation=cv2.INTER_NEAREST)
    return small_pixels

def get_raw_state_mini_grid(env,obs,info):
    """In mini_grid Environment, return the pixels
    
    Arguments:
        env: CartPole environment
    
    Returns: Numpy array of size 1x84x84"""

    obs_ch_first = np.transpose(info['observation']['image'], (2, 0, 1))  # 3x7x7
    obs_ch_first = obs_ch_first.flatten()

    obs_ch_first = np.append(obs_ch_first,np.array([env.unwrapped.agent_pos[0],env.unwrapped.agent_pos[1],env.unwrapped.agent_dir]))
    return obs_ch_first

def get_raw_state_cartpole(env,obs):
    """Get the raw underlying state in a CartPole environment
    Arguments:
        env: CartPole environment
        obs: Current observation, 4-vector
    
    Returns: The same 4-vector observation"""

    return obs 

small_pixels = np.empty((84,84), dtype=np.uint8)

def get_raw_state_atari(env,obs):
    """Get the raw underlying state in an Atari environment
    
    Arguments:
        env: CartPole environment
        obs: Current observation, 4-vector
    
    Returns: A 1x84x84 numpy array representing the screen"""

    pixels = env.ale.getScreenGrayscale().astype(np.uint8)
    cv2.resize(pixels, (84,84), dst=small_pixels[0], interpolation=cv2.INTER_NEAREST)
    return small_pixels

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
            frameskip=1,  # Keep frameskip=1 so consecutive frames actually differ
            difficulty=0  # easiest opponent
        )
    env.ale = env._ale 
    
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

def get_environment(environment_string,concept_list,seed):
    """Get a specific environment based on a string + concept list
    
    Arguments:
        environment_string: String, mapping to one environment
        concept_list: List of functions mapping state -> concept, or None
    
    Returns: Gymasium environment, and a dictionary of additional information"""

    additional_info = {}
    num_envs = 4
    num_stack = 4

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
        # TODO: Fix MIMIC
        physpol, env, cluster_concept,new_concept_list = create_mimic_environment(concept_list,seed)
        gymnasium_env = env
        additional_info = {'physpol': physpol, 'cluster_concept': cluster_concept, 'concept_list': new_concept_list}
    elif environment_string  == "mini_grid":
        def make_env():
            if concept_list is None:
                env = gym.make("MiniGrid-DoorKey-5x5-v0",render_mode="rgb_array")
                env = FrameSkipWrapper(env, skip=4, get_pixels_fn=get_raw_pixels_cartpole)
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
            if concept_list is None:
                env = gym.make("CartPole-v1", render_mode="rgb_array")
                env = FrameSkipWrapper(env, skip=4, get_pixels_fn=get_raw_pixels_cartpole)
                env = ConceptWrapper(env,None,spaces.Box(
                        low=0, high=255,
                        shape=(84,84),  # Height x Width, no color channel
                        dtype=np.uint8
                    ),lambda env, obs: obs,use_info_obs=True)
                env = FrameStack(env,num_stack)
                env = LazyFramesToNumpy(env)
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
            vec_env = get_n_atari_env(num_envs,"PongNoFrameskip-v4",concept_list,spaces.Box(low=-255, high=255, shape=(len(concept_list),), dtype=np.float32))
        gymnasium_env = GymnasiumWrapper(vec_env)
    return vec_env, gymnasium_env, additional_info
