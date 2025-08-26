import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from skimage.transform import resize
import ocatari
import cv2 
from collections import Counter
from sklearn.model_selection import train_test_split

from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecEnvWrapper
from concept_abstraction.post_hoc import BinaryFeatureEnvironmentWrapper, CartPoleBinaryFeatureExtractor
from concept_abstraction.utils import one_hot_state
from concept_abstraction.mimic import *
from concept_abstraction.mimic import C_INPUT_STEP, C_MAX_DOSE_VASO

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
            return self.state 
        else:
            return np.array([concept(self.state) for concept in self.concept_list])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.choice(self.all_states,p=self.state_distro)
        self.steps = 0
        return self.get_observation(), {}

    def step(self, action):
        reward = self.rewards[self.state][action]
        if np.sum(self.transitions[self.state][action]) == 0:
            reward = -10
        else:
            self.state = np.random.choice(self.all_states, p=self.transitions[self.state][action])        
            self.steps += 1
        obs = self.get_observation()

        done = self.steps >= self.max_steps or self.done_map(self.state)
        return obs, reward, done, False, {}

    def render(self):
        pass 

    def close(self):
        pass

class ConceptWrapper(gym.ObservationWrapper):
    def __init__(self, env,concept_list,observation_space,get_raw_state):
        super().__init__(env)
        self.observation_space = observation_space
        self.concept_list = concept_list 
        self.get_raw_state = get_raw_state

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process(obs), info

    def observation(self, obs):
        processed = self._process(obs)
        return processed

    def _process(self, obs):
        if self.concept_list is None:
            return self.get_raw_state(self,obs)
        else:
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
                    transitions[state][action][0] = 1
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

    states_train = np.array([concept_list[0](i) for i in X_train.values])

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

    observation_space = spaces.Box(0,1, shape=(n_states,))
    action_space = spaces.Discrete(25)
    rewards = R
    transitions = transitionr.transpose((1,2,0)) 
    max_steps = 10000
    all_states = list(range(n_states)) 
    
    done_map = lambda s: s in [n_cluster_states,n_cluster_states+1]
    state_distro = state_distro
    env = ConceptEnv([lambda s: one_hot_state(s,n_states)],observation_space,action_space,rewards,transitions,all_states,max_steps,state_distro=state_distro,done_map=done_map)
    return physpol, env 

def eval_mimic_model(physpol,model,concept_list,seed):
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

    states_train = np.array([concept_list[0](i) for i in X_train.values])
    states_val = np.array([concept_list[0](i) for i in X_val.values])

    n_states = np.max(states_train)+3

    phys_probs = compute_physician_probabilities(physpol,np.max(states_train)+1,states=states_val, actions=actions_val)
    model_probs = compute_model_probabilities(model,[lambda s: one_hot_state(s,n_states)],states=states_val, actions=actions_val)
    val_bootwis, _,  _ = evaluate_policy_wis(
        metadata_val,
        phys_probs,
        model_probs,
        [100,-100],
        gamma,
        200
    )

    return np.quantile(val_bootwis, 0.05)


def get_raw_pixels_cartpole(env, obs=None):
    """In CartPole Environment, return the Greyscale pixels
    
    Arguments:
        env: CartPole environment
    
    Returns: Numpy array of size 1x84x84"""

    pixels = env.render()  # RGB uint8
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)  # still uint8
    small_pixels = resize(gray, (84, 84), anti_aliasing=True)  # float 0-1
    return (small_pixels * 255).astype(np.uint8).reshape((1, 84, 84))

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
    small_pixels = resize(pixels, (84, 84), anti_aliasing=True)
    return (small_pixels*255).astype(np.uint8).reshape((1,84,84))

def make_ocenv(env_name,concept_list,observation_space,seed=0):
    """Create an OCAtari environment for a given concept list
    
    Arguments:
        env_name: String for the Atari environment
        concept_list: List of functions mapping
            internal Atari states to some number
        observation_space: Type of environment observation
    
    Returns: Gym Environment wrapped with Concepts"""
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
    env.reset(seed=seed)
    return env

def get_n_atari_env(n_envs,atari_env_name,concept_list,observation_space):
    """Create a series of parallel Atari environments 
    
    Arguments:
        n_envs: Integer, number of parallel environments
        atari_env_name: String, which Atari environment we're using
        concept_list: List of concepts which we're using for mapping
        observation_space: Type of environment observation
    
    Returns: SubprocVecEnv with all the environments"""
    
    vec_env = SubprocVecEnv([
        lambda seed=i: make_ocenv(atari_env_name,concept_list,observation_space,seed=seed) for i in range(n_envs)
    ], start_method='spawn')

    if concept_list is None:
        vec_env = VecFrameStack(vec_env, n_stack=4)
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

