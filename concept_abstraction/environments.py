import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from concept_abstraction.post_hoc import BinaryFeatureEnvironmentWrapper, CartPoleBinaryFeatureExtractor


class Cyclic4StateEnv(gym.Env):
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

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self,environment_nodes,concept_list=[],acc_by_concept=None):
        super().__init__()
        self.environment_nodes = environment_nodes 
        self.concept_list = sorted(concept_list)
        self.acc_by_concept = acc_by_concept 

        self.observation_space = spaces.MultiBinary(len(self.concept_list))
        self.action_space = spaces.Discrete(3)
        self.all_states = list(range(environment_nodes))
        self.max_steps = 20
        self.state = np.random.randint(0,self.environment_nodes)
        self.steps = 0

        self.rewards = np.zeros((environment_nodes,3)) 
        for i in range(environment_nodes):
            if i%2 == 0:
                self.rewards[i,0] = self.rewards[i,1] = 1
            else:
                self.rewards[i,2] = 1
        self.rewards = np.array(self.rewards)

        self.transitions = []
        for i in range(len(self.all_states)):
            transitions_by_state = []
            for action in range(self.action_space.n):
                next_probs = [0.0 for i in range(len(self.all_states))]
                if action == 0:
                    next_probs[(i - 1) % self.environment_nodes] = 1.0
                if action == 1:
                    next_probs[(i + 1) % self.environment_nodes] = 1.0
                if action == 2:
                    next_probs[(i) % self.environment_nodes] = 1.0
                transitions_by_state.append(next_probs)
            self.transitions.append(transitions_by_state)
        self.transitions = np.array(self.transitions)

        self.concepts = []
        for i in range(2,environment_nodes+1):
            concept_vals = [int((j+1)%i == 0) for j in range(environment_nodes)]
            self.concepts.append(concept_vals)
        self.concepts = np.array(self.concepts)
        
    def get_observation(self):
        current_concepts = self.concepts[self.concept_list,self.state].copy()
        for i in range(len(current_concepts)):
            concept_num = self.concept_list[i]
            if self.acc_by_concept is not None and np.random.random() > self.acc_by_concept[concept_num]:
                current_concepts[i] = 1-current_concepts[i]         
        return current_concepts 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.randint(0,self.environment_nodes)
        self.steps = 0
        return self.get_observation(), {}

    def step(self, action):
        # Reward
        reward = self.rewards[self.state][action]

        # State 
        next_state_probs = self.transitions[self.state][action]
        self.state = np.random.choice(self.all_states, p=next_state_probs)        
        
        # Observation
        obs = self.get_observation()

        # Termination
        self.steps += 1
        done = self.steps >= self.max_steps
        return obs, reward, done, False, {}

    def render(self):
        pass 

    def close(self):
        pass


class TreeRepeatEnv(gym.Env):
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


    def __init__(self,environment_nodes,concept_list=[],acc_by_concept=None):
        super().__init__()
        self.environment_nodes = environment_nodes 
        self.concept_list = sorted(concept_list)
        self.acc_by_concept = acc_by_concept 

        self.num_layers = int(np.log2(self.environment_nodes+1))
        self.all_states = list(range(environment_nodes))
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.MultiBinary(len(self.concept_list))
        self.steps = 0
        self.state = 0
        self.max_steps = 20

        self.rewards = np.zeros((self.environment_nodes,self.action_space.n))
        for i in range(self.num_layers):
            self.rewards[2**i-1][0] = 1
        self.rewards[:,1] = 0.5

        self.transitions = np.zeros((len(self.all_states),
                                    self.action_space.n,
                                    len(self.all_states)))
        for state in range(len(self.transitions)):
            for action in range(len(self.transitions[state])):
                if state >= self.environment_nodes//2:
                    if state == self.environment_nodes//2:
                        self.transitions[state][action][0] = 1
                    else:
                        self.transitions[state][action][2] = 1
                else:
                    self.transitions[state][action][2 * (state+1) + action - 1] = 1

        self.concepts = []
        for i in range(self.num_layers):
            curr_concept = []
            for state in range(1,self.environment_nodes+1):
                binary_rep = bin(state)[2:]
                binary_rep = '0'*(self.num_layers-len(binary_rep)) + binary_rep
                curr_concept.append(int(binary_rep[i]))
            self.concepts.append(curr_concept)
        final_concept = [0 for i in range(2**self.num_layers-1)]
        for i in range(self.num_layers):
            final_concept[2**i-1] = 1
        self.concepts.append(final_concept)
        self.concepts = np.array(self.concepts)

    def get_observation(self):
        current_concepts = self.concepts[self.concept_list,self.state].copy()

        for i in range(len(current_concepts)):
            concept_num = self.concept_list[i]
            if self.acc_by_concept is not None and np.random.random() > self.acc_by_concept[concept_num]:
                current_concepts[i] = 1-current_concepts[i]         
        return current_concepts 

    def reset(self, seed=None,options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.state = 0
        return self.get_observation(), {}

    def step(self, action): 
        # Reward 
        reward = self.rewards[self.state][action]

        # State 
        next_state_probs = self.transitions[self.state][action]
        self.state = np.random.choice(self.all_states,p=next_state_probs)

        # Observation
        obs = self.get_observation()

        # Termination
        self.steps += 1
        done = self.steps >= self.max_steps  

        return obs, reward, done, False, {}

    def render(self):
        pass 

    def close(self):
        pass

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

def get_custom_binary_features(observation):
    """Given an observation (4 vector), convert
        this to a 13-length binary vector
        suggested by LLMs
    
    Arguments:
        observation: 4-vector of floats
            representing the position/velocity
            of the cart
    
    Returns: 13-length binary vector"""

    cart_pos, cart_vel, pole_angle, pole_ang_vel = observation
    binary_features = np.zeros(13, dtype=np.int32)
    binary_features[0] = int(abs(cart_pos) < 0.5)          # cart near center
    binary_features[1] = int(abs(cart_pos) >= 0.5)         # cart far from center
    binary_features[2] = int(cart_vel < 0)                 # cart moving left
    binary_features[3] = int(cart_vel > 0)                 # cart moving right
    binary_features[4] = int(abs(cart_vel) > 1.0)          # cart moving fast
    binary_features[5] = int(pole_angle < 0)               # pole leaning left
    binary_features[6] = int(pole_angle > 0)               # pole leaning right
    binary_features[7] = int(abs(pole_angle) < 0.02)       # pole near vertical
    binary_features[8] = int(abs(pole_angle) >= 0.1)       # pole far from vertical
    binary_features[9] = int(abs(pole_angle) >= 0.2)       # pole about to fall
    binary_features[10] = int(pole_ang_vel < 0)            # pole rotating clockwise
    binary_features[11] = int(pole_ang_vel > 0)            # pole rotating counterclockwise
    binary_features[12] = int(abs(pole_ang_vel) > 1.0)     # pole rotating fast
    return binary_features

class CustomBinaryFeatureWrapper(gym.ObservationWrapper):
    """
    Wrapper that converts CartPole observations to custom binary features
        These observations come from LLMs

    Features:
    0: Is cart near center? → |cart position| < 0.5
    1: Is cart far from center? → |cart position| ≥ 0.5
    2: Is cart moving left? → cart velocity < 0
    3: Is cart moving right? → cart velocity > 0
    4: Is cart moving fast? → |cart velocity| > 1.0
    5: Is pole leaning left? → pole angle < 0
    6: Is pole leaning right? → pole angle > 0
    7: Is pole near vertical? → |pole angle| < 0.02
    8: Is pole far from vertical? → |pole angle| ≥ 0.1
    9: Is pole about to fall? → |pole angle| ≥ 0.2
    10: Is pole rotating clockwise? → pole angular velocity < 0
    11: Is pole rotating counterclockwise? → pole angular velocity > 0
    12: Is pole rotating fast? → |pole angular velocity| > 1.0
    """
    
    def __init__(self, env,accuracies=None):
        super().__init__(env)
        self.observation_space = gym.spaces.MultiBinary(13)  # 13 binary features
        self.accuracies = accuracies

        self.feature_names = [
            "cart_near_center",      # 0
            "cart_far_from_center",  # 1
            "cart_moving_left",      # 2
            "cart_moving_right",     # 3
            "cart_moving_fast",      # 4
            "pole_leaning_left",     # 5
            "pole_leaning_right",    # 6
            "pole_near_vertical",    # 7
            "pole_far_from_vertical", # 8
            "pole_about_to_fall",    # 9
            "pole_rotating_clockwise", # 10
            "pole_rotating_ccw",     # 11
            "pole_rotating_fast"     # 12
        ]
    
    def observation(self, observation):
        """Convert continuous observation to binary features"""
        binary_features = get_custom_binary_features(observation)
        if self.accuracies is not None: 
            for i in range(len(self.accuracies)):
                if np.random.random() > self.accuracies[i]:
                    binary_features[i] = 1-binary_features[i]

        return binary_features
    
    def get_feature_names(self):
        """Return list of feature names"""
        return self.feature_names.copy()
    
    def print_observation(self, observation):
        """Print human-readable observation"""
        binary_obs = self.observation(observation)
        print("Binary Features:")
        for i, (name, value) in enumerate(zip(self.feature_names, binary_obs)):
            if value:
                print(f"  {i:2d}: {name} = True")

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
