import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.feature_selection import mutual_info_classif

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
    def __init__(self, env, indices):
        super().__init__(env)
        self.indices = indices
        original_space = env.observation_space

        # Define new observation space
        low = original_space.low[indices]
        high = original_space.high[indices]
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        return observation[self.indices]

class DiscretizeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, bins_per_feature=4):
        super().__init__(env)
        self.bins_per_feature = bins_per_feature
        self.n_features = env.observation_space.shape[0]

        # Define bin edges for each feature (replace inf with reasonable bounds)
        self.bin_edges = [
            np.linspace(-4.8, 4.8, bins_per_feature + 1),         # Cart position
            np.linspace(-3.0, 3.0, bins_per_feature + 1),         # Cart velocity
            np.linspace(-0.418, 0.418, bins_per_feature + 1),     # Pole angle
            np.linspace(-3.5, 3.5, bins_per_feature + 1)          # Pole angular velocity
        ]

        # New observation space is a flat MultiBinary vector
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
    def __init__(self, env, indices):
        super().__init__(env)
        self.indices = indices

        if not isinstance(env.observation_space, gym.spaces.MultiBinary):
            raise ValueError("BinaryObservationSubsetWrapper requires MultiBinary observation space.")

        orig_n = env.observation_space.n
        if max(indices) >= orig_n:
            raise ValueError("Subset indices exceed original observation length.")

        # Define new observation space
        self.observation_space = gym.spaces.MultiBinary(len(indices))

    def observation(self, observation):
        return observation[self.indices]

class CustomBinaryFeatureWrapper(gym.ObservationWrapper):
    """
    Wrapper that converts CartPole observations to custom binary features
    
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
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.MultiBinary(13)  # 13 binary features
        
        # Feature names for reference
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
        cart_pos, cart_vel, pole_angle, pole_ang_vel = observation
        
        binary_features = np.zeros(13, dtype=np.int32)
        
        # Cart position features
        binary_features[0] = int(abs(cart_pos) < 0.5)          # cart near center
        binary_features[1] = int(abs(cart_pos) >= 0.5)         # cart far from center
        
        # Cart velocity features
        binary_features[2] = int(cart_vel < 0)                 # cart moving left
        binary_features[3] = int(cart_vel > 0)                 # cart moving right
        binary_features[4] = int(abs(cart_vel) > 1.0)          # cart moving fast
        
        # Pole angle features
        binary_features[5] = int(pole_angle < 0)               # pole leaning left
        binary_features[6] = int(pole_angle > 0)               # pole leaning right
        binary_features[7] = int(abs(pole_angle) < 0.02)       # pole near vertical
        binary_features[8] = int(abs(pole_angle) >= 0.1)       # pole far from vertical
        binary_features[9] = int(abs(pole_angle) >= 0.2)       # pole about to fall
        
        # Pole angular velocity features
        binary_features[10] = int(pole_ang_vel < 0)            # pole rotating clockwise
        binary_features[11] = int(pole_ang_vel > 0)            # pole rotating counterclockwise
        binary_features[12] = int(abs(pole_ang_vel) > 1.0)     # pole rotating fast
        
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

def get_recordable(env):
    """Add recording and save the recording to logs/videos
    
    Arguments:
        env: Gymnasium environment
    
    Returns: Gymnasium Environment
    
    Side Effects: Wraps the environment for video recording"""

    env = gym.wrappers.RecordVideo(env, video_folder="../../runs/videos/", episode_trigger=lambda e: True)
    return env 


class BinaryObservationSubsetWrapper(gym.ObservationWrapper):
    def __init__(self, env, indices):
        super().__init__(env)
        self.indices = indices
        if not isinstance(env.observation_space, gym.spaces.MultiBinary):
            raise ValueError("BinaryObservationSubsetWrapper requires MultiBinary observation space.")
        orig_n = env.observation_space.n
        if max(indices) >= orig_n:
            raise ValueError("Subset indices exceed original observation length.")
        # Define new observation space
        self.observation_space = gym.spaces.MultiBinary(len(indices))
    
    def observation(self, observation):
        return observation[self.indices]

class CartPoleBinaryFeatureExtractor:
    def __init__(self, percentiles=[20, 40, 60, 80]):
        self.percentiles = percentiles
        self.feature_names = ["cart_pos", "cart_vel", "pole_angle", "pole_vel"]
        self.thresholds = {}  # Will store {feature_idx: [thresholds]}
        self.binary_feature_names = []
        self.feature_rankings = None
        
    def _generate_training_data(self, golden_model, env, n_samples=10000):
        """Generate realistic training data from golden model"""
        states = []
        actions = []
        
        # Generate data from actual episodes
        samples_collected = 0
        while samples_collected < n_samples:
            state, _ = env.reset()
            
            for _ in range(200):  # Max episode length
                action, _ = golden_model.predict(state, deterministic=True)
                states.append(state.copy())
                actions.append(action)
                samples_collected += 1
                
                if samples_collected >= n_samples:
                    break
                    
                state, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
                    
        return np.array(states[:n_samples]), np.array(actions[:n_samples]).flatten()
    
    def fit_thresholds(self, golden_model, env, n_samples=10000):
        """Learn thresholds for each feature based on training data"""
        print("Generating training data for threshold learning...")
        states, actions = self._generate_training_data(golden_model, env, n_samples)
        
        print(f"State ranges:")
        for i, name in enumerate(self.feature_names):
            min_val, max_val = states[:, i].min(), states[:, i].max()
            print(f"  {name}: [{min_val:.3f}, {max_val:.3f}]")
        
        # Calculate thresholds for each feature
        for feature_idx in range(len(self.feature_names)):
            feature_values = states[:, feature_idx]
            thresholds = np.percentile(feature_values, self.percentiles)
            self.thresholds[feature_idx] = thresholds
            
            print(f"\n{self.feature_names[feature_idx]} thresholds:")
            for i, (percentile, threshold) in enumerate(zip(self.percentiles, thresholds)):
                feature_name = f"{self.feature_names[feature_idx]}_p{percentile}"
                self.binary_feature_names.append(feature_name)
                print(f"  {percentile}th percentile: {threshold:.3f}")
        
        return states, actions
    
    def convert_to_binary(self, states):
        """Convert continuous states to binary features"""
        if not self.thresholds:
            raise ValueError("Must call fit_thresholds first")
            
        n_samples = len(states)
        n_binary_features = len(self.binary_feature_names)
        binary_features = np.zeros((n_samples, n_binary_features), dtype=int)
        
        feature_idx = 0
        for orig_feature_idx in range(len(self.feature_names)):
            feature_values = states[:, orig_feature_idx]
            thresholds = self.thresholds[orig_feature_idx]
            
            for threshold in thresholds:
                binary_features[:, feature_idx] = (feature_values <= threshold).astype(int)
                feature_idx += 1
                
        return binary_features
    
    def rank_features(self, states, actions, method='mutual_info'):
        """Rank binary features by their predictive power"""
        binary_features = self.convert_to_binary(states)
        
        if method == 'mutual_info':
            # Use mutual information
            scores = mutual_info_classif(binary_features, actions, random_state=42)
        elif method == 'individual_accuracy':
            # Use individual feature accuracy
            scores = []
            for i in range(binary_features.shape[1]):
                feature = binary_features[:, i]
                # Predict majority class for each binary value
                pred_0 = np.bincount(actions[feature == 0]).argmax() if np.sum(feature == 0) > 0 else 0
                pred_1 = np.bincount(actions[feature == 1]).argmax() if np.sum(feature == 1) > 0 else 1
                
                predictions = np.where(feature == 0, pred_0, pred_1)
                accuracy = accuracy_score(actions, predictions)
                scores.append(accuracy)
            scores = np.array(scores)
        else:
            raise ValueError("Method must be 'mutual_info' or 'individual_accuracy'")
            
        # Create ranking dataframe
        ranking_data = []
        for i, (name, score) in enumerate(zip(self.binary_feature_names, scores)):
            orig_feature = "_p".join(name.split('_p')[0:-1])
            print(name)
            percentile = int(name.split('_p')[-1])
            ranking_data.append({
                'feature_idx': i,
                'feature_name': name,
                'original_feature': orig_feature,
                'percentile': percentile,
                'score': score
            })
        
        self.feature_rankings = pd.DataFrame(ranking_data).sort_values('score', ascending=False)
        return self.feature_rankings
    
    def get_top_features(self, n_features):
        """Get indices of top n features"""
        if self.feature_rankings is None:
            raise ValueError("Must call rank_features first")
        return self.feature_rankings.head(n_features)['feature_idx'].tolist()
    
    def print_feature_rankings(self, top_n=None):
        """Print feature rankings"""
        if self.feature_rankings is None:
            raise ValueError("Must call rank_features first")
            
        df_to_show = self.feature_rankings.head(top_n) if top_n else self.feature_rankings
        
        print("\nFeature Rankings:")
        print("=" * 60)
        for _, row in df_to_show.iterrows():
            print(f"{row['feature_name']:15} | Score: {row['score']:.4f} | "
                  f"Orig: {row['original_feature']:10} | Percentile: {row['percentile']:2d}")

class BinaryFeatureEnvironmentWrapper(gym.ObservationWrapper):
    """Wrapper that converts CartPole observations to binary features"""
    def __init__(self, env, feature_extractor):
        super().__init__(env)
        self.feature_extractor = feature_extractor
        
        if not feature_extractor.thresholds:
            raise ValueError("Feature extractor must be fitted first")
            
        n_binary_features = len(feature_extractor.binary_feature_names)
        self.observation_space = gym.spaces.MultiBinary(n_binary_features)
    
    def observation(self, observation):
        # Convert single observation to binary features
        observation_2d = observation.reshape(1, -1)
        binary_obs = self.feature_extractor.convert_to_binary(observation_2d)
        return binary_obs[0]  # Return 1D array

class SimpleDecisionTree:
    """Simple decision tree for binary features"""
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = {}
        self.feature_names = None
        
    def fit(self, binary_features, actions, feature_names=None):
        self.feature_names = feature_names
        self.tree = self._build_tree(binary_features, actions, depth=0)
        
    def _build_tree(self, features, actions, depth, used_features=None):
        if used_features is None:
            used_features = set()
            
        # Base cases
        if depth >= self.max_depth or len(np.unique(actions)) == 1 or len(actions) == 0:
            return int(np.bincount(actions).argmax()) if len(actions) > 0 else 0
            
        # Find best split
        best_feature = None
        best_score = -1
        
        for feature_idx in range(features.shape[1]):
            if feature_idx in used_features:
                continue
                
            feature_col = features[:, feature_idx]
            
            # Calculate accuracy for this binary split
            left_mask = feature_col == 0
            right_mask = feature_col == 1
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
                
            left_actions = actions[left_mask]
            right_actions = actions[right_mask]
            
            left_pred = np.bincount(left_actions).argmax()
            right_pred = np.bincount(right_actions).argmax()
            
            left_acc = np.mean(left_actions == left_pred)
            right_acc = np.mean(right_actions == right_pred)
            
            weighted_acc = (len(left_actions) * left_acc + len(right_actions) * right_acc) / len(actions)
            
            if weighted_acc > best_score:
                best_score = weighted_acc
                best_feature = feature_idx
        
        if best_feature is None:
            return int(np.bincount(actions).argmax())
            
        # Split data
        used_features.add(best_feature)
        feature_col = features[:, best_feature]
        left_mask = feature_col == 0
        right_mask = feature_col == 1
        
        return {
            'feature': best_feature,
            'left': self._build_tree(features[left_mask], actions[left_mask], depth + 1, used_features.copy()),
            'right': self._build_tree(features[right_mask], actions[right_mask], depth + 1, used_features.copy())
        }
    
    def predict(self, binary_features):
        if len(binary_features.shape) == 1:
            return self._predict_single(binary_features, self.tree)
        else:
            return np.array([self._predict_single(bf, self.tree) for bf in binary_features])
    
    def _predict_single(self, binary_feature, node):
        if isinstance(node, int):
            return node
            
        feature_idx = node['feature']
        if binary_feature[feature_idx] == 0:
            return self._predict_single(binary_feature, node['left'])
        else:
            return self._predict_single(binary_feature, node['right'])
    
    def print_tree(self, node=None, depth=0, prefix="Root"):
        if node is None:
            node = self.tree
            
        if isinstance(node, int):
            print(f"{'  ' * depth}{prefix}: Action {node}")
            return
            
        feature_name = self.feature_names[node['feature']] if self.feature_names else f"Feature_{node['feature']}"
        print(f"{'  ' * depth}{prefix}: {feature_name}")
        
        self.print_tree(node['left'], depth + 1, "├─ False")
        self.print_tree(node['right'], depth + 1, "└─ True")
def create_binary_feature_system(golden_model, env, percentiles=[20, 40, 60, 80]):
    """Complete workflow for binary feature extraction and ranking"""
    
    # Step 1: Create feature extractor and fit thresholds
    extractor = CartPoleBinaryFeatureExtractor(percentiles=percentiles)
    states, actions = extractor.fit_thresholds(golden_model, env, n_samples=10000)
    
    # Step 2: Rank features
    print(f"\nRanking {len(extractor.binary_feature_names)} binary features...")
    rankings = extractor.rank_features(states, actions, method='mutual_info')
    extractor.print_feature_rankings(top_n=10)
    
    # Step 3: Create binary environment wrapper
    binary_env = BinaryFeatureEnvironmentWrapper(env, extractor)
    
    return extractor, binary_env, rankings

# Minimum example to get binary subset environment
def get_binary_subset_env(golden_model, env, indices):
    """
    Minimal function to get an environment that returns binary features at specified indices
    
    Args:
        golden_model: Your trained stable_baselines model
        env: Original CartPole environment  
        indices: List of indices to select from the binary features (e.g., [0, 3, 7, 12])
    
    Returns:
        subset_env: Environment that returns only the selected binary features
    """
    # Create feature extractor and fit thresholds
    extractor = CartPoleBinaryFeatureExtractor(percentiles=[20, 40, 60, 80])
    extractor.fit_thresholds(golden_model, env, n_samples=5000)
    
    # Create binary environment
    binary_env = BinaryFeatureEnvironmentWrapper(env, extractor)
    
    # Create subset environment with your specified indices
    subset_env = BinaryObservationSubsetWrapper(binary_env, indices)
    
    return subset_env

