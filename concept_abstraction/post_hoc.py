
import gymnasium as gym
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif

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
        observation_2d = observation.reshape(1, -1)
        binary_obs = self.feature_extractor.convert_to_binary(observation_2d)
        return binary_obs[0]


class CartPoleBinaryFeatureExtractor:
    """Wrapper that allows us to extract concepts post-hoc from 
        a Gymnasium Environment"""

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
