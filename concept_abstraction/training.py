import random

import numpy as np
import torch
import torch.distributions as D
import torch.nn.functional as F
from concept_abstraction.env_utils import get_average_reward
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import time
import gymnasium as gym
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.torch_layers import NatureCNN


import numpy as np
import wandb

from io import StringIO
from contextlib import redirect_stderr
stderr_buffer = StringIO()
with redirect_stderr(stderr_buffer):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

class ConceptPredictorCNN(nn.Module):
    def __init__(self, num_concepts, num_frames=4, features_dim=512, height=84, width=84):
        super().__init__()
        obs_space = gym.spaces.Box(low=0, high=255, shape=(num_frames, height, width), dtype=np.uint8)
        self.feature_extractor = NatureCNN(obs_space, features_dim)
        self.classifier = nn.Linear(features_dim, num_concepts)
    
    def forward(self, x):
        if x.ndim == 4 and x.shape[1] not in [1, 3, 4]:
            x = x.permute(0, 3, 1, 2)
        features = self.feature_extractor(x)
        return self.classifier(features)

class WandbLoggingCallback(BaseCallback):
    def __init__(self, smooth_alpha=0.9, log_freq=10):
        super().__init__()
        self.smoothed_avg_norm_reward = 0
        self.alpha = smooth_alpha
        self.log_freq = log_freq
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_episodes_completed = 0  # <- accumulate over whole run

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                raw_reward = info["episode"]["r"]
                self.smoothed_avg_norm_reward = (
                    self.alpha * self.smoothed_avg_norm_reward
                    + (1 - self.alpha) * raw_reward
                )
                self.episode_rewards.append(raw_reward)
                self.episode_lengths.append(info["episode"]["l"])
                self.total_episodes_completed += 1  # <- increment global counter

        if self.n_calls % self.log_freq == 0:
            metrics = {}
            if self.episode_rewards:
                metrics.update({
                    "episode_reward_mean": np.mean(self.episode_rewards),
                    "episode_reward_max": np.max(self.episode_rewards),
                    "episode_reward_min": np.min(self.episode_rewards),
                    "episode_length_mean": np.mean(self.episode_lengths),
                    "episodes_completed": self.total_episodes_completed,  # <- use global counter
                    "ema_norm_reward": self.smoothed_avg_norm_reward
                })
                self.episode_rewards.clear()
                self.episode_lengths.clear()

            # PPO metrics logging (unchanged)
            logger_data = getattr(self.model.logger, "name_to_value", {})
            ppo_metrics = {
                "explained_variance": logger_data.get("train/explained_variance"),
                "value_loss": logger_data.get("train/value_loss"),
                "approx_kl": logger_data.get("train/approx_kl"),
                "clip_fraction": logger_data.get("train/clip_fraction"),
                "entropy_loss": logger_data.get("train/entropy_loss"),
                "grad_norm": logger_data.get("diagnostics/grad_norm"),
            }
            metrics.update({k: v for k, v in ppo_metrics.items() if v is not None})

            if metrics:
                wandb.log(metrics, step=self.num_timesteps)

        return True

def get_model(environment_string,policy,custom_name="",override={}):
    default_model_dict = {
        'n_steps': 512,
        'batch_size': 128,
        'learning_rate': 3e-4,
        'device': 'cpu',
        'n_epochs': 10,
        'policy_kwargs': None ,
        'ent_coef': 0.01,
    }

    if policy == "MlpPolicy":
        if "cyclic" in environment_string:
            default_model_dict['policy_kwargs'] = {'net_arch': [16]}
            default_model_dict['n_steps'] = 32
            default_model_dict['batch_size'] = 32
            default_model_dict['learning_rate'] = 1e-3
        elif "tree" in environment_string:
            default_model_dict['policy_kwargs'] = {'net_arch': [16]}
            default_model_dict['n_steps'] = 32
            default_model_dict['batch_size'] = 32
            default_model_dict['learning_rate'] = 1e-3
            default_model_dict['ent_coef'] = 0.02
        elif environment_string == "glucose":
            print("Loading regular glucose")
            default_model_dict['learning_rate'] = 3e-4     # or even 1e-4 (important)
            default_model_dict['ent_coef'] = 0.01          # not 0.1
            default_model_dict['n_steps'] = 1024           # from 256 --> 1024 or 2048
            default_model_dict['batch_size'] = 256         # if n_steps increased
            default_model_dict['n_epochs'] = 5             # more steps = fewer epochs
        elif environment_string == "glucose_raw":
            default_model_dict['learning_rate'] = 3e-4     # or even 1e-4 (important)
            default_model_dict['ent_coef'] = 0.01          # not 0.1
            default_model_dict['n_steps'] = 1024           # from 256 --> 1024 or 2048
            default_model_dict['batch_size'] = 256         # if n_steps increased
            default_model_dict['n_epochs'] = 5             # more steps = fewer epochs
        elif environment_string == "cart_pole":
            default_model_dict['policy_kwargs'] = {'net_arch': [64,64]}
            default_model_dict['batch_size'] = 128
            default_model_dict['n_steps'] = 256
            default_model_dict['n_epochs'] = 10
            default_model_dict['ent_coef'] = 0
            default_model_dict['learning_rate'] = 3e-4
            default_model_dict['clip_range'] = 0.1
            default_model_dict['target_kl'] = 0.01
        elif environment_string == "mini_grid":
            default_model_dict['learning_rate'] = 3e-4
            default_model_dict['n_steps'] = 1024
            default_model_dict['batch_size'] = 1024
            default_model_dict['n_epochs'] = 4
        elif environment_string == "pong":
            default_model_dict['policy_kwargs'] = {'net_arch': [256,256]}
            default_model_dict['n_steps'] = 1024
            default_model_dict['batch_size'] = 512
            default_model_dict['n_epochs'] = 10
        elif environment_string == "boxing":
            if "imperfect" in custom_name or "all_concepts_real" in custom_name or "intervention" in custom_name:
                print("Using imperfect boxing")
                default_model_dict['policy_kwargs'] = {'net_arch': [512, 256]} # Wider first layer
                default_model_dict['n_steps'] = 2048                          # Larger rollout for stability
                default_model_dict['batch_size'] = 512                        # Better gradient estimate
                default_model_dict['n_epochs'] = 10                           # Standard for PPO, helps critic
                default_model_dict['learning_rate'] = 2e-4                    # Slightly lower for stability
                default_model_dict['ent_coef'] = 0.01                         # Keep for now, watch entropy_loss           
            else:
                default_model_dict['policy_kwargs'] = {'net_arch': [128,128]}
                default_model_dict['n_steps'] = 128
                default_model_dict['batch_size'] = 256
                default_model_dict['n_epochs'] = 4
                default_model_dict['ent_coef'] = 0.01
    else:
        if environment_string == "cart_pole":
            default_model_dict['n_steps'] = 4096
            default_model_dict['batch_size'] = 512
            default_model_dict['n_epochs'] = 10
            default_model_dict['ent_coef'] = 0.01
            default_model_dict['learning_rate'] = 3*10**-4
            default_model_dict['vf_coef'] = 0.5
            default_model_dict['device'] = 'cuda'
        elif environment_string == 'mini_grid':
            default_model_dict['n_steps'] = 1024
            default_model_dict['batch_size'] = 1024
            default_model_dict['n_epochs'] = 4
            default_model_dict['device'] = 'cuda'
        elif environment_string == "pong":
            default_model_dict['n_steps'] = 1024
            default_model_dict['batch_size'] = 512
            default_model_dict['n_epochs'] = 10
            default_model_dict['device'] = 'cuda'
        elif environment_string == "boxing":
            default_model_dict['n_steps'] = 1024
            default_model_dict['batch_size'] = 1024
            default_model_dict['n_epochs'] = 4
            default_model_dict['device'] = 'cuda'
        else:
            default_model_dict['n_steps'] = 128
            default_model_dict['batch_size'] = 256
            default_model_dict['n_epochs'] = 4
            default_model_dict['device'] = 'cuda'
            default_model_dict['learning_rate'] = 2.5e-4
    for i in override:
        default_model_dict[i] = override[i]

    return default_model_dict

def train_ppo_model(env,environment_string,seed=42,total_timesteps=150_000,policy="MlpPolicy",override={},custom_name="",model=None,silent=False):
    """Train an environment according to a stable baseline policy
    
    Arguments:
        env: Gymnasium environment
    
    Returns: Stable Baseline3 PPO Model"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_params = get_model(environment_string,policy,override=override,custom_name=custom_name)
    model_params['env'] = environment_string

    name = "{}_{}".format(environment_string,policy)
    if custom_name != "":
        name = custom_name
    
    if not silent:
        wandb.init(
            project="Concept Decisions",
            name=name,
            config=model_params
        )

    if model is None:
        model = PPO(
            policy,
            env,
            policy_kwargs=model_params['policy_kwargs'],
            n_steps=model_params['n_steps'],
            batch_size=model_params['batch_size'],
            n_epochs=model_params['n_epochs'],
            learning_rate=model_params['learning_rate'],
            ent_coef=model_params['ent_coef'],
            gamma=0.99,
            verbose=0,
            device=model_params['device'],
        )

    if silent:
        model.learn(total_timesteps=total_timesteps)
    else:
        model.learn(total_timesteps=total_timesteps, callback=WandbLoggingCallback())
        wandb.finish()
    return model 

class DistributionWrapper:
    def __init__(self, logits):
        # logits = unnormalized scores (before softmax)
        self.logits = logits
        self.distribution = D.Categorical(logits=logits)


class Policy:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def get_distribution(self, obs):
        batch_size = len(obs)
        # uniform logits: shape (batch_size, num_actions)
        logits = torch.zeros((batch_size, self.num_actions))
        return DistributionWrapper(logits)


class RandomAgent:
    """Random policy that randomly selects actions"""

    def __init__(self, vec_env):
        self.action_space = vec_env.action_space
        self.num_envs = vec_env.num_envs
        self.device = "cpu"

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        # Sample one action per environment
        actions = np.array([self.action_space.sample() for _ in range(self.num_envs)])
        return actions, None

def evaluate_model(environment_string,env,model,seed,max_steps=50_000):
    """Evaluation Function that tailors the evaluation
        based on the environment
        
    Arguments:
        environment_string: String, such as cart_pole, describing the environment
        env: Environment where we are evaluating the polciy
        additional_info: Any additional info needed to evaluate
            For example, for MIMIC, we need the Physician policy
        model: Policy function \pi
    
    Returns: Average Reward (for most environments) or MIMIC WIQ score"""

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if environment_string == "glucose":
        return get_average_reward(env,model,max_steps=5000,max_steps_per=500)
    else:
        return get_average_reward(env,model,max_steps=max_steps)
#----------------------------
# Lightweight CNN for 1-channel input
# ----------------------------
def collect_cartpole_data(ground_truth_gym_env, groundtruth_model, concept_list, num_episodes=100, max_episode_length=500,use_gold=False):
    """
    Collect data from CartPole environment - Memory efficient version
    Returns:
        X: array of shape (N, 4, 84, 84) - frame sequences (uint8)
        Y: array of shape (N, ...) - observations (float32)
    """
    X_data = []
    Y_data = []
    
    print(f"Collecting data from {num_episodes} episodes...")

    for episode in range(num_episodes):
        # Use gold model every 20th episode for better data
        use_gold_this_episode = (episode % 2 == 0)
        
        obs, info = ground_truth_gym_env.reset()
        done = [False for i in range(8)]
        step_count = 0

        for i in range(max_episode_length):        
            concepts = [[c(info[i]['observation']) for c in concept_list] for i in range(len(info)) if not done[i]]
            actions = [ground_truth_gym_env.action_space.sample() for _ in range(len(concepts))]
            if use_gold_this_episode:
                actions = groundtruth_model.predict(obs)[0] 
            obs, _, _, _, info = ground_truth_gym_env.step(actions)
            for j in range(len(obs)):
                if np.random.random() < 0.3 and 'observation' in info[j]:
                    # Store as uint8 (0-255) instead of float32 (0-1)
                    X_data.append(obs[j].astype(np.uint8))
                    Y_data.append(info[j]['observation'])
            
            step_count += 1
        
            if (i+1)%1_000 == 0:
                print("Iteration {}/{}".format(i+1,max_episode_length))
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, steps: {step_count}, samples collected: {len(X_data)}")
            # Print memory usage estimate
            if len(X_data) > 0:
                mem_mb = len(X_data) * np.prod(X_data[0].shape) / (1024**2)
                print(f"  Estimated memory: {mem_mb:.1f} MB")
    
    # Convert to numpy arrays
    # Keep X as uint8 (4x smaller than float32)
    X_data = np.array(X_data, dtype=np.uint8)
    Y_data = np.array(Y_data, dtype=np.float32)
    
    print(f"\n✅ Collection complete!")
    print(f"X shape: {X_data.shape}, dtype: {X_data.dtype}, size: {X_data.nbytes / (1024**2):.1f} MB")
    print(f"Y shape: {Y_data.shape}, dtype: {Y_data.dtype}")
    
    # Only compute these stats if Y_data is 1D (single value per sample)
    if len(Y_data.shape) == 1 or Y_data.shape[1] == 1:
        print(f"Value range: [{Y_data.min():.3f}, {Y_data.max():.3f}]")
    else:
        print(f"Y contains {Y_data.shape[1]} features per sample")
    
    return X_data, Y_data
class FrameSequenceDataset(Dataset):
    def __init__(self, X, Y, normalize=True):
        """
        X: numpy array of shape (N, num_frames, 84, 84)
        Y: numpy array of shape (N,) - velocities
        """
        # Normalize pixel values to [0, 1]
        if normalize:
            X = X / 255.0
        
        self.X = torch.FloatTensor(X)
        # Binary classification: 1 if velocity < -0.02, else 0
        self.Y = torch.FloatTensor(Y)
        
        # Store original velocities for debugging
        self.Y_original = Y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def train_concept_predictor(ground_truth_gym_env, groundtruth_model, concept_list, idx, environment_string,epochs=25,NUM_EPISODES=5,max_episode_length=10_000): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ----------------------------
    # Initialize model, criterion, optimizer
    # ----------------------------

    height = width = 84

    if environment_string == "mini_grid":
        num_frames = 1
    else:
        num_frames = 4
    
    if environment_string == "cart_pole":
        height = 160
        width = 240

    model = ConceptPredictorCNN(len(idx), num_frames=num_frames,height=height,width=width).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    best_f1 = 0.0
    
    # Collect data
    X_data, Y_data = collect_cartpole_data(ground_truth_gym_env, groundtruth_model, 
                                           concept_list=concept_list, num_episodes=NUM_EPISODES, 
                                           use_gold=True,max_episode_length=max_episode_length)
    
    # Process concepts

    Y_data = [[c(i) for c in concept_list] for i in Y_data] 
    Y_data = np.array(Y_data, dtype=np.float32)[:, idx]  # Use float32 explicitly
    
    print(f"X_data shape: {X_data.shape}, dtype: {X_data.dtype}")
    print(f"Y_data shape: {Y_data.shape}, dtype: {Y_data.dtype}")
    
    # ----------------------------
    # MEMORY EFFICIENT: Use indices instead of copying arrays
    # ----------------------------
    train_size = int(0.8 * len(X_data))
    indices = np.random.permutation(len(X_data))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Don't create copies - use SubsetDataset or index on-the-fly
    class IndexedDataset(Dataset):
        def __init__(self, X, Y, indices, normalize=True):
            self.X = X  # Keep reference, don't copy
            self.Y = Y  # Keep reference, don't copy
            self.indices = indices
            self.normalize = normalize
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            actual_idx = self.indices[idx]
            x = self.X[actual_idx]
            y = self.Y[actual_idx]
            
            # Normalize on-the-fly to save memory
            if self.normalize:
                x = x.astype(np.float32) / 255.0
            
            return torch.from_numpy(x), torch.from_numpy(y)
    
    # Create datasets using indices (no data copying)
    train_dataset = IndexedDataset(X_data, Y_data, train_indices)
    val_dataset = IndexedDataset(X_data, Y_data, val_indices)
    
    BATCH_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=2, pin_memory=True)  # Speed up data loading
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=2, pin_memory=True)
    
    # ----------------------------
    # Training loop
    # ----------------------------
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss, train_count = 0.0, 0
        
        for Xb, Yb in train_loader:
            X_tensor = Xb.to(device, non_blocking=True)
            Y_tensor = Yb.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            logits = model(X_tensor)
            loss = criterion(logits, Y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_count += 1
        
        avg_train_loss = train_loss / max(train_count, 1)
        
        # ----------------------------
        # Validation
        # ----------------------------
        model.eval()
        val_loss, val_count = 0.0, 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for Xb, Yb in val_loader:
                X_tensor = Xb.to(device, non_blocking=True)
                Y_tensor = Yb.to(device, non_blocking=True)
                
                logits = model(X_tensor)
                loss = criterion(logits, Y_tensor)
                val_loss += loss.item()
                val_count += 1
                
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(Y_tensor.cpu().numpy())
        
        # Compute F1 once on all data
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        avg_val_f1 = f1_score(all_targets.ravel(), all_preds.ravel())
        
        avg_val_loss = val_loss / max(val_count, 1)
        epoch_time = time.time() - epoch_start
        
        print(f"📉 Epoch {epoch+1}/{epochs} | "
              f"Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | "
              f"Val F1 {avg_val_f1:.4f} | Took {epoch_time:.2f}s")
        
        if avg_val_f1 > best_f1:
            best_f1 = avg_val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # ----------------------------
    # Calculate per-concept accuracy
    # ----------------------------
    model.eval()
    acc_list = np.zeros(len(idx))
    tot = 0
    
    with torch.no_grad():
        for Xb, Yb in val_loader:
            X_tensor = Xb.to(device)
            Y_tensor = Yb.to(device)
            
            logits = model(X_tensor)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            
            acc_list += np.sum(preds == Y_tensor.cpu().numpy(), axis=0)
            tot += len(logits)
    
    acc_list /= tot
    
    print(f"\n📊 Per-concept accuracy:")
    for i, acc in enumerate(acc_list):
        status = "✅" if acc >= 0.85 else "⚠️" if acc >= 0.75 else "❌"
        print(f"  {status} Concept {i}: {acc:.3f}")
    
    return model, acc_list


def evaluate_concept_predictor(
    concept_predictor,
    ground_truth_gym_env,
    groundtruth_model,
    concept_list,
    NUM_EPISODES=5,
    max_episode_length=10_000,
    batch_size=64
):
    """
    Evaluate a trained concept predictor on fresh data.
    
    Args:
        concept_predictor: Trained ConceptPredictorCNN model
        ground_truth_gym_env: The gym environment to collect data from
        groundtruth_model: The ground truth policy model
        concept_list: List of concept functions
        NUM_EPISODES: Number of episodes to collect for evaluation
        max_episode_length: Maximum length of each episode
        batch_size: Batch size for evaluation
    
    Returns:
        acc_list: Per-concept accuracy array
        overall_f1: Overall F1 score across all concepts
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Collect fresh evaluation data
    print("Collecting evaluation data...")
    X_data, Y_data = collect_cartpole_data(
        ground_truth_gym_env, 
        groundtruth_model,
        concept_list=concept_list,
        num_episodes=NUM_EPISODES,
        use_gold=True,
        max_episode_length=max_episode_length
    )
    
    # Process concepts
    Y_data = [[c(i) for c in concept_list] for i in Y_data]
    Y_data = np.array(Y_data, dtype=np.float32)
    
    print(f"Evaluation data - X shape: {X_data.shape}, Y shape: {Y_data.shape}")
    
    # Create evaluation dataset
    class EvalDataset(Dataset):
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            x = self.X[idx].astype(np.float32) / 255.0
            y = self.Y[idx]
            return torch.from_numpy(x), torch.from_numpy(y)
    
    eval_dataset = EvalDataset(X_data, Y_data)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Evaluate
    concept_predictor.eval()
    concept_predictor.to(device)
    
    acc_list = np.zeros(len(concept_list))
    all_preds = []
    all_targets = []
    total_samples = 0
    
    print("Evaluating concept predictor...")
    with torch.no_grad():
        for Xb, Yb in eval_loader:
            X_tensor = Xb.to(device, non_blocking=True)
            Y_tensor = Yb.to(device, non_blocking=True)
            
            logits = concept_predictor(X_tensor)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            targets = Y_tensor.cpu().numpy()
            
            # Accumulate per-concept accuracy
            acc_list += np.sum(preds == targets, axis=0)
            total_samples += len(preds)
            
            # Store for F1 calculation
            all_preds.append(preds)
            all_targets.append(targets)
    
    # Calculate per-concept accuracy
    acc_list /= total_samples
    
    # Calculate overall F1 score
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    from sklearn.metrics import f1_score
    overall_f1 = f1_score(all_targets.ravel(), all_preds.ravel())
    
    # Print results
    print(f"\n📊 Evaluation Results:")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    print(f"\nPer-concept accuracy:")
    for i, acc in enumerate(acc_list):
        status = "✅" if acc >= 0.85 else "⚠️" if acc >= 0.75 else "❌"
        print(f"  {status} Concept {i}: {acc:.3f}")
    
    return acc_list