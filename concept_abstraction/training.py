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


import numpy as np
import wandb

from io import StringIO
from contextlib import redirect_stderr
stderr_buffer = StringIO()
with redirect_stderr(stderr_buffer):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

class WandbLoggingCallback(BaseCallback):
    """
    Logs PPO diagnostics to wandb.
    Works with Monitor-wrapped environments.
    """
    def __init__(self, smooth_alpha=0.9):
        super().__init__()
        self.smoothed_avg_norm_reward = 0
        self.alpha = smooth_alpha  # EMA smoothing for normalized reward

    def _on_step(self) -> bool:
        # ---- Episode info from Monitor ----
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info.keys():
                raw_reward = info["episode"]["r"]
                # normalized reward if you implement reward normalization
                norm_reward = raw_reward
                # EMA smoothing
                self.smoothed_avg_norm_reward = (
                    self.alpha * self.smoothed_avg_norm_reward
                    + (1 - self.alpha) * norm_reward
                )

                # Log episode metrics to wandb
                wandb.log({
                    "episode_reward": raw_reward,
                    "episode_length": info["episode"]["l"],
                    "avg_norm_reward": norm_reward,
                    "ema_norm_reward": self.smoothed_avg_norm_reward
                }, step=self.num_timesteps)

        # ---- PPO internal diagnostics ----
        logger_data = getattr(self.model.logger, "name_to_value", {})
        ev = logger_data.get("train/explained_variance", None)
        vl = logger_data.get("train/value_loss", None)
        kl = logger_data.get("train/approx_kl", None)
        cf = logger_data.get("train/clip_fraction", None)
        ent = logger_data.get("train/entropy_loss", None)
        gn = logger_data.get("diagnostics/grad_norm", None)

        # Log PPO diagnostics to wandb
        wandb.log({
            "explained_variance": ev,
            "value_loss": vl,
            "approx_kl": kl,
            "clip_fraction": cf,
            "entropy_loss": ent,
            "grad_norm": gn
        }, step=self.num_timesteps)

        return True

def get_model(environment_string,policy,override={}):
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
            default_model_dict['policy_kwargs'] = {'net_arch': [64,64]}
            default_model_dict['batch_size'] = 256
            default_model_dict['n_epochs'] = 5
            default_model_dict['learning_rate'] = 1e-5
        elif environment_string == "cart_pole":
            default_model_dict['policy_kwargs'] = {'net_arch': [128]}
            default_model_dict['batch_size'] = 1024
            default_model_dict['n_epochs'] = 5
            default_model_dict['ent_coef'] = 0.005
        elif environment_string == "mini_grid":
            default_model_dict['learning_rate'] = 3e-4
            default_model_dict['n_steps'] = 1024
            default_model_dict['batch_size'] = 1024
            default_model_dict['n_epochs'] = 4
        elif environment_string == "pong":
            default_model_dict['policy_kwargs'] = {'net_arch': [128,128]}
            default_model_dict['n_steps'] = 4096
            default_model_dict['batch_size'] = 256
            default_model_dict['n_epochs'] = 6
            default_model_dict['learning_rate'] = 2e-3
            default_model_dict['ent_coef'] = 0.02
        elif environment_string == "boxing":
            default_model_dict['policy_kwargs'] = {'net_arch': [128,128]}
            default_model_dict['n_steps'] = 4096
            default_model_dict['batch_size'] = 256
            default_model_dict['n_epochs'] = 4
            default_model_dict['learning_rate'] = 1e-3
            default_model_dict['ent_coef'] = 0.015
    else:
        if environment_string == "cart_pole" or environment_string == "mini_grid":
            default_model_dict['n_steps'] = 512
            default_model_dict['batch_size'] = 4096
            default_model_dict['n_epochs'] = 10
            default_model_dict['device'] = 'cuda'
        elif environment_string == "pong":
            default_model_dict['n_steps'] = 1024
            default_model_dict['batch_size'] = 512
            default_model_dict['n_epochs'] = 10
            default_model_dict['device'] = 'cuda'
        elif environment_string == "boxing":
            default_model_dict['n_steps'] = 2048
            default_model_dict['batch_size'] = 256
            default_model_dict['n_epochs'] = 3
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

    model_params = get_model(environment_string,policy,override=override)
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

def evaluate_model(environment_string,env,additional_info,model,seed):
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
        return get_average_reward(env,model)
#----------------------------
# Lightweight CNN for 1-channel input
# ----------------------------
class CartPoleConceptCNN(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_outputs)
        )

    def forward(self, x):
        return self.fc(self.features(x))

class TemporalCartPoleCNN(nn.Module):
    """
    CNN that explicitly models temporal information for stacked frames.
    Uses both spatial convolutions and temporal processing.
    """
    def __init__(self, num_outputs, num_frames=4, input_size=84):
        super().__init__()
        self.num_frames = num_frames
        
        # Process each frame independently with shared weights
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=2),  # 84x84 -> 21x21
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # 21x21 -> 11x11
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 11x11 -> 6x6
            nn.ReLU(),
        )
        
        # Calculate spatial feature size
        # Run a dummy forward pass to get the dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_size, input_size)
            spatial_out = self.spatial_encoder(dummy)
            spatial_feature_size = spatial_out.view(1, -1).shape[1]
        
        # Temporal convolution over the frame dimension
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(spatial_feature_size, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=num_frames),  # Reduces temporal dim to 1
            nn.ReLU(),
        )
        
        # Final decision layers
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_outputs)
        )
    
    def forward(self, x):
        # x shape: (batch, num_frames, H, W)
        batch_size = x.shape[0]
        
        # Process each frame through spatial encoder
        # Reshape to process all frames at once
        x = x.view(batch_size * self.num_frames, 1, x.shape[2], x.shape[3])
        spatial_features = self.spatial_encoder(x)  # (batch*frames, C, H', W')
        
        # Reshape for temporal processing
        spatial_features = spatial_features.view(batch_size, self.num_frames, -1)  # (batch, frames, C*H'*W')
        spatial_features = spatial_features.transpose(1, 2)  # (batch, C*H'*W', frames)
        
        # Apply temporal convolution
        temporal_features = self.temporal_conv(spatial_features)  # (batch, 128, 1)
        temporal_features = temporal_features.squeeze(-1)  # (batch, 128)
        
        # Final output
        return self.fc(temporal_features)




class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits, targets):
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()
    
def get_concept_labels_avg(env, model, concept_list, num_steps=5000, batch_size=100):
    obs, infos = env.reset()
    steps = 0

    while steps < num_steps:
        X_batch, Y_batch = [], []
        batch_steps = 0
        while batch_steps < batch_size and steps < num_steps:
            # Average 4 frames
            X_avg = obs  # (batch, 1, 84,84)
            X_batch.append(X_avg)
            Yb = np.array([[c(inf['observation']) for c in concept_list] for inf in infos])
            Y_batch.append(Yb)

            # Take action
            if np.random.random() < 1.0:
                action = [env.action_space.sample() for _ in range(len(obs))]
            else:
                concepts = [[c(inf['observation']) for c in concept_list] for inf in infos]
                action = model.predict(concepts)[0]

            obs, _, terminated, truncated, infos = env.step(action)
            steps += 1
            batch_steps += 1

            if np.random.random() < 0.05:
                obs, infos = env.reset()

        X_batch = np.concatenate(X_batch, axis=0)
        Y_batch = np.concatenate(Y_batch, axis=0)
        yield X_batch, Y_batch
        del X_batch, Y_batch

def collect_cartpole_data(ground_truth_gym_env,num_episodes=100):
    """
    Collect data from CartPole environment
    Returns:
        X: array of shape (N, num_frames, 84, 84) - frame sequences
        Y: array of shape (N,) - cart velocities
    """
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    
    X_data = []
    Y_data = []
    
    print(f"Collecting data from {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs = ground_truth_gym_env.reset()[0]
        
        done = False
        step_count = 0
        
        while not done:
            # Take random action
            action = ground_truth_gym_env.action_space.sample()
            
            obs, reward, terminated, truncated, info = ground_truth_gym_env.step([action for i in range(8)])
            done = terminated[0] or truncated[0]
            # Store frame sequence and velocity (obs[1] is cart velocity)
            X_data.append(np.array(obs[0]))
            Y_data.append(info[0]['observation'])  # Cart velocity
            
            step_count += 1
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, steps: {step_count}")
    
    env.close()
    
    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)
    
    print(f"\nCollected {len(X_data)} samples")
    print(f"X shape: {X_data.shape}")
    print(f"Y shape: {Y_data.shape}")
    print(f"Velocity range: [{Y_data.min():.3f}, {Y_data.max():.3f}]")
    print(f"Samples with velocity < -0.02: {(Y_data < -0.02).sum()} ({100*(Y_data < -0.02).mean():.2f}%)")
    
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
        self.Y = torch.FloatTensor((Y < 0.02).astype(np.float32))
        
        # Store original velocities for debugging
        self.Y_original = Y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train_concept_predictor(ground_truth_gym_env, gold_model, concept_list, idx, epochs=25): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    # ----------------------------
    # Initialize model, criterion, optimizer
    # ----------------------------
    model = TemporalCartPoleCNN(len(idx)).to('cuda')
    criterion = FocalLoss(alpha=1.0, gamma=2.0, smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_f1 = 0.0
    device = 'cuda'
    num_steps = 1000
    batch_size = 64
    env = ground_truth_gym_env
    
    # ----------------------------
    # Generate validation set ONCE before training
    # ----------------------------
    # val_X_list, val_Y_list = [], []
    # for Xb, Yb in get_concept_labels_avg(env, gold_model, concept_list,
    #                                     num_steps=1000, batch_size=batch_size):
    #     val_X_list.append(Xb)
    #     val_Y_list.append(Yb[:,idx])

    # val_X = np.concatenate(val_X_list, axis=0)
    # val_Y = np.concatenate(val_Y_list, axis=0)

    # def val_loader(val_X, val_Y, batch_size=100):
    #     for i in range(0, len(val_X), batch_size):
    #         yield val_X[i:i+batch_size], val_Y[i:i+batch_size]

    # train_X_list, train_Y_list = [], []
    # for Xb, Yb in get_concept_labels_avg(env, gold_model, concept_list,
    #                                     num_steps=1000, batch_size=batch_size):
    #     train_X_list.append(Xb)
    #     train_Y_list.append(Yb[:,idx])

    # train_X_list = val_X_list 
    # train_Y_list = val_Y_list

    # train_X = np.concatenate(train_X_list, axis=0)
    # train_Y = np.concatenate(train_Y_list, axis=0)

    # def train_loader(train_X, train_Y, batch_size=100):
    #     for i in range(0, len(train_X), batch_size):
    #         yield train_X[i:i+batch_size], train_Y[i:i+batch_size]
    NUM_EPISODES = 200
    NUM_FRAMES=4
    X_data, Y_data = collect_cartpole_data(env,num_episodes=NUM_EPISODES)
    Y_data = [[c(i) for c in concept_list] for i in Y_data] 
    Y_data = np.array(Y_data)[:,idx]
    print(Y_data.shape)

    # Split data
    train_size = int(0.8 * len(X_data))
    indices = np.random.permutation(len(X_data))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    X_train, Y_train = X_data[train_indices], Y_data[train_indices]
    X_val, Y_val = X_data[val_indices], Y_data[val_indices]
    train_dataset = FrameSequenceDataset(X_train, Y_train)
    val_dataset = FrameSequenceDataset(X_val, Y_val)
    BATCH_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss, train_count = 0.0, 0

        for Xb, Yb in train_loader:
            X_tensor = torch.tensor(Xb, dtype=torch.float32, device=device)
            Y_tensor = torch.tensor(Yb, dtype=torch.float32, device=device)
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
        # Validation using the FIXED validation set
        # ----------------------------
        # model.eval()
        val_loss, val_count, f1_list = 0.0, 0, []
        with torch.no_grad():
            for Xb, Yb in val_loader:
                X_tensor = Xb.to(device)
                Y_tensor = Yb.to(device)
                logits = model(X_tensor)
                loss = criterion(logits, Y_tensor)
                val_loss += loss.item()
                val_count += 1

                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                f1_list.append(f1_score(Y_tensor.cpu().numpy().ravel(), preds.ravel()))
        avg_val_loss = val_loss / max(val_count, 1)
        avg_val_f1 = np.mean(f1_list)

        epoch_time = time.time() - epoch_start
        print(f"📉 Epoch {epoch+1}/{epochs} | "
                f"Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f}, "
                f"Val Macro F1 {avg_val_f1:.4f} | Took {epoch_time:.2f} sec")

        if avg_val_f1 > best_f1:
            best_f1 = avg_val_f1
            best_model_state = model.state_dict()

    acc_list = np.zeros(len(idx))
    tot = 0
    for Xb, Yb in val_loader:
        X_tensor = torch.tensor(Xb, dtype=torch.float32, device=device)
        Y_tensor = torch.tensor(Yb, dtype=torch.float32, device=device)
        logits = model(X_tensor)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        acc_list += np.sum(preds == Y_tensor.cpu().numpy(),axis=0)
        tot += len(logits)
    acc_list /= tot 
    
    return model,acc_list 