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

def train_ppo_model(env,environment_string,seed=42,total_timesteps=150_000,policy="MlpPolicy",override={},custom_name=""):
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
    wandb.init(
        project="Concept Decisions",
        name=name,
        config=model_params
    )
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
            if np.random.random() < 0.1:
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

def train_concept_predictor(ground_truth_gym_env, gold_model, concept_list, idx, epochs=25): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sample_X, sample_Y = [], []
    for Xb, Yb in get_concept_labels_avg(ground_truth_gym_env, gold_model, concept_list, num_steps=500, batch_size=100):
        sample_X.append(Xb)
        sample_Y.append(Yb[:,idx])

    sample_Y = np.concatenate(sample_Y, axis=0)
    pos_counts = sample_Y.sum(axis=0)
    neg_counts = len(sample_Y) - pos_counts
    pos_weights = torch.tensor([max(min(neg/p, 3.0), 0.5) if p>0 else 1.0
                                for p, neg in zip(pos_counts, neg_counts)], dtype=torch.float32, device='cuda')

    # ----------------------------
    # Initialize model, criterion, optimizer
    # ----------------------------
    model = CartPoleConceptCNN(len(idx)).to('cuda')
    criterion = FocalLoss(alpha=1.0, gamma=2.0, smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    best_f1 = 0.0
    device = 'cuda'
    num_steps = 1000
    batch_size = 64
    env = ground_truth_gym_env
    
    # ----------------------------
    # Generate validation set ONCE before training
    # ----------------------------
    val_X_list, val_Y_list = [], []
    for Xb, Yb in get_concept_labels_avg(env, gold_model, concept_list,
                                        num_steps=1000, batch_size=batch_size):
        val_X_list.append(Xb)
        val_Y_list.append(Yb[:,idx])

    val_X = np.concatenate(val_X_list, axis=0)
    val_Y = np.concatenate(val_Y_list, axis=0)

    def val_loader(val_X, val_Y, batch_size=100):
        for i in range(0, len(val_X), batch_size):
            yield val_X[i:i+batch_size], val_Y[i:i+batch_size]

    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss, train_count = 0.0, 0

        for Xb, Yb in get_concept_labels_avg(env, gold_model, concept_list, num_steps=num_steps, batch_size=batch_size):
            X_tensor = torch.tensor(Xb, dtype=torch.float32, device=device)
            Y_tensor = torch.tensor(Yb, dtype=torch.float32, device=device)[:,idx]

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
        model.eval()
        val_loss, val_count, f1_list = 0.0, 0, []
        with torch.no_grad():
            for Xb, Yb in val_loader(val_X, val_Y):
                X_tensor = torch.tensor(Xb, dtype=torch.float32, device=device)
                Y_tensor = torch.tensor(Yb, dtype=torch.float32, device=device)
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
    for Xb, Yb in val_loader(val_X, val_Y):
        X_tensor = torch.tensor(Xb, dtype=torch.float32, device=device)
        Y_tensor = torch.tensor(Yb, dtype=torch.float32, device=device)
        logits = model(X_tensor)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        acc_list += np.sum(preds == Y_tensor.cpu().numpy(),axis=0)
        tot += len(logits)
    acc_list /= tot 
    
    return model,acc_list 