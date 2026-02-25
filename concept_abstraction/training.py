import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from stable_baselines3.common.torch_layers import NatureCNN

from io import StringIO
from contextlib import redirect_stderr
stderr_buffer = StringIO()
with redirect_stderr(stderr_buffer):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

from concept_abstraction.env_utils import evaluate_policy


# ── Concept predictor CNN ─────────────────────────────────────────────────────

class ConceptPredictorCNN(nn.Module):
    """CNN that predicts binary concept values from raw pixel observations."""

    def __init__(self, num_concepts, num_frames=4, features_dim=512, height=84, width=84):
        super().__init__()
        obs_space = gym.spaces.Box(
            low=0, high=255, shape=(num_frames, height, width), dtype=np.uint8
        )
        self.feature_extractor = NatureCNN(obs_space, features_dim)
        self.classifier = nn.Linear(features_dim, num_concepts)

    def forward(self, x):
        if x.ndim == 4 and x.shape[1] not in [1, 3, 4]:
            x = x.permute(0, 3, 1, 2)
        return self.classifier(self.feature_extractor(x))


# ── PPO training ──────────────────────────────────────────────────────────────

def _get_ppo_params(environment_string, policy, custom_name="", override={}):
    """Return PPO hyperparameters for the given environment and policy type."""
    params = {
        "n_steps":     512,
        "batch_size":  128,
        "learning_rate": 3e-4,
        "device":      "cpu",
        "n_epochs":    10,
        "policy_kwargs": None,
        "ent_coef":    0.01,
    }

    if policy == "MlpPolicy":
        if environment_string == "glucose":
            params.update(learning_rate=3e-4, ent_coef=0.01, n_steps=1024, batch_size=256, n_epochs=5)
        elif environment_string == "cart_pole":
            params.update(policy_kwargs={"net_arch": [64, 64]}, batch_size=128, n_steps=256,
                          n_epochs=4, ent_coef=0, learning_rate=1e-4, clip_range=0.1, target_kl=0.01)
        elif environment_string == "mini_grid":
            params.update(learning_rate=3e-4, n_steps=1024, batch_size=1024, n_epochs=4)
        elif environment_string == "pong":
            params.update(policy_kwargs={"net_arch": [256, 256]}, n_steps=1024, batch_size=256,
                          n_epochs=5, learning_rate=1e-4)
        elif environment_string == "boxing":
            if any(tag in custom_name for tag in ("imperfect", "all_concepts_real", "intervention")):
                params.update(policy_kwargs={"net_arch": [512, 256]}, n_steps=2048, batch_size=512,
                              n_epochs=10, learning_rate=2e-4, ent_coef=0.01, clip_fraction=0.1)
            else:
                params.update(policy_kwargs={"net_arch": [128, 128]}, n_steps=128, batch_size=256,
                              n_epochs=4, ent_coef=0.01)
    else:  # CnnPolicy
        if environment_string == "cart_pole":
            params.update(n_steps=1024, batch_size=256, n_epochs=4, ent_coef=5e-4,
                          learning_rate=1e-4, vf_coef=0.5, device="cuda")
        elif environment_string == "mini_grid":
            params.update(n_steps=1024, batch_size=1024, n_epochs=4, device="cuda")
        elif environment_string == "pong":
            params.update(n_steps=1024, batch_size=256, n_epochs=5, learning_rate=1e-4, device="cuda")
        elif environment_string == "boxing":
            params.update(n_steps=1024, batch_size=1024, n_epochs=4, device="cuda")
        else:
            params.update(n_steps=128, batch_size=256, n_epochs=4, learning_rate=2.5e-4, device="cuda")

    params.update(override)
    return params


class _WandbCallback(BaseCallback):
    def __init__(self, smooth_alpha=0.9, log_freq=10):
        super().__init__()
        self.ema = 0.0
        self.alpha = smooth_alpha
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_episodes  = 0

    def _on_step(self):
        import wandb
        for info in self.locals.get("infos", []):
            if "episode" in info:
                r = info["episode"]["r"]
                self.ema = self.alpha * self.ema + (1 - self.alpha) * r
                self.episode_rewards.append(r)
                self.episode_lengths.append(info["episode"]["l"])
                self.total_episodes += 1

        if self.n_calls % self.log_freq == 0:
            metrics = {}
            if self.episode_rewards:
                metrics.update({
                    "episode_reward_mean": np.mean(self.episode_rewards),
                    "episode_reward_max":  np.max(self.episode_rewards),
                    "episode_reward_min":  np.min(self.episode_rewards),
                    "episode_length_mean": np.mean(self.episode_lengths),
                    "episodes_completed":  self.total_episodes,
                    "ema_norm_reward":     self.ema,
                })
                self.episode_rewards.clear()
                self.episode_lengths.clear()

            logger_data = getattr(self.model.logger, "name_to_value", {})
            for key in ("train/explained_variance", "train/value_loss", "train/approx_kl",
                        "train/clip_fraction", "train/entropy_loss", "diagnostics/grad_norm"):
                if key in logger_data:
                    metrics[key.split("/")[-1]] = logger_data[key]

            if metrics:
                wandb.log(metrics, step=self.num_timesteps)
        return True


def train_ppo(
    env,
    environment_string,
    seed=42,
    total_timesteps=150_000,
    policy="MlpPolicy",
    override={},
    custom_name="",
    model=None,
    use_wandb=False,
):
    """Train a PPO policy on the given environment.

    Args:
        env: SB3-compatible VecEnv
        environment_string: Environment name for hyperparameter lookup
        seed: Random seed
        total_timesteps: Training budget
        policy: SB3 policy string ('MlpPolicy' or 'CnnPolicy')
        override: Dict of hyperparameter overrides
        custom_name: Run name for logging
        model: Existing PPO model to continue training (optional)
        use_wandb: Whether to log metrics to Weights & Biases

    Returns:
        Trained PPO model
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    params = _get_ppo_params(environment_string, policy, custom_name=custom_name, override=override)
    env.seed(seed)

    run_name = custom_name or f"{environment_string}_{policy}"

    if use_wandb:
        import wandb
        wandb.init(project="Concept Decisions", name=run_name, config=params)

    if model is None:
        ppo_kwargs = {k: params[k] for k in
                      ("policy_kwargs", "n_steps", "batch_size", "n_epochs",
                       "learning_rate", "ent_coef", "device")
                      if k in params}
        model = PPO(policy, env, gamma=0.99, verbose=0, seed=seed, progress_bar=True,**ppo_kwargs)

    callback = _WandbCallback() if use_wandb else None
    model.learn(total_timesteps=total_timesteps, callback=callback)

    if use_wandb:
        import wandb
        wandb.finish()

    return model


class RandomAgent:
    """Uniform random policy."""

    def __init__(self, vec_env):
        self.action_space = vec_env.action_space
        self.num_envs     = vec_env.num_envs
        self.device       = "cpu"

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        return np.array([self.action_space.sample() for _ in range(self.num_envs)]), None


def evaluate_model(environment_string, env, model, seed, max_steps=100_000):
    """Evaluate a trained model and return mean episode reward.

    Uses a shorter episode budget for glucose to match its episode structure.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    if hasattr(model, "set_random_seed"):
        model.set_random_seed(seed)

    if environment_string == "glucose":
        return evaluate_policy(env, model, seed, max_steps=10_000, max_steps_per_episode=1_000)
    return evaluate_policy(env, model, seed, max_steps=max_steps)


# ── Concept predictor training ────────────────────────────────────────────────

def _collect_training_data(gym_env, policy, concept_list, num_episodes=5, max_episode_length=10_000):
    """Roll out a policy and collect (pixel_obs, raw_state) pairs.

    Args:
        gym_env: GymnasiumWrapper environment
        policy: SB3-compatible policy
        concept_list: Concept functions (used to determine Y labels)
        num_episodes: Number of episodes to collect
        max_episode_length: Max steps per episode

    Returns:
        X: uint8 pixel observations, shape (N, frames, H, W)
        Y: raw state observations, shape (N, state_dim)
    """
    X_data, Y_data = [], []

    for episode in range(num_episodes):
        use_gold = episode % 2 == 0
        obs, info = gym_env.reset()
        done = [False] * 8

        for step in range(max_episode_length):
            actions = (policy.predict(obs)[0] if use_gold
                       else [gym_env.action_space.sample() for _ in range(8)])
            obs, _, _, _, info = gym_env.step(actions)

            for j in range(len(obs)):
                if np.random.random() < 0.3 and "observation" in info[j]:
                    X_data.append(obs[j].astype(np.uint8))
                    Y_data.append(info[j]["observation"])

            if (step + 1) % 1_000 == 0:
                print(f"  Episode {episode+1}/{num_episodes}, step {step+1}/{max_episode_length}")

        print(f"Episode {episode+1}/{num_episodes} done, {len(X_data)} samples collected")

    return np.array(X_data, dtype=np.uint8), np.array(Y_data, dtype=np.float32)


def train_concept_predictor(
    gym_env,
    policy,
    concept_list,
    concept_idx,
    environment_string,
    epochs=25,
    num_episodes=5,
    max_episode_length=10_000,
):
    """Train a CNN to predict binary concept values from pixel observations.

    Args:
        gym_env: GymnasiumWrapper environment
        policy: SB3-compatible policy for data collection
        concept_list: Full list of concept functions
        concept_idx: Indices of concepts to predict
        environment_string: Environment name (affects CNN input shape)
        epochs: Maximum training epochs
        num_episodes: Episodes to collect for training data
        max_episode_length: Max steps per episode

    Returns:
        model: Trained ConceptPredictorCNN
        acc_list: Per-concept accuracy on the validation split
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    height = width = 84
    num_frames = 1 if environment_string == "mini_grid" else 4
    if environment_string == "cart_pole":
        height, width = 160, 240

    model = ConceptPredictorCNN(len(concept_idx), num_frames=num_frames,
                                height=height, width=width).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # Collect data
    X_data, Y_data_raw = _collect_training_data(
        gym_env, policy, concept_list,
        num_episodes=num_episodes, max_episode_length=max_episode_length,
    )
    Y_data = np.array([[c(obs) for c in concept_list] for obs in Y_data_raw],
                      dtype=np.float32)[:, concept_idx]

    # Train/val split
    indices     = np.random.permutation(len(X_data))
    train_idx   = indices[:int(0.8 * len(X_data))]
    val_idx     = indices[int(0.8 * len(X_data)):]

    class _DS(Dataset):
        def __init__(self, X, Y, idx):
            self.X, self.Y, self.idx = X, Y, idx
        def __len__(self): return len(self.idx)
        def __getitem__(self, i):
            j = self.idx[i]
            return torch.from_numpy(self.X[j].astype(np.float32) / 255.0), torch.from_numpy(self.Y[j])

    train_loader = DataLoader(_DS(X_data, Y_data, train_idx), batch_size=64, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(_DS(X_data, Y_data, val_idx),   batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    best_f1    = 0.0
    best_state = None
    patience   = 10
    no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for Xb, Yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb.to(device)), Yb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for Xb, Yb in val_loader:
                logits = model(Xb.to(device))
                preds  = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(Yb.numpy())

        all_preds   = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        val_f1      = f1_score(all_targets.ravel(), all_preds.ravel())

        print(f"Epoch {epoch+1}/{epochs} | val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)

    # Per-concept accuracy on validation set
    model.eval()
    acc_list = np.zeros(len(concept_idx))
    total    = 0
    with torch.no_grad():
        for Xb, Yb in val_loader:
            preds     = (torch.sigmoid(model(Xb.to(device))) > 0.5).cpu().numpy()
            acc_list += np.sum(preds == Yb.numpy(), axis=0)
            total    += len(Xb)
    acc_list /= total

    print("Per-concept accuracy:")
    for i, acc in enumerate(acc_list):
        status = "✓" if acc >= 0.85 else "~" if acc >= 0.75 else "✗"
        print(f"  [{status}] concept {i}: {acc:.3f}")

    return model, acc_list