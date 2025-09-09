import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.buffers import RolloutBuffer, RolloutBufferSamples
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from typing import Callable
import numpy as np
import gymnasium as gym
import random 
from collections import namedtuple
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

from concept_abstraction.env_utils import get_average_reward, get_average_reward_mimic
from concept_abstraction.environments import GymnasiumWrapper

InfoRolloutBufferSamples = namedtuple(
    "InfoRolloutBufferSamples",
    ["observations", "actions", "old_values", "old_log_prob", "advantages", "returns", "infos"]
)


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, update_interval=0.01, verbose=0):
        """
        update_interval: fraction of total timesteps to update progress bar (e.g., 0.01 = 1%)
        """
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.update_interval = update_interval
        self.pbar = None
        self.last_update = 0

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training")

    def _on_step(self):
        # Only update when enough progress has been made
        if self.num_timesteps - self.last_update >= self.total_timesteps * self.update_interval:
            self.pbar.n = self.num_timesteps
            self.pbar.refresh()
            self.last_update = self.num_timesteps
        return True

    def _on_training_end(self):
        self.pbar.close()


class InfoRolloutBuffer(RolloutBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device,
                 gamma=0.99, gae_lambda=1.0, n_envs=1):
        super().__init__(buffer_size, observation_space, action_space, device,
                         gamma=gamma, gae_lambda=gae_lambda, n_envs=n_envs)
        # One info dict per timestep
        self.infos = [None] * buffer_size

    def add(self, obs, action, reward, episode_start, value, log_prob, info=None):
        super().add(obs, action, reward, episode_start, value, log_prob)
        # Store info aligned with current position
        self.infos[self.pos - 1] = info

    def get(self, batch_size):
        """
        Mimics RolloutBuffer.get(), but attaches infos to the batch.
        """
        assert self.full, "Rollout buffer must be full before sampling"

        # Flatten (n_steps, n_envs, *) into (buffer_size, *)
        obs = self.observations.reshape((-1,) + self.observation_space.shape)
        actions = self.actions.reshape((-1,) + self.action_space.shape)
        values = self.values.flatten()
        log_probs = self.log_probs.flatten()
        advantages = self.advantages.flatten()
        returns = self.returns.flatten()

        indices = np.random.permutation(self.buffer_size)
        start_idx = 0
        while start_idx < self.buffer_size:
            batch_inds = indices[start_idx:start_idx + batch_size]
            start_idx += batch_size

            yield InfoRolloutBufferSamples(
                observations=torch.as_tensor(obs[batch_inds]).to(self.device),
                actions=torch.as_tensor(actions[batch_inds]).to(self.device),
                old_values=torch.as_tensor(values[batch_inds]).to(self.device),
                old_log_prob=torch.as_tensor(log_probs[batch_inds]).to(self.device),
                advantages=torch.as_tensor(advantages[batch_inds]).to(self.device),
                returns=torch.as_tensor(returns[batch_inds]).to(self.device),
                infos=[self.infos[i] for i in batch_inds],  # stays as list of dicts
            )

def train_model(env,total_timesteps=10000):
    """Train an environment according to a stable baseline policy
    
    Arguments:
        env: Gymnasium environment
    
    Returns: Stable Baseline3 DQN Model"""
    model = DQN("MlpPolicy", env, device="cuda",verbose=0)
    model.learn(total_timesteps=total_timesteps, log_interval=4)
    return model 

def train_ppo_model(env,environment_string,seed=42,total_timesteps=150_000,policy="MlpPolicy",batch_size=256, n_steps=128):
    """Train an environment according to a stable baseline policy
    
    Arguments:
        env: Gymnasium environment
    
    Returns: Stable Baseline3 PPO Model"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if policy == "MlpPolicy":
        if "cyclic" in environment_string or "tree" in environment_string:
            model = PPO(
                "MlpPolicy",
                env,
                policy_kwargs={"net_arch": [16]},
                n_steps=32,            # smaller rollout
                batch_size=32,         # match rollout size
                n_epochs=10,           # more passes per batch
                learning_rate=3e-4,    # safer LR
                device="cpu",
                ent_coef=0.0,
                verbose=0
            )
        else:
            model = PPO(
                "MlpPolicy",
                env,
                policy_kwargs={"net_arch": [16,16]},  # Single layer is actually fastest
                n_steps=256,
                batch_size=2048,        # Match n_steps for single batch processing
                n_epochs=1,           # KEY: Single epoch only
                learning_rate=5e-3,   # Higher LR to compensate for fewer epochs
                device='cpu',        # Your GPU is working fine
                verbose=0
            )

    else:
        model = PPO(policy, env, verbose=0,n_steps=256,device='cuda',batch_size=256*8,n_epochs=1)
    model.learn(total_timesteps=total_timesteps,callback=ProgressBarCallback(total_timesteps))  
    return model 

def train_two_stage_ppo_model(environment_string,env,concept_list,total_timesteps):
    """Train an environment by first predicting the observation
        then predicting the action
    
    Arguments:
        env: Gymnasium environment
        total_timesteps: Integer, number of timesteps to train for
    
    Returns: Stable Baseline3 PPO Model"""

    if environment_string in ["cart_pole"]:
        model = TwoStagePPO(create_two_stage_policy(TwoStageCartPoleExtractor,concept_list), env, state_loss_weight=1.0)
    elif environment_string in ["pong","boxing"]:
        model = TwoStagePPO(create_two_stage_policy(TwoStagePongExtractor,concept_list), env, state_loss_weight=1.0) 
    model.learn(total_timesteps=total_timesteps)
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

    


class SimpleQEstimator:
    """
    Simple neural network Q-function estimator for continuous state spaces.
    Uses Monte Carlo returns as training targets.
    """
    
    def __init__(self, state_dim: int, action_dim: int, policy: Callable):
        self.policy = policy
        
        # Simple 3-layer neural network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
    
    def collect_and_train(self, env, num_episodes: int = 100, gamma: float = 0.99):
        """Collect episodes and train Q-network in one go"""        
        all_transitions = []
        
        # Collect episodes
        for ep in range(num_episodes):            
            state, _ = env.reset()
            episode_transitions = []
            
            # Run episode
            while True:
                action = self.policy.predict(state)[0]
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                episode_transitions.append({
                    'state': state,
                    'action': [action],
                    'reward': reward
                })
                
                state = next_state
                if terminated or truncated:
                    break
            
            # Calculate returns (Monte Carlo targets)
            G = 0
            for i in reversed(range(len(episode_transitions))):
                G = episode_transitions[i]['reward'] + gamma * G
                episode_transitions[i]['return'] = G
            
            all_transitions.extend(episode_transitions)
        
        # Train on all collected data
        self._train_network(all_transitions)
    
    def _train_network(self, transitions, batch_size: int = 256, epochs: int = 50):
        """Train the Q-network"""
        print("Training Q-network...")
        
        # Prepare training data
        states = []
        actions = []
        returns = []
        
        for trans in transitions:
            states.append(trans['state'])
            actions.append(trans['action'])
            returns.append(trans['return'])
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        returns = torch.FloatTensor(returns)
        
        # Training loop
        dataset_size = len(states)
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, dataset_size, batch_size):
                batch_states = states[i:i+batch_size]
                batch_actions = actions[i:i+batch_size]
                batch_returns = returns[i:i+batch_size].reshape((-1,1))

                # Concatenate state and action
                inputs = torch.cat([batch_states, batch_actions], dim=1)
                
                # Forward pass
                predicted_q = self.q_network(inputs).squeeze()
                loss = self.loss_fn(predicted_q, batch_returns)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if epoch % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        input_tensor = torch.cat([state_tensor, action_tensor], dim=1)
        
        with torch.no_grad():
            return self.q_network(input_tensor).item()
    
    def save(self, filename: str):
        """Save the Q-network"""
        torch.save(self.q_network.state_dict(), filename)
    
    def load(self, filename: str):
        """Load a saved Q-network"""
        self.q_network.load_state_dict(torch.load(filename))


class TwoStageCartPoleExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space,concept_list,features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        self.cnn = nn.Sequential( nn.Conv2d(1, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(), nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten() )

        
        with torch.no_grad():
            cnn_out_size = self.cnn(torch.zeros(1, 1, 84, 84)).shape[1]

        self.state_predictor = nn.Sequential(
            nn.Linear(cnn_out_size, 128), nn.ReLU(),
            nn.Linear(128, len(concept_list))
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(len(concept_list), 64), nn.ReLU(),
            nn.Linear(64, features_dim), nn.ReLU()
        )
        
        self.last_predicted_state = None
        self.concept_list = concept_list
        
    def forward(self, obs):
        cnn_features = self.cnn(obs)
        predicted_state = self.state_predictor(cnn_features)
        self.last_predicted_state = predicted_state
        return self.mlp(predicted_state)


class TwoStagePongExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space,concept_list,features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        self.concept_list = concept_list
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        
        with torch.no_grad():
            cnn_out_size = self.cnn(torch.zeros(1, 4, 84, 84)).shape[1]
            

        self.state_predictor = nn.Sequential(
            nn.Linear(cnn_out_size, 128), nn.ReLU(),
            nn.Linear(128, len(concept_list))
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(len(concept_list), 64), nn.ReLU(),
            nn.Linear(64, features_dim), nn.ReLU()
        )
        
        self.last_predicted_state = None
        
    def forward(self, obs):
        cnn_features = self.cnn(obs)
        predicted_state = self.state_predictor(cnn_features)
        self.last_predicted_state = predicted_state
        return self.mlp(predicted_state)


def create_two_stage_policy(extractor,concept_list):
    class TwoStagePolicy(ActorCriticPolicy):
        def __init__(self,*args,**kwargs):
            extractor_kwargs = kwargs.get("features_extractor_kwargs", {})
            extractor_kwargs["concept_list"] = concept_list
            kwargs["features_extractor_kwargs"] = extractor_kwargs

            kwargs["features_extractor_class"] = extractor
            super().__init__(*args, **kwargs)
    return TwoStagePolicy

class TwoStagePPO(PPO):
    def __init__(self, *args, state_loss_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_loss_weight = state_loss_weight
        self.true_states = []
        self.rollout_buffer = InfoRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
    
    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, gym.spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value
            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                info=infos
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True



    
    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        
        clip_range = self.clip_range(self._current_progress_remaining)
        
        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                value_loss = F.mse_loss(rollout_data.returns, values)
                entropy_loss = -torch.mean(entropy) if entropy is not None else -torch.mean(-log_prob)

                # State prediction loss
                extractor = self.policy.features_extractor
                _ = extractor(rollout_data.observations)
                
                if hasattr(extractor, 'last_predicted_state') and extractor.last_predicted_state is not None:
                    true_states = torch.Tensor([[c(i[0]['observation']) for c in extractor.concept_list] for i in rollout_data.infos]).to(self.device)
                    state_loss = F.mse_loss(extractor.last_predicted_state, true_states)
                    self.per_state_loss = np.mean(np.abs(extractor.last_predicted_state.cpu().detach().numpy()-true_states.cpu().detach().numpy()),axis=0)

                else:
                    state_loss = torch.tensor(0.0, device=self.device)

                loss = (policy_loss + self.vf_coef * value_loss + 
                       self.ent_coef * entropy_loss + self.state_loss_weight * state_loss)

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

        self._n_updates += self.n_epochs


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
    if environment_string == "mimic":
        return get_average_reward_mimic(env,model,max_steps_per=100)

    return get_average_reward(env,model)

def get_concept_labels(env,model,concept_list,num_steps = 5000):
    X_all, Y_all = [], []

    obs, infos = env.reset()
    done, steps = False, 0
    while steps < num_steps:
        # store raw observation
        X_all.append(obs)
        # store concept values
        Y_all.append([[c(inf['observation']) for c in concept_list] for inf in infos])
        # take a random action just to explore
        if np.random.random() < 0.1:
            action = [env.action_space.sample() for i in range(len(obs))]
        else:
            action = model.predict(obs,deterministic=True)[0]
        obs, _, terminated, truncated, infos = env.step(action)
        steps += 1

    X = np.stack(X_all)   # shape (N, obs_dim)
    Y = np.stack(Y_all)   # shape (N, num_concepts)
    print(X.shape,Y.shape)

    X = X.reshape(-1, *X.shape[2:])   # (100*8, 4, 84, 84)
    Y = Y.reshape(-1, Y.shape[2])     # (100*8, 24)

    return X, Y

class MultiLabelCNN(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # BCEWithLogitsLoss will handle sigmoid internally

def train_concept_predictor(X,Y):

    # X: (N,4,84,84), Y: (N,K)
    X_torch = torch.tensor(X, dtype=torch.float32)
    Y_torch = torch.tensor(Y, dtype=torch.float32)  # float for BCE

    dataset = TensorDataset(X_torch, Y_torch)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Modified CNN

    num_outputs = Y.shape[1]
    model = MultiLabelCNN(num_outputs)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    def f1_score_multilabel(y_true, y_pred, threshold=0.5, eps=1e-8):
        y_pred = (torch.sigmoid(y_pred) > threshold).float()
        tp = (y_true * y_pred).sum(dim=0)
        fp = ((1 - y_true) * y_pred).sum(dim=0)
        fn = (y_true * (1 - y_pred)).sum(dim=0)
        f1 = 2 * tp / (2 * tp + fp + fn + eps)
        return f1.mean().item()

    # Training loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        f1_total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            f1_total += f1_score_multilabel(yb, out)
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}, F1: {f1_total/len(loader):.4f}")
    return model 

def score_concept_predictors(model,X,Y):
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.long)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    num_outputs = Y.shape[1]

    # --- 1️⃣ Compute per-output accuracy / F1 ---
    acc_list = []
    f1_list = []

    with torch.no_grad():
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X,Y),
                                            batch_size=64, shuffle=False)
        # accumulate predictions and labels
        all_preds = []
        all_labels = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = torch.sigmoid(model(xb))  # convert logits to 0-1 probabilities
            all_preds.append(out)
            all_labels.append(yb)
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

    for i in range(num_outputs):
        pred_col = (all_preds[:, i] > 0.5).float()
        label_col = all_labels[:, i]
        acc = (pred_col == label_col).float().mean().item()
        acc_list.append(acc)

        # F1 per output
        tp = (pred_col * label_col).sum()
        fp = (pred_col * (1 - label_col)).sum()
        fn = ((1 - pred_col) * label_col).sum()
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        f1_list.append(f1.item())
    return acc_list, f1_list 