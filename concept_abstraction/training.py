import time
import random
from collections import namedtuple, deque
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
from sklearn.metrics import accuracy_score

import gymnasium as gym
from tqdm import tqdm

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.callbacks import BaseCallback

from concept_abstraction.env_utils import get_average_reward, get_average_reward_mimic
from concept_abstraction.environments import eval_mimic_model

InfoRolloutBufferSamples = namedtuple(
    "InfoRolloutBufferSamples",
    ["observations", "actions", "old_values", "old_log_prob", "advantages", "returns", "infos"]
)

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, update_interval=0.01, print_interval=0.1,
                 reward_window=100, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.update_interval = update_interval
        self.print_interval = print_interval
        self.ep_rewards = deque(maxlen=reward_window)
        self.last_update = 0
        self.last_print = 0

    def _on_step(self) -> bool:
        # collect episode rewards from infos (Monitor must be used)
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
        # refresh progress bar if you have one
        return True

    def _log_grad_norm(self):
        # compute gradient L2 norm (policy + value nets)
        total_norm = 0.0
        for p in self.model.policy.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.logger.record("diagnostics/grad_norm", total_norm)

    def _on_rollout_end(self):
        # record diagnostics available from SB3 logger and also compute extra stats
        logger_data = self.model.logger.name_to_value

        # record avg reward from our buffer
        avg_reward = np.mean(self.ep_rewards) if self.ep_rewards else np.nan
        self.logger.record("diagnostics/avg_reward_window", float(avg_reward))

        # copy some SB3 metrics if present
        for k in ("train/explained_variance", "train/value_loss",
                  "train/policy_gradient_loss", "train/approx_kl",
                  "train/clip_fraction"):
            if k in logger_data:
                self.logger.record(k, float(logger_data[k]))

        # compute & log grad norm (requires grads to exist; SB3 computes grads during train step)
        try:
            self._log_grad_norm()
        except Exception:
            pass

        # If you can access sampled returns/advantages from locals, log their stats
        returns = self.locals.get("returns", None)
        advantages = self.locals.get("advantages", None)
        if returns is not None:
            r = np.array(returns)
            self.logger.record("diagnostics/returns_mean", float(r.mean()))
            self.logger.record("diagnostics/returns_std", float(r.std()))
        if advantages is not None:
            a = np.array(advantages)
            self.logger.record("diagnostics/advantages_mean", float(a.mean()))
            self.logger.record("diagnostics/advantages_std", float(a.std()))

        # print a compact line
        if self.num_timesteps - self.last_print >= self.total_timesteps * self.print_interval:
            ev = logger_data.get("train/explained_variance", None)
            vl = logger_data.get("train/value_loss", None)
            kl = logger_data.get("train/approx_kl", None)
            cf = logger_data.get("train/clip_fraction", None)
            gn = self.logger.name_to_value.get("diagnostics/grad_norm", None)
            self.logger.record("diagnostics/last_printed_steps", self.num_timesteps)
            print(f"Step: {self.num_timesteps} | AvgR: {avg_reward:.3f} | EV: {ev} | VLoss: {vl} | KL: {kl} | ClipF: {cf} | GradN: {gn}")
            self.last_print = self.num_timesteps

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
        elif environment_string == "pong" or environment_string == "boxing":
            model = PPO( "MlpPolicy", env, policy_kwargs=dict(net_arch=[128,128], activation_fn=torch.nn.ReLU), n_steps=4096, batch_size=256, n_epochs=4, learning_rate=1e-3, gamma=0.995, ent_coef=0.015, verbose=0, device='cpu' )
        elif environment_string == "mimic":
            model = PPO(
                "MlpPolicy", 
                env,
                policy_kwargs=dict(
                    net_arch=[128,128],        # Standard size for 141-dim observations
                    activation_fn=torch.nn.ReLU # Back to ReLU - fine for this env
                ),
                n_steps=512,               # Standard rollout length
                batch_size=256,              # Reasonable batch size
                n_epochs=4,                 # Moderate epochs
                learning_rate=1e-3,         # Standard PPO learning rate
                gamma=0.99,                 # Standard discount
                ent_coef=0.02,              # Moderate entropy
                vf_coef=0.5,                # Standard value function weight
                clip_range=0.2,             # Standard PPO clip
                verbose=0,
                device='cpu',
                normalize_advantage=True
            )
        elif environment_string == "mimic_raw":
            print("Running MIMIC RAW")
            model = PPO(
                "MlpPolicy",
                env,
                policy_kwargs={"net_arch": [128,128]},  # Single layer is actually fastest
                n_steps=128,
                batch_size=32,        # Match n_steps for single batch processing
                n_epochs=4,           # KEY: Single epoch only
                learning_rate=6e-4,   # Higher LR to compensate for fewer epochs
                device='cpu',        # Your GPU is working fine
                verbose=0,
                ent_coef=0.02,
                gamma=0.99,
            )
        else:
            model = PPO(
                "MlpPolicy",
                env,
                policy_kwargs={"net_arch": [32,32]},  # Single layer is actually fastest
                n_steps=256,
                batch_size=1024,        # Match n_steps for single batch processing
                n_epochs=1,           # KEY: Single epoch only
                learning_rate=5e-3,   # Higher LR to compensate for fewer epochs
                device='cpu',        # Your GPU is working fine
                verbose=0
            )

    else:
        model = PPO(
            "CnnPolicy",
            env,                          # your vec_env with 8 parallel pixel envs
            n_steps=128,                  # per env
            batch_size=256,               # divides total rollout = 1024
            n_epochs=4,
            learning_rate=2.5e-4,
            gamma=0.99,
            clip_range=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            gae_lambda=0.95,
            max_grad_norm=0.5,
            device="cuda",                # highly recommended with pixels
            verbose=0
        )
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
        # return get_average_reward_mimic(env,model,max_steps_per=100)
        return eval_mimic_model(additional_info['physpol'],model,additional_info['concept_list'],additional_info['clusterer'],seed)
    # TODO: Add back in 
    return get_average_reward(env,model)

# ----------------------------
# Lightweight CNN for 1-channel input
# ----------------------------
class CartPoleConceptCNN(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 84x84 -> 42x42
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 42x42 -> 21x21
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 21x21 -> 11x11
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64*11*11, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )

    def forward(self, x):
        return self.fc(self.features(x))

class WeightedBCEWithLogits(nn.Module):
    def __init__(self, pos_weights=None, smoothing=0.0):
        super().__init__()
        self.pos_weights = pos_weights
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    def forward(self, logits, targets):
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)

def get_concept_labels_avg(env, model, concept_list, num_steps=5000, batch_size=100):
    obs, infos = env.reset()
    steps = 0

    while steps < num_steps:
        X_batch, Y_batch = [], []
        batch_steps = 0
        while batch_steps < batch_size and steps < num_steps:
            # Average 4 frames
            X_avg = obs.mean(axis=1, keepdims=True)  # (batch, 1, 84,84)
            X_batch.append(X_avg)
            Yb = np.array([[c(inf['observation']) for c in concept_list] for inf in infos])
            Y_batch.append(Yb)

            # Take action
            if np.random.random() < 0.1:
                action = [env.action_space.sample() for _ in range(len(obs))]
            else:
                concepts = [[c(inf['observation']) for c in concept_list] for inf in infos]
                action = model.predict(concepts, deterministic=True)[0]

            obs, _, terminated, truncated, infos = env.step(action)
            steps += 1
            batch_steps += 1

            if np.random.random() < 0.05:
                obs, infos = env.reset()

        X_batch = np.concatenate(X_batch, axis=0)
        Y_batch = np.concatenate(Y_batch, axis=0)
        yield X_batch, Y_batch
        del X_batch, Y_batch

def train_concept_predictor(ground_truth_gym_env,gold_model,concept_list,idx): 
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
    criterion = WeightedBCEWithLogits(pos_weights=pos_weights, smoothing=0)
    optimizer = optim.Adam(model.parameters(), lr=5e-3)

    best_f1 = 0.0
    epochs = 20
    device = 'cuda'
    num_steps= 1000
    batch_size = 100
    env = ground_truth_gym_env
    # ----------------------------
    # Training loop
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


    start_time = time.time()
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
        # Validation on the same generator (or separate if available)
        # ----------------------------
        model.eval()
        val_loss, val_count, f1_list = 0.0, 0, []
        with torch.no_grad():
            for Xb, Yb in val_loader(val_X, val_Y): #get_concept_labels_avg(env, gold_model, concept_list, num_steps=1000, batch_size=batch_size)
                X_tensor = torch.tensor(Xb, dtype=torch.float32, device=device)
                Y_tensor = torch.tensor(Yb, dtype=torch.float32, device=device)
                logits = model(X_tensor)
                loss = criterion(logits, Y_tensor)
                val_loss += loss.item()
                val_count += 1

                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                f1_list.append(accuracy_score(Y_tensor.cpu().numpy().ravel(), preds.ravel()))

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
    for Xb, Yb in val_loader(val_X, val_Y): #get_concept_labels_avg(env, gold_model, concept_list, num_steps=1000, batch_size=batch_size)
        X_tensor = torch.tensor(Xb, dtype=torch.float32, device=device)
        Y_tensor = torch.tensor(Yb, dtype=torch.float32, device=device)
        logits = model(X_tensor)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        acc_list += np.sum(preds == Y_tensor.cpu().numpy(),axis=0)
        tot += len(logits)


    return model, ((acc_list/tot).tolist())