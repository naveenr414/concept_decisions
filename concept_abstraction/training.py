import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Callable
import numpy as np
import gymnasium as gym


from concept_abstraction.env_utils import get_average_reward
from concept_abstraction.environments import eval_mimic_model


def train_model(env,total_timesteps=10000):
    """Train an environment according to a stable baseline policy
    
    Arguments:
        env: Gymnasium environment
    
    Returns: Stable Baseline3 DQN Model"""
    model = DQN("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps, log_interval=4)
    return model 

def train_ppo_model(env,total_timesteps=150_000,policy="MlpPolicy",batch_size=256, n_steps=128):
    """Train an environment according to a stable baseline policy
    
    Arguments:
        env: Gymnasium environment
    
    Returns: Stable Baseline3 PPO Model"""

    model = PPO(policy, env, verbose=0)
    model.learn(total_timesteps=total_timesteps)  
    return model 

def train_two_stage_ppo_model(env,total_timesteps):
    """Train an environment by first predicting the observation
        then predicting the action
    
    Arguments:
        env: Gymnasium environment
        total_timesteps: Integer, number of timesteps to train for
    
    Returns: Stable Baseline3 PPO Model"""

    model = TwoStagePPO(TwoStagePolicy, env, state_loss_weight=1.0)
    model.learn(total_timesteps=total_timesteps)
    return model 

class RandomAgent:
    """Random policy that randomly selections actions"""
    def __init__(self, env):
        self.action_space = env.action_space

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        action = self.action_space.sample()
        return action, None


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


class TwoStageExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),  
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            cnn_out_size = self.cnn(torch.zeros(1, 1, 84, 84)).shape[1]
            
        self.state_predictor = nn.Sequential(
            nn.Linear(cnn_out_size, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, features_dim), nn.ReLU()
        )
        
        self.last_predicted_state = None
        
    def forward(self, obs):
        cnn_features = self.cnn(obs)
        predicted_state = self.state_predictor(cnn_features)
        self.last_predicted_state = predicted_state
        return self.mlp(predicted_state)

class TwoStagePolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["features_extractor_class"] = TwoStageExtractor
        super().__init__(*args, **kwargs)

class TwoStagePPO(PPO):
    def __init__(self, *args, state_loss_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_loss_weight = state_loss_weight
        self.true_states = []
        
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        self.true_states = []
        
        # Call parent method and collect true states
        result = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
                
        return result
    
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
                    # Simplified: use zeros as placeholder for true states
                    # In practice, you'd store actual true states from info['observation']
                    batch_size = len(rollout_data.observations)
                    true_states = torch.zeros(batch_size, 4, device=self.device)
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

    if environment_string == "mimic":
        return eval_mimic_model(additional_info['physpol'],model,additional_info['concept_list'],seed) 
    else:
        return get_average_reward(env,model)
