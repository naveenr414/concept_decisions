import numpy as np
import gymnasium as gym
from collections import defaultdict

def get_average_reward(env,model):
    """Given an environment, get the average reward following a
        policy, model
        
    Arguments:
        model: Object with 'predict' function
        env: Gymnasium environment
    
    Returns: Float, the average reward"""

    total_reward = 0
    steps = 0
    num_restarts = 10
    max_steps = 10000

    for restart in range(num_restarts):
        observation, info = env.reset()
        for _ in range(max_steps):
            action = model.predict(observation,deterministic=True)[0]
            # Take a step in the environment            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward 
            steps += 1
            # End episode if done
            if terminated or truncated:
                break
    env.close()
    return total_reward/num_restarts

def rollout_pi_estimates(model,env,concept_list,num_rollouts=100, max_steps=1000):
    """Estimate the policy/action for different states"""

    pair_list = []

    for _ in range(num_rollouts):
        obs, info = env.reset()
        state = obs.copy()

        done = False
        steps = 0
        concept = [c(info['observation']) for c in concept_list]

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            pair_list.append((concept,action))
            obs, reward, terminated, truncated, info = env.step(action)
            concept = [c(info['observation']) for c in concept_list]

            done = terminated or truncated
            steps += 1
    return pair_list

import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QNetwork(nn.Module):
    """Simple neural network for Q-function approximation"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)

class TDQLearning:
    """TD Learning for Q-value estimation"""
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-network
        self.q_net = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        
    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def get_q_value(self, state, action):
        """Get Q-value for a specific state-action pair"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state_tensor)
            return q_values[0][action].item()
    
    def get_action(self, state, deterministic=False):
        """Get action using epsilon-greedy policy"""
        if not deterministic and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state_tensor)
            return q_values.argmax().item()
    
    def update(self, batch_size=32):
        """Update Q-network using TD learning"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Current Q-values
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values (for TD target)
        with torch.no_grad():
            next_q_values = self.q_net(next_states).max(1)[0]
            td_targets = rewards + (self.gamma * next_q_values * ~dones)
        
        # TD loss
        loss = nn.MSELoss()(current_q_values.squeeze(), td_targets)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def rollout_q_estimates_td(model, env, concept_list, states=None, gamma=0.99, 
                          num_episodes=100, max_steps=1000, epsilon=0.05,
                          learning_rate=0.001, update_freq=10):
    """
    Estimate Q-values using TD learning instead of Monte Carlo rollouts.
    Returns a list of (state, action, q_estimate) tuples.
    """
    
    # Determine dimensions
    if states is not None:
        state_dim = len(concept_list)  # Using concept representation
    else:
        obs, _ = env.reset()
        state_dim = len(concept_list)
    
    # Get action dimension
    if hasattr(env.action_space, 'n'):
        action_dim = env.action_space.n
    else:
        # For continuous action spaces, you'd need to discretize or use different approach
        raise ValueError("This implementation assumes discrete action space")
    
    # Initialize TD learner
    td_learner = TDQLearning(state_dim, action_dim, lr=learning_rate, gamma=gamma, epsilon=epsilon)
    
    # Collect experiences and learn
    all_state_actions = set()
    
    for episode in range(num_episodes):
        # Choose starting state
        if states is not None:
            state = states[np.random.randint(len(states))]
            obs, info = env.reset()
            if hasattr(env.env, 'state'):
                env.env.state = state.copy()
            elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'state'):
                env.unwrapped.state = state.copy()
            else:
                obs = state.copy()
        else:
            obs, info = env.reset()
            state = obs.copy()
        
        # Get concept representation
        concept = np.array([c(info['observation'] if 'observation' in info else obs) for c in concept_list])
        
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Choose action (mix of model policy and exploration)
            if steps == 0 and np.random.rand() < 0.5:
                # Sometimes use random action for first step (like original)
                action = env.action_space.sample()
            else:
                # Use model policy with some exploration
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)
            
            # Store current state-action for later Q-value extraction
            all_state_actions.add((tuple(concept), action))
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Get next concept representation
            if not done:
                next_concept = np.array([c(info['observation'] if 'observation' in info else next_obs) 
                                       for c in concept_list])
            else:
                next_concept = np.zeros_like(concept)  # Terminal state
            
            # Add experience to TD learner
            td_learner.add_experience(concept, action, reward, next_concept, done)
            
            # Update Q-network periodically
            if len(td_learner.replay_buffer) > 32 and steps % update_freq == 0:
                td_learner.update()
            
            # Update for next iteration
            obs = next_obs
            concept = next_concept
            steps += 1
    
    # Final training updates
    for _ in range(50):  # Extra training at the end
        if len(td_learner.replay_buffer) > 32:
            td_learner.update()
    
    # Extract Q-values for all encountered state-action pairs
    q_estimate_list = []
    for state_tuple, action in all_state_actions:
        state_array = np.array(state_tuple)
        q_value = td_learner.get_q_value(state_array, action)
        q_estimate_list.append((state_array, action, q_value))
    
    return q_estimate_list



# def rollout_q_estimates(model, env, concept_list,states=None, gamma=0.99, num_rollouts=100, max_steps=1000, epsilon=0.05):
#     """
#     Estimate Q-values for given states using Monte Carlo rollouts with optional exploration.
    
#     Returns a list of (state, action, q_estimate) tuples.
#     """
#     results = defaultdict(list)

#     for _ in range(num_rollouts):
#         # Choose starting state
#         if states is not None:
#             state = states[np.random.randint(len(states))]
#             obs, _ = env.reset()
#             if hasattr(env.env, 'state'):
#                 env.env.state = state.copy()
#             elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'state'):
#                 env.unwrapped.state = state.copy()
#             else:
#                 obs = state.copy()
#         else:
#             obs, info = env.reset()
#             state = obs.copy()
#         concept = [c(info['observation']) for c in concept_list]

#         total_reward = 0.0
#         discount = 1.0
#         done = False
#         steps = 0
#         first_action = None 

#         while not done and steps < max_steps:
#             # Epsilon-greedy exploration
#             if steps == 0:
#                 if np.random.rand() < 0.5:
#                     action = env.action_space.sample()
#                 else:
#                     action, _ = model.predict(obs, deterministic=True)
#                 first_action = int(action)
#             else:
#                 action, _ = model.predict(obs, deterministic=True)

#             obs, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated
#             total_reward += discount * reward
#             discount *= gamma
#             steps += 1

#         # Record first-step action for this state
#         results[tuple(concept)].append((first_action, total_reward))
#     # Aggregate Q-estimates (average across rollouts)
#     q_estimate_list = []

#     for s_key, vals in results.items():
#         # vals is a list of tuples: (first_action_taken, total_reward)
#         action_groups = defaultdict(list)
#         for first_action, total_reward in vals:
#             action_groups[first_action].append(total_reward)

#         # Aggregate by first action
#         for action, rewards in action_groups.items():
#             avg_q = np.mean(rewards)
#             q_estimate_list.append((np.array(s_key), action, avg_q))

#     return q_estimate_list

def get_average_reward_gym(env, model, n_episodes=10):
    total_reward = 0
    episode_count = 0

    while episode_count < n_episodes:
        obs = env.reset()  # VecEnv: returns batch of obs
        done = [False] * env.num_envs

        while not all(done):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            total_reward += sum(rewards)
            done = dones
        episode_count += 1

    env.close()
    return total_reward / n_episodes


def list_to_string(obs):
    """Convert a list of numbers to its string concatenated version
    
    Arguments:
        obs: Some numpy array or list
    
    Returns: A string concatenated version"""

    return " ".join([str(j) for j in list(obs)])

def get_transition_reward_rollout(model,env,restarts=10,steps=1000):
    """Given an environment, run a series of rollouts
        to get the transition and rewards
        
    Arguments:
        model: Object with 'predict' function
        env: Gymnasium environment
    
    Returns: Two things: transitions, a dictionary
        mapping states (as tuples) to a dictionary
        mapping actions -> next state (as a tuple)
        And Reward, a dictionary mapping states
            to rewards"""

    transition_dict = {}
    reward_dict = {}

    for _ in range(restarts):
        observation, _ = env.reset()
        for _ in range(steps):
            action = model.predict(observation)[0]
            next_observation, reward, terminated, truncated, _ = env.step(action)
            
            action_string = str(action)
            obs_string = list_to_string(observation)
            next_obs_string = list_to_string(next_observation)
            if obs_string not in transition_dict:
                transition_dict[obs_string] = {}
                reward_dict[obs_string] = {}
            
            if (action_string,next_obs_string) not in transition_dict[obs_string]:
                transition_dict[obs_string][(action_string,next_obs_string)] = 0
            
            if action_string not in reward_dict[obs_string]:
                reward_dict[obs_string][action_string] = reward

            transition_dict[obs_string][(action_string,next_obs_string)] += 1
            observation = next_observation
            if terminated or truncated:
                break
    return transition_dict, reward_dict

def get_recordable(env):
    """Add recording and save the recording to logs/videos
    
    Arguments:
        env: Gymnasium environment
    
    Returns: Gymnasium Environment
    
    Side Effects: Wraps the environment for video recording"""

    env = gym.wrappers.RecordVideo(env, video_folder="../../runs/videos/", episode_trigger=lambda e: True)
    return env 