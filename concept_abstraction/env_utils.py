from concept_abstraction.environments import TreeRepeatEnv, Cyclic4StateEnv
import numpy as np
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable


def create_environment_from_string(environment_string,environment_nodes,concept_list,error):
    """Initialize an environment based on a string
    
    Arguments:
        environment_string: String, e.g., tree or cycle
        concept_list: List of concepts which are used to define the state
        error: float, erorr in concept prediction
    
    Returns: Gymasium Environment"""
    
    models_by_string = {
        'tree': TreeRepeatEnv, 
        'cycle': Cyclic4StateEnv,
    }
    
    return models_by_string[environment_string](environment_nodes,concept_list,error)

def get_baseline_concept_sets(environment_string,environment_nodes):
    """Retrieve a list of potential concept sets from a string
    
    Arguments:
        environment_string: String, e.g., tree or cycle
    
    Returns: List of lists, representing different concept 
        combinations"""
    
    baseline_concepts_by_string = {
        'tree': [list(range(int(np.log2(environment_nodes+1)))),[int(np.log2(environment_nodes+1))]],
        'cycle': [list(range(0,environment_nodes-1))] + [[j] for j in list(range(0,environment_nodes-1))]
    }

    return baseline_concepts_by_string[environment_string]

def get_values(env, q_net, num_rollouts=10, max_steps=100, gamma=1):
    """
    Estimate V^{π}(s) for all states using rollouts from each state under the policy implied by q_net.
        Currently it does so with no discounting, but with average reward

    Args:
        env: The environment with .all_states and .state access.
        q_net: Trained Q-network (maps obs to Q-values).
        num_rollouts: Number of rollouts per state.
        max_steps: Max steps per rollout.
        gamma: Discount factor.
    
    Returns:
        List of value estimates V(s) for each s in env.all_states.
    """
    values = []

    for s in env.all_states:
        total_return = 0.0

        for _ in range(num_rollouts):
            env.reset()
            try:
                env.unwrapped.state = s
            except AttributeError:
                raise AttributeError("env must support setting env.unwrapped.state directly")
            total_reward = 0.0
            done = False
            steps = 0

            while not done and steps < max_steps:
                obs = env.get_observation()
                action, _ = q_net.predict(obs, deterministic=True)
                obs, reward, done, _, _ = env.step(action)

                total_reward += reward
                steps += 1

            avg_reward = total_reward / steps
            total_return += avg_reward

        values.append(total_return / num_rollouts)

    return values

def get_average_reward(env,model):
    """Given an environment, get the average reward following a
        policy, model
        
    Arguments:
        model: Object with 'predict' function
        env: Gymnasium environment
    
    Returns: Float, the average reward"""

    total_reward = 0

    for restart in range(10):
        observation, info = env.reset()
        for _ in range(1000):
            # Random action: 0 (left) or 1 (right)
            action = model.predict(observation)[0]

            # Take a step in the environment
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward 
            # End episode if done
            if terminated or truncated:
                break
    total_reward /= 10000
    return total_reward

def list_to_string(obs):
    """Convert a list of numbers to its string concatenated version
    
    Arguments:
        obs: Some numpy array or list
    
    Returns: A string concatenated version"""

    return " ".join([str(j) for j in list(obs)])

def get_observed_transition(model,env):
    """Given an environment, get the observed transition matrix
        
    Arguments:
        model: Object with 'predict' function
        env: Gymnasium environment
    
    Returns: Matrix of size State x Action x State"""

    transition_dict = {}
    reward_dict = {}

    for restart in range(10):
        observation, info = env.reset()
        for _ in range(1000):
            # Random action: 0 (left) or 1 (right)
            action = model.predict(observation)[0]

            # Take a step in the environment
            next_observation, reward, terminated, truncated, info = env.step(action)
            
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
            
            # End episode if done
            if terminated or truncated:
                break
    return transition_dict, reward_dict

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
    
    def collect_and_train(self, env: gym.Env, num_episodes: int = 100, gamma: float = 0.99):
        """Collect episodes and train Q-network in one go"""
        print(f"Collecting {num_episodes} episodes and training...")
        
        all_transitions = []
        
        # Collect episodes
        for ep in range(num_episodes):
            if ep % 10 == 0:
                print(f"Episode {ep}/{num_episodes}")
            
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
        print(f"Q-network saved to {filename}")
    
    def load(self, filename: str):
        """Load a saved Q-network"""
        self.q_network.load_state_dict(torch.load(filename))
        print(f"Q-network loaded from {filename}")


# Example usage
if __name__ == "__main__":
    # Create environment
    env = gym.make('Pendulum-v1')
    
    # Your golden_model policy (replace this with your actual policy)
    def golden_model(state):
        # Example: simple policy that tries to balance
        # Replace this with your actual trained policy
        return np.array([np.tanh(state[0] + state[1])])  # Simple heuristic
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create Q-function estimator
    q_estimator = SimpleQEstimator(state_dim, action_dim, golden_model)
    
    # Collect episodes and train
    q_estimator.collect_and_train(env, num_episodes=500)
    
    # Test the Q-function
    state, _ = env.reset()
    action = golden_model(state)
    q_value = q_estimator.get_q_value(state, action)
    
    print(f"\nExample Q-value:")
    print(f"State: {state}")
    print(f"Action: {action}")
    print(f"Q-value: {q_value:.4f}")
    
    # Save the trained Q-function
    q_estimator.save("simple_q_function.pth")
    
    env.close()