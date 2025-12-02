import numpy as np
import gymnasium as gym
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import random



class QNetwork(nn.Module):
    """Q-Network with proper initialization"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Proper initialization for stability
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x):
        return self.network(x)


class TDQLearning:
    """Stable TD Learning optimized for sparse reward environments"""
    def __init__(self, state_dim, action_dim, lr=0.0001, gamma=0.99, epsilon=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Networks with proper initialization
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # Copy weights to target network
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # Keep target network in eval mode
        
        # Replay buffer - ALWAYS add experiences immediately
        self.replay_buffer = deque(maxlen=50000)
        
        self.update_count = 0
        self.target_update_freq = 500  # Less frequent updates for stability
        
        # Track loss for monitoring
        self.recent_losses = deque(maxlen=100)
        
    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to replay buffer IMMEDIATELY"""
        # CRITICAL FIX: Add to replay buffer right away, not only when done
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def get_q_value(self, state, action):
        """Get Q-value for a specific state-action pair"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state_tensor)
            return q_values[0][action].item()
    
    def get_all_q_values(self, state):
        """Get all Q-values for a state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state_tensor)
            return q_values[0].numpy()
    
    def get_action(self, state, deterministic=False):
        """Get action using epsilon-greedy policy"""
        if not deterministic and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state_tensor)
            return q_values.argmax().item()
    
    def update(self, batch_size=32):
        """Stable update with loss monitoring"""
        if len(self.replay_buffer) < batch_size * 2:  # Need more experiences
            return None
        
        # Sample batch
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        
        # Current Q-values
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network (Double DQN style)
        with torch.no_grad():
            # Use online network to select actions
            next_actions = self.q_net(next_states).argmax(1)
            # Use target network to evaluate actions
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            
            # TD target: r + gamma * max_a Q(s', a) for non-terminal states
            td_targets = rewards + (self.gamma * next_q_values * ~dones)
            
            # Less aggressive clipping - allow Q-values up to 500 for long episodes
            td_targets = torch.clamp(td_targets, -500, 500)
        
        # Calculate loss
        loss = nn.MSELoss()(current_q_values.squeeze(), td_targets)
        
        # Check for exploding loss
        if loss.item() > 1000:
            print(f"Warning: High loss detected: {loss.item():.2f}")
            # Don't skip the update, but clip gradients more aggressively
        
        # Update network with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.update_count += 1
        self.recent_losses.append(loss.item())
        
        # Update target network less frequently
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            avg_recent_loss = np.mean(self.recent_losses) if self.recent_losses else 0
            print(f"Updated target network at step {self.update_count}, Avg recent loss: {avg_recent_loss:.4f}")
        
        return loss.item()
    
    def decay_epsilon(self, decay_rate=0.9995, min_epsilon=0.005):
        """Gradual epsilon decay - lower minimum for better convergence"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
    
    def get_loss_stats(self):
        """Get loss statistics for monitoring"""
        if not self.recent_losses:
            return 0, 0, 0
        losses = list(self.recent_losses)
        return np.mean(losses), np.std(losses), np.max(losses)

def rollout_q_estimates_td(model, env, concept_list, states=None, gamma=0.99, 
                                total_timesteps=100_000, epsilon=0.1,
                                learning_rate=1e-4, update_freq=20, initial_random=0.3,
                                mimic=False, final_training=1_000, get_td_learner=False,use_initial_random=True):
    """
    Stable Q-value estimation for sparse reward environments
    """
    num_envs = env.num_envs
    state_dim = len(concept_list)

    if hasattr(env.action_space, "n"):
        action_dim = env.action_space.n
    else:
        raise ValueError("This implementation assumes discrete action space")

    # Create stable TD learner
    td_learner = TDQLearning(state_dim, action_dim, lr=learning_rate, gamma=gamma, epsilon=epsilon)

    all_state_actions = set()
    losses = []
    episode_rewards = []

    episodes_completed = 0
    episode_reward_sums = [0 for _ in range(num_envs)]
    
    # Loss monitoring
    loss_explosion_count = 0
    max_allowed_explosions = 5

    obs, infos = env.reset()
    steps = 0

    # Initial concept extraction - obs is already the observation array
    concepts = np.zeros((num_envs, state_dim))
    for i in range(num_envs):
        # correct selection of observation
        info_i = infos[i] if infos and len(infos) > i else {}

        # use obs[i] by default; override if info has 'observation'
        raw_obs = info_i.get("observation", obs[i])

        # ensure torch tensor with correct shape (always at least 1 batch dim)
        # actual_obs = torch.as_tensor(raw_obs)
        # if actual_obs.ndim == 1:
        #     actual_obs = actual_obs.unsqueeze(0)   # (1, obs_dim)
        #     actual_obs = actual_obs.unsqueeze(0)   # (1, obs_dim)
        # elif actual_obs.ndim == 2:
        #     actual_obs = actual_obs.unsqueeze(0)   # (1, obs_dim)

        
        # compute concepts
        concept_values = []
        for c in concept_list:
                concept_values.append(c(raw_obs))
        concepts[i] = concept_values 

        # for c in concept_list:
        #     concept_values.append(c(actual_obs).squeeze(0))  # each concept returns a scalar/tensor
        # concepts[i] = torch.stack(concept_values).cpu().numpy()

    print("Starting stable training for sparse rewards...")
    
    print("Total of {} steps".format(total_timesteps))

    while steps < total_timesteps:
        if (steps+1)%100 == 0:
            print("On step {}".format(steps))

        if steps % 5000 == 0:
            loss_mean, _, _ = td_learner.get_loss_stats()
            print(f"Step {steps}/{total_timesteps}, Loss mean: {loss_mean:.4f}")
        actions = model.predict(obs)[0]
        use_random = use_initial_random and ((steps < total_timesteps // 5) and (np.random.rand() < initial_random))
        if use_random:
            actions = [env.action_space.sample() for i in range(num_envs)]
        for i in range(num_envs):
            all_state_actions.add((tuple(concepts[i]), actions[i]))
        
        # Step the environment
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        dones = np.logical_or(terms, truncs)

        # # Compute next concepts for all environments
        next_concepts = np.zeros_like(concepts)
        for i in range(num_envs):
            # correct selection of observation
            info_i = infos[i] if infos and len(infos) > i else {}

            # use obs[i] by default; override if info has 'observation'
            raw_obs = info_i.get("observation", next_obs[i])

            # ensure torch tensor with correct shape (always at least 1 batch dim)
            # actual_obs = torch.as_tensor(raw_obs)
            # if actual_obs.ndim == 1:
            #     actual_obs = actual_obs.unsqueeze(0)   # (1, obs_dim)
            #     actual_obs = actual_obs.unsqueeze(0)   # (1, obs_dim)
            # elif actual_obs.ndim == 2:
            #     actual_obs = actual_obs.unsqueeze(0)   # (1, obs_dim)


            # compute concepts
            concept_values = []
            for c in concept_list:
                    concept_values.append(c(raw_obs))
            next_concepts[i] = concept_values 
        episode_reward_sums += rewards

        # Store experiences and track rewards
        for i in range(num_envs):
            td_learner.add_experience(concepts[i], actions[i], rewards[i], next_concepts[i], dones[i])

        # Update less frequently for stability
        if len(td_learner.replay_buffer) >= 128 and steps % update_freq == 0:
            loss = td_learner.update(batch_size=64)
            if loss is not None:
                losses.append(loss)
                
                # Monitor for loss explosion
                if loss > 500:
                    loss_explosion_count += 1
                    print(f"Loss explosion detected: {loss:.2f} (count: {loss_explosion_count})")
                    
                    if loss_explosion_count > max_allowed_explosions:
                        print("Too many loss explosions, reducing learning rate")
                        for param_group in td_learner.optimizer.param_groups:
                            param_group['lr'] *= 0.5
                        loss_explosion_count = 0

        # Handle episode completion and logging
        for i in range(num_envs):
            if dones[i]:
                episodes_completed += 1
                episode_rewards.append(episode_reward_sums[i])
                episode_reward_sums[i] = 0  # Reset for next episode
                if episodes_completed % 25 == 0:
                    recent_rewards = episode_rewards[-25:] if len(episode_rewards) >= 25 else episode_rewards
                    avg_reward = np.mean(recent_rewards)
                    loss_mean, loss_std, loss_max = td_learner.get_loss_stats()
                    
                    print(f"Episode {episodes_completed}, Avg Reward: {avg_reward:.2f}, "
                          f"Loss (mean/std/max): {loss_mean:.2f}/{loss_std:.2f}/{loss_max:.2f}, "
                          f"Epsilon: {td_learner.epsilon:.3f}")

        # Update state
        obs = next_obs
        concepts = next_concepts
        steps += num_envs 

        # Gradual epsilon decay
        if steps % 20*num_envs == 0:
            td_learner.decay_epsilon()

    # Conservative final training
    print("\n" + "="*50)
    print("Final training phase...")
    print("="*50)
    for i in range(final_training):
        if len(td_learner.replay_buffer) >= 64:
            loss = td_learner.update(batch_size=32)  # Smaller batch size
            if loss is not None and i % 100 == 0:
                print(f"Final training step {i}/{final_training}, Loss: {loss:.4f}")

    # Collect Q-value estimates
    print("\n" + "="*50)
    print("Collecting Q-value estimates...")
    print("="*50)
    q_estimate_list = []
    for state_tuple, action in all_state_actions:
        state_array = np.array(state_tuple)
        q_value = td_learner.get_q_value(state_array, action)
        q_estimate_list.append((state_array, action, q_value))

    # Final statistics
    final_avg_reward = np.mean(episode_rewards[-25:]) if len(episode_rewards) >= 25 else np.mean(episode_rewards) if episode_rewards else 0
    loss_mean, loss_std, loss_max = td_learner.get_loss_stats()
    
    print("\n" + "="*50)
    print("Training Summary")
    print("="*50)
    print(f"Total episodes completed: {episodes_completed}")
    print(f"Final epsilon: {td_learner.epsilon:.3f}")
    print(f"Final average reward (last 25 episodes): {final_avg_reward:.2f}")
    print(f"Final loss stats - Mean: {loss_mean:.2f}, Std: {loss_std:.2f}, Max: {loss_max:.2f}")
    print(f"Total state-action pairs explored: {len(q_estimate_list)}")
    print(f"Replay buffer size: {len(td_learner.replay_buffer)}")
    print("="*50 + "\n")

    if get_td_learner:
        return td_learner 
    else:
        return q_estimate_list
def rollout_pi_estimates(model, env, concept_list, num_rollouts=200, max_steps=2500,mimic=False):
    """
    Estimate the policy/action pairs for different states using a vectorized environment.
    
    Arguments:
        model: policy with `predict(obs, deterministic=True)`
        env: vectorized Gymnasium environment (e.g., SubprocVecEnv)
        concept_list: list of concept extraction functions
        num_rollouts: total number of rollouts to collect (across all envs)
        max_steps: max steps per rollout
    
    Returns:
        pair_list: list of (concept, action) pairs
    """
    num_envs = env.num_envs
    pair_list = []

    rollouts_done = 0
    steps = 0

    # Initial reset
    obs, infos = env.reset()

    while rollouts_done < num_rollouts and steps < max_steps * num_rollouts:
        # Compute concepts for each env
        concepts = []
        for i in range(num_envs):
            info_i = infos[i] if infos and len(infos) > i else {}
            obs_i = obs[i]
            concepts.append([c(info_i.get("observation", obs_i)) for c in concept_list])

        # Predict actions
        actions = []
        for i in range(num_envs):
            if mimic:
                valid_action =  np.sum(env.envs[0].transitions[infos[i]['observation']],axis=1)
                valid_action = torch.Tensor(valid_action).to(model.device)
                action = model.policy.get_distribution(torch.Tensor(obs_i).unsqueeze(0).to(model.device)).distribution.probs 
                action *= valid_action
                action = torch.argmax(action).item()
            else:
                action, _ = model.predict(obs[i], deterministic=True)
            actions.append(int(action))
            pair_list.append((concepts[i], int(action)))

        # Step environments
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        dones = np.logical_or(terms, truncs)

        # Count finished rollouts
        for d in dones:
            if d:
                rollouts_done += 1
                print(rollouts_done,num_rollouts)

        obs = next_obs
        steps += 1

    return pair_list


def get_average_reward(vec_env, model, max_steps=50_000,max_steps_per=5000):
    """
    Evaluate a model on a SubprocVecEnv (or any VecEnv) and return the average reward.

    Arguments:
        vec_env: SB3 VecEnv (SubprocVecEnv, DummyVecEnv, etc.)
            But in Gymnasium Form
        model: Object with `predict` function
        num_restarts: Number of episodes to average over
        max_steps: Max steps per episode

    Returns:
        Float: average reward
    """
    num_envs = vec_env.num_envs
    episode_rewards = []
    rewards_accum = np.zeros(num_envs)
    obs, _ = vec_env.reset()
    total_steps = 0
    steps_per = np.zeros(num_envs)

    while total_steps < max_steps:
        actions, _ = model.predict(obs)
        obs, rewards, terminated, truncated, infos = vec_env.step(actions)
        rewards_accum += rewards
        total_steps += num_envs 
        steps_per += 1

        for i in range(num_envs):
            if terminated[i] or truncated[i] or steps_per[i] >= max_steps_per:
                episode_rewards.append(rewards_accum[i])
                rewards_accum[i] = 0  # reset for next episode
                steps_per[i] = 0
    return np.mean(episode_rewards)

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

def save_fqe(fqe,path):
    save_dict = {
        "state_dim": fqe.state_dim,
        "action_dim": fqe.action_dim,
        "gamma": fqe.gamma,
        "n_ensemble": fqe.n_ensemble,
        "hidden_dims": fqe.hidden_dims,  # or store manually
        "ensemble_state_dicts": [q.state_dict() for q in fqe.ensemble],
        "state_mean": fqe.state_mean,
        "state_std": fqe.state_std
    }
    torch.save(save_dict, path)

def load_fqe(path, eval_env, optimal_model, device='cuda'):
    checkpoint = torch.load(path, map_location=device)

    fqe = FittedQEvaluation(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"],
        policy_model=optimal_model,
        gamma=checkpoint["gamma"],
        hidden_dims=checkpoint["hidden_dims"],
        n_ensemble=checkpoint["n_ensemble"],
        device=device,
        state_mean=checkpoint['state_mean'],
        state_std=checkpoint['state_std']
    )

    state_mean = checkpoint['state_mean']
    state_std = checkpoint['state_std']

    for q_net, state_dict in zip(fqe.ensemble, checkpoint["ensemble_state_dicts"]):
        q_net.load_state_dict(state_dict)

    collector = TransitionCollector(eval_env, optimal_model)
    transitions = collector.collect_diverse_states(n_transitions=10000)
    data = collector.transitions_to_arrays(transitions)
    states = (data['states'] - state_mean) / state_std

    q_values, _ = fqe.get_q_values(states)


    return fqe, states, state_mean, state_std, q_values 
