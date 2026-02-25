import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import random


# ── Q-value estimation ────────────────────────────────────────────────────────

class _QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        return self.network(x)


class _TDLearner:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, epsilon=0.1):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_dim

        self.q_net = _QNetwork(state_dim, action_dim)
        self.target_net = _QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.replay_buffer = deque(maxlen=50_000)
        self.update_count = 0
        self.target_update_freq = 500
        self.recent_losses = deque(maxlen=100)

    def add(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def q_value(self, state, action):
        with torch.no_grad():
            return self.q_net(torch.FloatTensor(state).unsqueeze(0))[0][action].item()

    def act(self, state, deterministic=False):
        if not deterministic and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            return self.q_net(torch.FloatTensor(state).unsqueeze(0)).argmax().item()

    def update(self, batch_size=32):
        if len(self.replay_buffer) < batch_size * 2:
            return None

        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states))
        actions     = torch.LongTensor(actions)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones       = torch.BoolTensor(dones)

        current_q = self.q_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            targets = torch.clamp(rewards + self.gamma * next_q * ~dones, -500, 500)

        loss = nn.MSELoss()(current_q.squeeze(), targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.update_count += 1
        self.recent_losses.append(loss.item())

        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def decay_epsilon(self, decay=0.9995, min_eps=0.005):
        self.epsilon = max(min_eps, self.epsilon * decay)

    def loss_stats(self):
        if not self.recent_losses:
            return 0, 0, 0
        losses = list(self.recent_losses)
        return np.mean(losses), np.std(losses), np.max(losses)


def estimate_q_values(
    model,
    env,
    concept_list,
    total_timesteps=200_000,
    epsilon=0.1,
    learning_rate=1e-4,
    update_freq=20,
    initial_random_fraction=0.3,
    final_training_steps=1_000,
    gamma=0.99,
):
    """Estimate Q(s, a) values over the concept state space using TD learning.

    Runs the provided policy in the environment, collects transitions in the
    concept space, trains a small Q-network, and returns a list of
    (concept_vector, action, q_value) triples for the second half of training
    (when all state-action pairs are tracked).

    Args:
        model: SB3-compatible policy with a `predict(obs)` method
        env: Vectorised gym environment
        concept_list: List of concept functions f(obs) -> float
        total_timesteps: Total environment steps to collect
        epsilon: Initial exploration rate for the TD learner
        learning_rate: Adam learning rate
        update_freq: Steps between TD network updates
        initial_random_fraction: Fraction of steps to inject random actions at start
        final_training_steps: Extra gradient steps after rollout
        gamma: Discount factor

    Returns:
        List of (concept_vector, action, q_value) triples
    """
    num_envs  = env.num_envs
    state_dim = len(concept_list)
    action_dim = env.action_space.n

    learner = _TDLearner(state_dim, action_dim, lr=learning_rate, gamma=gamma, epsilon=epsilon)
    all_state_actions = {}
    episode_rewards = []
    episode_reward_sums = np.zeros(num_envs)

    def extract_concepts(obs, infos):
        concepts = np.zeros((num_envs, state_dim))
        for i in range(num_envs):
            info_i = infos[i] if infos and len(infos) > i else {}
            raw = info_i.get("observation", obs[i])
            concepts[i] = [c(raw) for c in concept_list]
        return concepts

    obs, infos = env.reset()
    concepts = extract_concepts(obs, infos)
    steps = 0

    while steps < total_timesteps:
        actions = model.predict(obs)[0]
        use_random = steps < total_timesteps // 5 and np.random.rand() < initial_random_fraction
        if use_random:
            actions = [env.action_space.sample() for _ in range(num_envs)]

        # Track all state-action pairs in second half of training
        if steps > total_timesteps // 2:
            for i in range(num_envs):
                key = tuple(concepts[i])
                if key not in all_state_actions:
                    all_state_actions[key] = set()
                for j in range(action_dim):
                    all_state_actions[key].add(j)

        next_obs, rewards, terms, truncs, infos = env.step(actions)
        dones = np.logical_or(terms, truncs)
        next_concepts = extract_concepts(next_obs, infos)
        episode_reward_sums += rewards

        for i in range(num_envs):
            learner.add(concepts[i], actions[i], rewards[i], next_concepts[i], dones[i])

        if len(learner.replay_buffer) >= 128 and steps % update_freq == 0:
            loss = learner.update(batch_size=64)
            if loss is not None and loss > 500:
                for pg in learner.optimizer.param_groups:
                    pg["lr"] *= 0.5

        for i in range(num_envs):
            if dones[i]:
                episode_rewards.append(episode_reward_sums[i])
                episode_reward_sums[i] = 0

        obs = next_obs
        concepts = next_concepts
        steps += num_envs

        if steps % (20 * num_envs) == 0:
            learner.decay_epsilon()

        if steps % 5000 == 0:
            mean_loss, _, _ = learner.loss_stats()
            avg_r = np.mean(episode_rewards[-25:]) if len(episode_rewards) >= 25 else 0
            print(f"Step {steps}/{total_timesteps} | avg_reward={avg_r:.2f} | loss={mean_loss:.4f} | eps={learner.epsilon:.3f}")

    # Final training phase
    for _ in range(final_training_steps):
        if len(learner.replay_buffer) >= 64:
            learner.update(batch_size=32)

    # Collect Q-value estimates
    q_estimates = [
        (np.array(state), action, learner.q_value(np.array(state), action))
        for state, actions in all_state_actions.items()
        for action in actions
    ]

    print(f"Q-estimation complete: {len(q_estimates)} state-action pairs")
    return q_estimates


# ── Policy rollouts ───────────────────────────────────────────────────────────

def rollout_policy(model, env, concept_list, num_rollouts=200, max_steps=2500):
    """Collect (concept_vector, action) pairs by rolling out a policy.

    Args:
        model: SB3-compatible policy
        env: Vectorised gym environment
        concept_list: List of concept functions
        num_rollouts: Number of episode completions to collect
        max_steps: Maximum total steps

    Returns:
        List of (concept_vector, action) pairs
    """
    num_envs = env.num_envs
    pair_list = []
    rollouts_done = 0
    steps = 0

    obs, infos = env.reset()

    while rollouts_done < num_rollouts and steps < max_steps * num_rollouts:
        concepts = []
        for i in range(num_envs):
            info_i = infos[i] if infos and len(infos) > i else {}
            raw = info_i.get("observation", obs[i])
            concepts.append([c(raw) for c in concept_list])

        actions = [int(model.predict(obs[i], deterministic=True)[0]) for i in range(num_envs)]
        for i in range(num_envs):
            pair_list.append((concepts[i], actions[i]))

        next_obs, _, terms, truncs, infos = env.step(actions)
        dones = np.logical_or(terms, truncs)
        rollouts_done += int(dones.sum())
        obs = next_obs
        steps += 1

    return pair_list


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_policy(vec_env, model, seed, max_steps=100_000, max_steps_per_episode=10_000):
    """Evaluate a policy and return mean episode reward.

    Args:
        vec_env: GymnasiumWrapper or VecEnv
        model: SB3-compatible policy
        seed: Random seed for reproducibility
        max_steps: Total steps to run
        max_steps_per_episode: Force episode reset after this many steps

    Returns:
        Mean episode reward (float)
    """
    if hasattr(vec_env, "training_mode"):
        vec_env.set_eval_mode()

    num_envs = vec_env.num_envs
    episode_rewards = []
    rewards_accum = np.zeros(num_envs)
    steps_per = np.zeros(num_envs)
    total_steps = 0

    vec_env.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    obs, _ = vec_env.reset()

    while total_steps < max_steps:
        actions, _ = model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, _ = vec_env.step(actions)
        rewards_accum += rewards
        total_steps += num_envs
        steps_per += 1

        for i in range(num_envs):
            if terminated[i] or truncated[i] or steps_per[i] >= max_steps_per_episode:
                episode_rewards.append(rewards_accum[i])
                rewards_accum[i] = 0
                steps_per[i] = 0

    return np.mean(episode_rewards)