import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random 
from collections import deque

class QNet(nn.Module):
    def __init__(self, obs_size, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions)
        )

    def forward(self, x):
        return self.net(x.float())

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def push(self, obs, action, reward, next_obs, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done = zip(*samples)
        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(act, dtype=torch.int64),
            torch.tensor(rew, dtype=torch.float32),
            torch.tensor(next_obs, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

def train_model(env, steps=1000):
    obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    q_net = QNet(obs_size, num_actions)
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer()

    epsilon = 0.25
    gamma = 0.9
    batch_size = 32

    def select_action(obs):
        if random.random() < epsilon:
            return random.randint(0, num_actions - 1)
        with torch.no_grad():
            q_vals = q_net(torch.tensor(obs).unsqueeze(0))
            return int(torch.argmax(q_vals))

    for step in range(steps):
        obs, _ = env.reset()
        done = False

        for _ in range(32):
            action = select_action(obs)
            next_obs, reward, done, _, _ = env.step(action)

            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs

            if len(buffer) < batch_size:
                continue

            b_obs, b_act, b_rew, b_next_obs, b_done = buffer.sample(batch_size)

            q_vals = q_net(b_obs).gather(1, b_act.unsqueeze(1)).squeeze()
            with torch.no_grad():
                max_next_q = q_net(b_next_obs).max(dim=1)[0]
            target = b_rew + gamma * max_next_q * (1 - b_done)

            loss = nn.functional.mse_loss(q_vals, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return q_net
