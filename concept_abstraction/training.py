import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from gymnasium.vector import SyncVectorEnv
from torch.cuda.amp import GradScaler, autocast
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import gymnasium
from stable_baselines3 import DQN

def train_model(env):
    model = DQN("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10000, log_interval=4)
    return model 
