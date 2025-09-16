import numpy as np
import gymnasium
import pufferlib
import pufferlib.models
import pufferlib.cleanrl
import pufferlib.environments.classic_control
import pufferlib.vectorization
import time 
from stable_baselines3 import PPO
envs = pufferlib.vectorization.Serial(
    env_creator=pufferlib.environments.classic_control.env_creator('cartpole'),
    num_envs=4, envs_per_worker=2
)


# def test_performance(timeout=10, atn_cache=8192, continuous=True):
#     """Benchmark environment performance."""
#     total_timesteps = 100_000
#     start = time.time() 
#     num_envs = 8
#     env = Cartpole(num_envs=num_envs, continuous=continuous)
#     model = PPO(
#                 "MlpPolicy",
#                 env,
#                 policy_kwargs={"net_arch": [16,16]},  # Single layer is actually fastest
#                 n_steps=256,
#                 batch_size=2048,        # Match n_steps for single batch processing
#                 n_epochs=1,           # KEY: Single epoch only
#                 learning_rate=5e-3,   # Higher LR to compensate for fewer epochs
#                 device='cpu',        # Your GPU is working fine
#                 verbose=0
#             )

#     model.learn(total_timesteps=total_timesteps)
#     return time.time()-start
# if __name__ == '__main__':
#     test_performance()
