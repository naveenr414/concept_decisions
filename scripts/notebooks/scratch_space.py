import gymnasium as gym
from stable_baselines3 import PPO

# Create environment with video recording
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, video_folder="./cartpole-videos", episode_trigger=lambda e: True)

# Load a pre-trained PPO model or train one quickly
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=10_000)  # quick training

observation, info = env.reset()
try:
    for _ in range(500):
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
finally:
    env.close()