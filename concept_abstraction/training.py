from stable_baselines3 import DQN, PPO

def train_model(env,total_timesteps=10000):
    """Train an environment according to a stable baseline policy
    
    Arguments:
        env: Gymnasium environment
    
    Returns: Stable Baseline3 DQN Model"""
    model = DQN("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps, log_interval=4)
    return model 

def train_ppo_model(env,total_timesteps=150_000):
    """Train an environment according to a stable baseline policy
    
    Arguments:
        env: Gymnasium environment
    
    Returns: Stable Baseline3 PPO Model"""

    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps)  
    return model 
