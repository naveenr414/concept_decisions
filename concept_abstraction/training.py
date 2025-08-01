from stable_baselines3 import DQN

def train_model(env):
    """Train an environment according to a stable baseline policy
    
    Arguments:
        env: Gymnasium environment
    
    Returns: Stable Baseline3 DQN Model"""
    model = DQN("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10000, log_interval=4)
    return model 
