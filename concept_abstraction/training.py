import random

import numpy as np
import torch
import torch.distributions as D
from concept_abstraction.env_utils import get_average_reward

import numpy as np
import wandb

from io import StringIO
from contextlib import redirect_stderr
stderr_buffer = StringIO()
with redirect_stderr(stderr_buffer):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

class WandbLoggingCallback(BaseCallback):
    """
    Logs PPO diagnostics to wandb.
    Works with Monitor-wrapped environments.
    """
    def __init__(self, smooth_alpha=0.9):
        super().__init__()
        self.smoothed_avg_norm_reward = 0
        self.alpha = smooth_alpha  # EMA smoothing for normalized reward

    def _on_step(self) -> bool:
        # ---- Episode info from Monitor ----
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info.keys():
                raw_reward = info["episode"]["r"]
                # normalized reward if you implement reward normalization
                norm_reward = raw_reward
                # EMA smoothing
                self.smoothed_avg_norm_reward = (
                    self.alpha * self.smoothed_avg_norm_reward
                    + (1 - self.alpha) * norm_reward
                )

                # Log episode metrics to wandb
                wandb.log({
                    "episode_reward": raw_reward,
                    "episode_length": info["episode"]["l"],
                    "avg_norm_reward": norm_reward,
                    "ema_norm_reward": self.smoothed_avg_norm_reward
                }, step=self.num_timesteps)

        # ---- PPO internal diagnostics ----
        logger_data = getattr(self.model.logger, "name_to_value", {})
        ev = logger_data.get("train/explained_variance", None)
        vl = logger_data.get("train/value_loss", None)
        kl = logger_data.get("train/approx_kl", None)
        cf = logger_data.get("train/clip_fraction", None)
        ent = logger_data.get("train/entropy_loss", None)
        gn = logger_data.get("diagnostics/grad_norm", None)

        # Log PPO diagnostics to wandb
        wandb.log({
            "explained_variance": ev,
            "value_loss": vl,
            "approx_kl": kl,
            "clip_fraction": cf,
            "entropy_loss": ent,
            "grad_norm": gn
        }, step=self.num_timesteps)

        return True

def get_model(environment_string,policy,override={}):
    default_model_dict = {
        'n_steps': 512,
        'batch_size': 128,
        'learning_rate': 3e-4,
        'device': 'cpu',
        'n_epochs': 10,
        'policy_kwargs': None ,
        'ent_coef': 0.01,
    }

    if policy == "MlpPolicy":
        if "cyclic" in environment_string:
            default_model_dict['policy_kwargs'] = {'net_arch': [16]}
            default_model_dict['n_steps'] = 32
            default_model_dict['batch_size'] = 32
            default_model_dict['learning_rate'] = 1e-3
        elif "tree" in environment_string:
            default_model_dict['policy_kwargs'] = {'net_arch': [16]}
            default_model_dict['n_steps'] = 32
            default_model_dict['batch_size'] = 32
            default_model_dict['learning_rate'] = 1e-3
            default_model_dict['ent_coef'] = 0.02
        elif environment_string == "glucose":
            default_model_dict['policy_kwargs'] = {'net_arch': [64,64]}
            default_model_dict['batch_size'] = 64
            default_model_dict['n_epochs'] = 5
            default_model_dict['learning_rate'] = 6e-4
        elif environment_string == "cart_pole":
            default_model_dict['policy_kwargs'] = {'net_arch': [128]}
            default_model_dict['batch_size'] = 1024
            default_model_dict['n_epochs'] = 5
            default_model_dict['ent_coef'] = 0.005
        elif environment_string == "pong":
            default_model_dict['policy_kwargs'] = {'net_arch': [128,128]}
            default_model_dict['n_steps'] = 4096
            default_model_dict['batch_size'] = 256
            default_model_dict['n_epochs'] = 6
            default_model_dict['learning_rate'] = 2e-3
            default_model_dict['ent_coef'] = 0.02
        elif environment_string == "boxing":
            default_model_dict['policy_kwargs'] = {'net_arch': [128,128]}
            default_model_dict['n_steps'] = 4096
            default_model_dict['batch_size'] = 256
            default_model_dict['n_epochs'] = 4
            default_model_dict['learning_rate'] = 1e-3
            default_model_dict['ent_coef'] = 0.015
    else:
        if environment_string == "cart_pole" or environment_string == "mini_grid":
            default_model_dict['n_steps'] = 512
            default_model_dict['batch_size'] = 4096
            default_model_dict['n_epochs'] = 10
            default_model_dict['device'] = 'cuda'
        elif environment_string == "pong":
            default_model_dict['n_steps'] = 1024
            default_model_dict['batch_size'] = 512
            default_model_dict['n_epochs'] = 10
            default_model_dict['device'] = 'cuda'
        elif environment_string == "boxing":
            default_model_dict['n_steps'] = 2048
            default_model_dict['batch_size'] = 256
            default_model_dict['n_epochs'] = 3
            default_model_dict['device'] = 'cuda'
        else:
            default_model_dict['n_steps'] = 128
            default_model_dict['batch_size'] = 256
            default_model_dict['n_epochs'] = 4
            default_model_dict['device'] = 'cuda'
            default_model_dict['learning_rate'] = 2.5e-4
    for i in override:
        default_model_dict[i] = override[i]

    return default_model_dict

def train_ppo_model(env,environment_string,seed=42,total_timesteps=150_000,policy="MlpPolicy",override={},custom_name=""):
    """Train an environment according to a stable baseline policy
    
    Arguments:
        env: Gymnasium environment
    
    Returns: Stable Baseline3 PPO Model"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_params = get_model(environment_string,policy,override=override)
    model_params['env'] = environment_string

    name = "{}_{}".format(environment_string,policy)
    if custom_name != "":
        name = custom_name
    wandb.init(
        project="Concept Decisions",
        name=name,
        config=model_params
    )
    model = PPO(
        policy,
        env,
        policy_kwargs=model_params['policy_kwargs'],
        n_steps=model_params['n_steps'],
        batch_size=model_params['batch_size'],
        n_epochs=model_params['n_epochs'],
        learning_rate=model_params['learning_rate'],
        ent_coef=model_params['ent_coef'],
        gamma=0.99,
        verbose=0,
        device=model_params['device'],
    )
    model.learn(total_timesteps=total_timesteps, callback=WandbLoggingCallback())

    wandb.finish()
    return model 

class DistributionWrapper:
    def __init__(self, logits):
        # logits = unnormalized scores (before softmax)
        self.logits = logits
        self.distribution = D.Categorical(logits=logits)


class Policy:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def get_distribution(self, obs):
        batch_size = len(obs)
        # uniform logits: shape (batch_size, num_actions)
        logits = torch.zeros((batch_size, self.num_actions))
        return DistributionWrapper(logits)


class RandomAgent:
    """Random policy that randomly selects actions"""

    def __init__(self, vec_env):
        self.action_space = vec_env.action_space
        self.num_envs = vec_env.num_envs
        self.device = "cpu"

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        # Sample one action per environment
        actions = np.array([self.action_space.sample() for _ in range(self.num_envs)])
        return actions, None

def evaluate_model(environment_string,env,additional_info,model,seed):
    """Evaluation Function that tailors the evaluation
        based on the environment
        
    Arguments:
        environment_string: String, such as cart_pole, describing the environment
        env: Environment where we are evaluating the polciy
        additional_info: Any additional info needed to evaluate
            For example, for MIMIC, we need the Physician policy
        model: Policy function \pi
    
    Returns: Average Reward (for most environments) or MIMIC WIQ score"""

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if environment_string == "glucose":
        return get_average_reward(env,model,max_steps=5000,max_steps_per=500)
    else:
        return get_average_reward(env,model)