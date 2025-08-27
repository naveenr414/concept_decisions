import numpy as np
import numpy as np
import gymnasium as gym
import os 
from stable_baselines3 import PPO
import torch 

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
            action = model.predict(observation[None,:])[0]

            # Take a step in the environment
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward 
            # End episode if done
            if terminated or truncated:
                break
    env.close()
    total_reward /= 10000
    return total_reward

def rollout_q_estimates(model, env, concept_list,n_steps=100, gamma=0.99):
    """Estimate a list of Q values, along with concept values
        For a given groundtruth model + groundtruth environment
        combination
    
    Arguments:
        model: Groundtruth model that operates on the raw state space
        env: Groundtruth environment with no concepts
        concept_list: List of potential concepts from a concept bnak
    
    Returns: List of (concepts,action,Q_estimate) values"""
    
    q_estimates = []

    for _ in range(100):
        obs, info = env.reset()
        for _ in range(n_steps):
            # get action from PPO policy
            action, _ = model.predict(obs, deterministic=False)

            # step env
            next_obs, reward, terminated, truncated, info = env.step(action)

            # convert to tensor for value prediction
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(model.device)

            with torch.no_grad():
                v_next = model.policy.predict_values(next_obs_tensor).cpu().numpy()[0][0]
            # TD estimate of Q(s,a)
            q_val = reward + (0 if (terminated or truncated) else gamma * v_next)
            

            concepts = [concept(info['observation']) for concept in concept_list]

            q_estimates.append((concepts, action, q_val))

            obs = next_obs
            if terminated or truncated:
                obs, info = env.reset()

    return q_estimates


def get_average_reward_gym(env, model, n_episodes=10):
    total_reward = 0
    episode_count = 0

    while episode_count < n_episodes:
        obs = env.reset()  # VecEnv: returns batch of obs
        done = [False] * env.num_envs

        while not all(done):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            total_reward += sum(rewards)
            done = dones
        episode_count += 1

    env.close()
    return total_reward / n_episodes


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