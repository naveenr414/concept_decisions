import gymnasium as gym
from gymnasium import spaces
import numpy as np
import ocatari
import cv2 
from gymnasium.wrappers import FrameStackObservation as FrameStack
import random
import torch
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import VecNormalize

from io import StringIO
from contextlib import redirect_stderr
stderr_buffer = StringIO()
with redirect_stderr(stderr_buffer):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import minigrid
from minigrid.core.constants import DIR_TO_VEC
from concept_abstraction.glucose_env import GlucoseEnvironment
from concept_abstraction.environment_wrappers import *

cv2.setNumThreads(1)

def get_raw_state_cartpole(_,obs,__):
    """Get the raw underlying state in a CartPole environment
    Arguments:
        env: CartPole environment
        obs: Current observation, 4-vector
    
    Returns: The same 4-vector observation"""

    return obs 

def get_raw_pixels_cartpole(env, obs=None):
    pixels = env.render()
    small_pixels = cv2.resize(pixels, (240, 160), interpolation=cv2.INTER_NEAREST)

    # Convert to grayscale in C
    gray = cv2.cvtColor(small_pixels, cv2.COLOR_RGB2GRAY)
    return gray


def get_raw_pixels_mini_grid(env, obs=None):
    pixels = env.render()[:160, :160]
    gray = np.dot(pixels[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    small_pixels = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_NEAREST)
    return small_pixels

def can_move(position, direction, grid):
    """
    Helper function to determine if movement in a specific direction is possible.
    """
    next_pos = DIR_TO_VEC[direction]
    if 1 <= position[0] + next_pos[0] < grid.width-1 and 1 <= position[1] + next_pos[1] < grid.height-1:
        next_cell = grid.get(position[0] + next_pos[0], position[1] + next_pos[1])
        return next_cell is None or next_cell.can_overlap()
    else:
        return False  # Out of bounds

def get_raw_state_mini_grid(env,obs,info):
    agent_pos = env.unwrapped.agent_pos
    agent_dir = env.unwrapped.agent_dir
    grid = env.unwrapped.grid
    key_pos = (0, 0) # default one if not found (carrying)
    door_pos = None
    door_open = False

    # Locate door, key, and goal positions
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None:
                if cell.type == 'door':
                    door_pos = (x, y)
                    door_open = cell.is_open  # Check if the door is open
                elif cell.type == 'key':
                    key_pos = (x, y)

    # Check direction_movable in all four directions
    direction_movable = {
        'right': can_move(agent_pos, 0, grid),
        'down': can_move(agent_pos, 1, grid),
        'left': can_move(agent_pos, 2, grid),
        'up': can_move(agent_pos, 3, grid),
    }

    vec = [agent_pos[0],agent_pos[1],agent_dir,key_pos[0],key_pos[1],door_pos[0],door_pos[1],int(door_open)]+[int(direction_movable[i]) for i in ['right','down','left','up']]
    return vec 

def get_raw_state_atari(env,_, __):
    """Get the raw underlying state in an Atari environment
    
    Arguments:
        env: CartPole environment
        obs: Current observation, 4-vector
    
    Returns: A 1x84x84 numpy array representing the screen"""

    pixels = env.unwrapped._ale.getScreenGrayscale().astype(np.uint8)
    return cv2.resize(pixels, (84,84),interpolation=cv2.INTER_NEAREST)


def get_raw_state_glucose(_,obs,__):
    """Get the raw underlying state in a CartPole environment
    Arguments:
        env: CartPole environment
        obs: Current observation, 4-vector
    
    Returns: The same 4-vector observation"""

    return obs 

def make_ocenv(env_name,use_concepts,seed=0,recordable=False,num_stack=4,processed_concepts=None):
    """Create an OCAtari environment for a given concept list
    
    Arguments:
        env_name: String for the Atari environment
        concept_list: List of functions mapping
            internal Atari states to some number
        observation_space: Type of environment observation
    
    Returns: Gym Environment wrapped with Concepts"""
    if recordable:
        env = ocatari.OCAtari(
            env_name,
            mode="ram",
            render_mode="rgb_array",
            frameskip=1,
        )
    else:
        env = ocatari.OCAtari(
            env_name,
            mode="ram",
            render_mode=None, 
            frameskip=2, 
        )
    env = Monitor(env)
    env.ale = env.unwrapped._ale

    if use_concepts:
        if "pong" in env_name.lower():
            env = ConceptWrapper(env, gym.spaces.Box(
                        low=0, high=255,
                        shape=(4,6),  # Height x Width, no color channel
                        dtype=np.uint8
                    ), lambda env, obs, info: obs) 
        elif "boxing" in env_name.lower():
            env = ConceptWrapper(env, gym.spaces.Box(
                        low=0, high=255,
                        shape=(4,4),  # Height x Width, no color channel
                        dtype=np.uint8
                    ), lambda env, obs, info: obs) 

    else:
        env = ConceptWrapper(env, gym.spaces.Box(
                    low=0, high=255,
                    shape=(84,84),  # Height x Width, no color channel
                    dtype=np.uint8
                ), get_raw_state_atari) 
        env = FrameStack(env,num_stack)
        env = LazyFramesToNumpy(env)

    env.reset(seed=seed)
    return env

def get_n_atari_env(n_envs,atari_env_name,use_concepts,recordable=False,num_stack=4,processed_concepts=None):
    """Create a series of parallel Atari environments 
    
    Arguments:
        n_envs: Integer, number of parallel environments
        atari_env_name: String, which Atari environment we're using
        concept_list: List of concepts which we're using for mapping
        observation_space: Type of environment observation
    
    Returns: SubprocVecEnv with all the environments"""    
    def safe_make_env(seed):
        try:
            return make_ocenv(atari_env_name, use_concepts, seed=seed,num_stack=num_stack,processed_concepts=processed_concepts,recordable=recordable)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    if use_concepts:
        vec_env = DummyVecEnv([
            lambda seed=i: safe_make_env(seed=seed)
            for i in range(n_envs)
        ])
    else:
        vec_env = DummyVecEnv([
            lambda seed=i: safe_make_env(seed=seed)
            for i in range(n_envs)
        ], start_method='spawn')
    return vec_env

def get_environment(environment_string,concept_list,seed,concept_idx=[],use_processed=False,processed_concepts=None,fast_predictor=None,intervention_prob=0.0,concept_accuracy=1.0):
    """Get a specific environment based on a string + concept list
    
    Arguments:
        environment_string: String, mapping to one environment
        concept_list: List of functions mapping state -> concept, or None
    
    Returns: Gymasium environment, and a dictionary of additional information"""

    num_envs = 8
    num_stack = 4

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if environment_string == "cart_pole":
        def make_env():
            if concept_list is None or use_processed:
                env = gym.make("CartPole-v1", render_mode="rgb_array")
                env = Monitor(env)
                env = FrameSkipWrapper(env,skip=1,get_pixels_fn=get_raw_pixels_cartpole)
                env = ConceptWrapper(env,spaces.Box(
                        low=0, high=255,
                        shape=(160,240),  # Height x Width, no color channel
                        dtype=np.uint8
                    ),lambda env, obs, info: obs,use_info_obs=True)
                env = FrameStack(env,num_stack)
                env = LazyFramesToNumpy(env)
            else:
                env = Monitor(gym.make("CartPole-v1"))
                env = ConceptWrapper(env,spaces.Box(
                        low=0, high=255,
                        shape=(4,),  # Height x Width, no color channel
                        dtype=np.uint8
                    ),get_raw_state_cartpole)
            return env 
        if concept_list is None or use_processed:
            vec_env = DummyVecEnv([make_env for _ in range(8)])
        else:
            vec_env = DummyVecEnv([make_env for _ in range(num_envs)])
    elif environment_string  == "mini_grid":
        num_stack = 1
        
        def make_env():
            if concept_list is None or use_processed:
                env = Monitor(gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="rgb_array"))
                env = FrameSkipWrapper(env, skip=1, get_pixels_fn=get_raw_pixels_mini_grid)
                env = ConceptWrapper(
                    env,
                    spaces.Box(low=0, high=255, shape=(84,84), dtype=np.uint8),
                    lambda env,obs,info:obs, 
                    obs_function=lambda env, obs, info:  get_raw_state_mini_grid(env, info, {'observation': obs})

                )
                env = FrameStack(env, num_stack)
                env = LazyFramesToNumpy(env)
            else:
                env = Monitor(gym.make("MiniGrid-DoorKey-5x5-v0"))
                env = ConceptWrapper(
                    env,
                    spaces.Box(
                        low=0, high=5,
                        shape=(12,),  # Height x Width, no color channel
                        dtype=np.uint8
                    ),
                    lambda env,obs,info: get_raw_state_mini_grid(env, info, {'observation': obs}), 
                    obs_function=lambda env, obs, info:  get_raw_state_mini_grid(env, info, {'observation': obs})
                )
            return env
        
        if concept_list is None or use_processed:
            vec_env = DummyVecEnv([make_env for _ in range(num_envs)])
        else:
            vec_env = DummyVecEnv([make_env for _ in range(num_envs)])
    elif environment_string == "pong":
        if concept_list is None or use_processed:
            vec_env = get_n_atari_env(num_envs,"PongNoFrameskip-v4",False,num_stack=num_stack,processed_concepts=processed_concepts)
        else:
            vec_env = get_n_atari_env(num_envs,"PongNoFrameskip-v4",True,processed_concepts=processed_concepts)
    elif environment_string == "boxing":
        if concept_list is None or use_processed:
            vec_env = get_n_atari_env(num_envs,"BoxingNoFrameskip-v4",False,num_stack=num_stack,processed_concepts=processed_concepts)
        else:
            vec_env = get_n_atari_env(num_envs,"BoxingNoFrameskip-v4",True,processed_concepts=processed_concepts)
    elif environment_string == "glucose":
        def make_env():
            register(
                id="simglucose/adolescent2-custom-v0",
                entry_point=GlucoseEnvironment,  # adjust if using a module
                max_episode_steps=288,
                kwargs={"patient_name": "adolescent#002"},
            )

            env = gym.make("simglucose/adolescent2-custom-v0", render_mode=None)
            env = Monitor(env)
            if concept_list is not None:
                env = ConceptWrapper(env,spaces.Box(
                        low=0, high=255,
                        shape=(6,),  # Height x Width, no color channel
                        dtype=np.uint8
                    ),get_raw_state_glucose)

            return env 
        vec_env = DummyVecEnv([make_env for i in range(num_envs)])

    if use_processed:
        vec_env = VecConceptWrapper(vec_env, fast_predictor, concept_idx,intervention_prob=intervention_prob,processed_concepts=processed_concepts)
        vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=False,   # or True if you want reward normalization too
                clip_obs=10.0
            )
    elif concept_list is not None:
        vec_env = VecConceptWrapper(vec_env, None, concept_idx,intervention_prob=intervention_prob,processed_concepts=processed_concepts,concept_accuracy=concept_accuracy)
        vec_env.observation_space = spaces.MultiBinary(len(concept_idx))

    gymnasium_env = GymnasiumWrapper(vec_env)

    return vec_env, gymnasium_env
