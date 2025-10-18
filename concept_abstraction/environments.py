import gymnasium as gym
from gymnasium import spaces
import numpy as np
import ocatari
import cv2 
from gymnasium.wrappers import FrameStackObservation as FrameStack
import random
import torch
from gymnasium.envs.registration import register
from simglucose.envs import T1DSimGymnaisumEnv

from io import StringIO
from contextlib import redirect_stderr
stderr_buffer = StringIO()
with redirect_stderr(stderr_buffer):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import minigrid
from minigrid.core.constants import DIR_TO_VEC
from concept_abstraction.environment_wrappers import *

cv2.setNumThreads(1)

def create_cyclic_env(num_nodes,concept_list):
    """Simple Environment that captures a cyclic structure between states
    0 -> 1 -> 2 -> 3 -> 0
    
    Here, certain states have certain rewards; 
        e.g., 0 and 2 have the same reward
        as do 1 and 3
    Concepts should capture this, so [0,2], and [1,3] should be 
        split by the concept
        
    Example concept splits include
        [0,2],[1,3]
        [0,3],[1,2]
        [0,1,2], [3]"""

    environment_nodes = num_nodes 
    action_space = spaces.Discrete(3)

    if concept_list is None:
        observation_space = spaces.MultiBinary(environment_nodes)
    else:
        observation_space = spaces.MultiBinary(len(concept_list))
    all_states = list(range(environment_nodes))
    max_steps = 20

    rewards = np.zeros((environment_nodes,3)) 
    for i in range(environment_nodes):
        if i%2 == 0:
            rewards[i,0] = rewards[i,1] = 1
        else:
            rewards[i,2] = 1
    rewards = np.array(rewards)

    transitions = []
    for i in range(len(all_states)):
        transitions_by_state = []
        for action in range(3):
            next_probs = [0.0 for i in range(len(all_states))]
            if action == 0:
                next_probs[(i - 1) % environment_nodes] = 1.0
            if action == 1:
                next_probs[(i + 1) % environment_nodes] = 1.0
            if action == 2:
                next_probs[(i) % environment_nodes] = 1.0
            transitions_by_state.append(next_probs)
        transitions.append(transitions_by_state)
    transitions = np.array(transitions)
    return Monitor(ConceptEnv(concept_list,observation_space,action_space,rewards,transitions,all_states,max_steps))

def create_tree_env(num_nodes,concept_list):
    """Simple Environment that captures a tree structure between states
    0 -> (1,2)
    1 -> (3,4)
    2 -> (5,6)
    3 -> (7,8)
    etc. 
    then 7 -> 0

    The idea is to show that errors can propogate; the ideal path is to 
    go 0->1->3,etc.; this requires playing action LEFT each time
    However, error in the concepts lead to a large loss in the value
        as the agent will instead play RIGHT
            
    Example concept splits include
        Top: [0,1,3,7], all otehrs
        1st Binary Digit: [0,2,4,6,8]...[1,3,5,..]
        2nd Digit: [0,1,4,5],...
        etc."""

    environment_nodes = num_nodes 
    num_layers = int(np.log2(environment_nodes+1))
    all_states = list(range(environment_nodes))
    action_space = spaces.Discrete(2)
    if concept_list is None:
        observation_space = spaces.MultiBinary(environment_nodes)
    else:
        observation_space = spaces.MultiBinary(len(concept_list))
    max_steps = 20

    rewards = np.zeros((environment_nodes,action_space.n))
    for i in range(num_layers):
        rewards[2**i-1][0] = 1
    rewards[:,1] = 0.5

    transitions = np.zeros((len(all_states),
                                action_space.n,
                                len(all_states)))
    for state in range(len(transitions)):
        for action in range(len(transitions[state])):
            if state >= environment_nodes//2:
                if state == environment_nodes//2:
                    transitions[state][0][0] = 1
                    transitions[state][1][2] = 1
                else:
                    transitions[state][action][2] = 1
            else:
                transitions[state][action][2 * (state+1) + action - 1] = 1

    return Monitor(ConceptEnv(concept_list,observation_space,action_space,rewards,transitions,all_states,max_steps))

def get_raw_state_cartpole(env,obs):
    """Get the raw underlying state in a CartPole environment
    Arguments:
        env: CartPole environment
        obs: Current observation, 4-vector
    
    Returns: The same 4-vector observation"""

    return obs 

def get_raw_pixels_mini_grid(env,obs=None):
    pixels = env.render()[:160,:160]
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    small_pixels = cv2.resize(gray, (84,84), interpolation=cv2.INTER_NEAREST)
    return small_pixels

def get_raw_pixels_cartpole(env, obs=None):
    pixels = env.render()
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    small_pixels = cv2.resize(gray, (84,84), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
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



class GlucoseEnvironment(T1DSimGymnaisumEnv):
    def __init__(self, patient_name="adolescent#002", **kwargs):
        super().__init__(patient_name=patient_name, **kwargs)
        self.bg_history = []
        self.action_to_dose = {
            0: 0,
            1: 6,
            2: 12,
            3: 18,
            4: 24,
            5: 30,
        }
        self.action_space = spaces.Discrete(len(self.action_to_dose))

    def reset(self, *, seed=None, options=None):
        # Call the parent reset
        obs, info = super().reset(seed=seed, options=options)
        obs = np.random.random()*45+150
        
        # Initialize BG history
        self.bg_history = []
        if hasattr(obs, "__getitem__"):
            self.bg_history.append(obs[0])  # adjust if needed

        return obs/150, info
    def step(self, action,options={}):
        obs, reward, terminated, truncated, info = super().step(self.action_to_dose[action])
        info['observation'] = obs/150
        # Current BG (blood glucose)
        bg = obs[0]  # depends on observation structure
        self.bg_history.append(bg)
        if len(self.bg_history) > 2:
            self.bg_history.pop(0)
        
        # Compute delta
        if len(self.bg_history) < 2:
            delta = 0
        else:
            delta = self.bg_history[-1] - self.bg_history[-2]

        # ----- State reward -----
        if bg < 70 or bg > 200:
            r_state = 0  # extreme hypo/hyper, episode ends
        elif bg < 100 and delta < 0.5:
            r_state = 0.1  # mild hypoglycemia
        elif bg > 150 and delta > 0.5:
            r_state = 0.1  # mild hyperglycemia
        elif 100 <= bg <= 150:
            r_state = 1  # target zone
        else:
            r_state = 0.1

        # ----- Action penalty -----
        r_action = 0.1 * (action ** 2)/(30**2)  # penalize insulin dose magnitude

        # Total reward
        custom_reward = r_state - r_action

        return obs/150, custom_reward, terminated, truncated, info



def get_raw_state_atari(env,obs):
    """Get the raw underlying state in an Atari environment
    
    Arguments:
        env: CartPole environment
        obs: Current observation, 4-vector
    
    Returns: A 1x84x84 numpy array representing the screen"""

    pixels = env.unwrapped._ale.getScreenGrayscale().astype(np.uint8)
    return cv2.resize(pixels, (84,84),interpolation=cv2.INTER_NEAREST)

def make_ocenv(env_name,concept_list,observation_space,seed=0,recordable=False,num_stack=4):
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
            frameskip=1,  # Keep frameskip=1 so consecutive frames actually differ
            difficulty=0  # easiest opponent
        )
    else:
        env = ocatari.OCAtari(
            env_name,
            mode="ram",
            render_mode=None, 
            frameskip=2,  # Keep frameskip=1 so consecutive frames actually differ
            difficulty=0  # easiest opponent
        )
    env = Monitor(env)
    env.ale = env.unwrapped._ale 
    # Apply ConceptWrapper (assuming it returns the pixel observations)
    env = ConceptWrapper(env, concept_list, observation_space, get_raw_state_atari)
    if concept_list is None:
        env = FrameStack(env,num_stack)
        env = LazyFramesToNumpy(env)
    env.reset(seed=seed)
    return env

def get_n_atari_env(n_envs,atari_env_name,concept_list,observation_space,recordable=False,num_stack=4):
    """Create a series of parallel Atari environments 
    
    Arguments:
        n_envs: Integer, number of parallel environments
        atari_env_name: String, which Atari environment we're using
        concept_list: List of concepts which we're using for mapping
        observation_space: Type of environment observation
    
    Returns: SubprocVecEnv with all the environments"""
    def safe_make_env(seed):
        try:
            return make_ocenv(atari_env_name, concept_list, observation_space, seed=seed,num_stack=num_stack)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    if concept_list is None:
        vec_env = DummyVecEnv([
            lambda seed=i: safe_make_env(seed=seed)
            for i in range(n_envs)
        ])#, start_method='spawn')
    else:
        vec_env = SubprocVecEnv([
            lambda seed=i: safe_make_env(seed=seed)
            for i in range(n_envs)
        ], start_method='spawn')
    return vec_env

def get_environment(environment_string,concept_list,seed,concept_idx=[],use_processed=False,fast_predictor=None):
    """Get a specific environment based on a string + concept list
    
    Arguments:
        environment_string: String, mapping to one environment
        concept_list: List of functions mapping state -> concept, or None
    
    Returns: Gymasium environment, and a dictionary of additional information"""

    additional_info = {}
    num_envs = 8
    num_stack = 4

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if "cyclic" in environment_string:
        def make_env():
            num_nodes = int(environment_string.split("_")[-1])
            env = create_cyclic_env(num_nodes,concept_list)
            return env 
        vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])
        gymnasium_env = GymnasiumWrapper(vec_env)
    elif "tree" in environment_string:
        def make_env():
            num_nodes = int(environment_string.split("_")[-1])
            env = create_tree_env(num_nodes,concept_list)
            return env 
        vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])
        gymnasium_env = GymnasiumWrapper(vec_env)
    elif environment_string == "glucose":
        register(
            id="simglucose/adolescent2-custom-v0",
            entry_point="concept_abstraction.environments:GlucoseEnvironment",  # adjust if using a module
            max_episode_steps=288,
            kwargs={"patient_name": "adolescent#002"},
        )

        def make_env():
            env = gym.make("simglucose/adolescent2-custom-v0", render_mode=None)
            env = Monitor(env)  # required for wandb callback to track rewards/lengths
            return env
        vec_env = DummyVecEnv([make_env])
        gymnasium_env = GymnasiumWrapper(vec_env)

    elif environment_string  == "mini_grid":
        def make_env():
            if concept_list is None:
                env = Monitor(gym.make("MiniGrid-DoorKey-5x5-v0",render_mode="rgb_array"))
                env = FrameSkipWrapper(env, skip=4, get_pixels_fn=get_raw_pixels_mini_grid)
                env = ConceptWrapper(env,None,spaces.Box(
                        low=0, high=255,
                        shape=(84,84),  # Height x Width, no color channel
                        dtype=np.uint8
                    ),lambda env, obs: obs, obs_function=lambda e,o,i: get_raw_state_mini_grid(e,o,i))
                env = FrameStack(env,num_stack)
                env = LazyFramesToNumpy(env)

            else:
                env = Monitor(gym.make("MiniGrid-DoorKey-5x5-v0"))
                env = ConceptWrapper(env,concept_list,spaces.MultiBinary(len(concept_list)),lambda env, obs: obs, obs_function=lambda env, obs, info: get_raw_state_mini_grid(env,info,{'observation': obs}))
            return env 

        vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])
        gymnasium_env = GymnasiumWrapper(vec_env)

    elif environment_string == "cart_pole":
        def make_env():
            if concept_list is None or use_processed:
                env = gym.make("CartPole-v1", render_mode="rgb_array")
                env = Monitor(env)
                env = FrameSkipWrapper(env, skip=1, get_pixels_fn=get_raw_pixels_cartpole)
                env = ConceptWrapper(env,None,spaces.Box(
                        low=0, high=255,
                        shape=(84,84),  # Height x Width, no color channel
                        dtype=np.uint8
                    ),lambda env, obs: obs,use_info_obs=True)
                env = FrameStack(env,4)
                env = LazyFramesToNumpy(env)
                if use_processed:
                    env = OptimizedConceptWrapper(env, fast_predictor, spaces.MultiBinary(len(concept_idx)), lambda env, obs: obs, concept_idx,use_info_obs=True)
            else:
                env = Monitor(gym.make("CartPole-v1"))
                env = ConceptWrapper(env,concept_list,spaces.MultiBinary(len(concept_list)),get_raw_state_cartpole)
            return env 

        vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])

        gymnasium_env = GymnasiumWrapper(vec_env)

    elif environment_string == "boxing":
        if concept_list is None:
            vec_env = get_n_atari_env(num_envs,"BoxingNoFrameskip-v4",None,gym.spaces.Box(
                    low=0, high=255,
                    shape=(84,84),  # Height x Width, no color channel
                    dtype=np.uint8
                ),num_stack=num_stack)
        else:
            vec_env = get_n_atari_env(num_envs,"BoxingNoFrameskip-v4",concept_list,spaces.Box(low=-255, high=255, shape=(len(concept_list),), dtype=np.float32))
        gymnasium_env = GymnasiumWrapper(vec_env)

    elif environment_string == "pong":
        if concept_list is None:
            vec_env = get_n_atari_env(num_envs,"PongNoFrameskip-v4",None,gym.spaces.Box(
                    low=0, high=255,
                    shape=(84,84),  # Height x Width, no color channel
                    dtype=np.uint8
                ),num_stack=num_stack)
        else:
            vec_env = get_n_atari_env(num_envs,"PongNoFrameskip-v4",concept_list,spaces.Box(low=-1, high=1, shape=(len(concept_list),), dtype=np.float32))
        gymnasium_env = GymnasiumWrapper(vec_env)
    return vec_env, gymnasium_env, additional_info
