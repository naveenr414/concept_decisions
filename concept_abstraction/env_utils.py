from concept_abstraction.environments import TreeRepeatEnv, Cyclic4StateEnv, get_custom_binary_features, RewardPerturbationWrapper, ObservationSubsetWrapper, DiscretizeObservationWrapper, BinaryObservationSubsetWrapper, CustomBinaryFeatureWrapper, get_binary_subset_env
from concept_abstraction.training import train_ppo_model
import numpy as np
import numpy as np
import gymnasium as gym
import os 
from stable_baselines3 import PPO


def create_environment_from_string(environment_string,environment_nodes,concept_list,error):
    """Initialize an environment based on a string
    
    Arguments:
        environment_string: String, e.g., tree or cycle
        concept_list: List of concepts which are used to define the state
        error: float, erorr in concept prediction
    
    Returns: Gymasium Environment"""
    
    models_by_string = {
        'tree': TreeRepeatEnv, 
        'cycle': Cyclic4StateEnv,
    }
    
    return models_by_string[environment_string](environment_nodes,concept_list,error)

def create_environment_from_string_real_world(environment_string,concept_list,accuracies=None,reward_error=0):
    """Initialize an environment based on a string
    
    Arguments:
        environment_string: String, e.g., tree or cycle
        concept_list: List of concepts which are used to define the state
        error: float, erorr in concept prediction
    
    Returns: Gymasium Environment"""

    env = gym.make("CartPole-v1")

    if environment_string == "cart_pole":
        env = ObservationSubsetWrapper(env, indices=concept_list)
    elif environment_string == "cart_pole_binary":
        env = DiscretizeObservationWrapper(env, bins_per_feature=4)
        env = BinaryObservationSubsetWrapper(env, concept_list,accuracies)
    elif environment_string == "cart_pole_post_hoc":
        golden_model = get_golden_model(environment_string)
        env = get_binary_subset_env(golden_model, env, concept_list,accuracies=accuracies)
    elif environment_string == "cart_pole_llm":
        env = CustomBinaryFeatureWrapper(env)
        env = BinaryObservationSubsetWrapper(env, concept_list,accuracies=accuracies)
    else:
        raise Exception("Environment {} not implemented".format(environment_string))
    
    env.concepts = get_all_concepts(environment_string)

    if reward_error > 0:
        env = RewardPerturbationWrapper(env,reward_error)

    return env

def get_all_concepts(environment_string):
    """Get the list of all concepts from a string
    
    Arguments:
        environment_string: String, e.g., tree or cycle
    
    Returns: List with indices into all concepts"""

    if environment_string == "cart_pole":
        return list(range(4))
    elif environment_string == "cart_pole_binary":
        return list(range(16))
    elif environment_string == "cart_pole_post_hoc":
        return list(range(16))
    elif environment_string == "cart_pole_llm":
        return list(range(13))
    else:
        raise Exception("Environment {} not implemented".format(environment_string))

def get_golden_model(environment_string,reward_error=0):
    """Get the optimal model for a given environment
    
    Arguments:
        environment_string: String representing the environment
            which we want to instantiate
    
    Returns: StableBaseline model"""

    if "cart_pole" in environment_string:
        model_path = f"../../models/cart_pole/cart_pole_{reward_error}.zip"
        if os.path.exists(model_path):
            model = PPO.load(model_path)
            return model 
        else:
            env = create_environment_from_string_real_world("cart_pole",[0,1,2,3],None,reward_error)
            golden_model = train_ppo_model(env,total_timesteps=100000)
            golden_model.save(model_path)
            return golden_model 
    else:
        raise Exception("Environment {} not applicable for golden model".format(environment_string))

def convert_env_state_to_concept(environment_string,env,state):
    """Given a state as a 4-vector, convert this 
        to a concept vector, depending on the representation
    
    Arguments:
        environment_string: One of 
            cart_pole
            cart_pole_binary
            cart_pole_post_hoc
            cart_pole_llm
        env: A gymnasium environment
        state: 4 tuple state in CartPole

    Returns: Vector representing the 
        corresponding concept(s)"""

    if environment_string == "cart_pole_binary":
        bins_per_feature = 4
        n_features =  4
        bin_edges = [
            np.linspace(-4.8, 4.8, bins_per_feature + 1),         # Cart position
            np.linspace(-3.0, 3.0, bins_per_feature + 1),         # Cart velocity
            np.linspace(-0.418, 0.418, bins_per_feature + 1),     # Pole angle
            np.linspace(-3.5, 3.5, bins_per_feature + 1)          # Pole angular velocity
        ]

        binary_obs = np.zeros(n_features * bins_per_feature, dtype=np.int8)
        for i in range(n_features):
            bin_index = np.digitize(state[i], bin_edges[i]) - 1
            bin_index = np.clip(bin_index, 0, bins_per_feature - 1)
            offset = i * bins_per_feature
            binary_obs[offset + bin_index] = 1
        return binary_obs
    elif environment_string == "cart_pole_post_hoc":
        observation_2d = state.reshape(1, -1)
        binary_obs = env.feature_extractor.convert_to_binary(observation_2d)
        return binary_obs[0]
    elif environment_string == "cart_pole_llm":
        return get_custom_binary_features(state)
    else:
        return state 

def get_baseline_concept_sets(environment_string,environment_nodes):
    """Retrieve a list of potential concept sets from a string
    
    Arguments:
        environment_string: String, e.g., tree or cycle
    
    Returns: List of lists, representing different concept 
        combinations"""
    
    baseline_concepts_by_string = {
        'tree': [list(range(int(np.log2(environment_nodes+1)))),[int(np.log2(environment_nodes+1))]],
        'cycle': [list(range(0,environment_nodes-1))] + [[j] for j in list(range(0,environment_nodes-1))]
    }

    return baseline_concepts_by_string[environment_string]

def get_values(env, q_net, num_rollouts=10, max_steps=100, gamma=1):
    """
    Estimate V^{π}(s) for all states using rollouts from each state under the policy implied by q_net.
        Currently it does so with no discounting, but with average reward

    Args:
        env: The environment with .all_states and .state access.
        q_net: Trained Q-network (maps obs to Q-values).
        num_rollouts: Number of rollouts per state.
        max_steps: Max steps per rollout.
        gamma: Discount factor.
    
    Returns:
        List of value estimates V(s) for each s in env.all_states.
    """
    values = []

    for s in env.all_states:
        total_return = 0.0

        for _ in range(num_rollouts):
            env.reset()
            try:
                env.unwrapped.state = s
            except AttributeError:
                raise AttributeError("env must support setting env.unwrapped.state directly")
            total_reward = 0.0
            done = False
            steps = 0

            while not done and steps < max_steps:
                obs = env.get_observation()
                action, _ = q_net.predict(obs, deterministic=True)
                obs, reward, done, _, _ = env.step(action)

                total_reward += reward
                steps += 1

            avg_reward = total_reward / steps
            total_return += avg_reward

        values.append(total_return / num_rollouts)

    return values

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
            action = model.predict(observation)[0]

            # Take a step in the environment
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward 
            # End episode if done
            if terminated or truncated:
                break
    env.close()
    total_reward /= 10000
    return total_reward

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