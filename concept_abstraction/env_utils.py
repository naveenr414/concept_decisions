from concept_abstraction.environments import TreeRepeatEnv, Cyclic4StateEnv
import numpy as np


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