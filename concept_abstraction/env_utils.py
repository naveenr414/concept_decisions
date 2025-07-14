from concept_abstraction.environments import TreeRepeatEnv, Cyclic4StateEnv
import torch.nn.functional as F
import torch
import numpy as np


def create_environment_from_string(environment_string,concept_list,error):
    models_by_string = {
        'tree': TreeRepeatEnv, 
        'cycle': Cyclic4StateEnv,
    }
    
    return models_by_string[environment_string](concept_list,error)

def get_baseline_concept_sets(environment_string):
    baseline_concepts_by_string = {
        'tree': [[0,1,2,3],[4]],
        'cycle': [[0,1,2],[0],[1],[2]]
    }

    return baseline_concepts_by_string[environment_string]

def get_values(env,q_net):
    unique_obs = set()
    state_to_obs = {}
    obs_to_val = {}
    obs_to_q = {}

    # Get all unique observations and their value
    for s in env.all_states:
        env.state = s
        o = tuple(env.get_observation())
        unique_obs.add(o)
        state_to_obs[s] = o

    for o in unique_obs:
        o_tensor = torch.tensor(o).unsqueeze(0).float()
        q_vals = q_net(o_tensor).detach().squeeze().numpy()
        v = q_vals.max()
        obs_to_val[o] = v
        obs_to_q[o] = q_vals

    vals = []
    for s in env.all_states:
        o = state_to_obs[s]
        q_vals = obs_to_q[o]
        action_probs = F.softmax(torch.tensor(q_vals), dim=0).numpy()
        best_action = np.argmax(q_vals)
        vals.append(obs_to_val[o])
    return vals 