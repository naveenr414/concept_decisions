import numpy as np
import hashlib 
import torch
from dataclasses import dataclass
from copy import deepcopy
@dataclass
class ParsedConcept:
    name: str
    feature_fn: callable
    threshold: float
    concept_fn: callable
    meta: dict

def make_thresholded_feature(feature_fn, threshold):
    """Return a concept function that outputs (batch_size,) binary float tensor."""
    def concept_fn(obs_tensor):
        vals = feature_fn(obs_tensor)
        return (vals > threshold).float()
    return concept_fn

def make_equality_feature(feature_fn, target_value):
    """Return a concept function that checks equality (for discrete values)."""
    def concept_fn(obs_tensor):
        vals = feature_fn(obs_tensor)
        return (vals == target_value).float()
    return concept_fn

def build_concepts_with_parsing(feature_fns, thresholds, meta_map, use_equality=False):
    """
    Build concepts from features.
    
    Args:
        feature_fns: List of feature extraction functions
        thresholds: List of threshold/target values
        meta_map: Metadata for each feature function
        use_equality: If True, create equality concepts (feature == value) instead of threshold concepts (feature > threshold)
    
    Returns:
        concept_list: List of concept functions
        parsed: List of ParsedConcept objects with metadata
    """
    parsed = []
    
    for base_idx, fn in enumerate(feature_fns):
        base_meta = meta_map[fn]
        fn_name = fn.__name__
        
        for val in thresholds:
            val_float = float(val)
            
            if use_equality:
                # Create equality concept: feature == value
                concept_fn = make_equality_feature(fn, val_float)
                name = f"{fn_name} == {val_float:.0f}"
            else:
                # Create threshold concept: feature > threshold
                concept_fn = make_thresholded_feature(fn, val_float)
                name = f"{fn_name} > {val_float:.3f}"
            
            # Store the base feature metadata along with concept info
            meta = deepcopy(base_meta)
            meta["base_idx"] = base_idx
            
            if use_equality:
                meta["value"] = val_float  # Store target value for equality
            else:
                meta["thr"] = val_float    # Store threshold for comparison
            
            parsed.append(
                ParsedConcept(
                    name=name,
                    feature_fn=fn,
                    threshold=val_float if not use_equality else None,
                    concept_fn=concept_fn,
                    meta=meta
                )
            )
    
    concept_list = [p.concept_fn for p in parsed]
    
    return concept_list, parsed


### Cyclic Concepts
def cyclic_concept_mod(i):
    def get_concept(state):
        return int((state+1)%i == 0)
    return get_concept 

def get_all_cyclic_concepts(env_nodes):
    return [cyclic_concept_mod(i) for i in range(2,env_nodes+1)]

### Tree Concepts
def get_binary_tree_concept(i,num_layers):
    def get_concept(state):
        binary_rep = bin(state+1)[2:]
        binary_rep = '0'*(num_layers-len(binary_rep)) + binary_rep
        return int(binary_rep[i])
    return get_concept 

def get_final_tree_concept(state):
    n = state+1 
    if (n & (n-1) == 0) and n != 0:
        return 1 
    else:
        return 0

def get_all_tree_concepts(num_layers):
    return [get_binary_tree_concept(i,num_layers) for i in range(num_layers)]+[get_final_tree_concept]

### CartPole Concepts
def cartpole_position(obs):
    # obs: (N, num_frames, D)
    return obs[:, -1, 0]

def cartpole_velocity(obs):
    return obs[:, -1, 1]

def cartpole_angle(obs):
    return obs[:, -1, 2]

def cartpole_angular_velocity(obs):
    return obs[:, -1, 3]


def build_cartpole_meta():
    """
    Returns a dict mapping each cartpole_feature_fn to a metadata dictionary
    describing its type and indices.
    """
    meta_map = {}

    # ----- value-type concepts (direct state values) -----
    meta_map[cartpole_position] = {
        "type": "value",
        "frame": -1,
        "idx": 0,
        "scale": 1.0,
    }
    meta_map[cartpole_velocity] = {
        "type": "value",
        "frame": -1,
        "idx": 1,
        "scale": 1.0,
    }
    meta_map[cartpole_angle] = {
        "type": "value",
        "frame": -1,
        "idx": 2,
        "scale": 1.0,
    }
    meta_map[cartpole_angular_velocity] = {
        "type": "value",
        "frame": -1,
        "idx": 3,
        "scale": 1.0,
    }

    return meta_map


### Minigrid Concepts
def minigrid_feature_0(obs):
    # obs: (N, num_frames, D)
    return obs[:, -1, 0]

def minigrid_feature_1(obs):
    return obs[:, -1, 1]

def minigrid_feature_2(obs):
    return obs[:, -1, 2]

def minigrid_feature_3(obs):
    return obs[:, -1, 3]

def minigrid_feature_4(obs):
    return obs[:, -1, 4]

def minigrid_feature_5(obs):
    return obs[:, -1, 5]

def minigrid_feature_6(obs):
    return obs[:, -1, 6]

def minigrid_feature_7(obs):
    return obs[:, -1, 7]

def minigrid_feature_8(obs):
    return obs[:, -1, 8]

def minigrid_feature_9(obs):
    return obs[:, -1, 9]

def minigrid_feature_10(obs):
    return obs[:, -1, 10]

def minigrid_feature_11(obs):
    return obs[:, -1, 11]


def build_minigrid_meta():
    """
    Returns a dict mapping each minigrid_feature_fn to a metadata dictionary
    describing its type and indices.
    """
    meta_map = {}
    
    # All MiniGrid features are value-type concepts
    for i in range(12):
        fn = globals()[f'minigrid_feature_{i}']
        meta_map[fn] = {
            "type": "value",
            "frame": -1,
            "idx": i,
            "scale": 1.0,
        }
    
    return meta_map



## Pong Concepts
def pong_paddle_y(obs):
    # obs: (N, num_frames, D)
    return (obs[:, -1, 1] - 128) / 255.0

def pong_ball_x(obs):
    return (obs[:, -1, 2] - 128) / 255.0

def pong_ball_y(obs):
    return (obs[:, -1, 3] - 128) / 255.0

def pong_ball_v_x(obs):
    # velocity with clipping
    diff = obs[:, -1, 2] - obs[:, -2, 2]
    return torch.clamp(diff, -4, 4) / 4.0

def pong_ball_v_y(obs):
    diff = obs[:, -1, 3] - obs[:, -2, 3]
    return torch.clamp(diff, -4, 4) / 4.0

def pong_enemy_y(obs):
    return (obs[:, -1, 5] - 128) / 255.0

def pong_enemy_v_y(obs):
    diff = obs[:, -2, 5] - obs[:, -1, 5]
    return torch.clamp(diff, -4, 4) / 4.0

def pong_paddle_x_diff(obs):
    return (obs[:, -1, 0] - obs[:, -1, 2]) / 255.0

def pong_paddle_y_diff(obs):
    return (obs[:, -1, 1] - obs[:, -1, 3]) / 255.0

def pong_enemy_y_diff(obs):
    return (obs[:, -1, 1] - obs[:, -1, 5]) / 255.0

def pong_enemy_ball_x_diff(obs):
    return (obs[:, -1, 4] - obs[:, -1, 2]) / 255.0

def pong_enemy_ball_y_diff(obs):
    return (obs[:, -1, 5] - obs[:, -1, 3]) / 255.0


def build_pong_meta():
    """
    Returns a dict mapping each pong_feature_fn to a metadata dictionary
    describing its type and indices.
    """
    meta_map = {}

    # ----- value-type concepts (scaled positions) -----
    meta_map[pong_paddle_y] = {
        "type": "value",
        "frame": -1,
        "idx": 1,
        "scale": 1/255.0,
        "offset": -128,
    }
    meta_map[pong_ball_x] = {
        "type": "value",
        "frame": -1,
        "idx": 2,
        "scale": 1/255.0,
        "offset": -128,
    }
    meta_map[pong_ball_y] = {
        "type": "value",
        "frame": -1,
        "idx": 3,
        "scale": 1/255.0,
        "offset": -128,
    }
    meta_map[pong_enemy_y] = {
        "type": "value",
        "frame": -1,
        "idx": 5,
        "scale": 1/255.0,
        "offset": -128,
    }

    # ----- velocity concepts (diff between frames with clipping) -----
    meta_map[pong_ball_v_x] = {
        "type": "velocity",
        "frame1": -1, "idx1": 2,
        "frame2": -2, "idx2": 2,
        "clip_min": -4,
        "clip_max": 4,
        "scale": 1/4.0,
    }
    meta_map[pong_ball_v_y] = {
        "type": "velocity",
        "frame1": -1, "idx1": 3,
        "frame2": -2, "idx2": 3,
        "clip_min": -4,
        "clip_max": 4,
        "scale": 1/4.0,
    }
    meta_map[pong_enemy_v_y] = {
        "type": "velocity",
        "frame1": -2, "idx1": 5,
        "frame2": -1, "idx2": 5,
        "clip_min": -4,
        "clip_max": 4,
        "scale": 1/4.0,
    }

    # ----- difference concepts (same frame, different indices) -----
    meta_map[pong_paddle_x_diff] = {
        "type": "diff",
        "frame1": -1, "idx1": 0,
        "frame2": -1, "idx2": 2,
        "scale": 1/255.0,
    }
    meta_map[pong_paddle_y_diff] = {
        "type": "diff",
        "frame1": -1, "idx1": 1,
        "frame2": -1, "idx2": 3,
        "scale": 1/255.0,
    }
    meta_map[pong_enemy_y_diff] = {
        "type": "diff",
        "frame1": -1, "idx1": 1,
        "frame2": -1, "idx2": 5,
        "scale": 1/255.0,
    }
    meta_map[pong_enemy_ball_x_diff] = {
        "type": "diff",
        "frame1": -1, "idx1": 4,
        "frame2": -1, "idx2": 2,
        "scale": 1/255.0,
    }
    meta_map[pong_enemy_ball_y_diff] = {
        "type": "diff",
        "frame1": -1, "idx1": 5,
        "frame2": -1, "idx2": 3,
        "scale": 1/255.0,
    }

    return meta_map


### Boxing Concepts
def boxing_player_x(obs):
    # obs: (N, num_frames, D)
    return obs[:, -1, 0] / 255.0

def boxing_player_y(obs):
    return obs[:, -1, 1] / 255.0

def boxing_enemy_x(obs):
    return obs[:, -1, 2] / 255.0

def boxing_enemy_y(obs):
    return obs[:, -1, 3] / 255.0

def boxing_player_v_x(obs):
    # velocity = last_frame - second_last_frame
    return obs[:, -1, 0] - obs[:, -2, 0]

def boxing_player_v_y(obs):
    return obs[:, -1, 1] - obs[:, -2, 1]

def boxing_enemy_v_x(obs):
    return obs[:, -1, 2] - obs[:, -2, 2]

def boxing_enemy_v_y(obs):
    return obs[:, -1, 3] - obs[:, -2, 3]

def boxing_player_enemy_diff_x(obs):
    return obs[:, -1, 0] - obs[:, -1, 2]

def boxing_player_enemy_diff_y(obs):
    return obs[:, -1, 1] - obs[:, -1, 3]


def build_boxing_meta():
    """
    Returns a dict mapping each boxing_feature_fn to a metadata dictionary
    describing its type and indices, so compute_concepts_vectorized() can 
    evaluate them quickly.
    """
    meta_map = {}

    # ----- value-type concepts (just scaled last-frame positions) -----
    meta_map[boxing_player_x] = {
        "type": "value",
        "frame": -1,
        "idx": 0,
        "scale": 1/255.0,
    }
    meta_map[boxing_player_y] = {
        "type": "value",
        "frame": -1,
        "idx": 1,
        "scale": 1/255.0,
    }
    meta_map[boxing_enemy_x] = {
        "type": "value",
        "frame": -1,
        "idx": 2,
        "scale": 1/255.0,
    }
    meta_map[boxing_enemy_y] = {
        "type": "value",
        "frame": -1,
        "idx": 3,
        "scale": 1/255.0,
    }

    # ----- velocity concepts (diff between last and second-last frame) -----
    meta_map[boxing_player_v_x] = {
        "type": "diff",
        "frame1": -1, "idx1": 0,
        "frame2": -2, "idx2": 0,
    }
    meta_map[boxing_player_v_y] = {
        "type": "diff",
        "frame1": -1, "idx1": 1,
        "frame2": -2, "idx2": 1,
    }
    meta_map[boxing_enemy_v_x] = {
        "type": "diff",
        "frame1": -1, "idx1": 2,
        "frame2": -2, "idx2": 2,
    }
    meta_map[boxing_enemy_v_y] = {
        "type": "diff",
        "frame1": -1, "idx1": 3,
        "frame2": -2, "idx2": 3,
    }

    # ----- player–enemy difference concepts (same frame) -----
    meta_map[boxing_player_enemy_diff_x] = {
        "type": "diff",
        "frame1": -1, "idx1": 0,
        "frame2": -1, "idx2": 2,
    }
    meta_map[boxing_player_enemy_diff_y] = {
        "type": "diff",
        "frame1": -1, "idx1": 1,
        "frame2": -1, "idx2": 3,
    }

    return meta_map

## Glucose
def glucose_feature_0(obs):
    # obs: (N, num_frames, D)
    return obs[:, -1, 0]

def glucose_feature_1(obs):
    return obs[:, -1, 1]

def glucose_feature_2(obs):
    return obs[:, -1, 2]

def glucose_feature_3(obs):
    return obs[:, -1, 3]

def glucose_feature_4(obs):
    return obs[:, -1, 4]

def glucose_feature_5(obs):
    return obs[:, -1, 5]


def build_glucose_meta():
    """
    Returns a dict mapping each glucose_feature_fn to a metadata dictionary
    describing its type and indices.
    """
    meta_map = {}
    
    # All Glucose features are value-type concepts
    for i in range(6):
        fn = globals()[f'glucose_feature_{i}']
        meta_map[fn] = {
            "type": "value",
            "frame": -1,
            "idx": i,
            "scale": 1.0,
        }
    
    return meta_map

def get_concepts(environment_string,concept_source,seed):
    """Get concepts depending on the source
    
    Arguments:
        environment_string: String, such as Pong or Boxing
        num_concepts: The number of concepts to select from this environment
        concept_source: String, human_selected, llm_generated, etc.
        seed: Integer, random_state
    
    Returns: List of functions, each being a concept"""

    human_selected = {}
    human_selected_binary = {}
    parsed_human_selected = {}

    for num_nodes in [4,8,16,32]:
        human_selected['cyclic_{}'.format(num_nodes)] = get_all_cyclic_concepts(num_nodes)
        human_selected_binary['cyclic_{}'.format(num_nodes)] = get_all_cyclic_concepts(num_nodes)

    for num_nodes in [7,15,31,63]:
        num_layers = len(bin(num_nodes+1))-3
        human_selected['tree_{}'.format(num_nodes)] = get_all_tree_concepts(num_layers)
        human_selected_binary['tree_{}'.format(num_nodes)] = get_all_tree_concepts(num_layers)

    cartpole_feature_fns = [
        cartpole_position,
        cartpole_velocity,
        cartpole_angle,
        cartpole_angular_velocity,
    ]

    # Different thresholds for each feature
    cartpole_thresholds = [
        [-0.02, 0.02],                    # position
        [-0.2, -0.1, 0.1, 0.2],          # velocity
        [-0.02, 0.02],                    # angle
        [-0.3, -0.15, 0.15, 0.3],        # angular velocity
    ]

    # Build concepts with per-feature thresholds
    concept_list = []
    parsed_concepts = []
    meta_map = build_cartpole_meta()

    for fn, thresholds in zip(cartpole_feature_fns, cartpole_thresholds):
        base_meta = meta_map[fn]
        fn_name = fn.__name__
        
        for thr in thresholds:
            thr_val = float(thr)
            
            # thresholded concept fn
            concept_fn = make_thresholded_feature(fn, thr_val)
            concept_list.append(concept_fn)
            
            # build the parsed concept
            meta = {
                "type": base_meta["type"],
                "frame": base_meta["frame"],
                "idx": base_meta["idx"],
                "scale": base_meta["scale"],
                "thr": thr_val,
                "base": None,
            }
            
            # store a ParsedConcept
            parsed_concepts.append(
                ParsedConcept(
                    name=f"{fn_name} > {thr_val:.3f}",
                    feature_fn=fn,
                    threshold=thr_val,
                    concept_fn=concept_fn,
                    meta=meta
                )
            )

    human_selected_binary["cart_pole"] = concept_list
    human_selected["cart_pole"] = concept_list
    parsed_human_selected["cart_pole"] = parsed_concepts

    minigrid_feature_fns = [
        minigrid_feature_0, minigrid_feature_1, minigrid_feature_2,
        minigrid_feature_3, minigrid_feature_4, minigrid_feature_5,
        minigrid_feature_6, minigrid_feature_7, minigrid_feature_8,
        minigrid_feature_9, minigrid_feature_10, minigrid_feature_11
    ]

    minigrid_val_ranges = [
        [1, 2, 3, 4, 5],  # feature 0
        [1, 2, 3, 4, 5],  # feature 1
        [1, 2, 3, 4],     # feature 2
        [1, 2, 3, 4, 5],  # feature 3
        [1, 2, 3, 4, 5],  # feature 4
        [1, 2, 3, 4, 5],  # feature 5
        [1, 2, 3, 4, 5],  # feature 6
        [0, 1],           # feature 7
        [0, 1],           # feature 8
        [0, 1],           # feature 9
        [0, 1],           # feature 10
        [0, 1],           # feature 11
    ]

    # Build concepts with per-feature value ranges
    meta_map = build_minigrid_meta()
    parsed_concepts = []

    for base_idx, (fn, values) in enumerate(zip(minigrid_feature_fns, minigrid_val_ranges)):
        base_meta = meta_map[fn]
        fn_name = fn.__name__
        
        for val in values:
            val_float = float(val)
            concept_fn = make_equality_feature(fn, val_float)
            
            meta = deepcopy(base_meta)
            meta["base_idx"] = base_idx
            meta["value"] = val_float
            
            parsed_concepts.append(
                ParsedConcept(
                    name=f"{fn_name} == {val_float:.0f}",
                    feature_fn=fn,
                    threshold=None,
                    concept_fn=concept_fn,
                    meta=meta
                )
            )

    concept_list = [p.concept_fn for p in parsed_concepts]
    human_selected_binary["mini_grid"] = concept_list
    human_selected["mini_grid"] = minigrid_feature_fns
    parsed_human_selected["mini_grid"] = parsed_concepts

    pong_feature_fns = [
        pong_paddle_y,
        pong_ball_x,
        pong_ball_y,
        pong_ball_v_x,
        pong_ball_v_y,
        pong_enemy_y,
        pong_enemy_v_y,
        pong_paddle_x_diff,
        pong_paddle_y_diff,
        pong_enemy_y_diff,
        pong_enemy_ball_x_diff,
        pong_enemy_ball_y_diff,
    ]

    thresholds = torch.tensor(
        [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    concept_list, parsed_concepts = build_concepts_with_parsing(
        pong_feature_fns,
        thresholds,
        build_pong_meta()
    )

    human_selected_binary["pong"] = concept_list
    human_selected["pong"] = concept_list
    parsed_human_selected["pong"] = parsed_concepts

    boxing_feature_fns = [
        boxing_player_x,
        boxing_player_y,
        boxing_enemy_x,
        boxing_enemy_y,
        boxing_player_v_x,
        boxing_player_v_y,
        boxing_enemy_v_x,
        boxing_enemy_v_y,
        boxing_player_enemy_diff_x,
        boxing_player_enemy_diff_y,
    ]

    thresholds = torch.tensor(
        [-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,
        0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    )
    concept_list, parsed_concepts = build_concepts_with_parsing(
        boxing_feature_fns,
        thresholds,
        build_boxing_meta()
    )
    human_selected_binary["boxing"] = concept_list
    human_selected["boxing"] = concept_list
    parsed_human_selected["boxing"] = parsed_concepts

    glucose_thresholds = [
        [0.1,0.3,0.5,0.7,0.75],
        [-0.15,-0.1,-0.075,-0.05,-0.025,0,0.05,0.1,0.15],
        [-0.001,0.05,0.1,0.15,0.2],
        [15,30,45,50,60,75],
        [-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.8],
        [-0.75,-0.5,-0.25,0,0.25,0.5,0.75,0.9,0.95]
    ]

    glucose_feature_fns = [
        glucose_feature_0,
        glucose_feature_1,
        glucose_feature_2,
        glucose_feature_3,
        glucose_feature_4,
        glucose_feature_5,
    ]


    # Build concepts with per-feature thresholds
    concept_list = []
    parsed_concepts = []
    meta_map = build_glucose_meta()

    for fn, thresholds in zip(glucose_feature_fns, glucose_thresholds):
        base_meta = meta_map[fn]
        fn_name = fn.__name__
        
        for thr in thresholds:
            thr_val = float(thr)
            
            # thresholded concept fn
            concept_fn = make_thresholded_feature(fn, thr_val)
            concept_list.append(concept_fn)
            
            # build the parsed concept
            meta = {
                "type": base_meta["type"],
                "frame": base_meta["frame"],
                "idx": base_meta["idx"],
                "scale": base_meta["scale"],
                "thr": thr_val,
                "base": None,
            }
            
            # store a ParsedConcept
            parsed_concepts.append(
                ParsedConcept(
                    name=f"{fn_name} > {thr_val:.3f}",
                    feature_fn=fn,
                    threshold=thr_val,
                    concept_fn=concept_fn,
                    meta=meta
                )
            )

    human_selected_binary["glucose"] = concept_list
    human_selected["glucose"] = glucose_feature_fns
    parsed_human_selected["glucose"] = parsed_concepts


    if concept_source == 'human_selected':
        return human_selected[environment_string], parsed_human_selected[environment_string]
    elif concept_source == 'human_selected_binary':
        return human_selected_binary[environment_string], parsed_human_selected[environment_string]


def inaccurate_concepts_binary(concept_function,accuracy,seed):
    """Creates a new concept function that only agrees with the
        concept function x% of the time
    
    Arguments:
        concept_function: Some map from state -> observation
            Here, observation must be binary
        accuracy: Float, the accuracy of the predictor
    
    Returns: New Concept Function"""

    seen_states = {}

    def pred_function(state):
        hashed_state = int(hashlib.md5(np.array(state).tobytes()).hexdigest(), 16) % (2**32)
        if hashed_state not in seen_states:
            pred = concept_function(state)
            if np.random.random() > accuracy:
                seen_states[hashed_state] =1-pred 
            else:
                seen_states[hashed_state] = pred 
        return seen_states[hashed_state]
    return pred_function


def inaccurate_concepts_binary_intervention(concept_function,accuracy,intervention_accuracy,intervention_probability,seed):
    """Creates a new concept function that only agrees with the
        concept function x% of the time
    
    Arguments:
        concept_function: Some map from state -> observation
            Here, observation must be binary
        accuracy: Float, the accuracy of the predictor
    
    Returns: New Concept Function"""

    seen_states = {}

    def pred_function(state):
        hashed_state = (int(hashlib.md5(state.tobytes()).hexdigest(), 16)+seed) % (2**32)
        np.random.seed(hashed_state)
        if hashed_state not in seen_states:
            pred = concept_function(state)
            if np.random.random() > intervention_probability:
                if np.random.random() > accuracy:
                    seen_states[hashed_state] =1-pred 
                else:
                    seen_states[hashed_state] = pred 
            else:
                if np.random.random() > intervention_accuracy:
                    seen_states[hashed_state] =1-pred 
                else:
                    seen_states[hashed_state] = pred 

        return seen_states[hashed_state]
    return pred_function


def inaccurate_concepts_continuous(concept_function,error_mean,error_std):
    """Creates a new concept function that only agrees with the
        concept function x% of the time
    
    Arguments:
        concept_function: Some map from state -> observation
            Here, observation must be binary
        accuracy: Float, the accuracy of the predictor
    
    Returns: New Concept Function"""

    def pred_function(state):
        seed = int(hashlib.md5(state.tobytes()).hexdigest(), 16) % (2**32)
        rand_number = np.random.default_rng(seed).normal(error_mean,error_std)
        pred = concept_function(state)
        pred += rand_number
        return pred 
    return pred_function
