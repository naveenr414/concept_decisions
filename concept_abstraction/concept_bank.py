import numpy as np
import hashlib 

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
def get_cart_pole_concept(i):
    """Get the ith index of a concept
    
    Arguments:
        i: Integer, idx
    
    Returns: Function that returns the ith index into a vector"""

    def get_concept(obs):
        return obs[i]
    return get_concept

def get_glucose_concept(i):
    """Get the ith index of a concept
    
    Arguments:
        i: Integer, idx
    
    Returns: Function that returns the ith index into a vector"""

    def get_concept(obs):
        return obs[i]
    return get_concept



### Minigrid Concepts

def obj_row(obs,i,obj_id):
    obs = obs[:147].reshape((3,7,7))[0,:,:]
    return int(obj_id in list(obs[i,:]))

def obj_column(obs,i,obj_id):
    obs = obs[:147].reshape((3,7,7))[0,:,:]
    return int(obj_id in list(obs[:,i]))

def door_open(obs,obj_id):
    door_loc = (-1,-1)
    obs = obs[:147].reshape((3,7,7))

    for i in range(len(obs[0])):
        for j in range(len(obs[0][i])):
            if obs[0][i][j] == obj_id:
                door_loc = (i,j)
    return int(obs[1][door_loc[0]][door_loc[1]] == 1)

def mini_grid_position_x(obs,i):
    return int(obs[-3] == i)

def mini_grid_position_y(obs,j):
    return int(obs[-2] == j)

def mini_grid_direction(obs,d):
    return int(obs[-1] == d)

def get_all_mini_grid_concepts():
    all_concepts = [lambda obs,i=i: obs[i] for i in range(12)]
    return all_concepts

def get_all_mini_grid_binary_concepts():
    val_ranges = [(1,5),(1,5),(1,4),(1,5),(1,5),(1,5),(1,5),(0,1),(0,1),(0,1),(0,1),(0,1)]
    all_concepts = []

    for i in range(12):
        for j in range(val_ranges[i][0],val_ranges[i][1]+1):
            all_concepts.append(lambda obs,i=i,j=j: int(obs[i] == j))
    return all_concepts

def get_all_mini_grid_names():
    all_concepts = []
    val_ranges = [(1,5),(1,5),(1,4),(1,5),(1,5),(1,5),(1,5),(0,1),(0,1),(0,1),(0,1),(0,1)]
    vec = ["X Pos","Y Pos","Dir","Key X","Key Y","Door X","Door Y","Door Open","Right","Left","Down","Up"]

    for i in range(12):
        for j in range(val_ranges[i][0],val_ranges[i][1]+1):
            all_concepts.append(vec[i])
    return all_concepts


### Boxing Concepts
def boxing_player_x(obs):
    obs = np.array(obs)
    return obs[-1,0]/255

def boxing_player_y(obs):
    obs = np.array(obs)
    return obs[-1,1]/255

def boxing_enemy_x(obs):
    obs = np.array(obs)
    return obs[-1,2]/255

def boxing_enemy_y(obs):
    obs = np.array(obs)
    return obs[-1,3]/255

def boxing_player_v_x(obs):
    obs = np.array(obs)
    return obs[-1,0]-obs[-2,0]

def boxing_player_v_y(obs):
    obs = np.array(obs)
    return obs[-1,1]-obs[-2,1]

def boxing_enemy_v_x(obs):
    obs = np.array(obs)
    return obs[-1,2]-obs[-2,2]

def boxing_enemy_v_y(obs):
    obs = np.array(obs)
    return obs[-1,3]-obs[-2,3]

def boxing_player_enemy_diff_x(obs):
    return obs[-1,0]-obs[-1,2]

def boxing_player_enemy_diff_y(obs):
    return obs[-1,1]-obs[-1,3]


### Pong Concepts
def pong_paddle_y(obs):
    obs = np.array(obs)
    return (obs[-1,1]-128)/255

def pong_ball_x(obs):
    obs = np.array(obs)
    return (obs[-1,2]-128)/255

def pong_ball_y(obs):
    obs = np.array(obs)
    return (obs[-1,3]-128)/255

def pong_paddle_x_diff(obs):
    obs = np.array(obs)
    return (obs[-1,0]-obs[-1,2])/255

def pong_paddle_y_diff(obs):
    obs = np.array(obs)
    return (obs[-1,1]-obs[-1,3])/255

def pong_enemy_y_diff(obs):
    obs = np.array(obs)
    return (obs[-1,1]-obs[-1,5])/255

def pong_ball_v_x(obs):
    obs = np.array(obs)
    return np.clip(obs[-1,2]-obs[-2,2],-4,4)/4

def pong_ball_v_y(obs):
    obs = np.array(obs)
    return np.clip(obs[-1,3]-obs[-2,3],-4,4)/4

def pong_enemy_y(obs):
    obs = np.array(obs)
    return (obs[-1,5]-128)/255

def pong_enemy_v_y(obs):
    obs = np.array(obs)
    return np.clip(obs[-2,5]-obs[-1,5],-4,4)/4

def pong_enemy_ball_x_diff(obs):
    obs = np.array(obs)
    return (obs[-1,4]-obs[-1,2])/255

def pong_enemy_ball_y_diff(obs):
    obs = np.array(obs)
    return (obs[-1,5]-obs[-1,3])/255


def binarize_function(func,threshold):
    def f_greater(obs):
        return int(func(obs)>=threshold)
    
    def f_less(obs):
        return int(func(obs)<threshold)
    
    return [f_less,f_greater]

def less_function(func,threshold):
    def f_less(obs):
        return int(func(obs)<threshold)
    
    return f_less 

def binarize_function_list(func_list,threshold_list):
    new_funcs = []
    for (f,t) in zip(func_list,threshold_list):
        new_funcs += binarize_function(f,t)
    
    return new_funcs

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

    for num_nodes in [4,8,16,32]:
        human_selected['cyclic_{}'.format(num_nodes)] = get_all_cyclic_concepts(num_nodes)
        human_selected_binary['cyclic_{}'.format(num_nodes)] = get_all_cyclic_concepts(num_nodes)

    for num_nodes in [7,15,31,63]:
        num_layers = len(bin(num_nodes+1))-3
        human_selected['tree_{}'.format(num_nodes)] = get_all_tree_concepts(num_layers)
        human_selected_binary['tree_{}'.format(num_nodes)] = get_all_tree_concepts(num_layers)

    human_selected['cart_pole'] = [get_cart_pole_concept(i) for i in range(4)]
    full_lst = []

    thresholds = [[-0.02,0.02],[-0.2,-0.1,0.1,0.2],[-0.02,0.02],[-0.3,-0.15,0.15,0.3]]

    for i in range(4):
        for t in thresholds[i]:
            full_lst.append(less_function(get_cart_pole_concept(i),t))   
    human_selected_binary['cart_pole'] = full_lst

    full_lst = []
    thresholds = [
        [0.25, 0.35, 0.50, 0.65, 0.90, 1.00],
        [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20],
        [0.00, 0.10, 0.25, 0.50, 0.75, 1.00],
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
        [-0.90, -0.50, -0.10, 0.10, 0.50, 0.90],
        [-0.90, -0.50, -0.10, 0.10, 0.50, 0.90],
    ]

    for i in range(6):
        for t in thresholds[i]:
            full_lst.append(less_function(get_glucose_concept(i),t))   
    human_selected_binary['glucose'] = full_lst


    human_selected['mini_grid'] = get_all_mini_grid_concepts()
    human_selected_binary['mini_grid'] = get_all_mini_grid_binary_concepts()    
    

    human_selected['pong'] = [pong_paddle_y,pong_ball_x,pong_ball_y,pong_ball_v_x,pong_ball_v_y,pong_enemy_y,pong_enemy_v_y,pong_paddle_x_diff,pong_paddle_y_diff,pong_enemy_y_diff,pong_enemy_ball_x_diff,pong_enemy_ball_y_diff]
    full_lst = []
    for threshold in [-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        for func in human_selected['pong']:
            full_lst.append(less_function(func,threshold))
    human_selected_binary['pong'] = full_lst

    human_selected['boxing'] = [boxing_player_x,boxing_player_y,boxing_enemy_x,boxing_enemy_y,boxing_player_v_x,boxing_player_v_y,boxing_enemy_v_x,boxing_enemy_v_y,boxing_player_enemy_diff_x,boxing_player_enemy_diff_y]    
    full_lst = []
    for threshold in [-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        for func in human_selected['boxing']:
            full_lst.append(less_function(func,threshold))
    human_selected_binary['boxing'] = full_lst

    if concept_source == 'human_selected':
        return human_selected[environment_string]
    elif concept_source == 'human_selected_binary':
        return human_selected_binary[environment_string]


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
