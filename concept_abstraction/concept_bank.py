import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
import pandas as pd 
import gymnasium as gym

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

### MIMIC Concepts
def clustering_concept_mimic(X_train_concepts,n_clusters,seed):
    """Concept for MIMIC derived from clustering
    
    Arguments:
        n_clusters: Integer, number of clusters
        seed: Integer, random seed
    
    Returns: Function that maps state to cluster (integer)"""

    state_data = X_train_concepts
    sample = state_data[np.random.choice(len(state_data),
                                            size=int(len(state_data) * 0.25),
                                            replace=False)]
    clusterer = MiniBatchKMeans(n_clusters=n_clusters,
                                max_iter=30,n_init=32).fit(sample)
    centers = clusterer.cluster_centers_

    def cluster_concept(state_data):
        """
        Produces a clustering of the given state data, where each state is
        considered independent (even from the same patient).
        
        Returns: a clustering object that can be queried using a predict() function,
            and an array of clustering indexes ranging from 0 to n_clusters.
        """
        return clusterer.predict([state_data])[0]
    return cluster_concept, centers

def mimic_concept(i):
    """Get the ith index of a concept
    
    Arguments:
        i: Integer, idx
    
    Returns: Function that returns the ith index into a vector"""
    def mimic_index(obs):
        return obs[i] 
    
    return mimic_index

### CartPole Concepts
def get_cart_pole_concept(i):
    """Get the ith index of a concept
    
    Arguments:
        i: Integer, idx
    
    Returns: Function that returns the ith index into a vector"""

    def get_concept(obs):
        return obs[i]
    return get_concept 

def get_custom_binary_features(observation):
    """Given an observation (4 vector), convert
        this to a 13-length binary vector
        suggested by LLMs
    
    Arguments:
        observation: 4-vector of floats
            representing the position/velocity
            of the cart
    
    Returns: 13-length binary vector"""

    cart_pos, cart_vel, pole_angle, pole_ang_vel = observation
    binary_features = np.zeros(13, dtype=np.int32)
    binary_features[0] = int(abs(cart_pos) < 0.5)          # cart near center
    binary_features[1] = int(abs(cart_pos) >= 0.5)         # cart far from center
    binary_features[2] = int(cart_vel < 0)                 # cart moving left
    binary_features[3] = int(cart_vel > 0)                 # cart moving right
    binary_features[4] = int(abs(cart_vel) > 1.0)          # cart moving fast
    binary_features[5] = int(pole_angle < 0)               # pole leaning left
    binary_features[6] = int(pole_angle > 0)               # pole leaning right
    binary_features[7] = int(abs(pole_angle) < 0.02)       # pole near vertical
    binary_features[8] = int(abs(pole_angle) >= 0.1)       # pole far from vertical
    binary_features[9] = int(abs(pole_angle) >= 0.2)       # pole about to fall
    binary_features[10] = int(pole_ang_vel < 0)            # pole rotating clockwise
    binary_features[11] = int(pole_ang_vel > 0)            # pole rotating counterclockwise
    binary_features[12] = int(abs(pole_ang_vel) > 1.0)     # pole rotating fast
    return binary_features

class CustomBinaryFeatureWrapper(gym.ObservationWrapper):
    """
    Wrapper that converts CartPole observations to custom binary features
        These observations come from LLMs

    Features:
    0: Is cart near center? → |cart position| < 0.5
    1: Is cart far from center? → |cart position| ≥ 0.5
    2: Is cart moving left? → cart velocity < 0
    3: Is cart moving right? → cart velocity > 0
    4: Is cart moving fast? → |cart velocity| > 1.0
    5: Is pole leaning left? → pole angle < 0
    6: Is pole leaning right? → pole angle > 0
    7: Is pole near vertical? → |pole angle| < 0.02
    8: Is pole far from vertical? → |pole angle| ≥ 0.1
    9: Is pole about to fall? → |pole angle| ≥ 0.2
    10: Is pole rotating clockwise? → pole angular velocity < 0
    11: Is pole rotating counterclockwise? → pole angular velocity > 0
    12: Is pole rotating fast? → |pole angular velocity| > 1.0
    """
    
    def __init__(self, env,accuracies=None):
        super().__init__(env)
        self.observation_space = gym.spaces.MultiBinary(13)  # 13 binary features
        self.accuracies = accuracies

        self.feature_names = [
            "cart_near_center",      # 0
            "cart_far_from_center",  # 1
            "cart_moving_left",      # 2
            "cart_moving_right",     # 3
            "cart_moving_fast",      # 4
            "pole_leaning_left",     # 5
            "pole_leaning_right",    # 6
            "pole_near_vertical",    # 7
            "pole_far_from_vertical", # 8
            "pole_about_to_fall",    # 9
            "pole_rotating_clockwise", # 10
            "pole_rotating_ccw",     # 11
            "pole_rotating_fast"     # 12
        ]
    
    def observation(self, observation):
        """Convert continuous observation to binary features"""
        binary_features = get_custom_binary_features(observation)
        if self.accuracies is not None: 
            for i in range(len(self.accuracies)):
                if np.random.random() > self.accuracies[i]:
                    binary_features[i] = 1-binary_features[i]

        return binary_features
    
    def get_feature_names(self):
        """Return list of feature names"""
        return self.feature_names.copy()
    
    def print_observation(self, observation):
        """Print human-readable observation"""
        binary_obs = self.observation(observation)
        print("Binary Features:")
        for i, (name, value) in enumerate(zip(self.feature_names, binary_obs)):
            if value:
                print(f"  {i:2d}: {name} = True")

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


### Pong Concepts
def pong_paddle_y(obs):
    obs = np.array(obs)
    return obs[-1,1]/255

def pong_ball_x(obs):
    obs = np.array(obs)
    return obs[-1,2]/255

def pong_ball_y(obs):
    obs = np.array(obs)
    return obs[-1,3]/255

def pong_ball_v_x(obs):
    obs = np.array(obs)
    return obs[-1,2]-obs[-2,2]

def pong_ball_v_y(obs):
    obs = np.array(obs)
    return obs[-1,3]-obs[-2,3]

def pong_enemy_y(obs):
    obs = np.array(obs)
    return obs[-1,5]/255

def pong_enemy_v_y(obs):
    obs = np.array(obs)
    return obs[-2,5]-obs[-1,5]

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
    object_concepts = [lambda obs: obj_row(obs,i,1) for i in range(7)]+[lambda obs: obj_row(obs,i,2) for i in range(7)]+[lambda obs: obj_row(obs,i,4) for i in range(7)]+[lambda obs: obj_row(obs,i,5) for i in range(7)]+[lambda obs: obj_row(obs,i,8) for i in range(7)]
    object_concepts += [lambda obs: obj_column(obs,i,1) for i in range(7)]+[lambda obs: obj_column(obs,i,2) for i in range(7)]+[lambda obs: obj_column(obs,i,4) for i in range(7)]+[lambda obs: obj_column(obs,i,5) for i in range(7)]+[lambda obs: obj_column(obs,i,8) for i in range(7)]
    door_concepts = [lambda obs: door_open(obs,4)]
    agent_concepts = [lambda obs: mini_grid_position_x(obs,i) for i in range(7)] + [lambda obs: mini_grid_position_y(obs,i) for i in range(7)]+[lambda obs: mini_grid_direction(obs,i) for i in range(4)]
    all_concepts = object_concepts + door_concepts + agent_concepts
    return all_concepts

def binarize_function(func,threshold):
    def f_greater(obs):
        return int(func(obs)>=threshold)
    
    def f_less(obs):
        return int(func(obs)<threshold)
    
    return [f_less,f_greater]

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

    human_selected['mimic'] = [mimic_concept(i) for i in range(47)]
    human_selected['cart_pole'] = [get_cart_pole_concept(i) for i in range(4)]
    human_selected['boxing'] = [boxing_player_x,boxing_player_y,boxing_enemy_x,boxing_enemy_y,boxing_player_v_x,boxing_player_v_y,boxing_enemy_v_x,boxing_enemy_v_y]
    human_selected['pong'] = [pong_paddle_y,pong_ball_x,pong_ball_y,pong_ball_v_x,pong_ball_v_y,pong_enemy_y,pong_enemy_v_y]
    human_selected['mini_grid'] = get_all_mini_grid_concepts()
    human_selected_binary['mini_grid'] = get_all_mini_grid_concepts()

    human_selected_binary['mimic'] = binarize_function_list([mimic_concept(i) for i in range(47)],[0 for i in range(47)])
    human_selected_binary['cart_pole'] = binarize_function_list( [get_cart_pole_concept(i) for i in range(4)],[0,0,0,0])
    human_selected_binary['boxing'] = binarize_function_list([boxing_player_x,boxing_player_y,boxing_enemy_x,boxing_enemy_y,boxing_player_v_x,boxing_player_v_y,boxing_enemy_v_x,boxing_enemy_v_y],[0.5,0.5,0.5,0.5,0,0,0,0])
    human_selected_binary['pong'] = binarize_function_list([pong_paddle_y,pong_ball_x,pong_ball_y,pong_ball_v_x,pong_ball_v_y,pong_enemy_y,pong_enemy_v_y],[0.5,0.5,0.5,0,0,0,0])

    if concept_source == 'human_selected':
        return human_selected[environment_string]
    elif concept_source == 'human_selected_binary':
        return human_selected_binary[environment_string]

def inaccurate_concepts_binary(concept_function,accuracy):
    """Creates a new concept function that only agrees with the
        concept function x% of the time
    
    Arguments:
        concept_function: Some map from state -> observation
            Here, observation must be binary
        accuracy: Float, the accuracy of the predictor
    
    Returns: New Concept Function"""

    def pred_function(state):
        seed = hash(str(state)) % (2**32)
        rand_number = np.random.default_rng(seed).random()
        pred = concept_function(state)
        if rand_number > accuracy:
            return 1-pred 
        return pred 
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
        seed = hash(str(state)) % (2**32)
        rand_number = np.random.default_rng(seed).normal(error_mean,error_std)
        pred = concept_function(state)
        pred += rand_number
        return pred 
    return pred_function
