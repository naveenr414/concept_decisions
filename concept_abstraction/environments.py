import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Cyclic4StateEnv(gym.Env):
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

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self,environment_nodes,concept_list=[],acc_by_concept=None):
        super().__init__()
        concept_list = sorted(concept_list)
        self.environment_nodes = environment_nodes 
        self.action_space = spaces.Discrete(3)
        self.all_states = list(range(environment_nodes))
        self.concept_list = concept_list
        self.acc_by_concept = acc_by_concept 

        self.rewards = np.zeros((environment_nodes,3)) 
        for i in range(environment_nodes):
            if i%2 == 0:
                self.rewards[i,0] = self.rewards[i,1] = 3
            else:
                self.rewards[i,2] = 3
        self.rewards = np.array(self.rewards)

        self.transitions = []
        for i in range(len(self.all_states)):
            transitions_by_state = []
            for action in range(self.action_space.n):
                next_probs = [0.0 for i in range(len(self.all_states))]
                if action == 0:
                    next_probs[(i - 1) % self.environment_nodes] = 1.0
                if action == 1:
                    next_probs[(i + 1) % self.environment_nodes] = 1.0
                if action == 2:
                    next_probs[(i) % self.environment_nodes] = 1.0
                transitions_by_state.append(next_probs)
            self.transitions.append(transitions_by_state)
        self.transitions = np.array(self.transitions)

        self.concepts = []
        for i in range(2,environment_nodes+1):
            concept_vals = [int((j+1)%i == 0) for j in range(environment_nodes)]
            self.concepts.append(concept_vals)
        self.concepts = np.array(self.concepts)

        self.concepts_to_observation = {}
        idx = 0

        if acc_by_concept is None or np.min(acc_by_concept) == 1:
            for column in self.concepts.T:
                str_column = ''.join([str(column[j]) for j in concept_list])

                if str_column not in self.concepts_to_observation:
                    self.concepts_to_observation[str_column] = idx
                    idx += 1
        else:
            idx = 0
            for i in range(2**len(self.concepts)):
                num = bin(i)[2:]
                num = '0'*(len(self.concepts)-len(num))+num 
                num = ''.join([num[j] for j in concept_list])
                if num not in self.concepts_to_observation:
                    self.concepts_to_observation[num] = idx 
                    idx += 1

        self.observation_space = spaces.Discrete(len(self.concepts_to_observation))
        
        self.state = np.random.randint(0,self.environment_nodes)
        self.steps = 0

    def get_observation(self):
        current_concepts = self.concepts[self.concept_list,self.state].copy()

        for i in range(len(current_concepts)):
            if self.acc_by_concept is not None and np.random.random() > self.acc_by_concept[i]:
                current_concepts[i] = 1-current_concepts[i]
        str_column = ''.join([str(i) for i in current_concepts])
        return self.concepts_to_observation[str_column]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.randint(0,self.environment_nodes)
        self.steps = 0
        return self.get_observation(), {}

    def step(self, action):
        reward = self.rewards[self.state][action]
        next_state_probs = self.transitions[self.state][action]
        self.state = np.random.choice(self.all_states, p=next_state_probs)        
        obs = self.get_observation()
        self.steps += 1
        done = self.steps >= 20  # or any number of steps
        return obs, reward, done, False, {}

    def render(self):
        states = [chr(65+i) for i in range(self.environment_nodes)]
        print("Current state {}, observation {}".format(states[self.state],self.get_observation()))

    def close(self):
        pass


class TreeRepeatEnv(gym.Env):
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


    def __init__(self,environment_nodes,concept_list=[],acc_by_concept=None):
        super().__init__()
        self.acc_by_concept = acc_by_concept
        self.all_states = list(range(environment_nodes))
        self.action_space = spaces.Discrete(2)
        self.concept_list = concept_list
        self.observation_space = spaces.MultiBinary(len(concept_list))
        self.environment_nodes = environment_nodes
        num_layers = int(np.log2(environment_nodes+1))

        self.rewards = np.zeros((environment_nodes,self.action_space.n))
        for i in range(num_layers):
            self.rewards[2**i-1][0] = 1
        self.rewards[:,1] = 0.1

        self.transitions = np.zeros((len(self.all_states),
                                    self.action_space.n,
                                    len(self.all_states)))
        for state in range(len(self.transitions)):
            for action in range(len(self.transitions[state])):
                if state >= environment_nodes//2:
                    if state == environment_nodes//2:
                        self.transitions[state][action][0] = 1
                    else:
                        self.transitions[state][action][2] = 1
                else:
                    self.transitions[state][action][2 * (state+1) + action - 1] = 1

        self.concepts = []
        for i in range(num_layers):
            curr_concept = []
            for state in range(1,environment_nodes+1):
                binary_rep = bin(state)[2:]
                binary_rep = '0'*(num_layers-len(binary_rep)) + binary_rep
                curr_concept.append(int(binary_rep[i]))
            self.concepts.append(curr_concept)
        final_concept = [0 for i in range(2**num_layers-1)]
        for i in range(num_layers):
            final_concept[2**i-1] = 1
        self.concepts.append(final_concept)
        self.concepts = np.array(self.concepts)
        self.steps = 0

    def get_observation(self):
        current_concepts = self.concepts[self.concept_list,self.state].copy()

        for i in range(len(current_concepts)):
            if self.acc_by_concept is not None and np.random.random() > self.acc_by_concept[i]:
                current_concepts[i] = 1-current_concepts[i]         
        return current_concepts 

    def reset(self, seed=None,options=None):
        super().reset(seed=seed)
        self.state = 0
        self.steps = 0
        return self.get_observation(), {}

    def step(self, action): 
        reward = self.rewards[self.state][action]
        next_state_probs = self.transitions[self.state][action]
        self.state = np.random.choice(self.all_states,p=next_state_probs)

        obs = self.get_observation()
        self.steps += 1
        done = self.steps >= 20  

        return obs, reward, done, False, {}

    def render(self):
        print(f"State: {self.state}, Obs: {self.get_observation()}")

    def close(self):
        pass
