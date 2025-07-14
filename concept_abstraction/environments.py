import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Cyclic4StateEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self,concept_list=[],error=0):
        super().__init__()
        
        if len(concept_list) == 0:
            self.observation_space = spaces.Discrete(4)  # States: 0 (A), 1 (B), 2 (C), 3 (D)
        else:
            self.observation_space = spaces.MultiBinary(len(concept_list))
        self.action_space = spaces.Discrete(3)       # 0 = left, 1 = right
        self.concept_list = concept_list
        self.error = error 

        self.state = 0  # Start at A
        self.all_states = [0,1,2,3]
        self.rewards = [2.5, 2.85, 2.5, 2.85]
        
        self.concepts = np.array([[0,1,0,1],[1,0,0,1],[0,0,0,1]])

    def get_observation(self):

        state = self.state 
        error = self.error 

        if np.random.random() < error:
            state = np.random.randint(0,len(self.rewards))

        if len(self.concept_list) == 0:
            return state  
        
        current_concepts = self.concepts[self.concept_list,state]
        return current_concepts 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return self.get_observation(), {}

    def step(self, action):
        reward = self.rewards[self.state]
        if action == 0:  # left
            self.state = (self.state - 1) % 4
        elif action == 1:  # right
            self.state = (self.state + 1) % 4
        elif action == 2:
            self.state = self.state 
        else:
            raise ValueError("Invalid action")

        done = False  # Infinite loop unless defined otherwise
        info = {}
        return self.get_observation(), reward, done, False, info

    def render(self):
        states = ['A', 'B', 'C', 'D']
        print("Current state {}, observation {}".format(states[self.state],self.get_observation()))

    def close(self):
        pass


class TreeRepeatEnv(gym.Env):
    def __init__(self,concept_list=[],error=0):
        super().__init__()
        self.max_state = 15
        self.state = 1
        self.error = error
        self.all_states = list(range(1,15))

        # Actions: 0 = up (left), 1 = down (right)
        self.action_space = spaces.Discrete(2)

        self.concept_list = concept_list
        # 4-bit binary + is_power_of_2
        self.observation_space = spaces.MultiBinary(len(concept_list))

        # Rewards: only for some leaves
        self.rewards = [0 for i in range(15)]
        self.rewards[0] = self.rewards[1] = self.rewards[3] = self.rewards[7] = 1

        self.concepts = []
        for i in range(4):
            curr_concept = []
            for state in range(1,16):
                binary_rep = bin(state)[2:]
                binary_rep = '0'*(4-len(binary_rep)) + binary_rep
                curr_concept.append(int(binary_rep[i]))
            self.concepts.append(curr_concept)

        final_concept = [0 for i in range(15)]
        final_concept[0] = final_concept[1] = final_concept[3] = final_concept[7] = 1
        self.concepts.append(final_concept)

        self.concepts = np.array(self.concepts)


    def get_observation(self):
        if len(self.concept_list) == 1:
            error = self.error 
        else:
            error = 0

        state = self.state 
        if np.random.random() < error:
            state = np.random.randint(0,15)
        
        current_concepts = self.concepts[self.concept_list,state]
        return current_concepts 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return self.get_observation(), {}

    def step(self, action):
        # Determine next state based on tree transition

        if action == 0:
            reward = self.rewards[self.state]
        elif action == 1:
            reward = 0.1

        if self.state > 6:
            if self.state == 7:
                self.state = 0
            else:
                self.state = 2
        else:
            self.state = 2 * (self.state+1) + action - 1

        obs = self.get_observation()

        return obs, reward, False, False, {}

    def render(self):
        print(f"State: {self.state}, Obs: {self.get_observation()}")

    def close(self):
        pass
