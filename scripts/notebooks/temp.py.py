import numpy as np
import minigrid 
import gymnasium as gym
from minigrid.core.constants import DIR_TO_VEC
def get_all_concepts(env):
    """
    Returns high-level concepts about the environment state:
    - agent_position: tuple (x, y)
    - agent_direction: int (0: right, 1: down, 2: left, 3: up)
    - key_position: tuple (x, y) of the key
    - door_position: tuple (x, y) of the door
    - door_open: boolean value, if the door is open
    - direction_movable: dictionary with boolean values for 'right', 'down', 'left', 'up'
    """
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

    def can_move(position,direction,grid):
        next_pos = DIR_TO_VEC[direction]
        if 1 <= position[0] + next_pos[0] < grid.width-1 and 1 <= position[1] + next_pos[1] < grid.height-1:
            next_cell = grid.get(position[0] + next_pos[0], position[1] + next_pos[1])
            return next_cell is None or next_cell.can_overlap()
        else:
            return False  # Out of bounds

    # Check direction_movable in all four directions
    direction_movable = {
        'right': can_move(agent_pos, 0, grid),
        'down': can_move(agent_pos, 1, grid),
        'left': can_move(agent_pos, 2, grid),
        'up': can_move(agent_pos, 3, grid),
    }

    infos = {
        'agent_position': agent_pos,
        'agent_direction': agent_dir,
        'key_position': key_pos,
        'door_position': door_pos,
        'door_open': door_open,  # Add door_open to infos
        'direction_movable': direction_movable
    }

    numbers = []
    for key, value in infos.items():
        if key == 'direction_movable':
            for k, v in value.items():
                numbers.append(int(v))
        elif isinstance(value, tuple):
            numbers.extend([x for x in value])
        else:
            numbers.append(value)

    return np.array(numbers, dtype=np.float32)

env = gym.make("MiniGrid-DoorKey-5x5-v0")
env.reset()
print(get_all_concepts(env))