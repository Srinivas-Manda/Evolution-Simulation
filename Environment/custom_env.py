import functools
import random
from copy import copy
import pygame
import math
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box

from pettingzoo import ParallelEnv

class Entity:
    def __init__(self, type = None) -> None:
        self.pos_x = None
        self.pos_y = None
        self.type = type

    def update_pos(self, x, y):
        self.pos_x = x
        self.pos_y = y
class Pellet(Entity):
    def __init__(self, id) -> None:
        super().__init__("Pellet")
        self.pellet_id = id
class Agent(Entity):
    def __init__(self, name, id) -> None:
        super().__init__("Agent")
        self.strength = None
        self.vision_size = None
        self.movement_speed = None
        self.agent_id = id
        self.name = name
        self.reward = 0

class CustomEnvironment(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """ 

    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, env_config):
        """The init method takes in environment arguments.

        Should define the following attributes:
        - Number of agents
        - Number of pellets
        - All agents starting positions(randomised)
        - All agents positions
        - All pellets starting positions(randomised near center)
        - All agents vision range
        - All agents speeds
        - All agents stamina
        - Field Size
        - Agent Observation Space

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.agents = [Agent(agent_name,id) for id,agent_name in enumerate(env_config.agents)] #agent object list
        self.pellets = [Pellet(id) for id in range(env_config.num_pellets)]
        self.num_pellets = env_config.num_pellets

        self.grid_size_x = env_config.grid_size_x
        self.grid_size_y = env_config.grid_size_y
        self.num_agents = env_config.agents
        self.agents = []
        self.max_num_agents = env_config.max_num_agents
        self.observation_spaces = None
        self.action_spaces = None
        self.max_vision_size = env_config.max_vision_size


    # top is 0, right is 1, bottom is 2, left is 3. clockwise
    # self.np_random is defined in ParalleEnv
    def init_pos(self):

        edge = self.np_random.randint(low = 0, high = 4, size = len(self.agents))
        pos_x = self.np_random.choice(a = list(range(self.grid_size_x)), size = len(self.agents), replace= False)
        pos_y = self.np_random.choice(a = list(range(self.grid_size_y)), size = len(self.agents), replace= False)
        for i,agent in enumerate(self.agents):
            if(edge[i]==0 or edge[i] ==2):
                pos_x = pos_x[i]
                pos_y = self.grid_size_y-1 if edge[i]==2 else 0
                
            elif(edge[i]==1 or edge[i] ==3):
                pos_y = pos_y[i]
                pos_x = self.grid_size_x-1 if edge[i]==1 else 0
            
            agent.update_pos(pos_x,pos_y)

        points = set()
        while len(points) < 10*self.num_pellets: # k-means++ will handle this later
            point_x = self.np_random.rand(1) * self.grid_size_x
            point_y = self.np_random.rand(1) * self.grid_size_y
            points.add((point_x[0],point_y[0]))
        
        points = [list(point) for point in points]
        points = np.array(points)
        
        init_pellet = self.np_random.choice(range(len(points)))

        temp_pellets = np.zeros((self.num_pellets, 2))
        temp_pellets[0] = points[init_pellet]
        
        for i in range(1, self.num_pellets):
            distances = np.sqrt(((points - temp_pellets[:i, np.newaxis])**2).sum(axis=2)).min(axis=0)
            probs = distances ** 2
            probs /= probs.sum(axis=0)
            temp_pellets[i] = points[self.np_random.choice(points.shape[0], p=probs)]

        for i, pellet in enumerate(self.pellets):
            pellet.update_pos(temp_pellets[i, 0], temp_pellets[i, 1])


    #to ease the creation of observation space
    def make_observation_space(self,agent):
        temp = Box(low = -1, high = -1,shape=(3,self.max_vision_size*2,self.max_vision_size*2),dtype=np.int32)
        box = temp.sample()
        agent_pos = []
        pellet_pos = []

        for temp_agent in self.agents:
            if(temp_agent.agent_id != agent.agent_id):
                agent_pos.append(math.floor(temp_agent.pos_x), math.floor(temp_agent.pos_y))

        for temp_pellet in self.pellets:
            pellet_pos.append(math.floor(temp_pellet.pos_x), math.floor(temp_pellet.pos_y))

        #agents
        # using val to fix for the odd or even vision size
        for i in range(-agent.vision_size , agent.vision_size + 1 ):
            for j in range(-agent.vision_size , agent.vision_size + 1):

                x = math.floor(agent.pos_x)+i
                y = math.floor(agent.pos_y)+j

                #inside the vision size but outside the grid
                if  x < 0 or x > self.grid_size_x or y < 0 or y > self.grid_size_y:
                    box[0][x][y] = 0
                    box[1][x][y] = 0 
                    box[2][x][y] = 1

                #inside the vision size and grid and is another agent
                elif [x,y] in agent_pos:
                    box[0][x][y] = 1

                #inside the vision size and grid and is a pellet 
                elif [x,y] in pellet_pos:
                    box[1][x][y] = 1

                #inside the vision size and grid and is neither a pellet nor an agent meaning its empty space
                else:
                    box[0][i][j] = 0



    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - agents start x y coords
        - pellets x y coords
        - vision range
        - speed
        - agents starting stamina
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        super().reset(seed=seed)
        self.init_pos()

        self.observation_spaces = {
          a.id : self.make_observation_space(a) for a in self.agents
        }

        self.infos = {a.id : {} for a in self.agents}

        return self.observation_spaces, self.infos

    def step(self, actions):
        """Takes in an action for the current agent (specified by agent_selection).

        Needs to update:
        - prisoner x and y coordinates
        - guard x and y coordinates
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """
        oldAgents = copy(self.agents)
        # Execute actions
        # 1. Move all agents to new positions
        # 2. Check if any pellets were consumed, remove them from the array(or set their position out of bounds) 
        # 3. Update observations for agents
        # 4. Assign rewards
        terminations = {a: False for a in self.agents}
        rewards = {a : 0 for a in self.agents}
        truncations = None
        observations = None
        for agent in self.agents:
            #move agent
            action = actions[agent.name]
            
            #YAAD SE APPLY WORLD LIMIT
            #move left
            if(action == 0 and agent.pos_x > 0):
                agent.pos_x -= agent.movement_speed
            #move right
            elif(action == 2 and agent.pos_x < self.grid_size_x):
                agent.pos_x += agent.movement_speed
            #move up
            elif(action == 1 and agent.pos_y > 0):
                agent.pos_y -= agent.movement_speed
            #move down
            elif(action == 3 and agent.pos_y < self.grid_size_y):
                agent.pos_y += agent.movement_speed

            


        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        if self.prisoner_x == self.guard_x and self.prisoner_y == self.guard_y:
            rewards = {"prisoner": -1, "guard": 1}
            terminations = {a: True for a in self.agents}
            self.agents = []

        elif self.prisoner_x == self.escape_x and self.prisoner_y == self.escape_y:
            rewards = {"prisoner": 1, "guard": -1}
            terminations = {a: True for a in self.agents}
            self.agents = []

        # Check truncation conditions (overwrites termination conditions)
        truncations = {"prisoner": False, "guard": False}
        if self.timestep > 100:
            rewards = {"prisoner": 0, "guard": 0}
            truncations = {"prisoner": True, "guard": True}
            self.agents = []
        self.timestep += 1

        # Get observations
        observation = (
            self.prisoner_x + 7 * self.prisoner_y,
            self.guard_x + 7 * self.guard_y,
            self.escape_x + 7 * self.escape_y,
        )
        observations = {
            "prisoner": {
                "observation": observation,
                "action_mask": prisoner_action_mask,
            },
            "guard": {"observation": observation, "action_mask": guard_action_mask},
        }

        # Get dummy infos (not used in this example)
        infos = {"prisoner": {}, "guard": {}}

        # self.render()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Renders the environment."""
        grid = np.zeros((7, 7))
        # grid = map(str, grid)
        grid[self.prisoner_y, self.prisoner_x] = "1"
        grid[self.guard_y, self.guard_x] = "2"
        grid[self.escape_y, self.escape_x] = "9"
        print(f"{grid} \n")

    # Observation space should be defined here.
    #re-check this for if high = self.max_strength and if low = -1 
    def observation_space(self, agent_id):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # return Box(low = -1, high = self.max_strength,shape=(3,self.max_agent_vision,self.max_agent_vision),dtype=np.int32
        return self.observation_space[agent_id]

    # Action space should be defined here.
    def action_space(self, agent_id):
        # return Discrete(360)
        return self.action_space[agent_id]
        
    # closes the rendering window
    def close(self):
        return super().close()

    #returns the state
    def state(self) -> np.ndarray:
        return super().state()
