import functools
import random
from copy import copy
import pygame
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Dict, MultiBinary
from pettingzoo import ParallelEnv
from pettingzoo.utils import agent_selector
from pettingzoo.magents import wrappers


class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "evolution_v0",
    }

    def __init__(self, num_agents = 4, num_food = 3):
        self.start_x = [] # contains the start x co-ordinates of all of the agents
        self.start_y = [] # contains the start y co-ordinates of all of the agents
        self.agent_pos = None # 2d list of x,y co-ordinates
        self.num_agents = num_agents
        self.agent_algos = {agent_id: None for agent_id in range(self.num_agents)}
        self.grid_size = None # grid_size * grid_size sized grid
        self.speed = None # an agent can overpower a weaker agent if caught for food
        self.strength = None # an agent can run faster or slower 
        # randomize food spawn, decide on the number of foods to spawned after each day and use random.randit(x,y)
        self.num_food =  num_food
        self.food_spawn = None
        self.action_space = Discrete(5) # UDLRC
        self.observation_space = Dict({
            "agent_pos": MultiBinary(self.grid_size**2),
            "local_food": MultiBinary(self.agent_vision**2),
            "energy": Discrete(101), # 0 to 100
            "surrounding_agents": MultiBinary(self.agent_vision**2)
            #add more as per requirement
        })
        self.rewards = {agent_id: 0 for agent_id in range(self.num_agents)}
        self.dones = {agent_id: False for agent_id in range(self.num_agents)}
        self.reset()

    def reset(self, seed=None, options=None):
        self.grid = np.zeros((self.grid_size,self.grid_size))
        self.place_food()

        agent_positions = np.random.randint(0, self.grid_size, size=(self.num_agents, 2)) ## change here to make the agents only spawn at the edges of the grid
        self.agent_states = {
            agent_id: {"agent_pos": np.zeros_like(self.observation_space["agent_pos"].sample()),
                       "local_food": np.zeros_like(self.observation_space["local_food"].sample()),
                       "energy": 100,  # Initial energy
                       "surrounding_agents": np.zeros_like(self.observation_space["surrounding_agents"].sample())}
            for agent_id in range(self.num_agents)
        }
        for agent_id, pos in enumerate(agent_positions):
            self.agent_states[agent_id]["agent_pos"][pos[0] * self.grid_size + pos[1]] = 2 # 1 in the grid means, food whereas 2 means that an agent is present at that location
            self.update_local_food(agent_id, pos)
            self.update_surrounding_agents(agent_id, pos)

        self.dones = {agent_id: False for agent_id in range(self.num_agents)}
        self.infos = {agent_id: {} for agent_id in range(self.num_agents)}
        self.available_actions = {agent_id: self.action_space.n for agent_id in range(self.num_agents)}

        return self.observe(agent_selector.all)
    
    def place_food(self):
        num_food_placed = 0
        while num_food_placed < self.num_food:
            x, y = np.random.randint(0, self.grid_size, size=2)
            if self.grid[x, y] == 0:
                self.grid[x, y] = 1 # 1 signifies that food is placed
                num_food_placed += 1
    
    def update_local_food(self, agent_id, agent_pos):
        local_grid = self.grid[
        max(0, agent_pos[0] - self.agent_vision // 2): min(self.grid_size, agent_pos[0] + self.agent_vision // 2 + 1),
        max(0, agent_pos[1] - self.agent_vision // 2): min(self.grid_size, agent_pos[1] + self.agent_vision // 2 + 1)
        ]
        self.agent_states[agent_id]["local_food"] = local_grid.flatten()

    def step(self, action_n):
        # Discrete(5) is ordered as up, down, left, right, collect food
        agent_actions = {}
        for agent_id in range(self.num_agents):
                agent_observation = self.observe(agent_id)
                agent_actions[agent_id] = self.agent_algos[agent_id].act(agent_observation)



    def render(self):
        pass

    def observation_space(self, agent):
        return MultiDiscrete([self.grid_x*self.grid_y])

    def action_space(self, agent):
        return Discrete(5) #up, down ,left ,right, collect food