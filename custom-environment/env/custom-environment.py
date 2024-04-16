import functools
import random
from copy import copy
import pygame
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Dict, MultiBinary
from pettingzoo import ParallelEnv
from pettingzoo.utils import agent_selector
# from pettingzoo.magents import wrappers
from pettingzoo.test import parallel_api_test


class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "evolution_v0",
        "render_modes" : "human"
    }

    def __init__(self, env_config):
        self.start_x = [] # contains the start x co-ordinates of all of the agents
        self.start_y = [] # contains the start y co-ordinates of all of the agents
        self.agent_pos = None # 2d list of x,y co-ordinates
        self.num_agents = env_config.num_agents
        self.agent_algos = {agent_id: None for agent_id in range(self.num_agents)}
        self.grid_size = None # grid_size * grid_size sized grid
        self.speed = None # an agent can overpower a weaker agent if caught for food
        self.strength = None # an agent can run faster or slower 
        # randomize food spawn, decide on the number of foods to spawned after each day and use random.randit(x,y)
        self.num_food =  env_config.num_food
        self.food_spawn = None
        self.pickup = None
        self.stamina = None
        self.action_space = Discrete(360 ) # UDLRC
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


    # Resets the environment to an initial state, 
    # required before calling step. 
    # Returns the first agent observation for an episode and information, i.e. metrics, debug info.
    def reset(self, seed=None, options=None):
        if(seed):
            super().reset(seed=seed)
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
            x, y = np.random.randint(0, self.grid_size, size=2) # make sure only middle
            if self.grid[x, y] == 0:
                self.grid[x, y] = 1 # 1 signifies that food is placed
                num_food_placed += 1
    
    def update_local_food(self, agent_id, agent_pos):
        local_grid = self.grid[
        max(0, agent_pos[0] - self.agent_vision // 2): min(self.grid_size, agent_pos[0] + self.agent_vision // 2 + 1),
        max(0, agent_pos[1] - self.agent_vision // 2): min(self.grid_size, agent_pos[1] + self.agent_vision // 2 + 1)
        ]
        self.agent_states[agent_id]["local_food"] = local_grid.flatten()

    # Updates an environment with actions returning the next agent observation, 
    # the reward for taking that actions, 
    # if the environment has terminated or truncated due to the latest action and information from the environment about the step, i.e. metrics, debug info.
    # gymnasium.Env.step(self, action: ActType) → tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]
    def step(self, id, action_n):
        # Discrete(5) is ordered as up, down, left, right, collect food
        agent_actions = {}
        for agent_id in range(self.num_agents):
                agent_observation = self.observe(agent_id)
                agent_actions[agent_id] = self.agent_algos[agent_id].act(agent_observation)

        for agent_id in range(self.num_agents):
            if self.dones[agent_id]:
                continue

            agent_action = agent_actions[agent_id]
            current_pos = (self.agent_states[agent_id]["agent_pos"] == 2).nonzero()

            # write logic for handling invalid actions, movement, gathering, and energy

            # Check for agent death due to energy depletion
            if self.agent_states[agent_id]["energy"] <= 0:
                self.dones[agent_id] = True
                self.infos[agent_id]["reason"] = "die"

            # Implement sex logic - Discontinued

            # Provide the RL algorithm with reward and next observation
            next_observation = self.observe(agent_id)
            self.agent_algos[agent_id].step(agent_observation, self.rewards[agent_id], self.dones[agent_id], next_observation)

            # return self._step(action_n) 

    def render(self):
        pass
    # The Space object corresponding to valid observations, 
    # all valid observations should be contained within the space.
    def observation_space(self, agent):
        return MultiDiscrete([self.grid_x*self.grid_y])

    #The Space object corresponding to valid actions, 
    #all valid actions should be contained within the space.
    def action_space(self, agent):
        return Discrete(360) #up, down ,left ,right, collect food
    
    def close(self):
        #close pygame and clear data here
        pass

if __name__ == "__main__":
    print("Raghuram Learning : RL")
    env = CustomEnvironment()
    parallel_api_test(env, num_cycles=1_000_000)