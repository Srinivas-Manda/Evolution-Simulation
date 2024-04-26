import functools
import random
from copy import copy
import pygame
import math
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box
from Models.ActorCritic import ActorCritic
from Models.SoftActorCritic import SoftActorCritic
from Models.DDQN import DoubleDQN
from pettingzoo import ParallelEnv
import json
from time import sleep

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

ac_config = open('Environment/ac_config.json')
ac_variables = json.load(ac_config)
ac_config.close

sac_config = open('Environment/sac_config.json')
sac_variables = json.load(sac_config)
sac_config.close

ddqn_config = open('Environment/ddqn_config.json')
ddqn_variables = json.load(ddqn_config)
ddqn_config.close

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
        self.stamina = None
        self.id = id
        self.name = name
        self.reward = 0
        self.active = True
        self.brain1 = None
        self.set_brain_1(name)

    def set_brain_1(self, agent_type):

        if(agent_type == "AC"):
            self.brain1 = ActorCritic(ac_variables)
        
        elif(agent_type == "SAC"):
            self.brain1 = SoftActorCritic(sac_variables)

        elif(agent_type == "DDQN"):
            self.brain1 = DoubleDQN(ddqn_variables)

        else:
            raise(f"{agent_type} is invalid")

class CustomEnvironment(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """ 

    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, env_config, render_mode):
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

        """
        #THESE ATTRIBUTES SHOULD NOT CHANGE AFTER INTIALIZATION
        #temporary agents and pellets list to store names, ids and their respective attributes
        self.possible_agents_objects = {id : Agent(agent_name,id) for id,agent_name in enumerate(env_config["agents"])} #changed this to dict
        self.possible_agents = [i for i in range(len(self.possible_agents_objects))]
        self.possible_pellets = [Pellet(id) for id in range(env_config["num_pellets"])]
        self.n_agents = env_config["num_agents"]
        self.num_actions = env_config["num_actions"]
        self.justdie = None

        for id,a in self.possible_agents_objects.items():
            a.strength = env_config["default_strength"]
            a.vision_size = env_config["default_vision"]
            a.movement_speed = env_config["default_movement_speed"]

        #initial number of pellets and agent's starting stamina  
        self.num_pellets = env_config["num_pellets"]
        self.agents_starting_stamina = env_config['stamina']

        #grid sizes
        self.grid_size_x = env_config['grid_size_x'] # [0,x)
        self.grid_size_y = env_config['grid_size_y'] #[0,y)
        self.render_mode = render_mode
        

        #screen parameters for rendering
        self.screen_width = env_config['screen_width']
        self.screen_height = env_config['screen_height']
    
        # self.observation_spaces = None
        # self.action_spaces = None

        #rewards and penalties
        self.pellet_stamina_gain = env_config['pellet_stamina_gain']
        self.pellet_collect_reward = env_config['pellet_collect_reward']

        #max values of agent attributes
        self.move_penalty = env_config['move_penalty']
        self.move_stamina_loss = env_config['move_stamina_loss']
        self.max_vision_size = env_config['max_vision']
        # self.reset()

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
        # super().reset(seed=seed) #set seed if necessary

        self.agents = copy(self.possible_agents) #agent id list
        self.agents_objects = copy(self.possible_agents_objects) #agent object list
        self.pellets = copy(self.possible_pellets)
        for id,a in self.agents_objects.items():
            a.stamina = self.agents_starting_stamina
            a.active = True

        self.justdie = {}
        self.init_pos() # initialise the positions of all agents and all the pellets
        # print(self.agents_objects)
        # for a in self.agents_objects:
        #     print(a.stamina)

        self.timestep = 0

        #get the observation space of all of the agents using the make_observation_space function
        self.observation_spaces = {
          id : self.make_observation_space(a) for id,a in self.agents_objects.items()
        }
        
        #havent thought up of necessary infos
        self.infos = {
            id : {} for id,a in self.agents_objects.items()
        }

        if(self.render_mode == "human"):
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("THE GAME")
            self.agent_size = 5
            self.pellet_size = 2

        #DEBUG
        # print(self.move_stamina_loss)
        # for a in self.agents_objects:
        #     print(a.active)

        return self.observation_spaces, self.infos
        
    def step(self, actions):
        # print(self.timestep)
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
        # Execute actions
        # 1. Move all agents to new positions
        # 2. Check if any pellets were consumed, set them as inactive 
        # 3. Update observations for agents
        # 4. Assign rewards
        terminations = {a : False for a in self.agents}
        rewards = {a : 0 for a in self.agents}
        truncations = {a : False for a in self.agents}
        observations = {a : None for a in self.agents}
        infos = None
        
        # allDead = True
        # for agent in self.agents_objects:
        #     if agent.active == True:
        #         allDead = False
        #         break
        
        # if(allDead):
        #     rewards = {a.id : 0 for a in self.agents_objects}
        #     terminations = {a.id : True for a in self.agents_objects}
        #     self.agents_objects = []
        #     self.agents = []
        #     self.pellets = []
        #     return observations, rewards, terminations, truncations, infos
        to_delete = []
        for id,agent in self.agents_objects.items():
            # print(str(agent.id) + ": " + str(agent.stamina))
            #move agent
            #skip dead agents
            # if(agent.active == False): 
            #     terminations[agent.id] = True
            #     rewards[agent.id] = 0
            #     continue
            
            action = actions[id]
            
            #YAAD SE APPLY WORLD LIMIT
            move_x = np.cos(np.deg2rad(action))
            move_y = np.sin(np.deg2rad(action))
            move_x = np.sqrt(agent.movement_speed) * move_x
            move_y = np.sqrt(agent.movement_speed) * move_y
            agent.pos_x += move_x
            agent.pos_y += move_y

            #apply world limits here
            # X limits
            if(agent.pos_x < 0):
                agent.pos_x = 0
            elif(agent.pos_x >= self.grid_size_x):
                agent.pos_x = self.grid_size_x
            # Y limits
            if(agent.pos_y < 0):
                agent.pos_y = 0
            elif(agent.pos_y >= self.grid_size_y):
                agent.pos_y = self.grid_size_y
            

            #check for pellet consumption and assign reward
            flagPelletConsumed = False
            for pellet in self.pellets:
                if(self.get_entity_collision(agent, pellet)):
                    print("agent" + str(id) + " consumed pellet" + str(pellet.pellet_id))
                    for i in range(len(self.pellets)):
                        if(self.pellets[i].pellet_id == pellet.pellet_id):
                            self.pellets.pop(i)
                            break
                    flagPelletConsumed = True
                    break
            
            if(flagPelletConsumed == False):
                agent.stamina -= self.move_stamina_loss
                agent.reward -= self.move_penalty
                rewards[id] = self.move_penalty
            else:
                agent.stamina += self.pellet_stamina_gain
                agent.reward += self.pellet_collect_reward
                rewards[id] = self.pellet_collect_reward
                
            #update observation of agent after moving it and before terminating it
            observations[id] = self.make_observation_space(agent)

            #set termination of agent who's stamina is 0
            if(agent.stamina == 0):
                print("agent" + str(id) + " died at time " + str(self.timestep))
                agent.active = False
                self.justdie[id] = agent
                terminations[agent.id] = True
                # del self.agents_objects[id]
                to_delete.append(id)
                for a in self.agents:
                    if(a == agent.id):
                        self.agents.remove(a)
                
        #update observation
        # observations = {a.id : self.make_observation_space(a) for a in self.agents_objects}
        for id in to_delete:
            del self.agents_objects[id]
        
        to_delete = []
        # Check truncation conditions (overwrites termination conditions)
        if(self.timestep) > 500:
            rewards = {id : 0 for id,a in self.agents_objects.items()}
            truncations = {id : True for id,a in self.agents_objects.items()}
            self.agents_objects = []
            self.agents = []
            self.pellets = []
        self.timestep += 1

        # Get dummy infos (not used in this example)
        infos = {id : {} for id,a in self.agents_objects.items()}

        # self.render()
        if(self.render_mode == "human"):
            self.render()

        return observations, rewards, terminations, truncations, infos
    
    # top is 0, right is 1, bottom is 2, left is 3. clockwise
    # self.np_random is defined in ParalleEnv. it is used to seed the randomnes
    def init_pos(self):

        edge = np.random.randint(low = 0, high = 4, size = len(self.agents_objects)) # to pick an edge to initialise the agent on
        pos_x = np.random.choice(a = list(range(self.grid_size_x)), size = len(self.agents_objects), replace= False) # then to pick an x value for all agents
        pos_y = np.random.choice(a = list(range(self.grid_size_y)), size = len(self.agents_objects), replace= False) # then to pick an y value for all agents

        for i,id in enumerate(self.agents_objects): # iterating over all agents
            if(edge[i]==0 or edge[i] ==2): # if top or bottom edge
                x = pos_x[i]
                y = self.grid_size_y-1 if edge[i]==2 else 0
                
            elif(edge[i]==1 or edge[i] ==3): # if left or right edge
                y = pos_y[i]
                x = self.grid_size_x-1 if edge[i]==1 else 0
            
            self.agents_objects[id].update_pos(x,y) # setting the position of agent

        points = set() # creating a set for different pellets

        while len(points) < 10*self.num_pellets: # need 10 times necessary possible positions for the pellets for k-means++ sampling
            point_x = np.random.rand(1) * self.grid_size_x
            point_y = np.random.rand(1) * self.grid_size_y
            points.add((point_x[0],point_y[0]))
        
        points = [list(point) for point in points]
        points = np.array(points)
        
        #k-means++ start
        init_pellet = np.random.choice(range(len(points)))

        temp_pellets = np.zeros((self.num_pellets, 2))
        temp_pellets[0] = points[init_pellet]
        
        for i in range(1, self.num_pellets):
            distances = np.sqrt(((points - temp_pellets[:i, np.newaxis])**2).sum(axis=2)).min(axis=0)
            probs = distances ** 2
            probs /= probs.sum(axis=0)
            temp_pellets[i] = points[np.random.choice(points.shape[0], p=probs)]

        #k-means++ end

        #pellet positions are updated
        for i, pellet in enumerate(self.pellets):
            pellet.update_pos(temp_pellets[i, 0], temp_pellets[i, 1])

    #check for hit of an agent with any entity
    #if possible make changes by steps or line
    def get_entity_collision(self, agent, entity):
        #create a box centered on the agent of unit length. this is the bounds of the agent
        #based on vision(min = 3) based on this, the agent can see upto 3 boxes around itself(square fasion mai)
        #set the observation matrix of the agent based on the information from the boxes
        dist_x = agent.pos_x - entity.pos_x
        dist_y = agent.pos_y - entity.pos_y

        # #entity not within the agent's vision
        # if(abs(dist_x) > agent.vision_size+0.5 or abs(dist_y) > agent.vision_size+0.5):
        #     return "NOT_VISIBLE"

        #entity with the pellet's vision and hitting the agent
        if(abs(dist_x) <= 0.5 and abs(dist_y) <= 0.5):
            return True
        
        return False

        # #find the location in vision of the entity
        # dist_x = np.floor(abs(dist_x - 3.5))
        # dist_y = np.floor(abs(dist_y - 3.5))
        # return dist_x + agent.vision_size * dist_y

    #observation space generation given an agent
    def make_observation_space(self,agent):
        temp = Box(low = -1, high = -1, shape=(3, self.max_vision_size*2+1, self.max_vision_size*2+1), dtype=np.int32)
        box = temp.sample()
        agent_pos = {}
        pellet_pos = []

        for id,temp_agent in self.agents_objects.items():
            if(id != agent.id):
                agent_pos[id] = [math.floor(temp_agent.pos_x), math.floor(temp_agent.pos_y), temp_agent.strength]

        for temp_pellet in self.pellets:
            pellet_pos.append([math.floor(temp_pellet.pos_x), math.floor(temp_pellet.pos_y)])

        # using val to fix for the odd or even vision size
        for i in range(-agent.vision_size, agent.vision_size + 1 ):
            for j in range(-agent.vision_size, agent.vision_size + 1):

                #recheck these two
                x = math.floor(agent.pos_x + i)
                y = math.floor(agent.pos_y + j)
                
                #initiate nothing is of interest in these
                box[0,self.max_vision_size+i,self.max_vision_size+j] = 0
                box[1,self.max_vision_size+i,self.max_vision_size+j] = 0
                box[2,self.max_vision_size+i,self.max_vision_size+j] = 0
                
                for tup in agent_pos.items():
                    id,pos = tup
                    # if inside the vision size, then set the strength of the other agent as the value 
                    if [x,y] == pos:
                        box[0,self.max_vision_size+i,self.max_vision_size+j] = self.agents_objects[id].strength

                #inside the vision size and grid and is a pellet 
                if [x,y] in pellet_pos:
                    box[1,self.max_vision_size+i,self.max_vision_size+j] = 1

                #inside the vision size but outside the grid
                if  x < 0 or x >= self.grid_size_x or y < 0 or y >= self.grid_size_y:
                    box[2,self.max_vision_size+i,self.max_vision_size+j] = 1

        return box

    def render(self):
        agent_pos = [[] for a in self.agents]
        i = 0
        for id,agent in self.agents_objects.items():
            agent_pos[i] = [agent.pos_x, agent.pos_y]
            i += 1

        # Generate two food positions
        pellet_pos = [[] for a in self.pellets]
        i = 0
        for pellet in self.pellets:
            pellet_pos[i] = [pellet.pos_x, pellet.pos_y]
            i += 1

        self.screen.fill(WHITE)
        
        # Draw the player circle (relative to coordinate space position)
        for pos in agent_pos:
            screen_pos = [pos[0] * (self.screen_width / self.grid_size_x), pos[1] * (self.screen_height / self.grid_size_y)]
            pygame.draw.circle(self.screen, RED, screen_pos, 5)

        # Draw the food circles (relative to coordinate space position)
        for pos in pellet_pos:
            screen_pos = [pos[0] * (self.screen_width / self.grid_size_x), pos[1] * (self.screen_height / self.grid_size_y)]
            pygame.draw.circle(self.screen, GREEN, screen_pos, 2)

        # Update the display
        pygame.display.flip()

        sleep(0.1)


    # Observation space should be defined here.
    @functools.lru_cache(maxsize=None)
    def observation_space(self, id):
        return self.observation_spaces[id]

    # Action space should be defined here.
    @functools.lru_cache(maxsize=None)
    def action_space(self, id):
        return Discrete(360)
        
    # closes the rendering window
    def close(self):
        # self.env.close()
        pygame.quit()

    #returns the state
    def state(self) -> np.ndarray:
        return super().state()
