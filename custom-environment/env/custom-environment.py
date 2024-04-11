import functools
import random
from copy import copy
import pygame
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import ParallelEnv


class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "evolution_v0",
    }

    def __init__(self):
        self.x = None
        self.y = None
        self.agents_algos = []
        self.speed = None
        self.strength = None
        

    def reset(self, seed=None, options=None):
        pass

    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return MultiDiscrete([20*20])

    def action_space(self, agent):
        return Discrete(4) #up, down ,left ,right