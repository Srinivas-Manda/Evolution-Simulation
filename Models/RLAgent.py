import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium
from tqdm import tqdm_notebook
import numpy as np
from collections import deque

from ReplayBuffer import ReplayBuffer, Transition

# parent class to all the rl agents
# implements common functionality like buffer related tasks
class RLAgent:
    
    def __init__(self, config):
        self.replay_buffer = ReplayBuffer(config['capacity'])
        self.batch_size = config["batch_size"]
        self.discount_factor = config["discount_factor"]
        self.num_actions = config['num_actions']
        self.device = config['device']
                
                
    def push_to_buffer(self, *args):
        '''Given the state, action, reward, next_state, log probability, done (in that order), the buffer is updated after calculating the loss
        '''
        raise NotImplementedError("push_to_buffer is not implemented")
        
    def select_action(self, state):
        raise NotImplementedError("Select action function not implemented")
        
    def sample_from_buffer(self, batch_size, experience=True):
        '''Sample batch_size number of transitions from the replay buffer
        
        Args:
            - batch_size - int: the size of the sampled batch required.
            - experience - bool: whether loss needs to be used for sampling. Default: True
            
        Returns:
            - (dict): a batch of transitions sampled according to the experience input. It contains the sampled states, actions and rewards in tensor form. It also has a non_final_mask which tells which of the sampled transitions have non terminal next states. Accordingly, all the non terminal next states are given in order.
        '''
        
        batch = self.replay_buffer.sample(batch_size=batch_size, experience=experience)

        return batch