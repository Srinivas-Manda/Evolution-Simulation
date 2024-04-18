import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium
from tqdm import tqdm_notebook
import numpy as np
from collections import deque

# Policy Network / Actor
class Actor(nn.Module):
    
    def __init__(self, config) -> None:
        super(Actor, self).__init__()
        '''
        Args:
            config: dict - all important hyperparameters for the model
        '''
        
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=config['num_channels'], out_channels=8, kernel_size=1, padding=0, stride=0),
            # nn.Dropout(p=0.3),
            nn.BatchNorm2d(num_features=8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=0),
            # nn.Dropout(p=0.3),
            nn.BatchNorm2d(num_features=16),
        )
        
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp_block = nn.Sequential(
            nn.Linear(in_features=16, out_features=32),
            # nn.Dropout(p=0.3),
            nn.Linear(in_features=32, out_features=config['num_actions'])
        )
        
    def forward(self, observation):
        '''
        Args:
            observation: torch.tensor - observation space, a 3 channel image denoting - agent positions, pellet positions, illegal area
        '''
        feature_map = self.conv_block(observation)
        x = self.avg_pool(feature_map)
        probs = F.softmax(self.mlp_block(x.flatten(-3)), dim=-1)
        
        return probs
    
# Value Network / Critic
class Critic(nn.Module):
    
    def __init__(self, config):
        '''
        Args:
            config: dict - all important hyperparameters for the model
        '''
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=config['num_channels'], out_channels=8, kernel_size=1, padding=0, stride=0),
            # nn.Dropout(p=0.3),
            nn.BatchNorm2d(num_features=8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=0),
            # nn.Dropout(p=0.3),
            nn.BatchNorm2d(num_features=16),
        )
        
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp_block = nn.Sequential(
            nn.Linear(in_features=16, out_features=32),
            # nn.Dropout(p=0.3),
            nn.Linear(in_features=32, out_features=1)
        )
        
# class that combines both Actor and Critic
class ActorCritic:
    def __init__(self, actor, critic):
        '''Initialise the object with and instance of an actor and a critic
        
        Args:
            - actor - nn.Module: The Actor / Policy Network
            - critic - nn.Module: The Critic / Value Network
        '''
        self.actor = actor
        self.critic = critic
        
    def update_weights(self, state, action, reward, next_state, done):
        '''This function updates the weights of the actor and the critic network based on the given state, action, reward and next_state
        
        Args:
            - state - torch.tensor: The state of the environment given as the input
            - action - torch.tensor: The action selected using the Actor
            - reward - torch.tensor: The reward given by the environment
            - next_state - torch.tensor: The next_state given by the environment
            - done - torch.tensor: tensor which tells if the next state is terminal or not
        '''
        
        # TODO: Change this part from the ref file.
        # - Add replay buffer object to ActorCritic
        # - Use batches for weight update according to the ref code
        # - Create RL Agent superclass and move buffer and action selection related tasks there
        
        
        # state value as calculated by the critic
        state_val = self.critic(state)
        
        # next state value as calculated by the critic
        new_state_value = self.critic(next_state)
        
        
        
        
        
        
def select_action(actor, state):
    ''' selects an action based on the probabilities given by the actor
    Args:
        - actor - nn.Module: Policy Network / Actor
        - state - torch.tensor: State provided by the environment
        
    Returns:
        - (int): selected action
        - (float): log probability of selecting the action given the state 
    '''
    
    # get the probability distribution for the actions
    action_probs = actor(state)
    state = state.detach()
    
    # select an action based on the predicted probability
    m = Categorical(action_probs)
    action = m.sample()
    
    return action.item(), m.log_prob(action)
