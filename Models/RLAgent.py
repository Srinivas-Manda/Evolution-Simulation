import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium
from tqdm import tqdm_notebook
import numpy as np
from collections import deque

if __name__ == 'RLAgent':
    from ReplayBuffer import ReplayBuffer
elif __name__ == 'ActorCritic':
    from ReplayBuffer import ReplayBuffer
else:
    from Models.ReplayBuffer import ReplayBuffer
# if __name__ == '__main__':
#     from ReplayBuffer import ReplayBuffer
# else:
#     from .ReplayBuffer import ReplayBuffer

# parent class to all the rl agents
# implements common functionality like buffer related tasks
class RLAgent:
    
    def __init__(self, config):
        self.replay_buffer = ReplayBuffer.ReplayBuffer(config['capacity'])
        self.batch_size = config["batch_size"]
        self.discount_factor = config["discount_factor"]
        self.num_actions = config['num_actions']
        
        if config['device'] == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        
        if not "grad_clip" in config:
            self.grad_clip = None
        else:
            self.grad_clip = config['grad_clip']
        
        if not "tau" in config:
            self.tau = None
        else:
            self.tau = config['tau']
            if self.tau is None:
                raise("tau cannot be None")
                
                
    def push_to_buffer(self, *args):
        '''Given the state, action, reward, next_state, log probability, done (in that order), the buffer is updated after calculating the loss
        '''
        raise NotImplementedError("push_to_buffer is not implemented")
        
    def select_action(self, state):
        raise NotImplementedError("select_action is not implemented")
        
    def update_weights(self):
        raise NotImplementedError("update_weights is not implemented")    
    
    # soft target network update
    def soft_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(t.data * (1.0 - self.tau) + s.data * self.tau)
    
    
    def param_step(self, optim, network, loss, retain_graph=False):
        optim.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if self.grad_clip is not None:
            for p in network.modules():
                torch.nn.utils.clip_grad_norm_(p.parameters(), self.grad_clip)
        optim.step()
    
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