import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.special import softmax
import numpy as np

from collections import deque, namedtuple

# The transition that will be stored in the replay buffer
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "loss"))

# The replay buffer to sample from
class ReplayBuffer:
    def __init__(self, capacity):
        '''Initialises the replay buffer with given capacity. A transition memory buffer and a loss memory buffer are created.

        Args:
            - capacity - int: the maximum capacity of the replay buffer
        '''
        
        self.transition_memory = deque([], maxlen=capacity)
        self.loss_memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        '''Given the state, action, reward, next_state, loss (in that order), the queues are updated
        '''
        self.memory.append(Transition(*args[:-1]))
        self.loss_memory.append(args[-1])
        
    def sample(self, batch_size, experience=True):
        '''Sample batch_size number of transitions from the replay buffer
        
        Args:
            - batch_size - int: the size of the sampled batch required.
            - experience - bool: whether loss needs to be used for sampling. Default: True
            
        Returns:
            - (dict): a batch of transitions sampled according to the experience input. It contains the sampled states, actions and rewards in tensor form. It also has a non_final_mask which tells which of the sampled transitions have non terminal next states. Accordingly, all the non terminal next states are given in order.
        '''
        
        # first create a probability distribution using the loss memory. None in the case when experience is False
        if experience:
            probs = softmax(self.loss_memory)
        else:
            probs = None
        # then usng this probability, sample the indices from the transition memory
        batch_indices = np.random.choice(range(len(self.transition_memory)), size=batch_size, replace=False, p=probs)
        # create the batch using the indices
        batch = [self.transition_memory[i] for i in batch_indices]
        # the issue is that the input to the network should be a batch of states, etc. Currently, we have batch of transitions, so we convert it to transitions of batches
        batch = Transition(*zip(*batch))
        
        # tensors are obtained for the transition elements
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # mask which indicated non terminal next state is obtained
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            dtype=torch.bool,
        )
        # non terminal next states are obtained in order
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        
        return {
            "state_batch": state_batch,
            "action_batch": action_batch,
            "reward_batch": reward_batch,
            "non_final_next_states": non_final_next_states,
            "non_final_mask": non_final_mask
        }
    
    def __len__(self):
        '''Gives the length of the transition memory buffer
        
        Returns:
            - (int): the length of the transition memory
        '''
        return len(self.transition_memory)
        
        
        