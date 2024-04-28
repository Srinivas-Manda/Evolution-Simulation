import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.special import softmax
import numpy as np

from collections import deque, namedtuple
from tqdm import tqdm

# The transition that will be stored in the replay buffer
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done", "log_probability"))

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
        '''Given the state, action, reward, next_state, done, loss (in that order), the queues are updated
        '''
        trans = Transition(*args[:-1])
        # print(trans.state[0].shape)
        # print(trans.state[1].shape)
        # print(trans.action.shape)
        # print(trans.reward.shape)
        # print(trans.next_state[0].shape)
        # print(trans.next_state[1].shape)
        # print(trans.done.shape)
        
        self.transition_memory.append(trans)
        self.loss_memory.append(args[-1])
        
    def sample(self, batch_size, experience=True):
        '''Sample batch_size number of transitions from the replay buffer
        
        Args:
            - batch_size - int: the size of the sampled batch required.
            - experience - bool: whether loss needs to be used for sampling. Default: True
            
        Returns:
            - (dict): a batch of transitions sampled according to the experience input. It contains the sampled states, actions and rewards in tensor form. It also has a non_final_mask which tells which of the sampled transitions have non terminal next states. Accordingly, all the non terminal next states are given in order.
        '''
        # if number of samples stored are less than batch_size, skip this function
        if len(self.transition_memory) < batch_size:
            return None
        # first create a probability distribution using the loss memory. None in the case when experience is False
        if experience:
            probs = softmax(np.array(self.loss_memory).squeeze())
        else:
            probs = None
            
        # then usng this probability, sample the indices from the transition memory
        batch_indices = np.random.choice(range(len(self.transition_memory)), size=batch_size, replace=False, p=probs)
        # create the batch using the indices
        batch = [self.transition_memory[i] for i in batch_indices]
        # the issue is that the input to the network should be a batch of states, etc. Currently, we have batch of transitions, so we convert it to transitions of batches
        batch = Transition(*zip(*batch))
        # curr_obs, curr_stam = zip(*batch.state)
        
        # tensors are obtained for the transition elements
        # print(batch.action.shape)
        # print(batch.state[0].shape)
        # print(batch.state[1].shape)
        state_batch = []
        # print(zip(*batch.state))
        for i in zip(*batch.state):
            # print(i)
            state_batch.append(torch.cat(i))
        # print(state_batch[0].shape)
        state_batch = tuple(state_batch)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # done_batch = torch.cat(batch.done)
        log_prob_batch = torch.cat(batch.log_probability)
        # mask which indicated non terminal next state is obtained
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not True, batch.done)),
            dtype=torch.bool,
        )
        
        next_states = []
        # print(zip(*batch.state))
        for i in zip(*batch.next_state):
            # print(i)
            next_states.append(torch.cat(i))
        # print(next_states[0].shape)
        next_states = tuple(next_states)
        # # next states are obtained in order
        # next_states = (torch.cat([s[0] if s is not None else torch.zeros_like(state_batch[0]).unsqueeze(0) for s in batch.next_state]), torch.cat([s[1] if s is not None else torch.zeros_like(state_batch[0]).unsqueeze(0) for s in batch.next_state]))
        
        
        return {
            "states": state_batch,
            "actions": action_batch,
            "rewards": reward_batch,
            "log_probabilities": log_prob_batch,
            "next_states": next_states,
            "non_final_mask": non_final_mask
        }
    
    def __len__(self):
        '''Gives the length of the transition memory buffer
        
        Returns:
            - (int): the length of the transition memory
        '''
        return len(self.transition_memory)
        
        
if __name__ == "__main__":
    rb = ReplayBuffer(capacity=16)
    batch = None
    
    for i in tqdm(range(64)):
        state = torch.ones((3, 10, 10))
        action = 1
        log_prob = torch.tensor([-1])
        loss = np.random.rand(1)
        next_state = torch.ones((3, 10, 10))
        reward = torch.tensor([1])
        
        rb.push(state.unsqueeze(0), torch.tensor([action]), reward.unsqueeze(0), next_state.unsqueeze(0), log_prob.unsqueeze(0), loss)
        
        batch = rb.sample(4)
        
    print(batch['states'].shape)
    print(batch["states"].shape)
    print(batch["actions"].shape)
    print(batch["rewards"].shape)
    print(batch["log_probabilities"].shape)
    print(batch["next_states"].shape)
    print(batch["non_final_mask"].shape)