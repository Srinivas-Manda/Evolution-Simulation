import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

# import gymnasium
from tqdm import tqdm
import numpy as np
from collections import deque

if __name__ == '__main__':
    from RLAgent import RLAgent
    from BasicNetworks import create_convolutional_network, create_linear_network
else:
    from Models.RLAgent import RLAgent
    from Models.BasicNetworks import create_convolutional_network, create_linear_network

# Critic Networks
class QNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.feature_extractor = create_convolutional_network(input_channels=config['in_channels'], output_channels=config['out_channels'], hidden_channels=config['hidden_channels'])
        
        self.mlp_block = create_linear_network(input_dim=config['out_channels'], output_dim=config['num_actions'], hidden_dims=config['hidden_dims'])
        
    def forward(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        
        x = self.feature_extractor(state)

        x = self.mlp_block(x.flatten(-3))
        
        return x
        
# Actor Network
class GaussianPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.log_std_min = config['log_std_min']
        self.log_std_max = config['log_std_max']
        self.eps = config['epsilon']
        
        self.policy_conv = create_convolutional_network(input_channels=config['in_channels'], output_channels=config['out_channels'], hidden_channels=config['hidden_channels'])
        
        # first half of output is mean, the second is log_std
        self.policy_lin = create_linear_network(input_dim=config['out_channels'], output_dim=config['num_actions']*2, hidden_dims=config['hidden_dims'])
        
        
    # returns the mean and log_std of the gaussian
    def forward(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        
        x = self.policy_conv(state)

        x = self.policy_lin(x.flatten(-3))
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        # get the mean and log_std
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal_dist = Normal(mean, std)
        
        # sample actions using re-parametrisation trick
        xs = normal_dist.rsample()
        actions = F.gumbel_softmax(xs)
        
        # calculate entropies
        log_probs = normal_dist.log_prob(xs) - torch.log(1 - actions + self.eps)
        entropies = -log_probs.sum(dim=-1, keepdim=True)
        
        return actions, entropies, F.gumbel_softmax(mean)
        
                

class SoftActorCritic(RLAgent):
    def __init__(self, config):
        super().__init__(config)
        
        # actor
        self.actor_policy_net = GaussianPolicy(config=config)
        # critics
        self.critic_q_net = QNetwork(config=config)
        self.critic_q_target_net = QNetwork(config=config)
        
        # optimisers
        self.policy_step_size = config["actor_step_size"]
        self.q_step_size = config["critic_step_size"]
        
        self.policy_optim = optim.Adam(self.actor_policy_net.parameters(), lr=self.policy_step_size)
        self.q_optim = optim.Adam(self.critic_q_net.parameters(), lr=self.q_step_size)
        self.q_target_optim = optim.Adam(self.critic_q_target_net.parameters(), lr=self.q_step_size)

        # entropy related stuff
        # target entropy is -1 * # actions i.e. complete randomness
        self.target_entropy = -config['num_actions']
        # optimise log(alpha)
        self.log_alpha = torch.zeros(
            1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.policy_step_size)

    def select_action(self, state):
        # make sure that the input is in the batch format
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
            
        # the exploration probability distribution, entropies and the mean distribution
        probs, _, _ = self.actor_policy_net.sample(state)
        
        return torch.argmax(probs, dim=-1), torch.max(probs, dim=-1)
    
    def calc_current_q(self, states):
        curr_q1 = self.critic_q_net(states)
        return curr_q1

    def calc_target_q(self, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_entropies, _ = self.actor_policy_net.sample(next_states)
            
            print(next_entropies.shape)
            # get the indices of the selected actions
            next_action_ind = torch.tensor([[False for i in range(self.num_actions)] for i in range(next_states.shape[0])])
            for i, inds in enumerate(next_action_ind):
                print(next_actions.shape)
                inds[next_actions[:, i]] = True
                next_action_ind[i] = inds
                
            print(next_action_ind.dtype)
            print(next_action_ind.shape)
                
            next_q = self.critic_q_target_net(next_states)[next_action_ind]
            next_q += self.alpha * next_entropies

        target_q = rewards + (1.0 - dones) * self.gamma_n * next_q

        return target_q
    
    def push_to_buffer(self, *args):
        '''Given the state, action, reward, next_state, log probability, done (in that order), the buffer is updated after calculating the loss
        '''
        state, action, reward, next_state, log_prob, done = args
        
        current_q = self.calc_current_q(state)
        print(current_q.shape)
        target_q = self.calc_target_q(rewards=reward, next_states=next_state, dones=done)
        
        print(target_q.shape)
        
        
if __name__ == '__main__':
    config = {
        "in_channels": 3,
        "out_channels": 32,
        "hidden_channels": [8, 16],
        "hidden_dims": [32],
        "num_actions": 360,
        "actor_step_size": 1e-6,
        "critic_step_size": 1e-3,
        "batch_size": 1,
        "discount_factor": 0.9,
        "capacity": 1,
        "device": 'cpu',
        "log_std_min": -2,
        "log_std_max": 20,
        "epsilon": 1e-6
    }
    
    sac_agent = SoftActorCritic(config=config)
    
    state = torch.ones((3, 10, 10))
    
    action, log_prob = sac_agent.select_action(state=state)
    
    next_state = torch.ones_like(state)
    reward = torch.tensor([1])
    done = False if np.random.rand(1)[0] <= 0.95 else True
    
    next_actions, entropies, det_next_actions = sac_agent.actor_policy_net.sample(state=state)
    
    print(next_actions.shape)
    print(entropies.shape)
    print(det_next_actions.shape)
    
    # sac_agent.push_to_buffer(state, action, reward, next_state, log_prob, done)
    
    
    

        
        
        
        
        
    