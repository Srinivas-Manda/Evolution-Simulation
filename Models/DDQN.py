import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# import gymnasium
from tqdm import tqdm
import numpy as np
from collections import deque

# from BasicNetworks.RLAgent import RLAgent
# from BasicNetworks.BasicNetworks import create_convolutional_network, create_linear_network

if __name__ == '__main__':
    from RLAgent import RLAgent
    from BasicNetworks import create_convolutional_network, create_linear_network
else:
    from Models.RLAgent import RLAgent
    from Models.BasicNetworks import create_convolutional_network, create_linear_network
    
# QNetwork
class QNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.device = config['device']
        
        self.feature_extractor = create_convolutional_network(input_channels=config['in_channels'], output_channels=config['out_channels'], hidden_channels=config['hidden_channels'])
        
        self.mlp_block = create_linear_network(input_dim=config['out_channels'], output_dim=config['num_actions'], hidden_dims=config['hidden_dims'])
        
    def forward(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
            
        state = state.to(self.device)
        
        x = self.feature_extractor(state).flatten(-3)
        
        x = self.mlp_block(x)
        
        return x

# Double DQN Class       
class DoubleDQN(RLAgent):
    def __init__(self, config):
        super().__init__(config=config)
        
        # initialise the networks
        self.policy_net = QNetwork(config=config).to(self.device)
        self.target_net = QNetwork(config=config).to(self.device)
        
        # initialise the optimiser
        self.policy_step_size = config['policy_step_size']
        self.policy_optim = optim.Adam(params=self.policy_net.parameters(), lr=self.policy_step_size)
        
        # exploration control
        self.eps_start = config['eps_start']
        self.eps_end = config['eps_end']
        self.eps_decay = config['eps_decay']
        self.steps_done = 0

    # calculate current q values
    def calc_current_q_values(self, state):
        return self.policy_net(state)
    
    def calc_target_q_value(self, reward, next_state, done):
        with torch.no_grad():
            next_q = self.target_net(next_state)
            
        if type(done) is bool:
            done = torch.Tensor([done]).to(self.device)
            
        target_q = reward + torch.logical_not(done).reshape(-1, 1) * self.discount_factor * next_q
        
        return target_q
    
    # calculate the loss
    def calc_policy_loss(self, state, action, reward, next_state, done):
        curr_q = self.calc_current_q_values(state=state)
        target_q = self.calc_target_q_value(reward=reward, next_state=next_state, done=done)
        
        if type(action) is not torch.Tensor:
            action = torch.tensor([action], dtype=torch.int).to(self.device)
        
        loss = F.mse_loss(curr_q[np.arange(curr_q.shape[0]), action.squeeze()], target_q.max(dim=-1)[0])
        
        return loss

    # gives an action and the corresponding log prob        
    def select_action(self, state):
        # calculate the threshold
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * self.steps_done/self.eps_decay)
        
        with torch.no_grad():
            if len(state.shape) == 3:
                state = state.unsqueeze(0)
                
            # get the probabilities
            action_probs = F.softmax(self.policy_net(state), dim=-1)
            
            log_probs = torch.log(action_probs)
            
        # if action is to be chosen greedily
        if np.random.rand() > eps_threshold:
            log_prob, action = torch.max(log_probs, dim=-1)
            
            return action.item(), log_prob
        # if random sampling needs to be done
        else:
            action = np.random.randint(0, self.num_actions)
            
            return action, log_probs[:, action]
        
    def push_to_buffer(self, *args):
        '''Given the state, action, reward, next_state, log probability, done (in that order), the buffer is updated after calculating the loss
        '''
        state, action, reward, next_state, log_prob, done = args
        
        state, action, reward, next_state, done = state.unsqueeze(0), torch.tensor([action]).unsqueeze(0), torch.tensor([reward]).unsqueeze(0), next_state.unsqueeze(0), torch.tensor([done]).unsqueeze(0)
        
        with torch.no_grad():
            # curr_q = self.calc_current_q_values(state=state)
            # target_q = self.calc_target_q_value(reward=reward, next_state=next_state, done=done)
            
            # loss = F.mse_loss(curr_q[0, action], target_q.max())
            loss = self.calc_policy_loss(state.to(self.device), action.to(self.device), reward.to(self.device), next_state.to(self.device), done.to(self.device))
            
        
        # print(state.shape)
        # print(action)
        # print(reward)
        # print(next_state.shape)
        # print(done)
        # print(log_prob)
        # print(loss)
        
        self.replay_buffer.push(state, action, reward, next_state, done, log_prob, loss)
        
    def update_weights(self):
        '''This function updates the weights of the actor and the critic network based on the given state, action, reward and next_state
        '''
        
        '''
        Batch:
            - states - torch.tensor: The state of the environment given as the input
            - actions - torch.tensor: The action selected using the Actor
            - rewards - torch.tensor: The reward given by the environment
            - log_probabilities - torch.tensor: The log probabilities as calculated during action selection
            - next_states - torch.tensor: The next_state given by the environment
            - non_final_mask - torch.tensor: tensor of the same size as the next_states which tells if the next state is terminal or not
        '''
        
        batch = self.replay_buffer.sample(self.batch_size, experience=False)
        
        # when buffer is not filled enough
        if batch is None:
            return
        
        batch['states'] = batch['states'].to(self.device)
        batch['actions'] = batch['actions'].to(self.device)
        batch['rewards'] = batch['rewards'].to(self.device)
        batch['log_probabilities'] = batch['log_probabilities'].to(self.device)
        batch['non_final_mask'] = batch['non_final_mask'].to(self.device)
        
        # calculate policy loss and update policy weights
        policy_loss = self.calc_policy_loss(batch['states'], batch['actions'], batch['rewards'], batch['next_states'], torch.logical_not(batch['non_final_mask']))
        self.param_step(optim=self.policy_optim, network=self.policy_net, loss=policy_loss)
        
        # soft updates for the target net
        self.soft_update(target=self.target_net, source=self.policy_net)
        
        # update number of steps done
        self.steps_done += 1

        return policy_loss.detach().cpu()
        
if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    config = {
        "in_channels": 3,
        "out_channels": 32,
        "hidden_channels": [8, 16],
        "hidden_dims": [32],
        "num_actions": 360,
        "policy_step_size": 1e-3,
        "batch_size": 4,
        "discount_factor": 0.9,
        "capacity": 8,
        "device": 'cpu',
        "log_std_min": -2,
        "log_std_max": 20,
        "eps_start": 0.5,
        "eps_end": 0.05,
        "eps_decay": 1000,
        "tau": 1e-3
    }
    
    ddqn_agent = DoubleDQN(config=config)
    
    state = torch.ones((3, 10, 10))
    
    for i in tqdm(range(20)):
        # will be taken by agent
        action, log_prob = ddqn_agent.select_action(state=state)
        
        # will be available from the environment
        next_state = torch.ones_like(state)
        reward = 1
        done = False if np.random.rand(1)[0] <= 0.999 else True
        
        ddqn_agent.push_to_buffer(state, action, reward, next_state, log_prob, done)
        
        
        ret = ddqn_agent.update_weights()
        # break
        # if ret is not None:
        #     losses.append(ret)
        
        state = next_state
        
        if done:
            break
    
    # action, log_prob = ddqn_agent.select_action(state=state)
    
    # print(action)
    # print(log_prob)
    
    # next_state = torch.ones_like(state)
    # reward = torch.tensor([1])
    # done = False if np.random.rand(1)[0] <= 0.95 else True
    
    # done = False
    # next_actions, entropies, det_next_actions = ddqn_agent.actor_policy_net.sample(state=state)
    
    # print(next_actions.shape)
    # print(entropies.shape)
    # print(det_next_actions.shape)