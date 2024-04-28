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

        self.flatten = False
        if "flatten" in config and config['flatten'] == "True":
            self.flatten = True
            self.vision = 2 * config['vision_size'] + 1

        self.device = config['device']
        
        if self.flatten:
            self.feature_extractor = create_linear_network(input_dim=self.vision*self.vision*config['in_channels'], output_dim=(self.vision//8)*(self.vision//8)*config['out_channels'], hidden_dims=[(self.vision//2)*(self.vision//2)*config['out_channels'], (self.vision//4)*(self.vision//4)*config['out_channels']])
            self.mlp_block = create_linear_network(input_dim=(self.vision//8)*(self.vision//8)*config['out_channels'], output_dim=config['num_actions'], hidden_dims=config['hidden_dims'])
            self.stamina_embedding = nn.Embedding(num_embeddings=config['max_stamina'], embedding_dim=(self.vision//8)*(self.vision//8)*config['out_channels'])
            self.x_pos_embedding = nn.Embedding(num_embeddings=config['max_x'], embedding_dim=(self.vision//8)*(self.vision//8)*config['out_channels'])
            self.y_pox_embedding = nn.Embedding(num_embeddings=config['max_y'], embedding_dim=(self.vision//8)*(self.vision//8)*config['out_channels'])
        else:
            self.feature_extractor = create_convolutional_network(input_channels=config['in_channels'], output_channels=config['out_channels'], hidden_channels=config['hidden_channels'])
            self.mlp_block = create_linear_network(input_dim=config['out_channels'], output_dim=config['num_actions'], hidden_dims=config['hidden_dims'])
            self.stamina_embedding = nn.Embedding(num_embeddings=config['max_stamina'], embedding_dim=config['out_channels'])
            self.x_pos_embedding = nn.Embedding(num_embeddings=config['max_x'], embedding_dim=config['out_channels'])
            self.y_pox_embedding = nn.Embedding(num_embeddings=config['max_y'], embedding_dim=config['out_channels'])
        
        
    def forward(self, state):
        obs, stam, x, y = state
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)

        obs = obs.to(self.device)

        if self.flatten:
            # print(obs.shape)
            obs = obs.flatten(-3)
            # print(obs.shape)
            feats = self.feature_extractor(obs)
        else:
            feats = self.feature_extractor(obs).flatten(-3)
            
        # lst = [stam, x, y]
        # for i, item in enumerate(lst):
        #     if type(i) is not torch.Tensor:
        #         i = torch.tensor([i],  dtype=torch.int)
        #     if len(i.shape) == 1:
        #         i = i.unsqueeze(0)
                
        #     i = i.to(self.device)
            
        if type(stam) is not torch.Tensor:
            stam = torch.tensor([stam],  dtype=torch.int)
        # if len(stam.shape) == 1:
        #     stam = stam.unsqueeze(0)
        stam = stam.to(self.device)
                
        if type(x) is not torch.Tensor:
            x = torch.tensor([x],  dtype=torch.int)
        # if len(x.shape) == 1:
        #     x = x.unsqueeze(0)
        x = x.to(self.device)
        
        if type(y) is not torch.Tensor:
            y = torch.tensor([y],  dtype=torch.int)
        # if len(y.shape) == 1:
        #     y = y.unsqueeze(0)
        y = y.to(self.device)

        # stam = stam.to(self.device)
        # x = x.to(self.device)
        # y = y.to(self.device)

        # try:
        stam_embed = self.stamina_embedding(stam)
        x_embed = self.x_pos_embedding(x)
        y_embed = self.y_pox_embedding(y)
        # except IndexError:
        
        
        # print(feats.shape)
        # print(stam_embed.shape)
        # print(x_embed.shape)
        # print(y_embed.shape)
        # print(stam.shape)
        # print(x.shape)
        # print(y.shape)
        
        # print(x.shape)
        # print(stam.shape)
        feats = stam_embed + x_embed + y_embed
        
        outs = self.mlp_block(feats)
        
        return outs

# Double DQN Class       
class DoubleDQN(RLAgent):
    def __init__(self, config):
        super().__init__(config=config)

        
        config['device'] = self.device
        # initialise the networks
        self.policy_net = QNetwork(config=config).to(self.device)
        self.target_net = QNetwork(config=config).to(self.device)
        
        # initialise the optimiser
        self.policy_step_size = config['policy_step_size']
        self.policy_optim = optim.Adam(params=self.policy_net.parameters(), lr=self.policy_step_size)
        self.target_step_size = config['policy_step_size']
        self.target_optim = optim.Adam(params=self.policy_net.parameters(), lr=self.target_step_size)
        
        # exploration control
        self.eps_start = config['eps_start']
        self.eps_end = config['eps_end']
        self.eps_decay = config['eps_decay']
        self.steps_done = 0

    # calculate current q values
    def calc_current_q_values(self, state, network):
        # print("passing state")
        return F.softmax(network(state), dim=-1)
    
    def calc_target_q_value(self, reward, next_state, done, network):
        with torch.no_grad():
            # print("passing next state")
            # print(next_state[0].shape)
            next_q = F.softmax(network(next_state), dim=-1)
            
        if type(done) is bool:
            done = torch.Tensor([done]).to(self.device)
            
        target_q = reward + torch.logical_not(done).reshape(-1, 1) * self.discount_factor * next_q
        
        # print(target_q.shape)
        return target_q
    
    # calculate the loss
    def calc_policy_loss(self, state, action, reward, next_state, done, policy_net, target_net):
        curr_q = self.calc_current_q_values(state=state, network=policy_net)
        target_q = self.calc_target_q_value(reward=reward, next_state=next_state, done=done, network=target_net)
        
        if type(action) is not torch.Tensor:
            action = torch.tensor([action], dtype=torch.int).to(self.device)
        
        loss = F.mse_loss(curr_q[np.arange(curr_q.shape[0]), action.squeeze()], target_q.detach().max(dim=-1)[0])

        # print(f"Curr Q Val: {curr_q}")
        # print(f"All Curr Q Val: {curr_q[np.arange(curr_q.shape[0]), action.squeeze()]}")
        # print(f"Actions Selected: {action}")
        # print()
        
        # print(f"All Next Q Val: {target_q}")
        # print(f"Next Q Val: {target_q.max(dim=-1)[0]}")
        # print(f"Actions Selected: {target_q.max(dim=-1)[1]}")
        # print()

        return loss

    # gives an action and the corresponding log prob        
    def select_action(self, state):
        if np.random.rand() > 0.5:
            network = self.policy_net
        else:
            network = self.target_net
            
        if type(state[0]) is not torch.Tensor:
            state = (torch.tensor(state[0]), torch.tensor([state[1]]), torch.tensor([state[2]]), torch.tensor([state[3]]))
            
        
        # calculate the threshold
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * min(self.steps_done, self.eps_decay)/self.eps_decay)
        
        with torch.no_grad():
            # if len(state.shape) == 3:
            #     state = state.unsqueeze(0)
                
            # get the probabilities
            # print("passing state")
            action_probs = F.softmax(network(state), dim=-1)
            # print(action_probs.shape)
            
            log_probs = torch.log(action_probs)
            
        # if action is to be chosen greedily
        if np.random.rand() > eps_threshold:
            action_probs, action = torch.max(action_probs, dim=-1)
            
            return action.item(), torch.log(action_probs)
        # if random sampling needs to be done
        else:
            action = np.random.randint(0, self.num_actions)
            
            return action, log_probs[:, action]
        
    def push_to_buffer(self, *args):
        '''Given the state, action, reward, next_state, log probability, done (in that order), the buffer is updated after calculating the loss
        '''
        state, action, reward, next_state, log_prob, done = args
        
        if type(state[0]) is not torch.Tensor:
            state = (torch.tensor(state[0]).unsqueeze(0), torch.tensor([state[1]]), torch.tensor([state[2]]), torch.tensor([state[3]]))
            next_state = (torch.tensor(next_state[0]).unsqueeze(0), torch.tensor([next_state[1]]), torch.tensor([next_state[2]]), torch.tensor([next_state[3]]))
            
        else:
            state = (state[0].unsqueeze(0), torch.tensor([state[1]]), torch.tensor([state[2]]), torch.tensor([state[3]]))
            next_state = (next_state[0].unsqueeze(0), torch.tensor([next_state[1]]), torch.tensor([next_state[2]]), torch.tensor([next_state[3]]))
        # print(state)
        
        state, action, reward, next_state, done = state, torch.tensor([action]).unsqueeze(0), torch.tensor([reward]).unsqueeze(0), next_state, torch.tensor([done]).unsqueeze(0)
        
        with torch.no_grad():
            # curr_q = self.calc_current_q_values(state=state)
            # target_q = self.calc_target_q_value(reward=reward, next_state=next_state, done=done)
            
            # loss = F.mse_loss(curr_q[0, action], target_q.max())
            if np.random.rand() > 0.5:
                network_1 = self.policy_net
                network_2 = self.target_net
            else:
                network_1 = self.target_net
                network_2 = self.policy_net
            loss = self.calc_policy_loss(state, action.to(self.device), reward.to(self.device), next_state, done.to(self.device), policy_net=network_1, target_net=network_2)
            
        
        # print(state.shape)
        # print(action)
        # print(reward)
        # print(next_state.shape)
        # print(done)
        # print(log_prob)
        # print(loss)
        
        # print(state[0].shape)
        # print(state[1].shape)
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
        
        # batch['states'] = batch['states'].to(self.device)
        batch['actions'] = batch['actions'].to(self.device)
        batch['rewards'] = batch['rewards'].to(self.device)
        batch['log_probabilities'] = batch['log_probabilities'].to(self.device)
        batch['non_final_mask'] = batch['non_final_mask'].to(self.device)
        
        if np.random.rand() > 0.5:
            network_1 = self.policy_net
            network_2 = self.target_net
            optim = self.policy_optim
        else:
            network_1 = self.target_net
            network_2 = self.policy_net
            optim = self.target_optim
        
        # calculate policy loss and update policy weights
        policy_loss = self.calc_policy_loss(batch['states'], batch['actions'], batch['rewards'], batch['next_states'], torch.logical_not(batch['non_final_mask']), policy_net=network_1, target_net=network_2)
        self.param_step(optim=optim, network=network_1, loss=policy_loss)
        
        # # soft updates for the target net
        # self.soft_update(target=self.target_net, source=self.policy_net)
        
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
        "num_actions": 4,
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
        "tau": 1e-3,
        "max_stamina": 500,
        "max_x": 100,
        "max_y": 100
    }
    
    ddqn_agent = DoubleDQN(config=config)
    
    state = (torch.ones((3, 10, 10)), 1, 50, 50)
    
    for i in tqdm(range(20)):
        # will be taken by agent
        action, log_prob = ddqn_agent.select_action(state=state)
        # print(action.shape)
        
        # will be available from the environment
        next_state = (torch.ones_like(state[0]), 1, 50, 50)
        reward = 1
        done = False if np.random.rand(1)[0] <= 0.999 else True
        
        # print(state[0].shape)
        ddqn_agent.push_to_buffer(state, action, reward, next_state, log_prob, done)
        
        
        ret = ddqn_agent.update_weights()
        # break
        # if ret is not None:
        #     losses.append(ret)
        
        state = next_state
        
        if done:
            break
    
        # break
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