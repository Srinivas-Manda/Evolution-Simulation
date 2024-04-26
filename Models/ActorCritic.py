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

# Policy Network / Actor
class Actor(nn.Module):
    
    def __init__(self, config) -> None:
        super().__init__()
        '''
        Args:
            config: dict - all important hyperparameters for the model
        '''
        self.device = config['device']    
        
        self.feature_extractor = create_convolutional_network(input_channels=config['in_channels'], output_channels=config['out_channels'], hidden_channels=config['hidden_channels'])
        
        self.mlp_block = create_linear_network(input_dim=config['out_channels'], output_dim=config['num_actions'], hidden_dims=config['hidden_dims'])
        
    def forward(self, state):
        '''
        Args:
            observation: torch.tensor - observation space, a 3 channel image denoting - agent positions, pellet positions, illegal area
        '''
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
            
        state = state.to(self.device)
                    
        feature_map = self.feature_extractor(state)
        probs = F.gumbel_softmax(self.mlp_block(feature_map.flatten(-3)), dim=-1)
        
        return probs
    
# Value Network / Critic
class Critic(nn.Module):
    
    def __init__(self, config):
        '''
        Args:
            config: dict - all important hyperparameters for the model
        '''
        super().__init__()

        self.device = config['device']
        
        self.feature_extractor = create_convolutional_network(input_channels=config['in_channels'], output_channels=config['out_channels'], hidden_channels=config['hidden_channels'])
        
        self.mlp_block = create_linear_network(input_dim=config['out_channels'], output_dim=1, hidden_dims=config['hidden_dims'])
        
    def forward(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
            
        state = state.to(self.device)
        
        feature_map = self.feature_extractor(state)
        state_val = self.mlp_block(feature_map.flatten(-3))
        
        return state_val
        
# class that combines both Actor and Critic
class ActorCritic(RLAgent):
    def __init__(self, config):
        '''Initialise the object with and instance of an actor and a critic
        
        Args:
            - actor - nn.Module: The Actor / Policy Network
            - critic - nn.Module: The Critic / Value Network
        '''
        super().__init__(config)
        
        self.actor = Actor(config).to(self.device)
        self.critic = Critic(config).to(self.device)
        
        self.actor_step_size = config['actor_step_size']
        self.critic_step_size = config['critic_step_size']
        
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr = self.actor_step_size)
        self.critic_optimiser = optim.Adam(self.critic.parameters(), lr = self.critic_step_size)

        
    def select_action(self, state):
        ''' selects an action based on the probabilities given by the actor
        Args:
            - actor - nn.Module: Policy Network / Actor
            - state - torch.tensor: State provided by the environment
            
        Returns:
            - (int): selected action
            - (float): log probability of selecting the action given the state 
        '''
        state = state.to(self.device)
        # make sure the input is in batch format
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        
        # get the probability distribution for the actions
        action_probs = self.actor(state)
        state = state.detach()
        
        # select an action based on the predicted probability
        m = Categorical(action_probs)
        action = m.sample()
        
        return action.item(), m.log_prob(action)
    
    def push_to_buffer(self, *args):
        '''Given the state, action, reward, next_state, log probability, done (in that order), the buffer is updated after calculating the loss
        '''
        
        state, action, reward, next_state, log_prob, done = args
        
        state, action, reward, next_state, log_prob, done = state.to(self.device), action, reward, next_state.to(self.device), log_prob.to(self.device), done 
        
        # next_state = None if done else next_state.unsqueeze(0)
        
        self.replay_buffer.push(state.unsqueeze(0), torch.tensor([action]).unsqueeze(0), torch.tensor([reward]).unsqueeze(0), next_state.unsqueeze(0), torch.tensor([done]).unsqueeze(0), log_prob.unsqueeze(0), 1)
        
        
    def update_weights(self) -> None:
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
        
        # print(batch['states'].shape)
        # print(batch['actions'])
        # print(batch['rewards'].shape)
        # print(batch['log_probabilities'].shape)
        # print(batch['non_final_next_states'].shape)
        # print(batch['non_final_mask'])
        # print()
        if batch is None:
            return
        
        batch['states'] = batch['states'].to(self.device)
        batch['actions'] = batch['actions'].to(self.device)
        batch['rewards'] = batch['rewards'].to(self.device)
        batch['log_probabilities'] = batch['log_probabilities'].to(self.device)
        batch['non_final_mask'] = batch['non_final_mask'].to(self.device)
        
        state_values = self.critic(batch['states'])
        
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        
        with torch.no_grad():
            next_state_values_temp = self.critic(batch["next_states"][batch["non_final_mask"]])
            
            next_state_values[batch["non_final_mask"]] = next_state_values_temp if next_state_values_temp.shape[0] != 0 else 0       
            
        
        # calculate actor loss and update actor weights
        advantage = batch["rewards"] + self.discount_factor * next_state_values - state_values
        actor_loss = - batch['log_probabilities'] * advantage
        
        self.param_step(optim=self.actor_optimiser, network=self.actor, loss=actor_loss, retain_graph=True)
        
        # calculate critic loss and update critic weight
        critic_loss = F.mse_loss(batch["rewards"] + self.discount_factor * next_state_values, state_values)
        
        self.param_step(optim=self.critic_optimiser, network=self.critic, loss=critic_loss)
        
        return
if __name__ == "__main__":
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
        "device": 'cpu'
    }
    
    ac_agent = ActorCritic(config=config)

    # will be available from previous step/initialisation
    state = torch.ones((3, 10, 10))
    
    for i in tqdm(range(60)):
        # will be taken by agent
        action, log_prob = ac_agent.select_action(state=state)
        
        # will be available from the environment
        next_state = torch.ones_like(state)
        reward = 1
        done = False if np.random.rand(1)[0] <= 0.95 else True
        
        ac_agent.push_to_buffer(state, action, reward, next_state, log_prob, done)
        
        ac_agent.update_weights()
        
        state = next_state
        
        if done:
            break
    
    
        
        
        
        
        
