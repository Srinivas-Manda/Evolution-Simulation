import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# import gymnasium
from tqdm import tqdm
import numpy as np
from collections import deque

from ReplayBuffer import ReplayBuffer, Transition
from RLAgent import RLAgent
from BasicNetworks import create_linear_network, create_convolutional_network

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
        
        # self.feature_extractor = CNN(config=config)
        
        self.mlp_block = create_linear_network(input_dim=config['out_channels'], output_dim=config['num_actions'], hidden_dims=config['hidden_dims'])
        
        # self.mlp_block = nn.Sequential(
        #     nn.Linear(in_features=16, out_features=32),
        #     # nn.Dropout(p=0.3),
        #     nn.Linear(in_features=32, out_features=config['num_actions'])
        # )
        
    def forward(self, state):
        '''
        Args:
            observation: torch.tensor - observation space, a 3 channel image denoting - agent positions, pellet positions, illegal area
        '''
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
                    
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
        
        # self.feature_extractor = CNN(config=config)
        
        self.mlp_block = create_linear_network(input_dim=config['out_channels'], output_dim=1, hidden_dims=config['hidden_dims'])
        
        # self.feature_extractor = CNN(config=config)   
       
        # self.mlp_block = nn.Sequential(
        #     nn.Linear(in_features=16, out_features=32),
        #     # nn.Dropout(p=0.3),
        #     nn.Linear(in_features=32, out_features=1)
        # )
        
    def forward(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        
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
        
        self.actor = Actor(config)
        self.critic = Critic(config)
        
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
        
        # return None
        state, action, reward, next_state, log_prob, done = args
        
        next_state = None if done else next_state.unsqueeze(0)
        
        self.replay_buffer.push(state.unsqueeze(0), torch.tensor([action]), reward.unsqueeze(0), next_state, log_prob.unsqueeze(0), 1)
        
        # # get the state value
        # state_value = self.critic(state).to('cpu')

        # # if the next state is terminal, set value to zero
        # if done:
        #     next_state_value = torch.tensor([0]).float().unsqueeze(0).to('cpu')
        # # else get the next state value
        # else:
        #     next_state_value = self.critic(next_state).to('cpu')
            
        # # calculate the advantage and the loss
        # advantage = reward + self.discount_factor * next_state_value.item() - state_value.item()
        # loss = -log_prob * advantage
        
        # # push the transition and the loss to the buffer
        # self.replay_buffer.push(state.cpu(), action.cpu(), reward.cpu(), next_state.cpu(), log_prob.cpu(), loss)
        
        
        
    def update_weights(self) -> None:
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
        
        state_values = self.critic(batch['states'])
        
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        
        with torch.no_grad():
            next_state_values_temp = self.critic(batch["next_states"][batch["non_final_mask"]])
            
            next_state_values[batch["non_final_mask"]] = next_state_values_temp if next_state_values_temp.shape[0] != 0 else 0       
            
        
        # calculate actor loss
        advantage = batch["rewards"] + self.discount_factor * next_state_values - state_values
        actor_loss = - batch['log_probabilities'] * advantage
        
        # actor weight updates
        self.actor_optimiser.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimiser.step()
        
        # calculate critic loss
        critic_loss = F.mse_loss(batch["rewards"] + self.discount_factor * next_state_values, state_values)
        
        # critic weight updates
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()
        
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
        reward = torch.tensor([1])
        done = False if np.random.rand(1)[0] <= 0.95 else True
        
        ac_agent.push_to_buffer(state, action, reward, next_state, log_prob, done)
        
        ac_agent.update_weights()
        
        state = next_state
        
        if done:
            break
    
    
        
        
        
        
        
