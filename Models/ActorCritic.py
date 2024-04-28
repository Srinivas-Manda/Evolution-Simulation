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
        
        self.stamina_embedding = nn.Embedding(num_embeddings=config['max_stamina'], embedding_dim=config['out_channels'])
        self.x_pos_embedding = nn.Embedding(num_embeddings=config['max_x'], embedding_dim=config['out_channels'])
        self.y_pox_embedding = nn.Embedding(num_embeddings=config['max_y'], embedding_dim=config['out_channels'])

    def forward(self, state):
        '''
        Args:
            observation: torch.tensor - observation space, a 3 channel image denoting - agent positions, pellet positions, illegal area
        '''
        # if len(state.shape) == 3:
        #     state = state.unsqueeze(0)
            
        # state = state.to(self.device)
                    
        # feature_map = self.feature_extractor(state)
        # probs = F.softmax(self.mlp_block(feature_map.flatten(-3)), dim=-1)
        
        # return probs

        obs, stam, x, y = state
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
            
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

        obs = obs.to(self.device)
        # stam = stam.to(self.device)
        # x = x.to(self.device)
        # y = y.to(self.device)

        stam_embed = self.stamina_embedding(stam)
        x_embed = self.x_pos_embedding(x)
        y_embed = self.y_pox_embedding(y)
        
        feats = self.feature_extractor(obs).flatten(-3)
        
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
        
        return F.softmax(outs, dim=-1)
    
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
        
        self.stamina_embedding = nn.Embedding(num_embeddings=config['max_stamina'], embedding_dim=config['out_channels'])
        self.x_pos_embedding = nn.Embedding(num_embeddings=config['max_x'], embedding_dim=config['out_channels'])
        self.y_pox_embedding = nn.Embedding(num_embeddings=config['max_y'], embedding_dim=config['out_channels'])

    def forward(self, state):
        # if len(state.shape) == 3:
        #     state = state.unsqueeze(0)
            
        # state = state.to(self.device)
        
        # feature_map = self.feature_extractor(state)
        # state_val = self.mlp_block(feature_map.flatten(-3))
        
        # return state_val
        
        obs, stam, x, y = state
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
            
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

        obs = obs.to(self.device)
        # stam = stam.to(self.device)
        # x = x.to(self.device)
        # y = y.to(self.device)

        stam_embed = self.stamina_embedding(stam)
        x_embed = self.x_pos_embedding(x)
        y_embed = self.y_pox_embedding(y)
        
        feats = self.feature_extractor(obs).flatten(-3)
        
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
        
# class that combines both Actor and Critic
class ActorCritic(RLAgent):
    def __init__(self, config):
        '''Initialise the object with and instance of an actor and a critic
        
        Args:
            - actor - nn.Module: The Actor / Policy Network
            - critic - nn.Module: The Critic / Value Network
        '''
        super().__init__(config)
        
        config['device'] = self.device
        
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
        # state = state.to(self.device)
        # # make sure the input is in batch format
        # if len(state.shape) == 3:
        #     state = state.unsqueeze(0)
        
        if type(state[0]) is not torch.Tensor:
            state = (torch.tensor(state[0]), torch.tensor([state[1]]), torch.tensor([state[2]]), torch.tensor([state[3]]))

        # get the probability distribution for the actions
        action_probs = F.softmax(self.actor(state), dim=-1)
        
        # select an action based on the predicted probability
        m = Categorical(action_probs)
        action = m.sample()
        
        return action.item(), m.log_prob(action)
    
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
        
        state, action, reward, next_state, done = state, torch.tensor([action]).unsqueeze(0), torch.tensor([reward]).unsqueeze(0), next_state, torch.tensor([done]).unsqueeze(0)
        
        # next_state = None if done else next_state.unsqueeze(0)
        
        self.replay_buffer.push(state, action, reward, next_state, done, log_prob, -1*log_prob)
        
        
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
        
        # when buffer is not filled enough
        if batch is None:
            return
        
        # batch['states'] = batch['states'].to(self.device)
        batch['actions'] = batch['actions'].to(self.device)
        batch['rewards'] = batch['rewards'].to(self.device)
        batch['log_probabilities'] = batch['log_probabilities'].to(self.device)
        batch['non_final_mask'] = batch['non_final_mask'].to(self.device)
        
        state_values = self.critic(batch['states'])
        next_state_values = self.critic(batch['next_states'])
        done = torch.logical_not(batch['non_final_mask'])
        
        # next_state_values = torch.zeros(self.batch_size, device=self.device)
        
        # with torch.no_grad():
        #     next_state_values_temp = self.critic(batch["next_states"][batch["non_final_mask"]])
            
        #     next_state_values[batch["non_final_mask"]] = next_state_values_temp if next_state_values_temp.shape[0] != 0 else 0       
            
        
        # calculate actor loss and update actor weights
        advantage = batch["rewards"] + self.discount_factor * next_state_values - state_values.detach()
        actor_loss = - batch['log_probabilities'].unsqueeze(1) * advantage
        actor_loss = torch.mean(actor_loss)

        # print(batch["rewards"].shape)
        # print(next_state_values.shape)
        # print(state_values.shape)
        # print(advantage.shape)
        # print(actor_loss.shape)
        # print(batch['log_probabilities'].unsqueeze(1).shape)
        
        
        
        self.param_step(optim=self.actor_optimiser, network=self.actor, loss=actor_loss, retain_graph=True)
        
        # calculate critic loss and update critic weight
        critic_loss = F.mse_loss(state_values, batch["rewards"] + torch.logical_not(done).reshape(-1, 1) * self.discount_factor * next_state_values.detach())
        
        self.param_step(optim=self.critic_optimiser, network=self.critic, loss=critic_loss)
        
        return actor_loss.detach().cpu()
        
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
    
    
        
        
        
        
        
