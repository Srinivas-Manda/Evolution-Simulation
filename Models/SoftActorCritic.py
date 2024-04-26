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
        
        self.device = config['device']
        
        self.feature_extractor = create_convolutional_network(input_channels=config['in_channels'], output_channels=config['out_channels'], hidden_channels=config['hidden_channels'])
        
        self.mlp_block = create_linear_network(input_dim=config['out_channels']+config['num_actions'], output_dim=1, hidden_dims=config['hidden_dims'])
        
    def forward(self, state, actions):
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)
        
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
            
        actions = actions.to(self.device)
        state = state.to(self.device)
        
        x = self.feature_extractor(state).flatten(-3)
        
        x = torch.cat([x, actions], dim=1)

        x = self.mlp_block(x)
        
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
        actions = F.softmax(xs, dim=-1)
        
        # calculate entropies
        log_probs = normal_dist.log_prob(xs) - torch.log(1 - actions + self.eps)
        entropies = -log_probs.sum(dim=-1, keepdim=True)
        
        return actions, entropies, F.softmax(mean, dim=-1)
        

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
        # self.q_target_optim = optim.Adam(self.critic_q_target_net.parameters(), lr=self.q_step_size)

        # entropy related stuff
        # target entropy is -1 * # actions i.e. complete randomness
        self.target_entropy = -config['num_actions']
        # optimise log(alpha)
        self.log_alpha = torch.zeros(
            1, requires_grad=True, device=self.device)
        # the weight given to entropy while calculating the next state dist
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.policy_step_size)
        
    # gets the current state policy distribution from the current critic q net
    def calc_current_q(self, states, actions):
        curr_q = self.critic_q_net(states, actions)
        return curr_q

    # gets the next state policy distribution from the target critic q net
    def calc_target_q(self, rewards, next_states, dones):
        # get the next state distribution
        with torch.no_grad():
            next_actions, next_entropies, _ = self.actor_policy_net.sample(next_states)
            
            next_q = self.critic_q_target_net(next_states, next_actions)
            # alpha is the weight given to entropy
            next_q += self.alpha * next_entropies

        # happens when pushing just one transition
        if type(dones) is bool:
            dones = torch.Tensor([dones])

        # calculate the target 
        target_q = rewards + torch.logical_not(dones).reshape(-1, 1) * self.discount_factor * next_q

        return target_q
        
        
    def calc_critic_loss(self, batch):
        curr_q = self.calc_current_q(states=batch['states'], actions=batch['actions'])
        target_q = self.calc_target_q(rewards=batch["rewards"], next_states=batch["next_states"], dones=torch.logical_not(batch["non_final_mask"]))

        # critic loss is kl divergence
        q_loss = F.mse_loss(curr_q, target_q)
        return q_loss
    
    def calc_policy_loss(self, batch):
        # We re-sample actions to calculate expectations of Q.
        sampled_action, entropy, _ = self.actor_policy_net.sample(batch["states"])
        # expectations of Q 
        q = self.critic_q_net(batch["states"], sampled_action)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = torch.mean(- q - self.alpha * entropy)
        
        return policy_loss, entropy
    
    def calc_entropy_loss(self, entropy):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach())
        
        return entropy_loss

    # gives an action and the corresponding log prob
    def select_action(self, state):
        # make sure that the input is in the batch format
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
            
        # the exploration probability distribution, entropies and the mean distribution
        probs, _, _ = self.actor_policy_net.sample(state)
        
        return torch.argmax(probs, dim=-1), probs[:, torch.argmax(probs, dim=-1)]
    
    def push_to_buffer(self, *args):
        '''Given the state, action, reward, next_state, log probability, done (in that order), the buffer is updated after calculating the loss
        '''
        state, action, reward, next_state, log_prob, done = args
        
        action_in = torch.zeros(self.num_actions)
        action_in[action] = 1.
        
        # get current and next state distributions
        current_q = self.calc_current_q(state, action_in.reshape(1, -1))
        target_q = self.calc_target_q(rewards=reward, next_states=next_state, dones=done)
        
        # find the kl divergence between the distributions
        loss = torch.abs(target_q - current_q).item()
        
        # push the transition to the buffer
        self.replay_buffer.push(state.unsqueeze(0), torch.tensor([action]).unsqueeze(0), torch.tensor([reward]).unsqueeze(0), next_state.unsqueeze(0), torch.tensor([done]).unsqueeze(0), log_prob, loss)
    
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
        batch['non_final_next_states'] = batch['non_final_next_states'].to(self.device)
        batch['non_final_mask'] = batch['non_final_mask'].to(self.device)
        
        # create an action one hot vector
        actions = torch.zeros((batch['actions']).shape[0], self.num_actions)
        actions[np.arange((batch['actions']).shape[0]), batch['actions'].squeeze()] = 1
        batch['actions'] = actions.detach()
        
        # q loss calc and curr critic update
        q_loss = self.calc_critic_loss(batch=batch)
        # print(type(q_loss))
        self.param_step(optim=self.q_optim, network=self.critic_q_net, loss=q_loss, retain_graph=True)
        
        # policy loss calc and actor update
        policy_loss, entropy = self.calc_policy_loss(batch=batch)
        self.param_step(optim=self.policy_optim, network=self.actor_policy_net, loss=policy_loss, retain_graph=True)
        
        # entropy loss calc and update
        entropy_loss = self.calc_entropy_loss(entropy)
        self.param_step(optim=self.alpha_optim, network=None, loss=entropy_loss, retain_graph=True)
        
        self.soft_update(self.critic_q_target_net, self.critic_q_net)
        
        # return q_loss.detach(), policy_loss.detach(), entropy_loss.detach()
        
        
if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)
    
    config = {
        "in_channels": 3,
        "out_channels": 32,
        "hidden_channels": [8, 16],
        "hidden_dims": [32],
        "num_actions": 360,
        "actor_step_size": 1e-6,
        "critic_step_size": 1e-3,
        "batch_size": 4,
        "discount_factor": 0.9,
        "capacity": 8,
        "device": 'cpu',
        "log_std_min": -2,
        "log_std_max": 20,
        "epsilon": 1e-6,
        "tau": 1e-3
    }
    
    sac_agent = SoftActorCritic(config=config)
    
    state = torch.ones((3, 10, 10))
    
    losses = []
    
    for i in tqdm(range(20)):
        # will be taken by agent
        action, log_prob = sac_agent.select_action(state=state)
        
        # will be available from the environment
        next_state = torch.ones_like(state)
        reward = 1
        done = False if np.random.rand(1)[0] <= 0.999 else True
        
        sac_agent.push_to_buffer(state, action, reward, next_state, log_prob, done)
        
        ret = sac_agent.update_weights()
        if ret is not None:
            losses.append(ret)
        
        state = next_state
        
        if done:
            break
        
    # for loss in losses:
    #     print(loss)
    
    # action, log_prob = sac_agent.select_action(state=state)
    
    # next_state = torch.ones_like(state)
    # reward = torch.tensor([1])
    # done = False if np.random.rand(1)[0] <= 0.95 else True
    
    # done = False
    # next_actions, entropies, det_next_actions = sac_agent.actor_policy_net.sample(state=state)
    
    # print(next_actions.shape)
    # print(entropies.shape)
    # print(det_next_actions.shape)
    
    # state = torch.ones((2, 3, 10, 10))
    
    
    sac_agent.push_to_buffer(state, action, reward, next_state, log_prob, done)
    
    
    

        
        
        
        
        
    