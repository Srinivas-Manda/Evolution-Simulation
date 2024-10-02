# Evolution-Simulation
## Problem Statement
To design a custom environment used to simulate a naive interpretation of pellet foraging simulation by making the agent/s find food scattered throughout the grid-based environment. Three Models, namely, Double Deep Q-Learning, Actor Critic, and Soft Actor Critic models were used in this multi agent environment to compare their performance over 100 episodes.

## RL Algorithms
All RL Algoritms have been implemented from scratch and have a consistent API. The models are compatible with the OpenAI Gym environment and multiple agents can be created for these algorithms in two different modes - cooperative and competitive. Moreover, two different types of agents can be created - MLP Style, GRU Style. Each RL Agent inherits from the base RLAgent class.

## Environment Design
PettingZoo is a simple, pythonic interface capable of representing general multi-agent reinforcement learning (MARL) problems. A multi-agent custom environment was built from scratch using the PettingZoo library which is built on top of OpenAI Gym.
The following notations will be used throughout:
1. A set of agents, G ≡ {gi
|1 ≤ i ≤ ng} where ng is the number of agents.
2. State space, S
3. Action space, A
4. Reward function, r(s, a) where s ∈ S and a ∈ A
5. Environment decision process, P(s, a) where s ∈ S and a ∈ A
6. Value (State Value) Network V(s; w) where s ∈ S and w are the weights of the network
7. Policy (State, Action Value) Network Q(s, a; θ) where s ∈ S, a ∈ A and θ are the weights of the network

## State Space/Observation Space:
The environment on which the agents exist is a 2D square grid with size sgrid. We ensure that sgrid is greater than the agent vision size sagent. At each timestep, the part of the grid that is visible to the agent is provided as an input to the agent. We come up with various strategies of doing the same.
The initial positions of the agents and the pellets are chosen randomly. To ensure a good spread of the pellets, we initialise their positions using the initialisation strategy of KMeans++. For selecting npellets number of pellets, we first select 10 ∗ n<sub>pellets</sub> coordinates on the grid randomly. From these coordinates, we select n<sub>pellets</sub> pellets using the KMeans++ initialisation strategy, which is built to ensure maximum spread of the selected points.
