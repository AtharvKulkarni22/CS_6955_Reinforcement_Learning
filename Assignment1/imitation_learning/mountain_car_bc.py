import gym
import argparse
import pygame
from teleop import collect_demos
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


device = torch.device('cpu')


def collect_human_demos(num_demos):
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}
    env = gym.make("MountainCar-v0",render_mode='single_rgb_array') 
    demos = collect_demos(env, keys_to_action=mapping, num_demos=num_demos, noop=1)
    return demos


def torchify_demos(sas_pairs):
    states = []
    actions = []
    next_states = []
    for s,a, s2 in sas_pairs:
        states.append(s)
        actions.append(a)
        next_states.append(s2)

    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)

    obs_torch = torch.from_numpy(np.array(states)).float().to(device)
    obs2_torch = torch.from_numpy(np.array(next_states)).float().to(device)
    acs_torch = torch.from_numpy(np.array(actions)).long().to(device)

    return obs_torch, acs_torch, obs2_torch


def train_policy(obs, acs, nn_policy, num_train_iters):
    # """TODO: train the policy using standard behavior cloning. Feel free to add other helper methods if you'd like or restructure the code as desired."""

    # optimizer = Adam(nn_policy.parameters(), lr=0.001)  # No
    # optimizer = Adam(nn_policy.parameters(), lr=0.1)  # Yes -> Sometimes
    optimizer = Adam(nn_policy.parameters(), lr=0.01)  # Yes --> Best 
    # optimizer = Adam(nn_policy.parameters(), lr=0.005)  # Yes --> Oscillates 
    # optimizer = Adam(nn_policy.parameters(), lr=0.05)  # Yes - Sometimes
    # optimizer = Adam(nn_policy.parameters(), lr=1)  # No
    
    # optimizer = Adam(nn_policy.parameters(), lr=5)  # No
    
    loss_fn = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification

    nn_policy.to(device)
    obs = obs.to(device)
    acs = acs.to(device)

    for i in range(num_train_iters):
        nn_policy.train()  # Set the model to training mode
        optimizer.zero_grad()  # Zero the gradient

        logits = nn_policy(obs)  # Forward pass
        loss = loss_fn(logits, acs)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the model parameters

        if i % 10 == 0:  # Print loss every 10 iterations
            print(f"Iteration {i}/{num_train_iters}, Loss: {loss.item()}")

class PolicyNetwork(nn.Module):
    '''
        Simple neural network with two layers that maps a 2-d state to a prediction
        over which of the three discrete actions should be taken.
        The three outputs corresponding to the logits for a 3-way classification problem.

    '''
    # # Basic implementation:
    # def __init__(self):
    #     super().__init__()

    #     # """TODO: create the layers for the neural network. A two-layer network should be sufficient"""
    #     self.fc1 = nn.Linear(2, 64) 
    #     self.fc2 = nn.Linear(64, 64)
    #     self.fc3 = nn.Linear(64, 3)

    # def forward(self, x):
    #     # """TODO: this method performs a forward pass through the network, applying a non-linearity (ReLU is fine) on the hidden layers and should output logit values (since this is a discrete action task) for the 3-way classification problem"""
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x
    
    # Basic 2:
    # def __init__(self):
    #     super().__init__()

    #     # Define the layers for the network
    #     self.fc1 = nn.Linear(2, 128)  # Input layer to first hidden layer
    #     self.fc2 = nn.Linear(128, 128)  # Second hidden layer
    #     self.fc3 = nn.Linear(128, 128)  # Third hidden layer
    #     self.fc4 = nn.Linear(128, 128)  # Fourth hidden layer
    #     self.fc5 = nn.Linear(128, 3)  # Output layer for 3 actions

    # Not Working 1 (works only if we add pi.eval() at eval):
    # def forward(self, x):
    #     """
    #     Perform a forward pass through the network.
    #     Apply ReLU as the non-linearity on the hidden layers.
    #     Outputs logits for a 3-way classification problem.
    #     """
    #     x = F.relu(self.fc1(x))  # First hidden layer with ReLU
    #     x = F.relu(self.fc2(x))  # Second hidden layer with ReLU
    #     x = F.relu(self.fc3(x))  # Third hidden layer with ReLU
    #     x = F.relu(self.fc4(x))  # Fourth hidden layer with ReLU
    #     x = self.fc5(x)  # Output layer (logits)
    #     return x
    
    # def __init__(self, hidden_size=128, dropout_rate=0.2):
    #     super().__init__()
    #     self.fc1 = nn.Linear(2, hidden_size)  # First hidden layer
    #     self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization
    #     self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second hidden layer
    #     self.bn2 = nn.BatchNorm1d(hidden_size)  # Batch normalization
    #     self.fc3 = nn.Linear(hidden_size, 3)  # Output layer (3 actions)
    #     self.dropout = nn.Dropout(dropout_rate)  # Dropout layer
    #     self.activation = F.relu  # Default activation function (ReLU)

    #     # Initialize weights
    #     self.init_weights()

    # def init_weights(self):
    #     """Initialize weights using Xavier initialization."""
    #     for layer in [self.fc1, self.fc2, self.fc3]:
    #         nn.init.xavier_uniform_(layer.weight)
    #         nn.init.zeros_(layer.bias)

    # def forward(self, x):
    #     """
    #     Forward pass through the network, applying non-linearities and dropout.
    #     Outputs logits for the 3-way classification problem.
    #     """
    #     x = self.activation(self.bn1(self.fc1(x)))  # Hidden layer 1
    #     x = self.dropout(x)  # Apply dropout
    #     x = self.activation(self.bn2(self.fc2(x)))  # Hidden layer 2
    #     x = self.dropout(x)  # Apply dropout
    #     x = self.fc3(x)  # Output layer
    #     return x
    
    # Working 2:
    def __init__(self, hidden_size=128):
    # def __init__(self, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_size)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_size, 3)  # Output layer (3 actions)
        self.activation = F.relu  # Default activation function (ReLU)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights using Xavier initialization."""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass through the network.
        Outputs logits for the 3-way classification problem.
        """
        x = self.activation(self.fc1(x))  # Hidden layer 1 with ReLU
        x = self.activation(self.fc2(x))  # Hidden layer 2 with ReLU
        x = self.fc3(x)  # Output layer (logits)
        return x
    
    
    

    

#evaluate learned policy
def evaluate_policy(pi, num_evals, human_render=True):
    if human_render:
        env = gym.make("MountainCar-v0",render_mode='human') 
    else:
        env = gym.make("MountainCar-v0") 

    policy_returns = []
    for i in range(num_evals):
        done = False
        total_reward = 0
        obs = env.reset()
        while not done:
            #take the action that the network assigns the highest logit value to
            #Note that first we convert from numpy to tensor and then we get the value of the 
            #argmax using .item() and feed that into the environment
            action = torch.argmax(pi(torch.from_numpy(obs).unsqueeze(0))).item()
            # print(action)
            obs, rew, done, info = env.step(action)
            total_reward += rew
        print("reward for evaluation", i, total_reward)
        policy_returns.append(total_reward)

    print("average policy return", np.mean(policy_returns))
    print("min policy return", np.min(policy_returns))
    print("max policy return", np.max(policy_returns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_demos', default = 1, type=int, help="number of human demonstrations to collect")
    parser.add_argument('--num_bc_iters', default = 100, type=int, help="number of iterations to run BC")
    parser.add_argument('--num_evals', default=6, type=int, help="number of times to run policy after training for evaluation")

    args = parser.parse_args()

    #collect human demos
    demos = collect_human_demos(args.num_demos)

    #process demos
    obs, acs, _ = torchify_demos(demos)

    #train policy
    pi = PolicyNetwork()
    train_policy(obs, acs, pi, args.num_bc_iters)

    #evaluate learned policy
    evaluate_policy(pi, args.num_evals)

