import gym
import argparse
import pygame
from teleop import collect_demos
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mountain_car_bc import collect_human_demos, torchify_demos, train_policy, PolicyNetwork, evaluate_policy


device = torch.device('cpu')


def collect_random_interaction_data(num_iters):
    state_next_state = []
    actions = []
    env = gym.make('MountainCar-v0')
    for _ in range(num_iters):
        obs = env.reset()

        done = False
        while not done:
            a = env.action_space.sample()
            next_obs, reward, done, info = env.step(a)
            state_next_state.append(np.concatenate((obs,next_obs), axis=0))
            actions.append(a)
            obs = next_obs
    env.close()

    return np.array(state_next_state), np.array(actions)


def train_inverse_dynamics_model(inv_dyn, s_s2_torch, a_torch, num_iters=100, lr=0.001):
    optimizer = Adam(inv_dyn.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for i in range(num_iters):
        optimizer.zero_grad()
        logits = inv_dyn(s_s2_torch)  # Predict actions from state transitions
        loss = loss_fn(logits, a_torch)  # Compare with true actions
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")



class InvDynamicsNetwork(nn.Module):
    '''
        Neural network with that maps (s,s') state to a prediction
        over which of the three discrete actions was taken.
        The network should have three outputs corresponding to the logits for a 3-way classification problem.

    '''
    # Basic:
    # def __init__(self):
    #     super().__init__()

    #     #This network should take in 4 inputs corresponding to car position and velocity in s and s'
    #     # and have 3 outputs corresponding to the three different actions

    #     #################
    #     #TODO:
    #     self.fc1 = nn.Linear(4, 128)  # Input is concatenated (s, s') with 4 dimensions
    #     self.fc2 = nn.Linear(128, 128)  # Hidden layer
    #     self.fc3 = nn.Linear(128, 3)  # Output logits for 3 discrete actions
    #     #################

    # def forward(self, x):
    #     #this method performs a forward pass through the network
    #     ###############
    #     #TODO:
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     ###############
    #     return x
    
    # Partially Working 1: 
    # lower dropout the better
    # more 
    # def __init__(self, hidden_size=128, dropout_rate=0.2):
    # def __init__(self, hidden_size=256, dropout_rate=0.3):
    # def __init__(self, hidden_size=512, dropout_rate=0.001):
    #     super().__init__()
    #     self.fc1 = nn.Linear(4, hidden_size)  # Input layer for (s, s') with 4 dimensions
    #     self.fc2 = nn.Linear(hidden_size, hidden_size)  # First hidden layer
    #     self.fc3 = nn.Linear(hidden_size, hidden_size)  # Second hidden layer
    #     self.fc4 = nn.Linear(hidden_size, 3)  # Output logits for 3 discrete actions
    #     # self.activation = F.relu  # Activation function (ReLU)
    #     self.activation = F.leaky_relu  # Activation function (ReLU)
    #     self.dropout = nn.Dropout(dropout_rate)  # Dropout for regularization

    #     # Initialize weights
    #     self.init_weights()

    # def init_weights(self):
    #     """Initialize weights using Xavier initialization."""
    #     for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
    #         nn.init.xavier_uniform_(layer.weight)
    #         nn.init.zeros_(layer.bias)

    # def forward(self, x):
    #     """
    #     Perform a forward pass through the network.
    #     Outputs logits for the 3-way classification problem.
    #     """
    #     x = self.activation(self.fc1(x))  # First hidden layer with ReLU
    #     x = self.dropout(x)  # Apply dropout
    #     x = self.activation(self.fc2(x))  # Second hidden layer with ReLU
    #     x = self.dropout(x)  # Apply dropout
    #     x = self.activation(self.fc3(x))  # Third hidden layer with ReLU
    #     x = self.fc4(x)  # Output layer (logits)
    #     return x
    
    # Final Architecture:
    # def __init__(self, hidden_size=128, dropout_rate=0.0): # Ok
    def __init__(self, hidden_size=256, dropout_rate=0.0): # Good
    # def __init__(self, hidden_size=512, dropout_rate=0.0): # No
        super().__init__()
        self.fc1 = nn.Linear(4, hidden_size)  # Input layer for (s, s') with 4 dimensions
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # First hidden layer
        self.fc3 = nn.Linear(hidden_size, hidden_size)  # Second hidden layer
        self.fc4 = nn.Linear(hidden_size, hidden_size)  # Third hidden layer
        self.fc5 = nn.Linear(hidden_size, 3)  # Output logits for 3 discrete actions
        self.activation = F.leaky_relu  # Activation function (Leaky ReLU)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout for regularization

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights using Xavier initialization."""
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Perform a forward pass through the network.
        Outputs logits for the 3-way classification problem.
        """
        x = self.activation(self.fc1(x))  # First hidden layer
        x = self.dropout(x)  
        x = self.activation(self.fc2(x))  # Second hidden layer
        x = self.dropout(x)  
        x = self.activation(self.fc3(x))  # Third hidden layer
        x = self.dropout(x) 
        x = self.activation(self.fc4(x))  # Fourth hidden layer
        x = self.fc5(x)  # Output layer (logits)
        return x


    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_demos', default = 1, type=int, help="number of human demonstrations to collect")
    parser.add_argument('--num_bc_iters', default = 100, type=int, help="number of iterations to run BC")
    parser.add_argument('--num_evals', default=6, type=int, help="number of times to run policy after training for evaluation")

    args = parser.parse_args()


    #collect random interaction data
    num_interactions = 5
    s_s2, acs = collect_random_interaction_data(num_interactions)
    #put the data into tensors for feeding into torch
    s_s2_torch = torch.from_numpy(np.array(s_s2)).float().to(device)
    a_torch = torch.from_numpy(np.array(acs)).long().to(device)


    #initialize inverse dynamics model
    inv_dyn = InvDynamicsNetwork()  #TODO: need to fill in the blanks in this method
    
    ##################
    #TODO: Train the inverse dyanmics model, no need to be fancy you can do it in one full batch via gradient descent if you like
    # train_inverse_dynamics_model(inv_dyn, s_s2_torch, a_torch, num_iters=100, lr=0.0001) # No
    # train_inverse_dynamics_model(inv_dyn, s_s2_torch, a_torch, num_iters=100, lr=0.001) # Yes
    train_inverse_dynamics_model(inv_dyn, s_s2_torch, a_torch, num_iters=100, lr=0.0009) # Yes --> Best
    # train_inverse_dynamics_model(inv_dyn, s_s2_torch, a_torch, num_iters=100, lr=0.0005) # No?
    # train_inverse_dynamics_model(inv_dyn, s_s2_torch, a_torch, num_iters=100, lr=0.00075) # No
    # train_inverse_dynamics_model(inv_dyn, s_s2_torch, a_torch, num_iters=100, lr=0.005) # No
    # train_inverse_dynamics_model(inv_dyn, s_s2_torch, a_torch, num_iters=100, lr=0.002) # No
    # train_inverse_dynamics_model(inv_dyn, s_s2_torch, a_torch, num_iters=100, lr=0.1) # No
    # train_inverse_dynamics_model(inv_dyn, s_s2_torch, a_torch, num_iters=100, lr=0.01) # No
    # train_inverse_dynamics_model(inv_dyn, s_s2_torch, a_torch, num_iters=100, lr=1) # No
    ##################

    #collect human demos
    demos = collect_human_demos(args.num_demos)

    #process demos
    obs, acs_true, obs2 = torchify_demos(demos)

    #predict actions
    state_trans = torch.cat((obs, obs2), dim = 1)
    outputs = inv_dyn(state_trans)
    _, acs = torch.max(outputs, 1)

    #train policy using predicted actions for states this should use your train_policy function from your BC implementation
    pi = PolicyNetwork()
    train_policy(obs, acs, pi, args.num_bc_iters)

    #evaluate learned policy
    evaluate_policy(pi, args.num_evals)

