import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------
# Environment Setup for Training
# ---------------------------
env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode="human")

# Set up matplotlib for live plotting
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# Force CPU-only execution
device = torch.device("cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    """Experience replay buffer"""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a random batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """Deep Q-Network model"""
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

#######################################
# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
#######################################

# Get number of actions and observations from the training environment
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

# Experience replay memory
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# Lists to store metrics for live plotting
episode_durations = []
episode_rewards = []

def plot_metrics(show_result=False):
    # Plot Episode Duration
    plt.figure(1)
    plt.clf()
    plt.title('Result - Duration' if show_result else 'Training... (Duration)')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.plot(durations_t.numpy(), color='blue', label='Episode Duration')

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), color='green', linestyle='--', label='Moving Avg (100 episodes)')

    plt.legend()
    plt.pause(0.001)
    
    # Plot Episode Reward
    plt.figure(2)
    plt.clf()
    plt.title('Result - Reward' if show_result else 'Training... (Reward)')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    plt.plot(rewards_t.numpy(), color='orange', label='Cumulative Reward')

    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), color='red', linestyle='--', label='Moving Avg (100 episodes)')

    plt.legend()
    plt.pause(0.001)

    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# ---------------------------
# Training Loop
# ---------------------------
num_episodes = 500

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    episode_reward = 0  # Cumulative reward for this episode

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward_tensor = torch.tensor([reward], device=device)

        episode_reward += reward

        done = terminated or truncated
        next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        memory.push(state, action, next_state, reward_tensor)
        state = next_state

        optimize_model()

        # Soft update of the target network
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            episode_rewards.append(episode_reward)
            plot_metrics()  # Update live plots
            break

print('Training Complete')
plot_metrics(show_result=True)
plt.figure(1)
plt.savefig("training_duration_plot.png")  # Save duration plot

plt.figure(2)
plt.savefig("training_reward_plot.png")  # Save reward plot
plt.ioff()
plt.pause(5)
plt.close('all')

torch.save(policy_net.state_dict(), "lunar_lander_dqn_v3.pth")

# ---------------------------
env.close()

# ---------------------------
# Reinitialize environment for evaluation (without rendering)
env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode=None)

def evaluate_policy(policy, num_episodes=100):
    """Evaluate a given policy over num_episodes.
       If policy is 'random', actions are sampled randomly.
       Otherwise, 'policy' is assumed to be a neural network model.
    """
    total_rewards = []
    total_durations = []
    for i in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0
        episode_duration = 0
        done = False
        while not done:
            # Use random actions if policy is 'random'
            if policy == 'random':
                action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
            else:
                with torch.no_grad():
                    action = policy(state).max(1).indices.view(1, 1)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward
            episode_duration += 1
            done = terminated or truncated
            if not done:
                state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        total_rewards.append(episode_reward)
        total_durations.append(episode_duration)
    avg_reward = sum(total_rewards) / len(total_rewards)
    avg_duration = sum(total_durations) / len(total_durations)
    return avg_reward, avg_duration, total_rewards, total_durations

# Evaluate the learned policy
learned_avg_reward, learned_avg_duration, learned_rewards, learned_durations = evaluate_policy(policy_net, num_episodes=100)
# Evaluate a random policy
random_avg_reward, random_avg_duration, random_rewards, random_durations = evaluate_policy('random', num_episodes=100)

results = {
    "learned_policy": {
         "average_reward": learned_avg_reward,
         "average_duration": learned_avg_duration,
         "all_rewards": learned_rewards,
         "all_durations": learned_durations
    },
    "random_policy": {
         "average_reward": random_avg_reward,
         "average_duration": random_avg_duration,
         "all_rewards": random_rewards,
         "all_durations": random_durations
    }
}

with open("performance_lunar_lander.json", "w") as f:
    json.dump(results, f, indent=4)

print("Evaluation results:")
print(json.dumps(results, indent=4))

# ---------------------------
# Reinitialize environment for demonstration testing (with rendering)
env.close()
env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode="human")

print("Testing trained agent...")
state, _ = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

for _ in range(200):
    env.render()
    action = policy_net(state).max(1).indices.view(1, 1)
    observation, _, terminated, truncated, _ = env.step(action.item())
    if terminated or truncated:
        break
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

env.close()
