import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import csv
import matplotlib.pyplot as plt

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False, save_plot=False):

    # make training environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make another environment for visual rendering (human mode)
    render_env = gym.make(env_name, render_mode="human")

    # build core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # helper functions for policy
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    def get_action(obs):
        return get_policy(obs).sample().item()

    # policy gradient loss function
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    optimizer = Adam(logits_net.parameters(), lr=lr)

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []

        obs, info = env.reset()
        done = False
        ep_rews = []

        finished_rendering_this_epoch = False

        # collect experience in batches
        while True:
            if (not finished_rendering_this_epoch) and render:
                env.render()

            batch_obs.append(obs.copy())
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                batch_weights += [ep_ret] * ep_len

                obs, info = env.reset()
                done, ep_rews = False, []
                finished_rendering_this_epoch = True

                if len(batch_obs) > batch_size:
                    break

        # gradient step
        optimizer.zero_grad()
        batch_loss = compute_loss(
            obs=torch.as_tensor(batch_obs, dtype=torch.float32),
            act=torch.as_tensor(batch_acts, dtype=torch.int32),
            weights=torch.as_tensor(batch_weights, dtype=torch.float32)
        )
        batch_loss.backward()
        optimizer.step()

        return batch_loss.item(), batch_rets, batch_lens

    # Function to render a single test episode in render_env
    def render_one_episode():
        """Run one episode with the current policy in render_env."""
        obs, info = render_env.reset()
        done = False
        ep_rewards = []

        # Inference mode: don't track gradients
        with torch.no_grad():
            while not done:
                action = get_action(torch.as_tensor(obs, dtype=torch.float32))
                obs, reward, terminated, truncated, info = render_env.step(action)
                ep_rewards.append(reward)
                done = terminated or truncated

        return sum(ep_rewards)

    # Prepare a list to store training statistics
    training_stats = []

    # main training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        avg_ret, avg_len = np.mean(batch_rets), np.mean(batch_lens)

        print(f"epoch: {i:3d} \t loss: {batch_loss:.3f} \t "
              f"return: {avg_ret:.3f} \t ep_len: {avg_len:.3f}")

        # Test the current policy
        test_return = render_one_episode()
        print(f"   [Test run] return: {test_return:.2f}")

        # Record stats for saving
        training_stats.append({
            'epoch': i,
            'loss': batch_loss,
            'avg_return': avg_ret,
            'avg_ep_len': avg_len,
            'test_return': test_return
        })

    csv_filename = fr"policy_gradient_homework\results\part1b\training_results.csv"
    save_csv(training_stats, csv_filename)
    print(f"Training results have been saved to {csv_filename}")

    plot_training(training_stats, fr"policy_gradient_homework\results\part1b\training_plot.png")

    env.close()
    render_env.close()

def save_csv(stats_list, filename):
    """Save a list of dicts (with consistent keys) to a CSV file."""
    import csv
    if not stats_list:
        print("No stats to save!")
        return

    fieldnames = list(stats_list[0].keys())

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats_list:
            writer.writerow(row)

def plot_training(stats_list, filename):
    """Plot average return (during training) vs. epoch."""
    epochs = [s['epoch'] for s in stats_list]
    avg_returns = [s['avg_return'] for s in stats_list]

    plt.figure()
    plt.plot(epochs, avg_returns, label='Average Return per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Return')
    plt.title('Training Progress')
    plt.legend()
    plt.tight_layout()

    # Save the figure to file
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()

    print('\nUsing simplest formulation of policy gradient with 1-episode rendering after each epoch.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
