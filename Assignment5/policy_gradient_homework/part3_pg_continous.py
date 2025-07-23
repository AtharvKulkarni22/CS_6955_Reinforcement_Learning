"""
part3_pg_continuous.py

A simple policy gradient script using the MLPActorCritic class from core.py,
which supports both discrete and continuous Gymnasium environments.

We log training stats (epoch, avg_return, avg_ep_len) to a CSV file,
and we plot the learning curve (average return vs. epoch).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import gymnasium as gym
import csv
import os
import matplotlib.pyplot as plt
from core import MLPActorCritic
from gymnasium.spaces import Box, Discrete


def reward_to_go(rews):
    """
    Compute reward-to-go for a single trajectory:
    If rews = [r0, r1, ..., rT],
    then rtg[i] = r_i + r_{i+1} + ... + rT
    """
    n = len(rews)
    rtg = np.zeros_like(rews, dtype=np.float32)
    running_sum = 0.0
    for i in reversed(range(n)):
        running_sum += rews[i]
        rtg[i] = running_sum
    return rtg


def train_pg(
    env_name="CartPole-v1",
    hidden_sizes=(64, 64),
    lr=1e-3,
    epochs=50,
    batch_size=5000,
    reward_to_go_flag=False,
    render=False,
    seed=0,
    results_file=None,
):

    # 1) Create environment & set seed
    env = gym.make(env_name)
    env.reset(seed=seed)

    if isinstance(env.action_space, Box):
        print(f"[INFO] Environment {env_name}: continuous action space, shape = {env.action_space.shape}")
    elif isinstance(env.action_space, Discrete):
        print(f"[INFO] Environment {env_name}: discrete action space, n = {env.action_space.n}")
    else:
        raise TypeError(f"Unsupported action space type: {type(env.action_space)}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2) Build MLPActorCritic
    ac = MLPActorCritic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_sizes=hidden_sizes,
        activation=nn.Tanh
    )
    optimizer = Adam(ac.parameters(), lr=lr)

    # 3) Define the policy gradient objective
    def compute_loss(obs, act, weights):
        """
        obs: shape [N, obs_dim]
        act: shape [N] or [N, act_dim] (depending on discrete or continuous)
        weights: shape [N], either full return or reward-to-go
        """
        pi, logp = ac.pi(obs, act=act)
        return -(logp * weights).mean()

    # stats per epoch
    all_epoch_returns = []
    all_epoch_lengths = []

    for epoch in range(epochs):
        # 4) Collect a batch of data
        batch_obs = []
        batch_acts = []
        batch_weights = [] 
        batch_ep_returns = []
        batch_ep_lens = []

        obs, info = env.reset()
        ep_rews = []
        done = False

        while True:
            if render:
                env.render()

            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                a, v, logp = ac.step(obs_tensor)

            if isinstance(env.action_space, Discrete):
                a = int(a)  
            else: 
                if isinstance(a, torch.Tensor):
                    a = a.numpy()

            next_obs, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            # store data
            batch_obs.append(obs.copy())
            batch_acts.append(a)
            ep_rews.append(r)

            obs = next_obs

            if done:
                ep_ret = sum(ep_rews)
                ep_len = len(ep_rews)
                batch_ep_returns.append(ep_ret)
                batch_ep_lens.append(ep_len)

                # reward-to-go or full-episode
                if reward_to_go_flag:
                    rtgs = reward_to_go(ep_rews)
                    batch_weights.extend(rtgs)
                else:
                    batch_weights.extend([ep_ret]*ep_len)

                # Reset episode
                obs, info = env.reset()
                done = False
                ep_rews = []

                # if we reached the batch size, end collection
                if len(batch_obs) >= batch_size:
                    break

        # 5) Policy gradient update (one per epoch)
        optimizer.zero_grad()

        # Convert arrays to Tensors
        batch_obs_t = torch.as_tensor(batch_obs, dtype=torch.float32)
        if isinstance(env.action_space, Box):
            # continuous
            batch_acts_t = torch.as_tensor(batch_acts, dtype=torch.float32)
        else:
            # discrete
            batch_acts_t = torch.as_tensor(batch_acts, dtype=torch.int32)

        batch_weights_t = torch.as_tensor(batch_weights, dtype=torch.float32)

        loss = compute_loss(batch_obs_t, batch_acts_t, batch_weights_t)
        loss.backward()
        optimizer.step()

        # 6) Logging
        avg_ret = np.mean(batch_ep_returns)
        avg_len = np.mean(batch_ep_lens)
        all_epoch_returns.append(avg_ret)
        all_epoch_lengths.append(avg_len)

        print(f"Epoch {epoch} | Loss: {loss.item():.3f} | "
              f"AvgReturn: {avg_ret:.3f} | AvgEpLen: {avg_len:.2f}")

    env.close()

    if results_file is not None:
        # ensure directory exists
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "avg_return", "avg_length"])
            for i, (ret_i, len_i) in enumerate(zip(all_epoch_returns, all_epoch_lengths)):
                writer.writerow([i, ret_i, len_i])
        print(f"[INFO] Saved training results to {results_file}")

    return all_epoch_returns, all_epoch_lengths


#########################
#   MAIN DEMO EXAMPLES  #
#########################
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example 1: DISCRETE - CartPole
    cart_returns, cart_lengths = train_pg(
        env_name="CartPole-v1",         # Discrete environment
        hidden_sizes=(64, 64),
        lr=1e-4,
        epochs=500,
        batch_size=2000,
        reward_to_go_flag=False,
        seed=0,
        results_file=fr"policy_gradient_homework\results\part3\cartpole_v1_results.csv",
    )

    # Plot for CartPole
    plt.figure()
    plt.plot(cart_returns, label='CartPole-v1')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Return')
    plt.title("CartPole-v1 Learning Curve")
    plt.legend()
    plt.tight_layout()
    fname = fr"policy_gradient_homework\results\part3\CartPole-v1 Learning Curve.png"
    plt.savefig(fname)
    plt.show()

    # Example 2: CONTINUOUS - InvertedPendulum
    invp_returns, invp_lengths = train_pg(
        env_name="InvertedPendulum-v4",  # Continuous environment
        hidden_sizes=(64, 64),
        lr=1e-4,
        epochs=500,
        batch_size=4000,
        reward_to_go_flag=True,
        seed=1,
        results_file=fr"policy_gradient_homework\results\part3\invertedpendulum_v4_results.csv",
    )

    # Plot for InvertedPendulum
    plt.figure()
    plt.plot(invp_returns, label='InvertedPendulum-v4')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Return')
    plt.title("InvertedPendulum-v4 Learning Curve")
    plt.legend()
    plt.tight_layout()
    fname = fr"policy_gradient_homework\results\part3\InvertedPendulum-v4 Learning Curve.png"
    plt.savefig(fname)
    plt.show()
