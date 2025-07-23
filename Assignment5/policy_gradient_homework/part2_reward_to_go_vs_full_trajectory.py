import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import gymnasium as gym
import matplotlib.pyplot as plt
import csv
import os

SAVE_DIR = fr"policy_gradient_homework\results\part2"

##########################
#  MLP HELPER FUNCTION   #
##########################

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    """Build a feedforward neural network."""
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

################################
#  REWARD-TO-GO COMPUTATION    #
################################

def reward_to_go(rews):
    """Given a list of rewards from a single episode [r0, r1, ..., r_T],
       compute the reward-to-go for each time step:
         rtg[t] = r_t + r_{t+1} + ... + r_T
    """
    n = len(rews)
    rtgs = np.zeros_like(rews, dtype=np.float32)
    running_sum = 0
    # go backwards from last reward to first
    for i in reversed(range(n)):
        running_sum += rews[i]
        rtgs[i] = running_sum
    return rtgs

################################
#  POLICY GRADIENT TRAIN LOOP  #
################################

def train_pg(
    env_name='CartPole-v1',
    hidden_sizes=[32],
    lr=1e-2,
    epochs=50,
    batch_size=5000,
    render=False,
    reward_to_go_flag=False,
    seed=0
):
    """
    Train a policy using either
       1) Full-trajectory returns, or
       2) Reward-to-go returns,
    depending on reward_to_go_flag.

    Returns a dict containing lists for 'returns' and 'lengths' for each epoch.
    """

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Make environment
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # Build policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # Policy helper
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    def get_action(obs):
        return get_policy(obs).sample().item()

    # Loss function
    def compute_loss(obs_tensor, act_tensor, weights_tensor):
        """Simple policy gradient loss: - log π(a|s) * weight."""
        logp = get_policy(obs_tensor).log_prob(act_tensor)
        return -(logp * weights_tensor).mean()

    optimizer = Adam(logits_net.parameters(), lr=lr)

    output_stats = {
        'returns': [],   # average return in this epoch
        'lengths': [],   # average episode length in this epoch
    }

    def train_one_epoch():
        # Batch data
        batch_obs = []
        batch_acts = []
        batch_weights = []  # either full returns or reward-to-go
        batch_ep_returns = []
        batch_ep_lens = []

        # Reset episode-specific vars
        obs, info = env.reset(seed=seed)
        done = False
        ep_rews = []

        while True:
            # Collect experience
            batch_obs.append(obs.copy())
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                ep_ret = sum(ep_rews)
                ep_len = len(ep_rews)
                batch_ep_returns.append(ep_ret)
                batch_ep_lens.append(ep_len)

                # Either add full-trajectory return or reward-to-go for each step
                if reward_to_go_flag:
                    rtgs = reward_to_go(ep_rews)
                    batch_weights.extend(rtgs)
                else:
                    # the weight for each step is the total return from that episode
                    batch_weights.extend([ep_ret]*ep_len)

                # Reset
                obs, info = env.reset()
                done = False
                ep_rews = []

                # End collection when we have enough
                if len(batch_obs) > batch_size:
                    break

        # One gradient update
        optimizer.zero_grad()
        loss = compute_loss(
            obs_tensor=torch.as_tensor(batch_obs, dtype=torch.float32),
            act_tensor=torch.as_tensor(batch_acts, dtype=torch.int32),
            weights_tensor=torch.as_tensor(batch_weights, dtype=torch.float32)
        )
        loss.backward()
        optimizer.step()

        return loss.item(), batch_ep_returns, batch_ep_lens

    for epoch in range(epochs):
        loss_val, ep_returns, ep_lens = train_one_epoch()
        avg_ret = np.mean(ep_returns)
        avg_len = np.mean(ep_lens)
        output_stats['returns'].append(avg_ret)
        output_stats['lengths'].append(avg_len)

        if (epoch % 10 == 0) or (epoch == epochs - 1):
            print(f"Epoch {epoch} | Loss: {loss_val:.3f} | "
                  f"AvgRet: {avg_ret:.3f} | AvgLen: {avg_len:.2f}")

    env.close()
    return output_stats


##############################
#  EXPERIMENT COMPARISON     #
##############################

def run_and_plot(
    env_name="CartPole-v1",
    n_runs=3,
    epochs=50,
    batch_size=5000,
    seed=0,
    hidden_sizes=[32],
    lr=1e-2,
    results_dir=SAVE_DIR
):
    """
    Run policy gradient with (a) full-episode returns,
    and (b) reward-to-go returns, each for `n_runs` times.
    Save each run's data to CSV, then plot average return vs. total timesteps.
    """

    os.makedirs(results_dir, exist_ok=True)

    all_stats_full = []
    all_stats_rtg = []

    #########################################
    #   RUN FULL-TRAJECTORY PG n_runs TIMES #
    #########################################
    for run_i in range(n_runs):
        run_seed = seed + run_i * 100
        print(f"\n=== Run {run_i+1}/{n_runs} - FULL EP RETURN - seed={run_seed} ===")

        stats_full = train_pg(
            env_name=env_name,
            hidden_sizes=hidden_sizes,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            reward_to_go_flag=False,
            seed=run_seed,
        )
        all_stats_full.append(stats_full)

        # save stats to CSV
        csv_file = os.path.join(
            results_dir,
            f"results_full_seed{run_seed}.csv"
        )
        save_stats_to_csv(stats_full, csv_file)

    #########################################
    #   RUN REWARD-TO-GO PG n_runs TIMES    #
    #########################################
    for run_i in range(n_runs):
        run_seed = seed + run_i * 100 + 999
        print(f"\n=== Run {run_i+1}/{n_runs} - REWARD-TO-GO - seed={run_seed} ===")

        stats_rtg = train_pg(
            env_name=env_name,
            hidden_sizes=hidden_sizes,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            reward_to_go_flag=True,
            seed=run_seed,
        )
        all_stats_rtg.append(stats_rtg)

        # save stats to CSV
        csv_file = os.path.join(
            results_dir,
            f"results_rtg_seed{run_seed}.csv"
        )
        save_stats_to_csv(stats_rtg, csv_file)


    avg_returns_full = np.mean(
        [stats['returns'] for stats in all_stats_full], axis=0
    )
    avg_returns_rtg = np.mean(
        [stats['returns'] for stats in all_stats_rtg], axis=0
    )

    # For x-axis: each epoch collects ~batch_size steps
    x_vals = np.arange(1, epochs+1) * batch_size

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, avg_returns_full, label='Full-Trajectory Return')
    plt.plot(x_vals, avg_returns_rtg, label='Reward-to-Go')
    plt.xlabel("Timesteps")
    plt.ylabel("Average Return")
    plt.title("Policy Gradient: Full-Return vs. Reward-to-Go")
    plt.legend()
    plt.tight_layout()
    fname = fr"policy_gradient_homework\results\part2\Full Return vs Reward To Go learning_curve.png"
    plt.savefig(fname)
    if os.path.isfile(fname):
        print(f"Figure saved successfully as {fname} in {os.getcwd()}")
    else:
        print("Figure was not found after saving!")
    plt.show()


def save_stats_to_csv(stats, filename):
    n_epochs = len(stats['returns'])

    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        # header
        writer.writerow(["epoch", "avg_return", "avg_length"])
        # data rows
        for epoch_i in range(n_epochs):
            row = [
                epoch_i,
                stats['returns'][epoch_i],
                stats['lengths'][epoch_i]
            ]
            writer.writerow(row)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    run_and_plot(
        env_name="CartPole-v1",
        n_runs=3,
        epochs=50,
        batch_size=5000,
        seed=0,
        hidden_sizes=[32],
        lr=1e-2,
        results_dir=SAVE_DIR
    )
