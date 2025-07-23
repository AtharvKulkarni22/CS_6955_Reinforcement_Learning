


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# def run_bernoulli_bandit_simulation(n_arms=10, n_tasks=100, n_steps=1000, epsilons=[0, 0.01, 0.1]):
def run_bernoulli_bandit_simulation(n_arms=10, n_tasks=20000, n_steps=1000, epsilons=[0, 0.01, 0.1, 0.2, 0.3, 0.5]):
    avg_rewards = {epsilon: np.zeros(n_steps) for epsilon in epsilons}
    avg_rewards['UCB1'] = np.zeros(n_steps)

    # for task in range(n_tasks):
    for task in tqdm(range(n_tasks), desc='Running Bandit Tasks'):

        true_action_probs = np.random.uniform(0, 1, n_arms)  # Sample true probabilities for each arm

        for epsilon in epsilons:
            action_estimates = np.zeros(n_arms)
            action_counts = np.zeros(n_arms)
            rewards = []

            for step in range(n_steps):
                if np.random.rand() < epsilon:
                    action = np.random.choice(n_arms)  # Explore
                else:
                    action = np.argmax(action_estimates)  # Exploit

                reward = np.random.rand() < true_action_probs[action]
                action_counts[action] += 1
                action_estimates[action] += (reward - action_estimates[action]) / action_counts[action]
                rewards.append(reward)

            avg_rewards[epsilon] += np.array(rewards)
        
        # Implement UCB1
        action_estimates = np.zeros(n_arms)
        action_counts = np.ones(n_arms)  # Start with 1 to avoid division by zero
        rewards = []

        for step in range(n_steps):
            ucb_values = action_estimates + np.sqrt(2 * np.log(step + 1) / action_counts)
            action = np.argmax(ucb_values)
            reward = np.random.rand() < true_action_probs[action]
            action_counts[action] += 1
            action_estimates[action] += (reward - action_estimates[action]) / action_counts[action]
            rewards.append(reward)
        
        avg_rewards['UCB1'] += np.array(rewards)
    
    for key in avg_rewards:
        avg_rewards[key] /= n_tasks
    
    return avg_rewards

# def plot_results(avg_rewards):
#     plt.figure(figsize=(10, 6))
#     for label, rewards in avg_rewards.items():
#         plt.plot(rewards, label=f"{label}")
#     plt.xlabel("Steps")
#     plt.ylabel("Average Reward")
#     plt.title("Comparison of Epsilon-Greedy and UCB1 on Bernoulli Bandit")
#     plt.legend()
#     plt.grid()
#     plt.show()

# def plot_results(avg_rewards):
#     plt.figure(figsize=(10, 6))
#     for label, rewards in avg_rewards.items():
#         plt.plot(rewards, label=f"{label}")
#     plt.xlabel("Steps")
#     plt.ylabel("Average Reward")
#     plt.ylim(0.0, 1.1)  # Expanding the y-axis range
#     # plt.yticks(np.arange(0, 2.1, 0.2))  # Adding more distinct y-axis units
#     # plt.yticks(np.arange(0, 2.1, 0.1))  # Adding more distinct y-axis units
#     plt.title("Comparison of Epsilon-Greedy and UCB1 on Bernoulli Bandit")
#     plt.legend()
#     plt.grid()
#     plt.show()

def plot_results(avg_rewards):
    plt.figure(figsize=(10, 6))
    for label, rewards in avg_rewards.items():
        # plt.plot(rewards * 100, label=f"{label}")
        plt.plot(rewards, label=f"{label}")
    plt.xlabel("Plays")
    # plt.ylabel("% Optimal Action")
    plt.ylabel("Average Reward")    
    plt.ylim(0.0, 1.1)  # Expanding the y-axis range
    # plt.yticks(np.arange(0, 101, 10))  # Adding more distinct y-axis units
    plt.title("Comparison of Epsilon-Greedy and UCB1 on 10-armed Bernoulli bandit")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    avg_rewards = run_bernoulli_bandit_simulation()
    plot_results(avg_rewards)



