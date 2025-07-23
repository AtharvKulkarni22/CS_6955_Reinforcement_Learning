import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# def run_bandit_simulation(n_arms=10, n_tasks=200, n_steps=1000, epsilons=[0, 0.01, 0.1]):
def run_bandit_simulation(n_arms=10, n_tasks=2000, n_steps=1000, epsilons=[0, 0.01, 0.1]):
# def run_bandit_simulation(n_arms=10, n_tasks=200000, n_steps=1000, epsilons=[0, 0.01, 0.1]):
    # Initialize storage for results
    avg_rewards = {epsilon: np.zeros(n_steps) for epsilon in epsilons}

    # Simulate multiple tasks
    # for task in range(n_tasks):
    for task in tqdm(range(n_tasks), desc='Running Bandit Tasks'):

        # Generate true action values (mean rewards) for each arm
        true_action_values = np.random.normal(0, 1, n_arms)

        for epsilon in epsilons:
            # Initialize estimates and action counts
            action_estimates = np.zeros(n_arms)
            action_counts = np.zeros(n_arms)

            rewards = []

            for step in range(n_steps):
                # Choose action based on epsilon-greedy policy
                if np.random.rand() < epsilon:
                    action = np.random.choice(n_arms)  # Explore
                else:
                    action = np.argmax(action_estimates)  # Exploit

                # Get reward from the chosen action
                reward = np.random.normal(true_action_values[action], 1)

                # Update estimates using sample-average method
                action_counts[action] += 1
                action_estimates[action] += (reward - action_estimates[action]) / action_counts[action]

                rewards.append(reward)

            # Update average rewards across tasks
            avg_rewards[epsilon] += np.array(rewards)

    # Normalize rewards by number of tasks
    for epsilon in epsilons:
        avg_rewards[epsilon] /= n_tasks

    return avg_rewards

def plot_results(avg_rewards):
    plt.figure(figsize=(10, 6))
    for epsilon, rewards in avg_rewards.items():
        plt.plot(rewards, label=f"Epsilon = {epsilon}")
    plt.xlabel("Plays")
    plt.ylabel("Average Reward")
    # plt.ylim(0, )  # Expanding the y-axis range
    plt.title("Comparison of Epsilon-Greedy Strategies")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    avg_rewards = run_bandit_simulation()
    plot_results(avg_rewards)
