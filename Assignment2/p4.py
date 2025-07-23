import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def run_bandit_simulation(n_arms=10, n_tasks=2000, n_steps=1000, epsilons=[0, 0.01, 0.1, 0.2, 0.5], c=2):
    # Initialize storage for results
    avg_rewards = {epsilon: np.zeros(n_steps) for epsilon in epsilons}
    avg_rewards['UCB1'] = np.zeros(n_steps)

    # Simulate multiple tasks
    for task in tqdm(range(n_tasks), desc='Running Bandit Tasks'):
        # Generate true action values (mean rewards) for each arm
        true_action_values = np.random.normal(0, 1, n_arms)

        # Epsilon-Greedy Strategies
        for epsilon in epsilons:
            action_estimates = np.full(n_arms, 5.0)  # Optimistic initialization
            action_counts = np.zeros(n_arms)
            rewards = []

            for step in range(n_steps):
                if np.random.rand() < epsilon:
                    action = np.random.choice(n_arms)  # Explore
                else:
                    action = np.argmax(action_estimates)  # Exploit

                reward = np.random.normal(true_action_values[action], 1)
                action_counts[action] += 1
                action_estimates[action] += (reward - action_estimates[action]) / action_counts[action]
                rewards.append(reward)

            avg_rewards[epsilon] += np.array(rewards)

        # UCB1 Algorithm
        action_estimates = np.zeros(n_arms)
        action_counts = np.zeros(n_arms)
        rewards = []

        # Step 1: Initialization (play each arm once)
        for action in range(n_arms):
            reward = np.random.normal(true_action_values[action], 1)
            action_counts[action] += 1
            action_estimates[action] += (reward - action_estimates[action]) / action_counts[action]
            rewards.append(reward)

        # Step 2: Apply UCB1
        for step in range(n_arms, n_steps):
            total_counts = np.sum(action_counts)
            ucb_values = action_estimates + c * np.sqrt(np.log(total_counts) / (action_counts + 1e-5))  # Added small value to prevent division by zero
            action = np.argmax(ucb_values)

            reward = np.random.normal(true_action_values[action], 1)
            action_counts[action] += 1
            action_estimates[action] += (reward - action_estimates[action]) / action_counts[action]
            rewards.append(reward)

        avg_rewards['UCB1'] += np.array(rewards)

    # Normalize rewards by number of tasks
    for key in avg_rewards:
        avg_rewards[key] /= n_tasks

    return avg_rewards

def plot_results(avg_rewards):
    plt.figure(figsize=(10, 6))
    for label, rewards in avg_rewards.items():
        plt.plot(rewards, label=f"{label}")
    plt.xlabel("Plays")
    plt.ylabel("Average Reward")
    plt.title("Comparison of Epsilon-Greedy and UCB1 Strategies")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    avg_rewards = run_bandit_simulation()
    plot_results(avg_rewards)
