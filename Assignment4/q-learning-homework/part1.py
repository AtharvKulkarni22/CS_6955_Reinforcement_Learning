import gymnasium as gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

class BlackjackAgent:
    def __init__(
        self,
        env,
        learning_rate=0.01,
        initial_epsilon=1.0,
        epsilon_decay=1e-4,
        final_epsilon=0.1,
        discount_factor=0.95
    ):
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_errors = []

    def get_action(self, env, state):
        if np.random.rand() < self.epsilon:
            return env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[state]))

    def update(self, state, action, reward, next_state, done):
        future_q = 0 if done else np.max(self.q_values[next_state])
        td_error = reward + self.discount_factor * future_q - self.q_values[state][action]
        self.q_values[state][action] += self.lr * td_error
        self.training_errors.append(td_error)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

def evaluate_policy(agent, env, n_episodes=1000, use_learned=True):
    """
    Runs 'n_episodes' episodes and returns the average reward.
    If 'use_learned' is True, use the agent's greedy policy.
    Otherwise, take random actions.
    """
    total_reward = 0.0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            if use_learned:
                # Greedy action w.r.t. Q-values
                action = int(np.argmax(agent.q_values[state]))
            else:
                # Random action
                action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        total_reward += reward
    return total_reward / n_episodes

def main():
    env = gym.make("Blackjack-v1", sab=True)

    # Hyperparameters
    n_episodes = 100_000
    agent = BlackjackAgent(
        env=env,
        learning_rate=0.01,
        initial_epsilon=1.0,
        epsilon_decay=1e-4,
        final_epsilon=0.1,
        discount_factor=0.95
    )

    # Track per-episode training rewards
    all_rewards = []

    # For periodic evaluation
    # eval_every = 5000
    # n_eval_episodes = 1000
    
    eval_every = 1000
    n_eval_episodes = 1000

    # Lists to store evaluation data
    episodes_eval = []
    qlearning_eval = []
    random_eval = []

    # Training
    for episode in tqdm(range(n_episodes), desc="Training"):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.get_action(env, state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state, terminated or truncated)
            state = next_state
            episode_reward += reward
            done = terminated or truncated

        agent.decay_epsilon()
        all_rewards.append(episode_reward)

        # Periodically evaluate both Q-learning and random policy
        if (episode + 1) % eval_every == 0:
            q_avg = evaluate_policy(agent, env, n_episodes=n_eval_episodes, use_learned=True)
            r_avg = evaluate_policy(agent, env, n_episodes=n_eval_episodes, use_learned=False)

            episodes_eval.append(episode + 1)
            qlearning_eval.append(q_avg)
            random_eval.append(r_avg)

    # -------------------------------
    # Final Print of Average Rewards
    # -------------------------------
    learned_avg_reward = evaluate_policy(agent, env, n_episodes=1000, use_learned=True)
    random_avg_reward = evaluate_policy(agent, env, n_episodes=1000, use_learned=False)
    print(f"\nFinal Evaluation (1000 episodes):")
    print(f"Learned Q-Learning policy average reward: {learned_avg_reward:.3f}")
    print(f"Random policy average reward:          {random_avg_reward:.3f}")

    # -------------------------------
    # 1) Plot the Learning Curve (Q-Learning vs. Random)
    # -------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(episodes_eval, qlearning_eval, label="Q-Learning", marker="o")
    plt.plot(episodes_eval, random_eval, label="Random", marker="o")
    plt.xlabel("Training Episode")
    plt.ylabel(f"Avg Reward (over {n_eval_episodes} eval eps)")
    plt.title("Learning Curve Comparison for Q-Learning vs. Random")
    plt.grid(True)
    plt.legend()
    plt.show()

    # -------------------------------
    # 2) Plot Smoothed Rewards of Q-Learning During Training
    # -------------------------------
    all_rewards = np.array(all_rewards)
    window_size = 500
    cumsum_vec = np.cumsum(np.insert(all_rewards, 0, 0))
    moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

    plt.figure(figsize=(8,4))
    plt.plot(moving_avg)
    plt.title("Rolling Average for Q-Learning")
    plt.xlabel("Training Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

    env.close()

if __name__ == "__main__":
    main()
    
# Final Evaluation (1000 episodes):
# Learned Q-Learning policy average reward: -0.084
# Random policy average reward:          -0.346
