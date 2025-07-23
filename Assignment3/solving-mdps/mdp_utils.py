from mdp import MDP
import numpy as np
import math
import copy


def get_random_policy(env):
    n = env.num_states
    a = env.num_actions
    policy =  np.random.randint(0, a, size=n) 
    return policy

def action_to_string(act, UP=0, DOWN=1, LEFT=2, RIGHT=3):
    if act == UP:
        return "^"
    elif act == DOWN:
        return "v"
    elif act == LEFT:
        return "<"
    elif act == RIGHT:
        return ">"
    else:
        return NotImplementedError




def visualize_policy(policy, env):
    """
  prints the policy of the MDP using text arrows and uses a '.' for terminals
  """
    count = 0
    for r in range(env.num_rows):
        policy_row = ""
        for c in range(env.num_cols):
            if count in env.terminals:
                policy_row += ".\t"    
            else:
                policy_row += action_to_string(policy[count]) + "\t"
            count += 1
        print(policy_row)


def print_array_as_grid(array_values, env):
    """
  Prints array as a grid
  :param array_values:
  :param env:
  :return:
  """
    count = 0
    for r in range(env.num_rows):
        print_row = ""
        for c in range(env.num_cols):
            print_row += "{:.2f}\t".format(array_values[count])
            count += 1
        print(print_row)


def value_iteration(env, epsilon=0.0001):
    """
    Run value iteration to find optimal values for each state
    :param env: the MDP
    :param epsilon: numerical precision for values to determine stopping condition
    :return: the vector of optimal values for each state in the MDP 
    """
    n = env.num_states
    V = np.zeros(n)  
    while True:
        delta = 0 
        new_V = np.copy(V)
        for s in range(n):
            if s in env.terminals:
                continue  
            
            action_values = []
            for a in range(env.num_actions):
                value = sum(env.transitions[s, a, s2] * (env.rewards[s2] + env.gamma * V[s2]) for s2 in range(n))
                action_values.append(value)
            new_V[s] = max(action_values)
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        if delta < epsilon:
            break
    return V

def extract_optimal_policy(V, env):
    """ 
    Perform a one step lookahead to find optimal policy
    :param V: precomputed values from value iteration
    :param env: the MDP
    :return: the optimal policy
    """
    n = env.num_states
    optimal_policy = np.zeros(n, dtype=int)
    for s in range(n):
        if s in env.terminals:
            continue
        action_values = []
        for a in range(env.num_actions):
            value = sum(env.transitions[s, a, s2] * (env.rewards[s2] + env.gamma * V[s2]) for s2 in range(n))
            action_values.append(value)
        optimal_policy[s] = np.argmax(action_values)
    return optimal_policy

def policy_evaluation(policy, env, epsilon=0.0001):
    """
    Evalute the policy and compute values in each state when executing the policy in the mdp
    :param policy: the policy to evaluate in the mdp
    :param env: markov decision process where we evaluate the policy
    :param epsilon: numerical precision desired
    :return: values of policy under mdp
    """
    n = env.num_states
    V = np.zeros(n)  
    while True:
        delta = 0
        new_V = np.copy(V)
        for s in range(n):
            if s in env.terminals:
                continue  
            a = policy[s]
            new_V[s] = sum(env.transitions[s, a, s2] * (env.rewards[s2] + env.gamma * V[s2]) for s2 in range(n))
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        if delta < epsilon:
            break
    return V


def policy_iteration(env, epsilon=0.0001):
    """
    Run policy iteration to find optimal values and policy
    :param env: markov decision process where we evaluate the policy
    :param epsilon: numerical precision desired
    :return: values of policy under mdp
    """
    n = env.num_states
    policy = np.random.randint(0, env.num_actions, size=n) 
    while True:
        V = policy_evaluation(policy, env, epsilon)
        policy_stable = True
        for s in range(n):
            if s in env.terminals:
                continue
            action_values = [
                sum(env.transitions[s, a, s2] * (env.rewards[s2] + env.gamma * V[s2]) for s2 in range(n))
                for a in range(env.num_actions)
            ]
            best_action = np.argmax(action_values)
            if best_action != policy[s]:  
                policy[s] = best_action
                policy_stable = False
        if policy_stable:
            break
    return policy, V

