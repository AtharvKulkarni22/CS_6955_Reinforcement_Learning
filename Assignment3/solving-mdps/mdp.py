import numpy as np


class MDP:
    def __init__(self, num_rows, num_cols, terminals, rewards, gamma, noise = 0.1):

        """
        Markov Decision Processes (MDP):
        num_rows: number of row in a environment
        num_cols: number of columns in environment
        terminals: terminal states (sink states)
        noise: with probability 2*noise the agent but will move perpendicular, 
                e.g. if taking up action, then the agent has probability noise of going right and probability noise of going left.
        """
        self.gamma = gamma
        self.num_states = num_rows * num_cols 
        self.num_actions = 4  #up:0, down:1, left:2, right:3
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.terminals = terminals #vector of sink states that are modelled as having no transitions out of them
        self.rewards = rewards  # vector of rewards, one for each state going left to right, top to bottom
        
        #initialize transitions given desired noise level
        self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.init_transition_probabilities(noise)


    def init_transition_probabilities(self, noise):
        # 0: up, 1 : down, 2:left, 3:right

        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3
        # going UP
        for s in range(self.num_states):

            # possibility of going foward

            if s >= self.num_cols:
                self.transitions[s][UP][s - self.num_cols] = 1.0 - (2 * noise)
            else:
                self.transitions[s][UP][s] = 1.0 - (2 * noise)

            # possibility of going left

            if s % self.num_cols == 0:
                self.transitions[s][UP][s] = noise
            else:
                self.transitions[s][UP][s - 1] = noise

            # possibility of going right

            if s % self.num_cols < self.num_cols - 1:
                self.transitions[s][UP][s + 1] = noise
            else:
                self.transitions[s][UP][s] = noise

            # special case top left corner

            if s < self.num_cols and s % self.num_cols == 0.0:
                self.transitions[s][UP][s] = 1.0 - noise
            elif s < self.num_cols and s % self.num_cols == self.num_cols - 1:
                self.transitions[s][UP][s] = 1.0 - noise

        # going down
        for s in range(self.num_states):

            # self.num_rows = gridHeight
            # self.num_cols = gridwidth

            # possibility of going down
            if s < (self.num_rows - 1) * self.num_cols:
                self.transitions[s][DOWN][s + self.num_cols] = 1.0 - (2 * noise)
            else:
                self.transitions[s][DOWN][s] = 1.0 - (2 * noise)

            # possibility of going left
            if s % self.num_cols == 0:
                self.transitions[s][DOWN][s] = noise
            else:
                self.transitions[s][DOWN][s - 1] = noise

            # possibility of going right
            if s % self.num_cols < self.num_cols - 1:
                self.transitions[s][DOWN][s + 1] = noise
            else:
                self.transitions[s][DOWN][s] = noise

            # checking bottom right corner
            if s >= (self.num_rows - 1) * self.num_cols and s % self.num_cols == 0:
                self.transitions[s][DOWN][s] = 1.0 - noise
            elif (
                s >= (self.num_rows - 1) * self.num_cols
                and s % self.num_cols == self.num_cols - 1
            ):
                self.transitions[s][DOWN][s] = 1.0 - noise

        # going left
        # self.num_rows = gridHeight
        # self.num_cols = gridwidth
        for s in range(self.num_states):
            # possibility of going left

            if s % self.num_cols > 0:
                self.transitions[s][LEFT][s - 1] = 1.0 - (2 * noise)
            else:
                self.transitions[s][LEFT][s] = 1.0 - (2 * noise)

            # possibility of going up

            if s >= self.num_cols:
                self.transitions[s][LEFT][s - self.num_cols] = noise
            else:
                self.transitions[s][LEFT][s] = noise

            # possiblity of going down
            if s < (self.num_rows - 1) * self.num_cols:
                self.transitions[s][LEFT][s + self.num_cols] = noise
            else:
                self.transitions[s][LEFT][s] = noise

            # check  top left corner
            if s < self.num_cols and s % self.num_cols == 0:
                self.transitions[s][LEFT][s] = 1.0 - noise
            elif s >= (self.num_rows - 1) * self.num_cols and s % self.num_cols == 0:
                self.transitions[s][LEFT][s] = 1 - noise

        # going right
        for s in range(self.num_states):

            if s % self.num_cols < self.num_cols - 1:
                self.transitions[s][RIGHT][s + 1] = 1.0 - (2 * noise)
            else:
                self.transitions[s][RIGHT][s] = 1.0 - (2 * noise)

            # possibility of going up

            if s >= self.num_cols:
                self.transitions[s][RIGHT][s - self.num_cols] = noise
            else:
                self.transitions[s][RIGHT][s] = noise

            # possibility of going down

            if s < (self.num_rows - 1) * self.num_cols:
                self.transitions[s][RIGHT][s + self.num_cols] = noise
            else:
                self.transitions[s][RIGHT][s] = noise

            # check top right corner
            if (s < self.num_cols) and (s % self.num_cols == self.num_cols - 1):
                self.transitions[s][RIGHT][s] = 1 - noise
            # check bottom rihgt corner case
            elif (
                s >= (self.num_rows - 1) * self.num_cols
                and s % self.num_cols == self.num_cols - 1
            ):
                self.transitions[s][RIGHT][s] = 1.0 - noise

        for s in range(self.num_states):
            if s in self.terminals:
                for a in range(self.num_actions):
                    for s2 in range(self.num_states):
                        self.transitions[s][a][s2] = 0.0

    
    def set_rewards(self, _rewards):
        self.rewards = _rewards

    def set_gamma(self, gamma):
        assert(gamma < 1.0 and gamma > 0.0)
        self.gamma = gamma



def gen_simple_world():
    #four features, red (-1), blue (+5), white (0), yellow (+1)
    gamma = 0.95
    #create state features for a 2x2 grid (really just an array, but I'm formating it to look like the grid world)
    noise = 0.1 #noise in transitions applied in directions perpendicular to chosen action
    # e.g., noise = 0.1 means a 10% chance of going left and 10% chance of going right if action is up
    n_rows = 4
    n_cols = 4
    terminals = [0,1,5] #top left state is terminal sink state 
    rewards = [+1, -1,  0, 0,
                0, -1,  0, 0,
                0,  0,  0, 0,
                0,  0,  0, 0]
    env = MDP(n_rows, n_cols, terminals, rewards, gamma, noise)
    return env

def gen_simple_world2():
    """
    Modified MDP with different rewards and noise level
    """
    gamma = 0.95
    noise = 0.2  # Increased noise for more stochastic transitions
    n_rows = 4
    n_cols = 4
    terminals = [0, 3, 12]  # Changed terminal states
    rewards = [+1, -1,  0, +5,
                0, -1,  0,  0,
                0,  0,  0,  0,
               -1,  0, +2,  0]  # Different reward structure
    env = MDP(n_rows, n_cols, terminals, rewards, gamma, noise)
    return env

def gen_simple_world3():
    """
    Another variation with high penalties and low rewards
    """
    gamma = 0.85  # Slightly reduced gamma to encourage convergence
    noise = 0.1  # Increased noise for more varied transitions
    n_rows = 4
    n_cols = 4
    terminals = [0, 15]  # Only two terminal states

    rewards = [+5, -5,  -2, -2,
               -2, -2,  -2, -2,
               -2, -2,  -2, -100,
               -2, -2,  -2, +10]  # Adjusted penalties to help convergence
    env = MDP(n_rows, n_cols, terminals, rewards, gamma, noise)
    return env



def gen_simple_world4():
    """
    A larger, more complex 10x10 gridworld with varied rewards and obstacles.
    """
    gamma = 0.9  # Slightly higher gamma for long-term planning
    noise = 0.15  # More stochasticity in movements
    n_rows = 10
    n_cols = 10
    terminals = [0, 99]  # Start and end as terminal states
    rewards = np.full((n_rows, n_cols), -1)  # Default reward is -1 (small penalty for each step)
    rewards[0, 0] = 5  # High reward for the start state
    rewards[9, 9] = 10  # High reward for the goal state
    rewards[5, 5] = -100  # A large obstacle with a huge penalty
    rewards[3, 3] = -5  # Another smaller penalty zone
    rewards[7, 2] = 2  # A small reward to encourage detours
    rewards[8, 8] = 5  # Medium reward near the goal
    rewards = rewards.flatten()
    
    env = MDP(n_rows, n_cols, terminals, rewards, gamma, noise)
    return env