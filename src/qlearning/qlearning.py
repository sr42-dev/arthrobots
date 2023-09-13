import sys
sys.path.append("..")
import numpy as np
# from env.grid_world import GridWorld
# from algorithms.temporal_difference import qlearning
# from utils.plots import plot_gridworld
np.random.seed(1)


# from utils.helper_functions import row_col_to_seq
# from utils.helper_functions import seq_to_col_row
from math import floor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from utils.helper_functions import create_policy_direction_arrays

def add_policy(model, policy):

    if policy is not None:
        # define the gridworld
        X = np.arange(0, model.num_cols, 1)
        Y = np.arange(0, model.num_rows, 1)

        # define the policy direction arrows
        U, V = create_policy_direction_arrays(model, policy)
        # remove the obstructions and final state arrows
        ra = model.goal_states
        U[ra[:, 0], ra[:, 1]] = np.nan
        V[ra[:, 0], ra[:, 1]] = np.nan
        if model.obs_states is not None:
            ra = model.obs_states
            U[ra[:, 0], ra[:, 1]] = np.nan
            V[ra[:, 0], ra[:, 1]] = np.nan
        if model.restart_states is not None:
            ra = model.restart_states
            U[ra[:, 0], ra[:, 1]] = np.nan
            V[ra[:, 0], ra[:, 1]] = np.nan

        plt.quiver(X, Y, U, V, zorder=10, label="Policy")

def row_col_to_seq(row_col, num_cols):
    return row_col[:,0] * num_cols + row_col[:,1]

def seq_to_col_row(seq, num_cols):
    r = floor(seq / num_cols)
    c = seq - r * num_cols
    return np.array([[r, c]])

def create_policy_direction_arrays(model, policy):
    """
     define the policy directions
     0 - up    [0, 1]
     1 - down  [0, -1]
     2 - left  [-1, 0]
     3 - right [1, 0]
    :param policy:
    :return:
    """
    # action options
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    # intitialize direction arrays
    U = np.zeros((model.num_rows, model.num_cols))
    V = np.zeros((model.num_rows, model.num_cols))

    for state in range(model.num_states-1):
        # get index of the state
        i = tuple(seq_to_col_row(state, model.num_cols)[0])
        # define the arrow direction
        if policy[state] == UP:
            U[i] = 0
            V[i] = 0.5
        elif policy[state] == DOWN:
            U[i] = 0
            V[i] = -0.5
        elif policy[state] == LEFT:
            U[i] = -0.5
            V[i] = 0
        elif policy[state] == RIGHT:
            U[i] = 0.5
            V[i] = 0

    return U, V


def plot_gridworld(model, value_function=None, policy=None, state_counts=None, title=None, path=None):
    """
    Plots the grid world solution.

    Parameters
    ----------
    model : python object
        Holds information about the environment to solve
        such as the reward structure and the transition dynamics.

    value_function : numpy array of shape (N, 1)
        Value function of the environment where N is the number
        of states in the environment.

    policy : numpy array of shape (N, 1)
        Optimal policy of the environment.

    title : string
        Title of the plot. Defaults to None.

    path : string
        Path to save image. Defaults to None.
    """

    if value_function is not None and state_counts is not None:
        raise Exception("Must supple either value function or state_counts, not both!")

    fig, ax = plt.subplots()

    # add features to grid world
    if value_function is not None:
        add_value_function(model, value_function, "Value function")
    elif state_counts is not None:
        add_value_function(model, state_counts, "State counts")
    elif value_function is None and state_counts is None:
        add_value_function(model, value_function, "Value function")

    add_patches(model, ax)
    add_policy(model, policy)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
               fancybox=True, shadow=True, ncol=3)
    if title is not None:
        plt.title(title, fontdict=None, loc='center')
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.show()

def add_value_function(model, value_function, name):

    if value_function is not None:
        # colobar max and min
        vmin = np.min(value_function)
        vmax = np.max(value_function)
        # reshape and set obstructed states to low value
        val = value_function[:-1, 0].reshape(model.num_rows, model.num_cols)
        if model.obs_states is not None:
            index = model.obs_states
            val[index[:, 0], index[:, 1]] = -100
        plt.imshow(val, vmin=vmin, vmax=vmax, zorder=0)
        plt.colorbar(label=name)
    else:
        val = np.zeros((model.num_rows, model.num_cols))
        plt.imshow(val, zorder=0)
        plt.yticks(np.arange(-0.5, model.num_rows+0.5, step=1))
        plt.xticks(np.arange(-0.5, model.num_cols+0.5, step=1))
        plt.grid()
        plt.colorbar(label=name)

def add_patches(model, ax):

    start = patches.Circle(tuple(np.flip(model.start_state[0])), 0.2, linewidth=1,
                           edgecolor='b', facecolor='b', zorder=1, label="Start")
    ax.add_patch(start)

    for i in range(model.goal_states.shape[0]):
        end = patches.RegularPolygon(tuple(np.flip(model.goal_states[i, :])), numVertices=5,
                                     radius=0.25, orientation=np.pi, edgecolor='g', zorder=1,
                                     facecolor='g', label="Goal" if i == 0 else None)
        ax.add_patch(end)

    # obstructed states patches
    if model.obs_states is not None:
        for i in range(model.obs_states.shape[0]):
            obstructed = patches.Rectangle(tuple(np.flip(model.obs_states[i, :]) - 0.35), 0.7, 0.7,
                                           linewidth=1, edgecolor='orange', facecolor='orange', zorder=1,
                                           label="Obstructed" if i == 0 else None)
            ax.add_patch(obstructed)

    if model.bad_states is not None:
        for i in range(model.bad_states.shape[0]):
            bad = patches.Wedge(tuple(np.flip(model.bad_states[i, :])), 0.2, 40, -40,
                                linewidth=1, edgecolor='r', facecolor='r', zorder=1,
                                label="Bad state" if i == 0 else None)
            ax.add_patch(bad)

    if model.restart_states is not None:
        for i in range(model.restart_states.shape[0]):
            restart = patches.Wedge(tuple(np.flip(model.restart_states[i, :])), 0.2, 40, -40,
                                    linewidth=1, edgecolor='y', facecolor='y', zorder=1,
                                    label="Restart state" if i == 0 else None)
            ax.add_patch(restart)





class GridWorld:
    """
    Creates a gridworld object to pass to an RL algorithm.

    Parameters
    ----------
    num_rows : int
        The number of rows in the gridworld.

    num_cols : int
        The number of cols in the gridworld.

    start_state : numpy array of shape (1, 2), np.array([[row, col]])
        The start state of the gridworld (can only be one start state)

    goal_states : numpy arrany of shape (n, 2)
        The goal states for the gridworld where n is the number of goal
        states.
    """
    def __init__(self, num_rows, num_cols, start_state, goal_states):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.start_state = start_state
        self.goal_states = goal_states
        self.obs_states = None
        self.bad_states = None
        self.num_bad_states = 0
        self.p_good_trans = None
        self.bias = None
        self.r_step = None
        self.r_goal = None
        self.r_dead = None
        self.gamma = 1 # default is no discounting

    def add_obstructions(self, obstructed_states=None, bad_states=None, restart_states=None):
        """
        Add obstructions to the grid world.

        Obstructed states: walls that prohibit the agent from entering that state.

        Bad states: states that incur a greater penalty than a normal step.

        Restart states: states that incur a high penalty and transition the agent
                        back to the start state (but do not end the episode).

        Parameters
        ----------
        obstructed_states : numpy array of shape (n, 2)
            States the agent cannot enter where n is the number of obstructed states
            and the two columns are the row and col position of the obstructed state.

        bad_states: numpy array of shape (n, 2)
            States in which the agent incurs high penalty where n is the number of bad
            states and the two columns are the row and col position of the bad state.

        restart_states: numpy array of shape (n, 2)
            States in which the agent incurs high penalty and transitions to the start
            state where n is the number of restart states and the two columns are the
            row and col position of the restart state.
        """
        self.obs_states = obstructed_states
        self.bad_states = bad_states
        if bad_states is not None:
            self.num_bad_states = bad_states.shape[0]
        else:
            self.num_bad_states = 0
        self.restart_states = restart_states
        if restart_states is not None:
            self.num_restart_states = restart_states.shape[0]
        else:
            self.num_restart_states = 0

    def add_transition_probability(self, p_good_transition, bias):
        """
        Add transition probabilities to the grid world.

        p_good_transition is the probability that the agent successfully
        executes the intended action. The action is then incorrectly executed
        with probability 1 - p_good_transition and in tis case the agent
        transitions to the left of the intended transition with probability
        (1 - p_good_transition) * bias and to the right with probability
        (1 - p_good_transition) * (1 - bias).

        Parameters
        ----------
        p_good_transition : float (in the interval [0,1])
             The probability that the agents attempted transition is successful.

        bias : float (in the interval [0,1])
            The probability that the agent transitions left or right of the
            intended transition if the intended transition is not successful.
        """
        self.p_good_trans = p_good_transition
        self.bias = bias

    def add_rewards(self, step_reward, goal_reward, bad_state_reward=None, restart_state_reward = None):
        """
        Define which states incur which rewards.

        Parameters
        ----------
        step_reward : float
            The reward for each step taken by the agent in the grid world.
            Typically a negative value (e.g. -1).

        goal_reward : float
            The reward given to the agent for reaching the goal state.
            Typically a middle range positive value (e.g. 10)

        bad_state_reward : float
            The reward given to the agent for transitioning to a bad state.
            Typically a middle range negative value (e.g. -6)

        restart_state_reward : float
            The reward given to the agent for transitioning to a restart state.
            Typically a large negative value (e.g. -100)
        """
        self.r_step = step_reward
        self.r_goal = goal_reward
        self.r_bad = bad_state_reward
        self.r_restart = restart_state_reward

    def add_discount(self, discount):
        """
        Discount rewards so that recent rewards carry more weight than past rewards.

        Parameters
        ----------
        discount : float (in the interval [0, 1])
            The discount factor.
        """
        self.gamma = discount

    def create_gridworld(self):
        """
        Create the grid world with the specified parameters.

        Returns
        -------
        self : class object
            Holds information about the environment to solve
            such as the reward structure and the transition dynamics.
        """
        self.num_actions = 4
        self.num_states = self.num_cols * self.num_rows + 1
        self.start_state_seq = row_col_to_seq(self.start_state, self.num_cols)
        self.goal_states_seq = row_col_to_seq(self.goal_states, self.num_cols)

        # rewards structure
        self.R = self.r_step * np.ones((self.num_states, 1))
        self.R[self.num_states-1] = 0
        self.R[self.goal_states_seq] = self.r_goal
        for i in range(self.num_bad_states):
            if self.r_bad is None:
                raise Exception("Bad state specified but no reward is given")
            bad_state = row_col_to_seq(self.bad_states[i,:].reshape(1,-1), self.num_cols)
            self.R[bad_state, :] = self.r_bad
        for i in range(self.num_restart_states):
            if self.r_restart is None:
                raise Exception("Restart state specified but no reward is given")
            restart_state = row_col_to_seq(self.restart_states[i,:].reshape(1,-1), self.num_cols)
            self.R[restart_state, :] = self.r_restart

        # probability model
        if self.p_good_trans == None:
            raise Exception("Must assign probability and bias terms via the add_transition_probability method.")

        self.P = np.zeros((self.num_states,self.num_states,self.num_actions))
        for action in range(self.num_actions):
            for state in range(self.num_states):

                # check if state is the fictional end state - self transition
                if state == self.num_states-1:
                    self.P[state, state, action] = 1
                    continue

                # check if the state is the goal state or an obstructed state - transition to end
                row_col = seq_to_col_row(state, self.num_cols)
                if self.obs_states is not None:
                    end_states = np.vstack((self.obs_states, self.goal_states))
                else:
                    end_states = self.goal_states

                if any(np.sum(np.abs(end_states-row_col), 1) == 0):
                    self.P[state, self.num_states-1, action] = 1

                # else consider stochastic effects of action
                else:
                    for dir in range(-1,2,1):
                        direction = self._get_direction(action, dir)
                        next_state = self._get_state(state, direction)
                        if dir == 0:
                            prob = self.p_good_trans
                        elif dir == -1:
                            prob = (1 - self.p_good_trans)*(self.bias)
                        elif dir == 1:
                            prob = (1 - self.p_good_trans)*(1-self.bias)

                        self.P[state, next_state, action] += prob

                # make restart states transition back to the start state with
                # probability 1
                if self.restart_states is not None:
                    if any(np.sum(np.abs(self.restart_states-row_col),1)==0):
                        next_state = row_col_to_seq(self.start_state, self.num_cols)
                        self.P[state,:,:] = 0
                        self.P[state,next_state,:] = 1
        return self

    def _get_direction(self, action, direction):
        """
        Takes is a direction and an action and returns a new direction.

        Parameters
        ----------
        action : int
            The current action 0, 1, 2, 3 for gridworld.

        direction : int
            Either -1, 0, 1.

        Returns
        -------
        direction : int
            Value either 0, 1, 2, 3.
        """
        left = [2,3,1,0]
        right = [3,2,0,1]
        if direction == 0:
            new_direction = action
        elif direction == -1:
            new_direction = left[action]
        elif direction == 1:
            new_direction = right[action]
        else:
            raise Exception("getDir received an unspecified case")
        return new_direction

    def _get_state(self, state, direction):
        """
        Get the next_state from the current state and a direction.

        Parameters
        ----------
        state : int
            The current state.

        direction : int
            The current direction.

        Returns
        -------
        next_state : int
            The next state given the current state and direction.
        """
        row_change = [-1,1,0,0]
        col_change = [0,0,-1,1]
        row_col = seq_to_col_row(state, self.num_cols)
        row_col[0,0] += row_change[direction]
        row_col[0,1] += col_change[direction]

        # check for invalid states
        if self.obs_states is not None:
            if (np.any(row_col < 0) or
                np.any(row_col[:,0] > self.num_rows-1) or
                np.any(row_col[:,1] > self.num_cols-1) or
                np.any(np.sum(abs(self.obs_states - row_col), 1)==0)):
                next_state = state
            else:
                next_state = row_col_to_seq(row_col, self.num_cols)[0]
        else:
            if (np.any(row_col < 0) or
                np.any(row_col[:,0] > self.num_rows-1) or
                np.any(row_col[:,1] > self.num_cols-1)):
                next_state = state
            else:
                next_state = row_col_to_seq(row_col, self.num_cols)[0]

        return next_state
def qlearning(model, alpha=0.5, epsilon=0.1, maxiter=100, maxeps=1000):
    """
    Solves the supplied environment using Q-learning.

    Parameters
    ----------
    model : python object
        Holds information about the environment to solve
        such as the reward structure and the transition dynamics.

    alpha : float
        Algorithm learning rate. Defaults to 0.5.

    epsilon : float
         Probability that a random action is selected. epsilon must be
         in the interval [0,1] where 0 means that the action is selected
         in a completely greedy manner and 1 means the action is always
         selected randomly.

    maxiter : int
        The maximum number of iterations to perform per episode.
        Defaults to 100.

    maxeps : int
        The number of episodes to run SARSA for.
        Defaults to 1000.

    Returns
    -------
    q : numpy array of shape (N, 1)
        The state-action value for the environment where N is the
        total number of states

    pi : numpy array of shape (N, 1)
        Optimal policy for the environment where N is the total
        number of states.

    state_counts : numpy array of shape (N, 1)
        Counts of the number of times each state is visited
    """
    # initialize the state-action value function and the state counts
    Q = np.zeros((model.num_states, model.num_actions))
    state_counts = np.zeros((model.num_states, 1))

    for i in range(maxeps):

        if np.mod(i,1000) == 0:
            print("Running episode %i." % i)

        # for each new episode, start at the given start state
        state = int(model.start_state_seq)

        for j in range(maxiter):
            # sample first e-greedy action
            action = sample_action(Q, state, model.num_actions, epsilon)
            # initialize p and r
            p, r = 0, np.random.random()

            # sample the next state according to the action and the
            # probability of the transition
            for next_state in range(model.num_states):
                p += model.P[state, next_state, action]
                if r <= p:
                    break

            # Calculate the temporal difference and update Q function
            Q[state, action] += alpha * (model.R[state] + model.gamma * np.max(Q[next_state, :]) - Q[state, action])

            # count the state visits
            state_counts[state] += 1

            #Store the previous state
            state = next_state
            # End episode is state is a terminal state
            if np.any(state == model.goal_states_seq):
                break

    # determine the q function and policy
    q = np.max(Q, axis=1).reshape(-1,1)
    pi = np.argmax(Q, axis=1).reshape(-1,1)

    return q, pi, state_counts

def sample_action(Q, state, num_actions, epsilon):
    """
    Epsilon greedy action selection.

    Parameters
    ----------
    Q : numpy array of shape (N, 1)
        Q function for the environment where N is the total number of states.

    state : int
        The current state.

    num_actions : int
        The number of actions.

    epsilon : float
         Probability that a random action is selected. epsilon must be
         in the interval [0,1] where 0 means that the action is selected
         in a completely greedy manner and 1 means the action is always
         selected randomly.

    Returns
    -------
    action : int
        Number representing the selected action between 0 and num_actions.
    """
    if np.random.random() < epsilon:
        action = np.random.randint(0, num_actions)
    else:
        action = np.argmax(Q[state, :])

    return action
###########################################################
#            Run Q-Learning on cliff walk                 #
###########################################################

# specify world parameters
# num_rows = 4
# num_cols = 12
# restart_states = np.array([[3,1],[3,2],[3,3],[3,4],[3,5],
#                            [3,6],[3,7],[3,8],[3,9],[3,10]])
# start_state = np.array([[3,0]])
# goal_states = np.array([[3,11]])

# # create model
# gw = GridWorld(num_rows=num_rows,
#                num_cols=num_cols,
#                start_state=start_state,
#                goal_states=goal_states)
# gw.add_obstructions(restart_states=restart_states)
# gw.add_rewards(step_reward=-1,
#                goal_reward=10,
#                restart_state_reward=-100)
# gw.add_transition_probability(p_good_transition=1,
#                               bias=0)
# gw.add_discount(discount=0.9)
# model = gw.create_gridworld()

# # solve with Q-Learning
# q_function, pi, state_counts = qlearning(model, alpha=0.9, epsilon=0.2, maxiter=100, maxeps=10000)

# # plot the results
# path = "./doc/imgs/qlearning_cliffworld.png"
# plot_gridworld(model, policy=pi, state_counts=state_counts, title="Q-Learning", path=path)