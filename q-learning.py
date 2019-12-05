import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

plt.style.use('ggplot')
np.set_printoptions(precision=3)

# let's define the world
WORLD = np.zeros((9, 9))

WALL = [(1,2), (1,3), (1,4), (1,5), (1,6), (2,6), (3,6), (4,6), (5,6), (7,1), (7,2), (7,3), (7,4)]

SNAKEPIT = (6, 5)
TERMINAL_STATE = (8, 8)
TERMINAL_STATES = [(8,8), (6, 5)]
START = (0, 0)

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

def isGameOver(state):
    return state in TERMINAL_STATE or state in SNAKEPIT


def getNextState(pos, action):
    '''
    Gives next state for specific action
    :param pos:
    :param action:
    :return:
    '''

    if action == UP:
        if pos[0] > 0:
            new_pos = (pos[0] - 1, pos[1])
        else:
            new_pos = pos

    elif action == DOWN:
        if pos[0] < 8:
            new_pos = (pos[0] + 1, pos[1])
        else:
            new_pos = pos

    elif action == LEFT:
        if pos[1] > 0:
            new_pos = (pos[0], pos[1] - 1)
        else:
            new_pos = pos

    elif action == RIGHT:
        if pos[1] < 8:
            new_pos = (pos[0], pos[1] + 1)
        else:
            new_pos = pos

    try:
        #print(new_pos)
        if new_pos in WALL:
            new_pos = pos
    except:
        return pos


    return new_pos

def get_next_state(pos, action, R, done=False):
    '''
    Gives next state for specific action
    :param pos:
    :param action:
    :return:
    '''

    if action == UP:
        if pos[0] > 0:
            new_pos = (pos[0] - 1, pos[1])
        else:
            new_pos = pos

    elif action == DOWN:
        if pos[0] < 8:
            new_pos = (pos[0] + 1, pos[1])
        else:
            new_pos = pos

    elif action == LEFT:
        if pos[1] > 0:
            new_pos = (pos[0], pos[1] - 1)
        else:
            new_pos = pos

    elif action == RIGHT:
        if pos[1] < 8:
            new_pos = (pos[0], pos[1] + 1)
        else:
            new_pos = pos

    if new_pos in WALL:
        new_pos = pos

    if new_pos in TERMINAL_STATES:
        done = True

    reward = R[new_pos]
    print(reward)
    return new_pos, done, reward




def initialize(init=0):

    R = np.zeros((9, 9))

    states = []

    actions = [UP, RIGHT, DOWN, LEFT]

    V = np.ones((9, 9))
    V *= init

    for i in range(9):
        for j in range(9):
            if (i, j) == TERMINAL_STATE:
                R[i, j] = 50
                V[i, j] = 0
            elif (i, j) == SNAKEPIT:
                R[i, j] = -50
            elif (i, j) not in WALL:
                states.append((i, j))
                R[i, j] = -1

    Q = np.ones((len(states), len(actions)))
    Q *= init
    return R, V, Q, states, actions


def initialize_Q(init=0):

    R = np.zeros((9, 9))

    states = []

    actions = [UP, RIGHT, DOWN, LEFT]

    V = np.ones((9, 9))
    V *= init

    for i in range(9):
        for j in range(9):
            states.append((i, j))
            if (i, j) == TERMINAL_STATE:
                R[i, j] = 50
                V[i, j] = 0
            elif (i, j) == SNAKEPIT:
                R[i, j] = -50
            elif (i, j) not in WALL:

                R[i, j] = -1

    Q = np.ones((len(states), len(actions)))
    Q *= init
    return R, V, Q, states, actions



def policy_evaluation(R, V, Q, states, actions, GAMMA):
    '''
    return value function for equiprobable policy
    '''

    theta = 0.01

    pos = START

    newV = np.zeros((9, 9))

    done = False
    delta = 0
    i = 0

    #while delta < theta:
    while not done:
    #for i in range(1000):
        delta = 0
        i += 1
        print(f"iteration : {i}")

        for state in states:
            v = V[state]

            intermediate = 0

            for action in actions:
                next_state = getNextState(state, action)
                intermediate += (0.25 * (R[next_state] + (GAMMA * V[next_state])))

            V[state] = intermediate

            delta = max(delta, abs(v - newV[state]))

        if delta < theta:
            done = True

    print()
    print("Policy Evaluation - Value function")
    print()
    print(V)
    np.savetxt("V-policy-evaluation.csv", V,  delimiter=",")
    return V



def value_iteration(R, V, Q, states, actions, GAMMA):
    '''
    return v*  for equiprobable policy
    '''

    theta = 0.01
    delta = 0

    pos = START

    newV = np.zeros((9, 9))

    done = False
    i = 0

    while not done:
        delta = 0
        for state in states:
            v = V[state]

            temp = []
            for action in actions:
                next_state = getNextState(state, action)
                temp.append(-1 + R[next_state] + (GAMMA * V[next_state]))

            newV[state] = max(temp)

            V[state] = newV[state]

            delta = max(delta, abs(v - newV[state]))

        if delta < theta:
            done = True

        print(f"Iteration : {i}")
        print(f"Delta : {delta}")
        print(V)
        i += 1

    print()
    print("Solving the Bellman Equation using DP - Optimal Value function V* ")
    print()
    print(V)
    return V


def q_learning(R, V, Q, states, actions, gamma=0.9, epsilon=0.05, alpha=0.1, n_episodes=1000):
    #Simple Q-learning

    Gs = []

    for e in np.arange(1, n_episodes+1):

        state = START
        G = 0
        #print(states)
        while state not in TERMINAL_STATES:

            # This is epsilon-greedy
            if np.random.binomial(1, epsilon):
                # with chance of epsilon, pick a random action
                action = np.random.choice(actions)
            else:
                # otherwise pick a random action amongst the highest q value only
                best = np.argwhere(Q[states.index(state)] == np.max(Q[states.index(state)])).flatten()
                #print(f"best : {best}")
                action = np.random.choice(best)

            #print(f"action : {action}")
            # get next state
            next_state, done, reward = get_next_state(state, action, R)

            # get reward
            #r = R[next_state]
            G = reward + (G * gamma)

            #print(f"next state : {next_state}")



            # update q
            Q[states.index(state)][action] += alpha * (reward + gamma * np.max(Q[states.index(next_state)]) - Q[states.index(state)][action])
            #Q[states.index(state)][action] += alpha * (r + gamma * np.max(Q[states[next_state_]]) - Q[states.index(state)][action])

            # check if done
            if done == True:
                break

            state = next_state

        # decay epsilon if needed
        #epsilon *= e_decay

        Gs.append(G)

        for state in states:
            V[state] = np.max(Q[states.index(state)])
        #print('V:\n', V)
        print()

    print()
    return V


def plot_V(V, states, name='V_Table.png', save_fig=True):

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('PiYG')
    heatmap = ax.matshow(V, cmap=cmap)
    plt.colorbar(heatmap)

    for state in states:
        x, y = state[0], state[1]
        if state in WALL:
            ax.text(y, x, 'wall', ha='center', va='center')
        else:
            ax.text(y, x, '{:0.1f}'.format(V[x, y]), ha='center', va='center')

    if save_fig:
        plt.savefig(name)
    plt.show()

# START THE PROGRAM
model = initialize()
GAMMA = 0.9
R = model[0]
V = model[1]
Q = model[2]
states = model[3]
actions = model[4]
print(R)

# ASS 1: policy Evaluation
policy_evaluation(R, V, Q, states, actions, GAMMA)
print()

# ASS 2: Bellman
value_iteration(R, V, Q, states, actions, GAMMA)

# ASS 3: Q - learning
model = initialize_Q()
GAMMA = 0.9
R = model[0]
V = model[1]
Q = model[2]
states = model[3]
actions = model[4]
V = q_learning(R, V, Q, states, actions)

plot_V(V, states)








