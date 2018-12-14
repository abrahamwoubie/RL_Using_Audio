import os, sys, random, operator
import numpy as np
import matplotlib.pyplot as plt
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from GlobalVariables import  GlobalVariables
grid_size=GlobalVariables

class AgentQL:

    def __init__(self, state_size, action_size):
        # Store state and action dimension
        self.state_dim = grid_size.nRow * grid_size.nCol
        self.action_dim = 4
        # Agent learning parameters
        self.epsilon = 1.0  # initial exploration probability
        self.epsilon_decay = 0.99  # epsilon decay after each episode
        self.learning_rate = 0.99  # learning rate
        self.discount_factor = 0.99  # reward discount factor
        # Initialize Q[s,a] table
        #self.Q = np.zeros(self.state_dim + self.action_dim, dtype=float)
        self.Q = np.zeros([grid_size.nRow * grid_size.nCol, 4])


    def get_action(self,env):
        # Epsilon-greedy agent policy
        if random.uniform(0, 1) < self.epsilon:
            # explore
            return np.random.choice(env.allowed_actions())
        else:

        # exploit on allowed actions
            state = env.state;
            actions_allowed = env.allowed_actions()
            Q_s = self.Q[state[0]*grid_size.nRow+state[1], actions_allowed]
            actions_greedy = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            return np.random.choice(actions_greedy)

    def train(self, memory):

        (state, action, state_next, reward, done) = memory
        #print("Memory",memory)
        self.Q[state, action] += self.learning_rate * (
                              reward + self.discount_factor * np.max(self.Q[state_next, :]) - self.Q[state, action])


