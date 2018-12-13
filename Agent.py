import os, sys, random, operator
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Convolution2D, MaxPooling2D


class Agent:

    def __init__(self, env, state_size, action_size,nRow, nCol):
        # Store state and action dimension


        # Agent learning parameters
        self.epsilon = 1.0  # initial exploration probability
        self.epsilon_decay = 0.99  # epsilon decay after each episode
        self.learning_rate = 0.99  # learning rate
        self.discount_factor = 0.99  # reward discount factor
        self.Q=np.zeros([nRow*nCol,action_size])

        self.batch_size = 32
        self.train_start = 100

        self.state_size = state_size
        self.action_size = action_size

        # create replay memory using deque
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=10000)
        self.model = self.build_model()
        self.target_model = self.build_model()
        # copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.update_target_model()

    #
    # def build_model(self):
    #     model = Sequential()
    #     model.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='random_uniform'))
    #     #model.add(Dense(24, input_shape=(100,), activation='relu', kernel_initializer='random_uniform'))
    #     model.add(Dense(24, activation='relu', kernel_initializer='random_uniform'))
    #     model.add(Dense(self.action_size, activation='linear', kernel_initializer='random_uniform'))
    #     model.summary()
    #     model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
    #     return model

    def build_model(self):
        model = Sequential()

        #model.add(Convolution2D(32, (1, 1), activation='relu', input_shape=(1, 1, 100), data_format='channels_first'))

        model.add(Conv2D(64, kernel_size=1, activation='relu', input_shape = (1, 1, 100))) #, data_format='channels_first'))
        model.add(Conv2D(32, kernel_size=1, activation='relu'))
        model.add(Flatten())
        model.add(Dense(4, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

        # '''
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action_next(self,env,nRow):
        return np.random.choice(env.allowed_actions())

    def get_action(self, env,nRow):
        # Epsilon-greedy agent policy
        if random.uniform(0, 1) < self.epsilon:
            # explore
            return np.random.choice(env.allowed_actions())
        else:

        # exploit on allowed actions
            state = env.state;
            actions_allowed = env.allowed_actions()
            Q_s = self.Q[state[0]*nRow+state[1], actions_allowed]
            actions_greedy = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            return np.random.choice(actions_greedy)

    def train(self, memory):

        (state, action, state_next, reward, done) = memory
        #print("Memory",memory)
#        self.Q[state,action]+=self.learning_rate * \
  #                            (reward + self.discount_factor * np.max(self.Q[state_next,:]) - self.Q[state,action])


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.discount_factor * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.action_size))


        for i in range(batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            reward = np.float32(reward)
            state = np.float32(state)
            next_state = np.float32(next_state)
            target = self.model.predict(state)[0]

            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done:
                target[action] = reward
            else:
                target = reward + self.discount_factor * \
                                  np.amax(self.model.predict(next_state)[0])

            update_input[i] = state
            update_target[i] = target

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(update_input, update_target, batch_size=batch_size, epochs=1, verbose=0)

    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
             self.epsilon *= self.epsilon_decay