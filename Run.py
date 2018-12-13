import os, sys, random, operator
import numpy as np
import matplotlib.pyplot as plt

from Agent import Agent
from Environment import Environment
from Environment import *

# Settings

state_size=100
action_size=4

env = Environment(nRow, nCol)
agent = Agent(env,state_size,action_size,nRow-1, nCol-1)

number_of_iterations_per_episode=[]
number_of_episodes=[]

# Train agent
print("\nTraining agent...\n")
N_episodes =50
batch_size = 32
reward_List=[]
samples=Extract_Features
for episode in range(N_episodes):

    done = False
    score = 0

    # Generate an episode
    reward_episode = 0
    state = env.reset()  # starting state
    state=samples.Extract_Samples(state[0],state[1],nRow-1,nCol-1)
    #state=samples.Extract_Raw_Data(state[0],state[1],nRow-1,nCol-1)
    #state = samples.Extract_Spectrogram(state[0], state[1], nRow - 1, nCol - 1)
    state = np.reshape(state, [1, state_size])
    number_of_episodes.append(episode)
    iteration=0
    while iteration < 100:
        iteration+=1
        #action = agent.get_action(env,nRow)  # get action
        action=agent.get_action_next(env,nRow)
        features, reward, done = env.step(action)  # evolve state by action
        features = np.reshape(features, [1, state_size])
        agent.replay_memory(state, action, reward, features, done)
        agent.train_replay()
        agent.train((state, action, features, reward, done))  # train agent
        reward_episode += reward
        if done==True:
            break
        state = features  # transition to next state

    # Decay agent exploration parameter
    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, 0.01)

    number_of_iterations_per_episode.append(iteration)
    reward_List.append(reward)

    print("[episode {}/{}] Number of Iterations = {}, Reward = {}".format(
            episode, N_episodes, iteration, reward_episode))

percentage_of_successful_episodes=(sum(reward_List)/N_episodes)*100

print("Percentage of Successful Episodes is {} {}".format(percentage_of_successful_episodes,'%'))
fig = plt.figure()
fig.suptitle('Deep Q-Learning', fontsize=12)
plt.plot(np.arange(len(number_of_episodes)), number_of_iterations_per_episode)
plt.ylabel('Number of Iterations')
plt.xlabel('Episode')
# plt.grid(True)
#plt.savefig("Q_Learning_10_10.png")
plt.show()
