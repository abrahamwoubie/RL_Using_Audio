import os, sys, random, operator
import numpy as np
import matplotlib.pyplot as plt
import gym
#from gym_Audio.envs import discrete


from AgentQL import AgentQL
from Environment import Environment
from Environment import *

from GlobalVariables import GlobalVariables


grid_size=GlobalVariables
parameter=GlobalVariables

# Settings
env = Environment(grid_size.nRow, grid_size.nCol)
agent = AgentQL(grid_size.nRow,grid_size.nCol)

number_of_iterations_per_episode=[]
number_of_episodes=[]

# Train agent
print("\nTraining agent...\n")
reward_List=[]
samples=Extract_Features
samples_goal = samples.Extract_Samples(grid_size.nRow - 1, grid_size.nCol - 1)
for episode in range(parameter.Number_of_episodes):

    # Generate an episode
    reward_episode = 0
    state = env.reset()  # starting state
    state = state[0] * grid_size.nRow + state[1]
    number_of_episodes.append(episode)
    iteration=0
    while iteration < parameter.timesteps:
        iteration+=1
        action = agent.get_action(env)  # get action
        state_next, reward, done = env.step(action,samples_goal)  # evolve state by action
        state_next=state_next[0]*grid_size.nRow+state_next[1]
        agent.train((state, action, state_next, reward, done))  # train agent
        reward_episode += reward
        if done:
            break
        state = state_next  # transition to next state

    # Decay agent exploration parameter
    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, 0.01)

    number_of_iterations_per_episode.append(iteration)
    reward_List.append(reward)

    print("[episode {}/{}] Number of Iterations = {}, Reward per episode = {}".format(
            episode + 1, parameter.Number_of_episodes, iteration, reward_episode))

percentage_of_successful_episodes=(sum(reward_List)/parameter.Number_of_episodes)*100

print("Percentage of Successful Episodes is {} {}".format(percentage_of_successful_episodes,'%'))
fig = plt.figure()
title="Q-Learning "+str(grid_size.nRow) + "X" + str(grid_size.nRow)
fig.suptitle(title, fontsize=12)
plt.plot(np.arange(len(number_of_episodes)), number_of_iterations_per_episode)
plt.ylabel('Number of Iterations')
plt.xlabel('Episode')
# plt.grid(True)
plt.savefig(title+'.png')
plt.show()