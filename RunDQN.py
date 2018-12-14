from DQNAgent import  DQNAgent
from Environment import Environment
import numpy as np
import matplotlib.pyplot as plt
from GlobalVariables import GlobalVariables


size=GlobalVariables

from Environment import *


options=GlobalVariables # To access global variables from GlobalVariable.py
parameter=GlobalVariables # To access parameters from GlobalVariables.py


reward_List = []
number_of_iterations_per_episode = []
number_of_episodes = []

samples=Extract_Features # to access the member functions of the ExtractFeatures class

#Extract the samples of the goal state

if (options.use_samples):
    samples_goal = samples.Extract_Samples(grid_size.nRow - 1, grid_size.nCol - 1)
elif (options.use_pitch):
    samples_goal = samples.Extract_Pitch(grid_size.nRow - 1, grid_size.nCol - 1)
elif (options.use_spectrogram):
    samples_goal = samples.Extract_Spectrogram(grid_size.nRow - 1, grid_size.nCol - 1)
else:
    samples_goal = samples.Extract_Raw_Data(grid_size.nRow - 1, grid_size.nCol - 1)

env = Environment(size.nRow,size.nCol)
agent = DQNAgent(parameter.state_size, parameter.action_size)

for episode in range(parameter.Number_of_episodes):
    done = False
    reward_per_episode=0
    state = env.reset()
    #options to run the experiment using samples, spectrogram or raw data
    if(options.use_samples):
        state = samples.Extract_Samples(state[0], state[1])
    elif (options.use_pitch):
        state = samples.Extract_Pitch(state[0], state[1])
    elif(options.use_spectrogram):
        state = samples.Extract_Spectrogram(state[0], state[1])
    else:
        state = samples.Extract_Raw_Data(state[0], state[1])

    state = np.reshape(state, [1, parameter.state_size])
    number_of_iterations=0
    number_of_episodes.append(episode)
    for iterations in range(parameter.timesteps):
        number_of_iterations+=1
        action = agent.get_action(env)
        feature, reward, done = env.step(action,samples_goal)
        reward_per_episode+=reward
        feature = np.reshape(feature, [1, parameter.state_size])
        agent.replay_memory(state, action, reward, feature, done)
        state=feature
        if done:
            break
        if len(agent.memory) > parameter.batch_size:
            agent.replay(parameter.batch_size)
    print("episode: {}/{}, Number of Iterations: {}, Reward: {}"\
          .format(episode, parameter.Number_of_episodes, number_of_iterations, reward_per_episode))

    #append number of iteration and reward for plotting
    number_of_iterations_per_episode.append(number_of_iterations)
    reward_List.append(reward)

percentage_of_successful_episodes = (sum(reward_List) / parameter.Number_of_episodes) * 100

print("Percentage of Successful Episodes is {} {}".format(percentage_of_successful_episodes, '%'))
fig = plt.figure()
fig.suptitle('Q-Learning', fontsize=12)
plt.plot(np.arange(len(number_of_episodes)), number_of_iterations_per_episode)
plt.ylabel('Number of Iterations')
plt.xlabel('Episode')
plt.show()