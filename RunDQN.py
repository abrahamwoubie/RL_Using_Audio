from DQNAgent import  DQNAgent
from Environment import Environment
import numpy as np
import matplotlib.pyplot as plt

from Environment import  *
env = Environment(nRow,nCol)


# Observation and Action Sizes
state_size = 100
action_size = 4

agent = DQNAgent(state_size, action_size,nRow,nCol)

#Options for running using samples, spectrogram and raw data
use_samples=1
use_spectrogram=0
use_raw_data=0

#parameters
batch_size=32
N_episodes = 50

reward_List = []
number_of_iterations_per_episode = []
number_of_episodes = []

samples=Extract_Features # to access the member functions of the ExtractFeatures class

for episode in range(N_episodes):
    done = False
    reward_per_episode=0
    state = env.reset()

    #options to run the experiment using samples, spectrogram or raw data
    if(use_samples):
        state = samples.Extract_Samples(state[0], state[1], nRow-1, nCol-1)
    elif(use_spectrogram):
        state = samples.Extract_Spectrogram(state[0], state[1], nRow - 1, nCol - 1)
    else:
        state = samples.Extract_Raw_Data(state[0], state[1], nRow - 1, nCol - 1)

    state = np.reshape(state, [1, state_size])
    #print("Shape is ",state.shape)
    number_of_iterations=0
    number_of_episodes.append(episode)
    for iterations in range(100):
        number_of_iterations+=1
        action = agent.get_action_next(env,nRow)
        feature, reward, done = env.step(action,use_samples,use_spectrogram,use_raw_data)
        reward_per_episode+=reward
        feature = np.reshape(feature, [1, state_size])
        agent.replay_memory(state, action, reward, feature, done)
        state=feature
        if done:
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    print("episode: {}/{}, Number of Iterations: {}, Reward: {}"\
          .format(episode, N_episodes, number_of_iterations, reward_per_episode))

    #append number of iteration and reward for plotting
    number_of_iterations_per_episode.append(number_of_iterations)
    reward_List.append(reward)

percentage_of_successful_episodes = (sum(reward_List) / N_episodes) * 100

print("Percentage of Successful Episodes is {} {}".format(percentage_of_successful_episodes, '%'))
fig = plt.figure()
fig.suptitle('Q-Learning', fontsize=12)
plt.plot(np.arange(len(number_of_episodes)), number_of_iterations_per_episode)
plt.ylabel('Number of Iterations')
plt.xlabel('Episode')
plt.show()