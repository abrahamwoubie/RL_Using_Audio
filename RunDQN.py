from DQNAgent import  DQNAgent
from Environment import Environment
import numpy as np
import matplotlib.pyplot as plt

from Environment import  *
env = Environment(nRow,nCol)
state_size = 100
action_size = 4
agent = DQNAgent(state_size, action_size,nRow,nCol)
# agent.load("./save/cartpole-dqn.h5")

batch_size = 32
N_episodes = 50
reward_List = []
number_of_iterations_per_episode = []
number_of_episodes = []
samples=Extract_Features
for episode in range(N_episodes):
    done = False
    reward_per_episode=0
    state = env.reset()
    state = samples.Extract_Samples(state[0], state[1], nRow-1, nCol-1)
    state = np.reshape(state, [1, state_size])
    number_of_iterations=0
    number_of_episodes.append(episode)
    for iterations in range(100):
        number_of_iterations+=1
        # env.render()
        #action = agent.act(state)
        action = agent.get_action_next(env,nRow)
        feature, reward, done = env.step(action)
        #next_state, reward, done = env.step(action)
        reward_per_episode+=reward
        #next_state = np.reshape(next_state, [1, state_size])
        feature = np.reshape(feature, [1, state_size])
        agent.replay_memory(state, action, reward, feature, done)
        #agent.replay_memory(state, action, reward, next_state, done)
        state=feature
        if done:
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    print("episode: {}/{}, Number of Iterations, {}, Reward: {}"
          .format(episode, N_episodes, number_of_iterations, reward_per_episode))
    number_of_iterations_per_episode.append(number_of_iterations)
    reward_List.append(reward)
percentage_of_successful_episodes = (sum(reward_List) / N_episodes) * 100

print("Percentage of Successful Episodes is {} {}".format(percentage_of_successful_episodes, '%'))
fig = plt.figure()
fig.suptitle('Q-Learning', fontsize=12)
plt.plot(np.arange(len(number_of_episodes)), number_of_iterations_per_episode)
plt.ylabel('Number of Iterations')
plt.xlabel('Episode')
# plt.grid(True)
#plt.savefig("Q_Learning_10_10.png")
plt.show()