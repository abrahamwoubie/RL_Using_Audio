import os, sys, random, operator
import numpy as np
import matplotlib.pyplot as plt
from Agent import Agent
from ExtractFeatures import Extract_Features
from scipy.spatial import distance
from GlobalVariables import GlobalVariables

options=GlobalVariables
grid_size=GlobalVariables


"""
  action['up'] = 0
  action['right'] = 1
  action['down'] = 2
  action['left'] = 3 
"""


class Environment:
    
    def __init__(self, nRow, nCol):
        # Define state space
        self.nRow = nRow  # x grid size
        self.nCol = nCol  # y grid size
        #self.state_dim = (nRow, nCol)
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}

    def reset(self):
        # Reset agent state to top-left grid corner
        start_row=random.choice(range(0,grid_size.nRow-1))
        start_col=random.choice(range(0,grid_size.nCol-1))
        self.state = (start_row, start_col)

        #goal_row = random.choice(range(0, nRow - 1))
        #goal_col = random.choice(range(0, nCol - 1))
        #self.goal_state=(goal_row,goal_col)

        return self.state #,self.goal_state


    def step(self, action,samples_goal):
        # Evolve agent state

        reward = 0
        done = False
        if(action==0): # up
            state_next =  (self.state[0]-1) , self.state[1]

        if(action==1): #right
            state_next = self.state[0] , (self.state[1] + 1)

        if(action==2): # down
            state_next = (self.state[0] + 1) , self.state[1]

        if(action==3): # left
            state_next = self.state[0]  , (self.state[1] - 1)

        # samples=Extract_Features
        #
        # # options to run the experiment using samples, spectrogram or raw data
        # if(options.use_samples):
        #     samples_current=samples.Extract_Samples(state_next[0],state_next[1])
        #     if (distance.euclidean(samples_goal, samples_current) == 0):
        #         reward = 1
        #         done = True
        #
        # elif(options.use_pitch):
        #     samples_current=samples.Extract_Pitch(state_next[0],state_next[1])
        #     if (distance.euclidean(samples_goal, samples_current) == 0):
        #         reward = 1
        #         done = True
        #
        # elif(options.use_spectrogram):
        #     samples_current=samples.Extract_Spectrogram(state_next[0],state_next[1])
        #     if (np.mean(samples_goal)==np.mean(samples_current)):
        #         reward = 1
        #         done = True
        #
        # else:
        #     samples_current=samples.Extract_Raw_Data(state_next[0],state_next[1])
        #     if (np.mean(samples_goal)==np.mean(samples_current) == 0):
        #         reward = 1
        #         done = True

        if(state_next==(self.nRow-1,self.nCol-1)):
            reward=1
            done=True

        self.state = state_next
        #return samples_current, reward, done
        return state_next,reward,done


    def allowed_actions(self):
        # Generate list of actions allowed depending on agent grid location
        actions_allowed = []
        row, col = self.state[0], self.state[1]
        if (row > 0):  # It can not got to the top-boundary
            actions_allowed.append(self.action_dict["up"])
        if (row < self.nRow - 1):  # It can not go to bottom-boundary
            actions_allowed.append(self.action_dict["down"])
        if (col > 0):  # It can not go to the left-boundary
            actions_allowed.append(self.action_dict["left"])
        if (col < self.nCol - 1):  # It can not go to right-boundary
            actions_allowed.append(self.action_dict["right"])
        actions_allowed = np.array(actions_allowed, dtype=int)
        return actions_allowed