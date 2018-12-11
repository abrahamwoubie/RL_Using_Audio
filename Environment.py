import os, sys, random, operator
import numpy as np
import matplotlib.pyplot as plt
from Agent import Agent
from ExtractFeatures import Extract_Features


from scipy.spatial import  distance


"""

  action['up'] = 0
  action['right'] = 1
  action['down'] = 2
  action['left'] = 3 

"""

nRow=7
nCol=7

class Environment:
    
    def __init__(self, nRow, nCol):
        # Define state space
        self.nRow = nRow  # x grid size
        self.nCol = nCol  # y grid size
        #self.state_dim = (nRow, nCol)
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}

    def reset(self):
        # Reset agent state to top-left grid corner
        start_row=random.choice(range(0,nRow-1))
        start_col=random.choice(range(0,nRow-1))
        self.state = (start_row, start_col)

        #goal_row = random.choice(range(0, nRow - 1))
        #goal_col = random.choice(range(0, nCol - 1))
        #self.goal_state=(goal_row,goal_col)

        return self.state #,self.goal_state


    def step(self, action):
        # Evolve agent state
        reward = 0
        done = False

        if(action==0):
            state_next =  (self.state[0]-1) , self.state[1] # up

        if(action==1):
            state_next = self.state[0] , (self.state[1] + 1) # right

        if(action==2):
            state_next = (self.state[0] + 1) , self.state[1] # down

        if(action==3):
            state_next = self.state[0]  , (self.state[1] - 1) # left

        # samples=Extract_Features
        # samples_state_next=samples.Extract_Samples(state_next[0],state_next[1],nRow-1,nCol-1)
        # samples_goal_state = samples.Extract_Samples(nRow - 1, nCol - 1,nRow-1,nCol-1)

        #Collect reward
        if(state_next==(self.nRow-1,self.nCol-1)):
        #if(state_next==self.goal_state):
            reward=1
            done=True

        # if(distance.euclidean(samples_state_next,samples_goal_state)==0):
        #     reward=1
        #     done=True

        self.state = state_next
        return state_next, reward, done
        #return samples_state_next, next, reward, done
    
    def allowed_actions(self):
        # Generate list of actions allowed depending on agent grid location
        actions_allowed = []
        row, col = self.state[0], self.state[1]
        #print("y",self.state[0])
        #print("x", self.state[1])
        if (row > 0):  # no passing top-boundary
            actions_allowed.append(self.action_dict["up"])
        if (row < self.nRow - 1):  # no passing bottom-boundary
            actions_allowed.append(self.action_dict["down"])
        if (col > 0):  # no passing left-boundary
            actions_allowed.append(self.action_dict["left"])
        if (col < self.nCol - 1):  # no passing right-boundary
            actions_allowed.append(self.action_dict["right"])
        actions_allowed = np.array(actions_allowed, dtype=int)
        return actions_allowed