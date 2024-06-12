# Code from following tutorial at https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html; 
# https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py; with 
# some modifications

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium

class PoAgent():

    def __init__(self, envName, hiddenSizes = [32] learningRate = 1e-2, epochs = 50, batchSize = 5000):
        self.env = gymnasium.make(envName)
        self.obsDim = self.env.observation_space.shape[0]
        self.numberOfActions = self.env.action_space.n
        self.policyNetwork = self.CreateNetwork(sizes=[self.obsDim]+[self.numberOfActions]+hiddenSizes)
        self.Optimizer = Adam(self.policyNetwork.parameters())

    def CreateNetwork(sizes, activation=nn.Tanh, outputActivation = nn.Identity ):
        layers = []
        for l in range(len(sizes)-1):
            #if it's anything other than the last layer, set it to be nn.Tahn. For the last layer set it to be nn.Identity 
            act = activation if l < len(sizes)-2 else outputActivation
            layers += [nn.layer(sizes[l], sizes[l+1], act())]
        return nn.Sequential(*layers)
    
    def getPolicy(self,  observation):
        pNetwork = self.policyNetwork(observation)
        return Categorical(logits=pNetwork)
    
    def getAction(self, observation):
        return self.getPolicy(observation).sample().item()
    
    def computeLoss(self, observation, action, weights):
        logOfP = self.getPolicy(observation).log_prob(action)
        return -(logOfP * weights).mean()
    
    def trainEpoch():
        pass



