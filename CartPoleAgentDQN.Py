import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import random 
import math

import torch
import torch.nn as neuralNetwork
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
        
class DQN(neuralNetwork.Module):
    def __init__(self, n_obs, n_acts, feats=128):
        super(DQN, self).__init__()
        self.layer1 = neuralNetwork.Linear(n_obs, feats)
        self.layer2 = neuralNetwork.Linear(feats, feats)
        self.layer3 = neuralNetwork.Linear(feats, n_acts)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class Agent():

    def __init__(self, num_actions, num_observations, device, batch_size = 256, gamma = 0.99, epsilon_start = 0.999, epsilon_end = 0.05, epsilon_decay = 2, tau = 0.005, lr = 1e-4):
        self.Num_Actions = num_actions
        self.Num_Observations = num_observations
        self.Device = device
        self.Batch_Size = batch_size
        self.Gamma = gamma
        self.Epsilon_Start = epsilon_start
        self.Epsilon_End = epsilon_end
        self.Epsilon_Decay = epsilon_decay
        self.Tau = tau
        self.Lr = lr

        self.policy_net = DQN(num_observations, num_actions).to(self.Device)
        self.target_net = DQN(num_observations, num_actions).to(self.Device)
        self.action_net = DQN(num_observations, num_actions).to(self.Device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.action_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.Lr, amsgrad=True)
        self.memory = ReplayMemory(10000)


    def PickAction(self, state, steps_done, env):
        sample = random.random()
        epsilion_threshold = self.Epsilon_End + (self.Epsilon_Start - self.Epsilon_End) * math.exp(-1. * steps_done / self.Epsilon_Decay)
        if sample > epsilion_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1,1)
                #return self.action_net(state).max(1).indices.view(1,1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=self.Device, dtype=torch.long)
        
    def PrintAction(self, state):
        print(self.policy_net(state).max(1).indices.view(1,1))
        print(self.policy_net(state))
        print(self.policy_net(state).tolist()[0][self.policy_net(state).max(1).indices.view(1,1).item()])
        #print(self.policy_net(state)[self.policy_net(state).max(1).indices])
        #print(self.policy_net(state)[0])
        # print(self.policy_net(state).max(1))


    def optimizeModel(self):
        if len(self.memory) < self.Batch_Size:
            return
        transitions = self.memory.sample(self.Batch_Size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.Device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_b = torch.cat(batch.state)
        action_b = torch.cat(batch.action)
        reward_b = torch.cat(batch.reward)

        state_actions_values = self.policy_net(state_b).gather(1, action_b)

        next_state_values = torch.zeros(self.Batch_Size, device=self.Device) 
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.Gamma) + reward_b

        criterion = neuralNetwork.SmoothL1Loss()
        loss = criterion(state_actions_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def updateMemory(self, s, a, nS, r):
        self.memory.push(s,a,nS,r)

    def updateWeights(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.Tau + target_net_state_dict[key]*(1-self.Tau)
        
        self.target_net.load_state_dict(target_net_state_dict)

    def updateEpisode(self):
        self.action_net.load_state_dict(self.policy_net.state_dict())