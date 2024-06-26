#Code from following tutorial at https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#training with 
#some modifications
import gymnasium as gym
import CartPoleAgentDQN
import CartPoleAgentDQNSingleNN
import random 
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as neuralNetwork
import torch.optim as optim
import torch.nn.functional as F

is_ipython = "inline" in matplotlib.get_backend()

if is_ipython:
    from IPython import display

plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


episode_durations = []
episode_rewards = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('rewards')
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


#environment = gym.make('LunarLander-v2', render_mode='human')
environment = gym.make('LunarLander-v2')

state, info = environment.reset()


games = 600

a = CartPoleAgentDQN.Agent(environment.action_space.n, len(state), device)

#a = CartPoleAgentDQNSingleNN.Agent(environment.action_space.n, len(state), device)

for game in range(games):
    state, info = environment.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    stepsDone = 0
    current_episode_rewards = 0
    for t in count():
        
        action = a.PickAction(state, stepsDone, environment)
        stepsDone += 1
        observation, reward, terminated, truncated, info = environment.step(action.item())
        reward = torch.tensor([reward], device=device)
        finished = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        a.updateMemory(state, action, next_state, reward)
        old_state = state
        state = next_state

        a.optimizeModel()
        a.updateWeights()
        current_episode_rewards += reward
        if finished:
            episode_durations.append(t+1)
            episode_rewards.append(current_episode_rewards)
            plot_durations()
            break

print("complete") 

#a.PrintValues(old_state)
plot_durations(show_result=True)
plt.ioff()
plt.savefig("./resultsLunarLander5_20_24.png")    
plt.show()

environment = gym.make('LunarLander-v2', render_mode='human')

for game in range(10):
    state, info = environment.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    stepsDone = 0
    current_episode_rewards = 0
    for t in count():
        
        action = a.PickAction(state, stepsDone, environment)
        stepsDone += 1
        observation, reward, terminated, truncated, info = environment.step(action.item())
        finished = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        if finished:
            break