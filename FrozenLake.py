import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
MapSize = 4
MapSeed = 10000
Slipery = True
#env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=Slipery, render_mode="human")
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=Slipery)
#env = gym.make('FrozenLake-v1', desc=generate_random_map(size=MapSize, seed=MapSeed), is_slippery=Slipery)

import random as rng
import Agent as a



FrozenAgent = a.Agent(4,MapSize,.9,.333)
games = 10000
wins = 0
deaths = 0
truncations = 0
gamma = .5
q_val = 0
for itterations in range(games):
    state = env.reset()[0]
    done = False
    rewards = 0
    while not done:
        action = FrozenAgent.PickCurrentBest(state)
        #action = FrozenAgent.PickWithFutureBest(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if terminated:
            if reward == 0:
                reward = -1
                deaths += 1
            else:
                wins += reward
        else:
            reward = -.5
            if truncated:
                truncations += 1   


        FrozenAgent.UpdateQDictValue(reward,state, action, next_state, .4)
        
        state = next_state
        FrozenAgent.UpdateEpsilon(.999)
    if itterations % 1000 == 0:
        print(f"current values at {itterations/1000:.0f}k runs:")
        FrozenAgent.PrintQDict()

print(f" Wins: {wins:.0f} \n Deaths: {deaths} \n Truncations: {truncations}")
#FrozenAgent.PrintQDict()


print("=================================================================================================================================")
print("after training")
#env = gym.make('FrozenLake-v1', desc=generate_random_map(size=MapSize, seed=MapSeed), is_slippery=Slipery, render_mode="human")
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=Slipery, render_mode="human")
games = 10
wins = 0
deaths = 0
truncations = 0
for itterations in range(games):
    state = env.reset()[0]
    done = False
    rewards = 0

    while not done:
        action = FrozenAgent.PickCurrentBest(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if terminated:
            if reward == 0:
                deaths += 1
            else:
                wins += reward
        else:
            if truncated:
                truncations += 1   
        state = next_state

print(f" Wins: {wins:.0f} \n Deaths: {deaths} \n Truncations: {truncations}")