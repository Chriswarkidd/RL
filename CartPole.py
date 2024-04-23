import gymnasium as gym
import CartPoleAgent

environment = gym.make('CartPole-v1', render_mode='human')

games = 5

a = CartPoleAgent.Agent(2)

for game in range(games):
    print(f"\ngame #{game}: \n")
    current_state = environment.reset()
    finished = False
    action = 1
    while finished == False:
        action = a.PickAction()
        next_state, reward, terminated, truncated, info = environment.step(action)
        a.UpdateState(next_state)
        a.PrintState()
        finished = terminated or truncated