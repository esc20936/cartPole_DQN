import gymnasium as gym
import random

env = gym.make("CartPole-v1",render_mode="human")

def Random_games():
    for episode in range(100):
        env.reset()
        for t in range(500):
            action = env.action_space.sample()

            next_state, reward, done, info,_ = env.step(action)

            print(t, next_state, reward, done, info, action)

            if done:
                break

Random_games()