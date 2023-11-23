#importamos modulos
import gymnasium as gym
import random

# Funcion para jugar aleatoriamente
def random_game(render_mode=None, episodes=1000, goal=500):
    env = gym.make("CartPole-v1",render_mode=render_mode)
    for episode in range(episodes):
        env.reset()
        for t in range(goal):
            action = env.action_space.sample()
            next_state, reward, done, info,_ = env.step(action)
            print(t, next_state, reward, done, info, action)
            if done:
                break
