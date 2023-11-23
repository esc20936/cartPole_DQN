import gymnasium as gym
from gymnasium.utils.play import play


def let_user_play():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    play(env, keys_to_action={"a": 0, "d": 1}, fps=15)