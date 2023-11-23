import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random
import gymnasium as gym
import numpy as np
from collections import deque
from keras.models import  load_model
from CartPole.Create_DQN_Model.Create_DQN_Model import create_DQN_model

class DQNAgent:
    def __init__(self):
        self.env = gym.make("CartPole-v1",render_mode="human")
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 1000
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 1000

        self.model = create_DQN_model(input_shape=(self.state_size,), action_space = self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def clear_log(self,filename="log.txt"):
        # clear the file
        with open(filename, "w") as myfile:
            myfile.write("")

    def save_log(self,string,filename="log.txt"):
        # append to the file
        with open(filename, "a") as myfile:
            myfile.write(string)

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)
            
    def run(self):
        self.clear_log(filename="train_log.txt")
        for e in range(self.EPISODES):
            state = self.env.reset()
            observation, info = state
            state = np.array(observation)
            state = state.reshape(1, self.state_size)
            done = False
            i = 0
            while not done:
                action = self.act(state)
                next_state, reward, done,info, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    data = "episode: {}/{}, score: {}, e: {:.2}\n".format(e, self.EPISODES, i, self.epsilon)
                    self.save_log(data,filename="train_log.txt")                  
                    if i >= 500:
                        print(f"Saving trained model as cartpole-{i}-score.h5")
                        self.save(f"cartpole-{i}-score.h5")
                        return
                self.replay()

    def test(self):
        self.load("cartpole-dqn.h5")
        self.clear_log(filename="test_log.txt")
        for e in range(self.EPISODES):
            state = self.env.reset()
            observation, info = state
            state = np.array(observation)
            state = state.reshape(1, self.state_size)
            done = False
            i = 0
            while not done:
                action = np.argmax(self.model.predict(state))
                next_state, reward, done,info , _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    data = "episode: {}/{}, score: {}\n".format(e, self.EPISODES, i)
                    self.save_log(data,filename="test_log.txt")
                    break