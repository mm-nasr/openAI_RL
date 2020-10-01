# Obtained from: https://github.com/shivaverma/OpenAIGym/blob/master/cart-pole/CartPole-v0.py
# Few comments added and switched to PyTorch
#  A pole is attached by an un-actuated joint to a cart,
#  which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart.
#  The pendulum starts upright, and the goal is to prevent it from falling over.
#  A reward of +1 is provided for every timestep that the pole remains upright.
#  The episode ends when the pole is more than 15 degrees from vertical,
#  or the cart moves more than 2.4 units from the center.

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from collections import deque
import matplotlib.pyplot as plt

import numpy as np
env = gym.make('CartPole-v0')
env.seed(0)
np.random.seed(0)

class network(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.dense1 = nn.Linear (input_size,24)
        self.dense2 = nn.Linear(24,24)
        self.output = nn.Linear(24,output_size)

        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x = torch.tensor(x,dtype=torch.float)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.output(x)
        return x

    def fit(self, states, targets_full, epochs=1, verbose=0):
        for i in range(0,len(targets_full)):
            self.optimizer.zero_grad()
            output = self.forward(states[i])
            target = torch.tensor(targets_full[i],dtype=torch.float)
            mse_loss = nn.MSELoss()
            loss = mse_loss(output,target)
            loss.backward()
            self.optimizer.step()

    def predict(self,x):
        return self.forward(x).detach().numpy()

    def predict_on_batch(self,X):
        return np.array([self.predict(x) for x in X])

class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1
        self.gamma = .95
        self.batch_size = 64
        self.epsilon_min = .01
        self.epsilon_decay = .995
        self.learning_rate = 0.001
        self.memory = deque(maxlen=10000)
        self.model = self.build_model()

    def build_model(self):
        model = network(self.state_space, self.action_space)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # epsilon-greedy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        # PREDICT ON BATCH
        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay_torch(self):
        
        if len(self.memory) < self.batch_size:
            return

        minibatch = DataLoader(self.memory, batch_size=self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(episode):

    loss = []
    agent = DQN(env.action_space.n, env.observation_space.shape[0])
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 4))
        score = 0
        max_steps = 1000
        for i in range(max_steps):
            # Comment this out to stop showing the game while training
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, 4))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            # train using stored memory if size of memory exceeds minimum required for a minibatch
            # form of online learning?
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        loss.append(score)
    return loss


def random_policy(episode, step):

    for i_episode in range(episode):
        env.reset()
        for t in range(step):
            env.render()
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
            print("Starting next episode")


if __name__ == '__main__':

    ep = 100
    loss = train_dqn(ep)
    plt.plot([i+1 for i in range(0, ep, 2)], loss[::2])
    plt.show()