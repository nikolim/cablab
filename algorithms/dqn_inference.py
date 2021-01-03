import os
import time
import torch
import math

import gym
import gym_cabworld

from dqn_model import DQN
from features import feature_engineering

torch.manual_seed(42)
env = gym.make("Cabworld-v6")

n_state = 6
n_actions = 6
episodes = 5
max_timesteps = 10000
dqn = DQN(n_state, n_actions)

dqn.load_model('../runs/dqn/44/dqn.pth')

for episode in range(episodes):

    state = env.reset()
    state = tuple((list(state))[:n_state])
    episode_reward = 0

    for _ in range(max_timesteps):

        state = tuple((list(state))[:n_state])
        action = dqn.deploy(state)
        print(action)
        state, reward, done, _ = env.step(action)
        episode_reward += reward

        env.render()
        #time.sleep(0.05)

        if done:
            print(f"Reward {episode_reward}")
            break
    
