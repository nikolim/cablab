import os
import time
import torch
import math

import gym
import gym_cabworld

from m_dqn_model import M_DQN_Agent
from features import feature_engineering

torch.manual_seed(42)
env = gym.make("Cabworld-v6")

episodes = 5
max_timesteps = 10000
mdqn = M_DQN_Agent()

mdqn.load_model("../runs/mdqn/24/mdqn.pth")

for episode in range(episodes):

    state = env.reset()
    episode_reward = 0

    for _ in range(max_timesteps):

        # state = feature_engineering(state)
        action = mdqn.act(state)
        state, reward, done, _ = env.step(action)
        episode_reward += reward

        env.render()
        # time.sleep(0.05)

        if done:
            print(f"Reward {episode_reward}")
            break
