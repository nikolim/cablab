import os
import time
import torch
import math

import gym
import gym_cabworld

from ppo_models import PPO
from features import feature_engineering

torch.manual_seed(42)
env = gym.make("Cabworld-v6")

episodes = 5
max_timesteps = 10000

n_states = 64
n_actions = 4

ppo = PPO(n_states, n_actions)

ppo.load_model('../runs/ppo/31/ppo.pth')

for episode in range(episodes):

    state = env.reset()
    episode_reward = 0

    for _ in range(max_timesteps):

        #state = feature_engineering(state)
        action = ppo.policy.deploy(state)
        state, reward, done, _ = env.step(action)
        episode_reward += reward

        env.render()
        #time.sleep(0.05)

        if done:
            print(f"Reward {episode_reward}")
            break
    
