import os
import torch
import gym
import gym_cabworld
from ppo_models import PPO

torch.manual_seed(42)
env = gym.make("Cabworld-v6")

episodes = 5
max_timesteps = 10000
ppo = PPO()
ppo.load_model(PATH='')

for episode in range(episodes):

    state = env.reset()
    episode_reward = 0

    for _ in range(max_timesteps):

        action = ppo.policy.deploy(state)
        state, reward, done, _ = env.step(action)
        episode_reward += reward

        if done:
            print(f"Reward {episode_reward}")
            break
    