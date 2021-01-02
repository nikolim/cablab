import os
import time
import torch
import math
import gym
import gym_cabworld
from ppo_models import PPO

torch.manual_seed(42)
env = gym.make("Cabworld-v6")

episodes = 5
max_timesteps = 10000
ppo = PPO()

ppo.load_model('../runs/ppo/24/ppo.pth')

def euclidean_distance(p1, p2): 
    return round(math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)/math.sqrt(2), 5)

def feature_engineering(state): 

    dist_pos_pass1 = euclidean_distance((state[4], state[5]), (state[6], state[7]))
    dist_dest_pas1 = euclidean_distance((state[4], state[5]), (state[8], state[9]))
    dist_pos_pass2 = euclidean_distance((state[4], state[5]), (state[10], state[11]))
    dist_dest_pas2 = euclidean_distance((state[4], state[5]), (state[12], state[13]))
    dist_pos_pass3 = euclidean_distance((state[4], state[5]), (state[14], state[15]))
    dist_dest_pas3 = euclidean_distance((state[4], state[5]), (state[16], state[17]))

    state = list(state)
    new_state = state[:6] + [dist_pos_pass1, dist_dest_pas1] + state[6:10] + [dist_pos_pass2, dist_dest_pas2] + state[10:14] + [dist_pos_pass3, dist_dest_pas3] + state[14:]
    return new_state


for episode in range(episodes):

    state = env.reset()
    episode_reward = 0

    for _ in range(max_timesteps):

        state = feature_engineering(state)
        action = ppo.policy.deploy(state)
        state, reward, done, _ = env.step(action)
        episode_reward += reward

        env.render()
        #time.sleep(0.05)

        if done:
            print(f"Reward {episode_reward}")
            break
    
