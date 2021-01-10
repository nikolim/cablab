import os
import torch
import numpy as np
import gym
import random
import math
import gym_cabworld
from torch.utils.tensorboard import SummaryWriter
from ppo_models import Memory, ActorCritic, PPO
from plotting import *
from features import *

from pyvirtualdisplay import Display

disp = Display().start()

torch.manual_seed(42)
env_name = "Cabworld-v6"
env = gym.make(env_name)

n_state = 20
n_actions = 6
episodes = 10
max_timesteps = 10000

dirname = os.path.dirname(__file__)
log_path = os.path.join(dirname, "../runs", "rand")
if not os.path.exists(log_path):
    os.mkdir(log_path)
log_folders = os.listdir(log_path)
if len(log_folders) == 0:
    folder_number = 0
else:
    folder_number = max([int(elem) for elem in log_folders]) + 1

log_path = os.path.join(log_path, str(folder_number))
os.mkdir(log_path)
with open(os.path.join(log_path, "info.txt"), "w+") as info_file:
    info_file.write(env_name + "\n")
    info_file.write("Episodes:" + str(episodes) + "\n")

illegal_pick_ups = []
illegal_moves = []
do_nothing = []
entropys = []
n_passengers = []
rewards = []
mean_pick_up_path = []
mean_drop_off_path = []

n_clip = 6

def random_policy(state): 
    state = list(state)

    if state[4] == 1: 
        return 4
    if state[5] == 1: 
        return 5 
    else: 
        move_flags = state[:4]
        legal_moves = [i for i,flag in enumerate(move_flags) if flag==1] 
        return random.sample(legal_moves, 1)[0]

for episode in range(episodes):

    state = env.reset()
    state = clip_state(state, n_clip)

    saved_rewards = [0, 0, 0, 0]
    episode_reward = 0
    uncertainty = None
    n_steps = 0
    pick_ups = 0
    mean_entropy = 0

    number_of_action_4 = 0
    number_of_action_5 = 0
    wrong_pick_up_or_drop_off = 0

    passenger = False
    pick_up_drop_off_steps = []
    drop_off_pick_up_steps = []
    passenger_steps = 0
    no_passenger_steps = 0

    for t in range(max_timesteps):
        action = random_policy(state)

        state, reward, done, _ = env.step(action)
        saved_rewards = track_reward(reward, saved_rewards)

        if passenger:
            passenger_steps += 1
        else:
            no_passenger_steps += 1

        if action == 4:
            number_of_action_4 += 1
        if action == 5:
            number_of_action_5 += 1
        if action == 6:
            saved_rewards[3] += 1

        if reward == -10:
            wrong_pick_up_or_drop_off += 1

        if reward == 100:
            if passenger:
                # drop-off
                drop_off_pick_up_steps.append(no_passenger_steps)
                no_passenger_steps = 0
            else:
                # pick-up
                pick_up_drop_off_steps.append(passenger_steps)
                passenger_steps = 0

            passenger = not passenger
            pick_ups += 1
            reward = 1000

        episode_reward += reward
        if done:
            print(
                f"Episode: {episode} Reward: {episode_reward} Passengers {pick_ups//2} N-Action-4: {number_of_action_4} N-Action-5: {number_of_action_5} Entropy {mean_entropy} Illegal-Pick-Ups {wrong_pick_up_or_drop_off}"
            )
            break

    rewards.append(episode_reward)
    illegal_pick_ups.append(saved_rewards[1])
    illegal_moves.append(saved_rewards[2])
    do_nothing.append(saved_rewards[3])
    entropys.append(mean_entropy)
    n_passengers.append(pick_ups // 2)
    mean_pick_up_path.append((np.array(drop_off_pick_up_steps).mean()))
    mean_drop_off_path.append((np.array(pick_up_drop_off_steps).mean()))

plot_rewards(rewards, log_path)
plot_rewards_and_entropy(rewards, entropys, log_path)
plot_rewards_and_passengers(rewards, n_passengers, log_path)
plot_rewards_and_illegal_actions(
    rewards, illegal_pick_ups, illegal_moves, do_nothing, log_path
)
plot_mean_pick_up_drop_offs(mean_pick_up_path, mean_drop_off_path, log_path)


