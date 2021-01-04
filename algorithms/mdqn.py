import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from collections import deque, namedtuple

import gym
import gym_cabworld

from plotting import *
from features import *

from m_dqn_model import M_DQN_Agent

from pyvirtualdisplay import Display

disp = Display().start()

np.random.seed(42)
torch.manual_seed(42)
env_name = "Cabworld-v6"
env = gym.make(env_name)

action_size = 6
state_size = 20
n_episodes = 100

layer_size = 64
buffer_size = 500000
batch_size = 25
eps_decay = 0.95
eps = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mdqn = M_DQN_Agent(
    state_size=state_size,
    action_size=action_size,
    layer_size=layer_size,
    BATCH_SIZE=batch_size,
    BUFFER_SIZE=buffer_size,
    LR=0.001,
    TAU=0.01,
    GAMMA=0.99,
    UPDATE_EVERY=5,
    device=device,
    seed=42,
)

dirname = os.path.dirname(__file__)
log_path = os.path.join(dirname, "../runs", "mdqn")
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
    info_file.write("Episodes:" + str(n_episodes) + "\n")


illegal_pick_ups = []
illegal_moves = []
do_nothing = []
epsilons = []
n_passengers = []
rewards = []
mean_pick_up_path = []
mean_drop_off_path = []

for episode in range(1, n_episodes + 1):

    state = env.reset()
    # state = feature_engineering(state)
    # state = tuple((list(state))[:state_size])
    running_reward = 0
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

    while True:
        action = mdqn.act(state, eps)
        next_state, reward, done, _ = env.step(action)
        # next_state = feature_engineering(next_state)
        # next_state = tuple((list(next_state))[:state_size])
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

        mdqn.step(state, action, reward, next_state, done)
        state = next_state
        running_reward += reward

        if done:
            print(
                f"Episode: {episode} Reward: {running_reward} Passengers {pick_ups//2} N-Action-4: {number_of_action_4} N-Action-5: {number_of_action_5} Epsilon {eps} Illegal-Pick-Ups {wrong_pick_up_or_drop_off}"
            )
            break

    eps = round(max(eps * eps_decay, 0.001), 3)

    rewards.append(running_reward)
    illegal_pick_ups.append(saved_rewards[1])
    illegal_moves.append(saved_rewards[2])
    do_nothing.append(saved_rewards[3])
    epsilons.append(eps)
    n_passengers.append(pick_ups // 2)
    mean_pick_up_path.append((np.array(drop_off_pick_up_steps).mean()))
    mean_drop_off_path.append((np.array(pick_up_drop_off_steps).mean()))

mdqn.save_model(log_path)

plot_rewards(rewards, log_path)
plot_rewards_and_epsilon(rewards, epsilons, log_path)
plot_rewards_and_passengers(rewards, n_passengers, log_path)
plot_rewards_and_illegal_actions(
    rewards, illegal_pick_ups, illegal_moves, do_nothing, log_path
)
plot_mean_pick_up_drop_offs(mean_pick_up_path, mean_drop_off_path, log_path)
