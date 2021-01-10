import os
import gym
import torch
import numpy as np

from collections import deque
import random

import copy
from torch.autograd import Variable

from features import *
from plotting import *

from dqn_model import DQN, gen_epsilon_greedy_policy

from pyvirtualdisplay import Display

disp = Display().start()

import gym_cabworld

env_name = "Cabworld-v6"
env = gym.envs.make(env_name)

def q_learning(
    env,
    estimator,
    n_episode,
    replay_size,
    target_update=5,
    gamma=0.99,
    epsilon=1,
    epsilon_decay=0.95,
):
    for episode in range(n_episode):

        if episode % target_update == 0:
            estimator.copy_target()

        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        state = env.reset()
        # state = feature_engineering(state)
        state = tuple((list(env.reset()))[:n_state])
        is_done = False

        saved_rewards = [0, 0, 0, 0]
        running_reward = 0
        pick_ups = 0
        number_of_action_4 = 0
        number_of_action_5 = 0
        wrong_pick_up_or_drop_off = 0

        passenger = False
        pick_up_drop_off_steps = []
        drop_off_pick_up_steps = []
        passenger_steps = 0
        no_passenger_steps = 0

        steps = 0

        while not is_done:
            action = policy(state)
            steps += 1

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

            next_state, reward, is_done, _ = env.step(action)
            next_state = tuple((list(next_state))[:n_state])
            # next_state = feature_engineering(next_state)
            saved_rewards = track_reward(reward, saved_rewards)

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

            running_reward += reward

            memory.append((state, action, next_state, reward, is_done))

            if steps % 10 == 0:
                estimator.replay(memory, replay_size, gamma)

            if is_done:
                print(
                    f"Episode: {episode} Reward: {running_reward} Passengers: {pick_ups//2} N-Action-4: {number_of_action_4} N-Action-5: {number_of_action_5} Illegal-Pick-Ups {wrong_pick_up_or_drop_off} Epsilon {epsilon}"
                )
                break

            state = next_state

        epsilon = max(epsilon * epsilon_decay, 0.01)

        rewards.append(running_reward)
        illegal_pick_ups.append(saved_rewards[1])
        illegal_moves.append(saved_rewards[2])
        do_nothing.append(saved_rewards[3])
        episolons.append(epsilon)
        n_passengers.append(pick_ups // 2)
        mean_pick_up_path.append((np.array(drop_off_pick_up_steps).mean()))
        mean_drop_off_path.append((np.array(pick_up_drop_off_steps).mean()))


counter = 0
n_state = 6
n_action = 6
n_hidden = 32
lr = 0.01

n_episode = 50
replay_size = 20
target_update = 5

illegal_pick_ups = []
illegal_moves = []
do_nothing = []
episolons = []
n_passengers = []
rewards = []
mean_pick_up_path = []
mean_drop_off_path = []

dqn = DQN(n_state, n_action, n_hidden, lr)
memory = deque(maxlen=50000)

dirname = os.path.dirname(__file__)
log_path = os.path.join(dirname, "../runs", "dqn")
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
    info_file.write("Episodes:" + str(n_episode) + "\n")


q_learning(env, dqn, n_episode, replay_size, target_update, gamma=0.99, epsilon=1)
dqn.save_model(log_path)

plot_rewards(rewards, log_path)
plot_rewards_and_epsilon(rewards, episolons, log_path)
plot_rewards_and_passengers(rewards, n_passengers, log_path)
plot_rewards_and_illegal_actions(
    rewards, illegal_pick_ups, illegal_moves, do_nothing, log_path
)
plot_mean_pick_up_drop_offs(mean_pick_up_path, mean_drop_off_path, log_path)
