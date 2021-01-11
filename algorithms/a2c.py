import os
import gym
import gym_cabworld
import random
import time
import torch
import numpy as np
from collections import deque
from features import track_reward, clip_state

from a2c_model import PolicyNetwork

from pyvirtualdisplay import Display

disp = Display().start()

def actor_critic(env, estimator, n_episode, gamma):

    for episode in range(n_episode):

        n_clip = 6

        if episode >= 50:
            n_clip = 6

        log_probs = []
        running_reward = 0
        state_values = []
        state = env.reset()
        # state = tuple((list(state))[:n_state])
        state = clip_state(state, n_clip)

        saved_rewards = [0, 0, 0, 0]
        pick_ups = 0
        number_of_action_4 = 0
        number_of_action_5 = 0
        wrong_pick_up_or_drop_off = 0

        passenger = False
        pick_up_drop_off_steps = []
        drop_off_pick_up_steps = []
        passenger_steps = 0
        no_passenger_steps = 0

        while True:
            action, log_prob, state_value = estimator.get_action(state)
            next_state, reward, is_done, _ = env.step(action)
            # next_state = tuple((list(next_state))[:n_state])
            next_state = clip_state(next_state, n_clip)

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

            log_probs.append(log_prob)
            state_values.append(state_value)
            rewards.append(reward)

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

            if is_done:
                print(
                f"Episode: {episode} Reward: {running_reward} Passengers {pick_ups//2} N-Action-4: {number_of_action_4} N-Action-5: {number_of_action_5} Illegal-Pick-Ups {wrong_pick_up_or_drop_off}"
                )
                returns = []
                Gt = 0
                pw = 0
                for reward in rewards[::-1]:
                    Gt += gamma ** pw * reward
                    pw += 1
                    returns.append(Gt)

                returns = returns[::-1]
                returns = torch.tensor(returns)
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
                estimator.update(returns, log_probs, state_values, episode)
                uncertainty = sum(log_probs) * -1
                rewards.append(running_reward)
                illegal_pick_ups.append(saved_rewards[1])
                illegal_moves.append(saved_rewards[2])
                do_nothing.append(saved_rewards[3])
                uncertainties.append(uncertainty)
                n_passengers.append(pick_ups // 2)
                mean_pick_up_path.append((np.array(drop_off_pick_up_steps).mean()))
                mean_drop_off_path.append((np.array(pick_up_drop_off_steps).mean()))
                break

            state = next_state


env_name = "Cabworld-v6"
env = gym.make(env_name)
n_action = 6
n_state = 20
n_episode = 100
lr = 0.001

dirname = os.path.dirname(__file__)
log_path = os.path.join(dirname, "../runs", "a2c")
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

illegal_pick_ups = []
illegal_moves = []
do_nothing = []
uncertainties = []
n_passengers = []
rewards = []
mean_pick_up_path = []
mean_drop_off_path = []

estimator = PolicyNetwork(n_state, n_action)
actor_critic(env, estimator, n_episode, 0.99)
estimator.save_model(log_path)

from plotting import *

plot_rewards(rewards, log_path)
plot_rewards_and_epsilon(rewards, uncertainties, log_path)
plot_rewards_and_passengers(rewards, n_passengers, log_path)
plot_rewards_and_illegal_actions(
    rewards, illegal_pick_ups, illegal_moves, do_nothing, log_path
)
plot_mean_pick_up_drop_offs(mean_pick_up_path, mean_drop_off_path, log_path)