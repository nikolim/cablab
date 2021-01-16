import os
import numpy as np

from common.plotting import *


def create_log_folder(algorithm):
    dirname = os.path.dirname(__file__)
    log_path = os.path.join(dirname, "../runs", str(algorithm))
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_folders = os.listdir(log_path)
    if len(log_folders) == 0:
        folder_number = 0
    else:
        folder_number = max([int(elem) for elem in log_folders]) + 1
    log_path = os.path.join(log_path, str(folder_number))
    os.mkdir(log_path)
    return log_path


def get_last_folder(algorithm):
    dirname = os.path.dirname(__file__)
    log_path = os.path.join(dirname, "../runs", str(algorithm))
    if not os.path.exists(log_path):
        return None
    log_folders = os.listdir(log_path)
    if len(log_folders) == 0:
        return None
    else:
        folder_number = max([int(elem) for elem in log_folders])
    current_log_path = os.path.join(log_path, str(folder_number))
    return current_log_path


class Tracker:
    def __init__(self) -> None:
        # storage for all episodes
        self.illegal_pick_ups = []
        self.illegal_moves = []
        self.epsilon = []
        self.n_passengers = []
        self.rewards = []
        self.mean_pick_up_path = []
        self.mean_drop_off_path = []
        self.total_number_passenger = 0
        self.eps_counter = 0
        self.init_episode_vars()

    def init_episode_vars(self):
        # storage for one episode
        self.episode_reward = 0
        self.pick_ups = 0
        self.illegal_pick_up_ep = 0
        self.illegal_moves_ep = 0
        self.passenger = False
        self.pick_up_drop_off_steps = []
        self.drop_off_pick_up_steps = []
        self.passenger_steps = 0
        self.no_passenger_steps = 0

    def new_episode(self):

        if self.eps_counter > 0:
            self.rewards.append(self.episode_reward)
            self.illegal_pick_ups.append(self.illegal_pick_up_ep)
            self.illegal_moves.append(self.illegal_moves_ep)
            self.n_passengers.append(self.pick_ups // 2)
            if len(self.drop_off_pick_up_steps) > 0:
                self.mean_pick_up_path.append(
                    (np.array(self.drop_off_pick_up_steps).mean())
                )
                self.mean_drop_off_path.append(
                    (np.array(self.pick_up_drop_off_steps).mean())
                )

        self.init_episode_vars()
        self.eps_counter += 1

    def track_reward(self, reward):
        if reward == -5:
            self.illegal_moves_ep += 1
        if reward == -10:
            self.illegal_pick_up_ep += 1
        if reward == 100:
            if self.passenger:
                self.drop_off_pick_up_steps.append(self.no_passenger_steps)
                self.no_passenger_steps = 0
            else:
                self.pick_up_drop_off_steps.append(self.passenger_steps)
                self.passenger_steps = 0

            self.passenger = not self.passenger
            self.pick_ups += 1

        self.episode_reward += reward

        if self.passenger:
            self.passenger_steps += 1
        else:
            self.no_passenger_steps += 1

    def get_pick_ups(self):
        return self.pick_ups // 2

    def plot(self, log_path):
        plot_rewards(self.rewards, log_path)
        plot_rewards_and_passengers(self.rewards, self.n_passengers, log_path)
        plot_rewards_and_illegal_actions(
            self.rewards, self.illegal_pick_ups, self.illegal_moves, log_path
        )
        plot_mean_pick_up_drop_offs(
            self.mean_pick_up_path, self.mean_drop_off_path, log_path
        )
