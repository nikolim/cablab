import os
import math
import logging
from os.path import join
from matplotlib.pyplot import plot
import pandas as pd
import numpy as np

from common.plotting import *


def create_log_folder(algorithm):
    dirname = os.path.dirname(__file__)
    log_path = os.path.join(dirname, "../runs", str(algorithm))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_folders = os.listdir(log_path)
    if len(log_folders) == 0:
        folder_number = 0
    else:
        folder_number = max([int(elem) for elem in log_folders]) + 1
    log_path = os.path.join(log_path, str(folder_number))
    os.mkdir(log_path)
    return log_path


def create_logger(path):
    LOG_FORMAT = "%(message)s"
    file_name = os.path.join(path, "logs.log")
    open(file_name, "w+")
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, filename=file_name)
    logger = logging.getLogger("cablab")
    return logger


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


total_values = [
    "illegal_pick_ups",
    "illegal_moves",
    "epsilon",
    "n_passengers",
    "rewards",
    "mean_pick_up_path",
    "mean_drop_off_path",
    "do_nothing_arr",
    "do_nothing_opt_arr",
    "do_nothing_sub_arr",
]


class Tracker:
    def __init__(self, logger=None) -> None:

        if logger:
            self.logger = logger
        else:
            self.logger = False

        # storage for all episodes
        self.total_values_dict = {}
        for value in total_values:
            self.total_values_dict[value] = np.array([])

        self.eps_counter = 0
        self.init_episode_vars()

    def init_episode_vars(self):

        # storage for one episode
        self.episode_reward = 0
        self.pick_ups = 0
        self.illegal_pick_up_ep = 0
        self.illegal_moves_ep = 0
        self.passenger_steps = 0
        self.no_passenger_steps = 0
        self.do_nothing = 0
        self.do_nothing_opt = 0
        self.do_nothing_sub = 0

        self.passenger = False
        self.pick_up_drop_off_steps = []
        self.drop_off_pick_up_steps = []

    def new_episode(self):

        if self.eps_counter > 0:
            if self.logger:
                logging.info(
                    f"Episode:{self.eps_counter} Reward: {self.episode_reward} Passengers: {self.pick_ups // 2}"
                )
            self.total_values_dict["rewards"] = np.append(
                self.total_values_dict["rewards"], self.episode_reward
            )
            self.total_values_dict["illegal_pick_ups"] = np.append(
                self.total_values_dict["illegal_pick_ups"], self.illegal_pick_up_ep
            )
            self.total_values_dict["illegal_moves"] = np.append(
                self.total_values_dict["illegal_moves"], self.illegal_moves_ep
            )
            self.total_values_dict["n_passengers"] = np.append(
                self.total_values_dict["n_passengers"], self.pick_ups // 2
            )
            self.total_values_dict["do_nothing_arr"] = np.append(
                self.total_values_dict["do_nothing_arr"], self.do_nothing
            )
            self.total_values_dict["do_nothing_opt_arr"] = np.append(
                self.total_values_dict["do_nothing_opt_arr"], self.do_nothing_opt
            )
            self.total_values_dict["do_nothing_sub_arr"] = np.append(
                self.total_values_dict["do_nothing_sub_arr"], self.do_nothing_sub
            )

            if len(self.drop_off_pick_up_steps) > 0:
                self.total_values_dict["mean_pick_up_path"] = np.append(
                    self.total_values_dict["mean_pick_up_path"],
                    (np.array(self.drop_off_pick_up_steps).mean()),
                )
                self.total_values_dict["mean_drop_off_path"] = np.append(
                    self.total_values_dict["mean_drop_off_path"],
                    (np.array(self.pick_up_drop_off_steps).mean()),
                )
            else:
                self.total_values_dict["mean_pick_up_path"] = np.append(
                    self.total_values_dict["mean_pick_up_path"], 0
                )
                self.total_values_dict["mean_drop_off_path"] = np.append(
                    self.total_values_dict["mean_drop_off_path"], 0
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

    def track_actions(self, state, action):
        if action == 6:
            self.do_nothing += 1
            if state[7] == -1 and state[8] == -1:
                self.do_nothing_opt += 1
            else:
                self.do_nothing_sub += 1

    def track_epsilon(self, epsilon):
        self.total_values_dict["epsilon"] = np.append(
            self.total_values_dict["epsilon"], epsilon
        )

    def plot(self, log_path):

        df = pd.DataFrame()

        for value in total_values:
            df[value] = self.total_values_dict[value]

        file_name = os.path.join(log_path, "logs.csv")
        df.to_csv(file_name)

        plot_values(df, ["rewards"], log_path)
        plot_values(df, ["rewards", "n_passengers"], log_path)
        plot_values(df, ["rewards", "illegal_pick_ups", "illegal_moves"], log_path)
        plot_values(
            df, ["do_nothing_arr", "do_nothing_opt_arr", "do_nothing_sub_arr"], log_path
        )
        plot_values(df, ["rewards", "epsilon"], log_path, double_scale=True)


class MultiTracker(Tracker):
    def __init__(self, n_agents):

        self.n_agents = n_agents

        # storage for all episodes
        self.illegal_pick_ups = [[] for _ in range(n_agents)]
        self.illegal_moves = [[] for _ in range(n_agents)]
        self.epsilon = [[] for _ in range(n_agents)]
        self.n_passengers = [[] for _ in range(n_agents)]
        self.rewards = [[] for _ in range(n_agents)]
        self.adv_rewards = [[] for _ in range(n_agents)]
        self.mean_pick_up_path = [[] for _ in range(n_agents)]
        self.mean_drop_off_path = [[] for _ in range(n_agents)]
        self.total_number_passenger = [[] for _ in range(n_agents)]
        self.eps_counter = 0
        self.do_nothing = [0, 0]
        self.init_episode_vars()

    def init_episode_vars(self):

        # storage for one episode
        self.episode_reward = [0] * self.n_agents
        self.episode_adv_reward = [0] * self.n_agents
        self.pick_ups = [0] * self.n_agents
        self.illegal_pick_up_ep = [0] * self.n_agents
        self.illegal_moves_ep = [0] * self.n_agents
        self.passenger = [False] * self.n_agents
        self.pick_up_drop_off_steps = [[] for _ in range(self.n_agents)]
        self.drop_off_pick_up_steps = [[] for _ in range(self.n_agents)]
        self.passenger_steps = [0] * self.n_agents
        self.no_passenger_steps = [0] * self.n_agents

    def new_episode(self):

        self.do_nothing = [0, 0]

        if self.eps_counter > 0:
            for i in range(self.n_agents):
                self.rewards[i].append(self.episode_reward[i])
                self.adv_rewards[i].append(self.episode_adv_reward[i])
                self.illegal_pick_ups[i].append(self.illegal_pick_up_ep[i])
                self.illegal_moves[i].append(self.illegal_moves_ep[i])
                self.n_passengers[i].append(self.pick_ups[i] // 2)
                if len(self.drop_off_pick_up_steps[i]) > 0:
                    self.mean_pick_up_path[i].append(
                        (np.array(self.drop_off_pick_up_steps[i]).mean())
                    )
                    self.mean_drop_off_path.append(
                        (np.array(self.pick_up_drop_off_steps[i]).mean())
                    )

        self.init_episode_vars()
        self.eps_counter += 1

    def track_reward(self, rewards):

        assert len(rewards) == self.n_agents

        for i in range(len(rewards)):
            if rewards[i] == -5:
                self.illegal_moves_ep[i] += 1
            if rewards[i] == -10:
                self.illegal_pick_up_ep[i] += 1
            if rewards[i] == 100:
                if self.passenger[i]:
                    self.drop_off_pick_up_steps[i].append(self.no_passenger_steps[i])
                    self.no_passenger_steps[i] = 0
                else:
                    self.pick_up_drop_off_steps[i].append(self.passenger_steps[i])
                    self.passenger_steps[i] = 0

                self.passenger[i] = not self.passenger[i]
                self.pick_ups[i] += 1

            self.episode_reward[i] += rewards[i]

            if self.passenger[i]:
                self.passenger_steps[i] += 1
            else:
                self.no_passenger_steps[i] += 1

    def track_adv_reward(self, rewards):
        for i in range(len(rewards)):
            self.episode_adv_reward[i] += rewards[i]

    def track_actions(self, actions):

        if actions[0] == 6:
            self.do_nothing[0] += 1
        if actions[1] == 6:
            self.do_nothing[1] += 1

    def get_pick_ups(self):
        return [picks // 2 for picks in self.pick_ups]

    def plot(self, log_path):
        plot_multiple_agents(self.adv_rewards, "adv_rewards", log_path)
        plot_multiple_agents(self.rewards, "rewards", log_path)
        plot_multiple_agents(self.n_passengers, "passengers", log_path)
