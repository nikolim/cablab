import os
import math
import logging
from os.path import join
from matplotlib.pyplot import plot
import pandas as pd
import numpy as np
from scipy.signal.filter_design import _nearest_real_complex_idx

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
    logging.basicConfig(format=LOG_FORMAT,
                        level=logging.INFO, filename=file_name)
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
    "useless_steps", 
    "opt_passengers", 
    "opt_ratio", 
    "idx_zero", 
    "idx_one", 
    "episode_loss"
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
    
    def write_to_log(self, str): 
        if self.logger: 
            logging.info(str)

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
        self.useless_steps = 0

        self.opt_passenger = 0

        self.passenger = False
        self.pick_up_drop_off_steps = []
        self.drop_off_pick_up_steps = []

        self.psng_idx_zero = 0 
        self.psng_idx_one = 0

        self.episode_loss = 0

    def new_episode(self):

        if self.eps_counter > 0:
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
                self.total_values_dict["n_passengers"], self.pick_ups
            )
            self.total_values_dict["opt_passengers"] = np.append(
                self.total_values_dict["opt_passengers"], self.opt_passenger
            )

            if self.opt_passenger > 0:
                ratio = round(self.opt_passenger/  self.pick_ups, 3)
            elif self.pick_ups > 0: 
                ratio = 0
            else: 
                ratio = 0.5

            self.total_values_dict["opt_ratio"] = np.append(
                self.total_values_dict["opt_ratio"], ratio
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
            self.total_values_dict["useless_steps"] = np.append(
                self.total_values_dict["useless_steps"], self.useless_steps
            )

            self.total_values_dict["idx_zero"] = np.append(
                self.total_values_dict["idx_zero"], self.psng_idx_zero
            )

            self.total_values_dict["idx_one"] = np.append(
                self.total_values_dict["idx_one"], self.psng_idx_one
            )

            self.total_values_dict["episode_loss"] = np.append(
                self.total_values_dict["episode_loss"], self.episode_loss
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

    def track_reward(self, reward, action, state, next_state):
        if reward == -5:
            self.illegal_moves_ep += 1
        if reward == -10:
            self.illegal_pick_up_ep += 1
        if reward == 25:
            if self.passenger:
                self.drop_off_pick_up_steps.append(self.no_passenger_steps)
                self.no_passenger_steps = 0
            else:
                self.pick_up_drop_off_steps.append(self.passenger_steps)
                self.passenger_steps = 0
            
            self.passenger = not self.passenger
            
            if action == 4: 
                self.pick_ups += 1
                idx = self.get_index_of_passenger(state)
                if idx == 0: 
                    self.opt_passenger += 1

            if action == 5: 
                self.save_dest_to_passengers(next_state)

        if self.passenger:
            self.passenger_steps += 1
        else:
            self.no_passenger_steps += 1


        self.episode_reward += reward


    def get_pick_ups(self):
        return self.pick_ups
    
    def get_opt_pick_ups(self): 
        return self.opt_passenger

    def track_actions(self, state, action):
        if action == 6:
            self.do_nothing += 1
            if state[7] == -1 and state[8] == -1:
                self.do_nothing_opt += 1
            else:
                self.do_nothing_sub += 1
        else: 
            if state[-1] == 1:
                self.useless_steps += 1


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
        plot_values(df, ["n_passengers"], log_path)
        plot_values(df, ["useless_steps"], log_path)
        plot_values(df, ["illegal_pick_ups", "illegal_moves"], log_path)
        plot_values(df, ["do_nothing_opt_arr", "do_nothing_sub_arr"], log_path)
        plot_values(df, ["rewards", "epsilon"], log_path, double_scale=True)
        plot_values(df, ["opt_ratio"], log_path)
        plot_values(df, ["n_passengers", "idx_zero", "idx_one"], log_path)
        plot_values(df, ["episode_loss"], log_path)



    def calc_distance(self, pos1, pos2): 
        return round(math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2),3)

    def save_dest_to_passengers(self, state): 
        n_passengers = 2
        self.dest_passengers = []        

        for i in range(n_passengers):
            self.dest_passengers.append(self.calc_distance((state[5],state[6]), (state[7 + i*2],state[8 + i*2])))

    def get_index_of_passenger(self, state):

        n_passengers = 2
        idx = None

        for i in range(n_passengers):
            if state[5] == state[7+2*i] and state[6] == state[8+2*i]: 
                idx = i
                break
        
        assert idx is not None
        
        if idx == 0: 
            self.psng_idx_zero += 1 
        else: 
            self.psng_idx_one += 1

        travelled_distance = self.dest_passengers[idx]
        self.dest_passengers.sort()
        travalled_idx = self.dest_passengers.index(travelled_distance)
        return travalled_idx
    
    def track_loss(self,loss): 
        self.episode_loss = loss


class MultiTracker:
    def __init__(self, n_agents, logger=None):

        if logger:
            self.logger = logger
        else:
            self.logger = False

        self.n_agents = n_agents
        self.trackers = []
        self.adv_rewards = []
        self.adv_episode_rewards = [0] * self.n_agents

        # storage for all episodes
        for i in range(n_agents):
            tracker = Tracker(logger)
            self.trackers.append(tracker)

        self.init_episode_vars()

    def write_to_log(self, str): 
        if self.logger: 
            logging.info(str)

    def init_episode_vars(self):
        # storage for one episode

        for tracker in self.trackers:
            tracker.init_episode_vars()

    def new_episode(self):

        self.adv_rewards.append(self.adv_episode_rewards)

        for tracker in self.trackers:
            tracker.new_episode()

        self.adv_episode_rewards = [0] * self.n_agents

    def track_reward(self, rewards):
        for i, tracker in enumerate(self.trackers):
            tracker.track_reward(rewards[i])

    def track_adv_reward(self, rewards):
        for i in range(len(rewards)):
            self.adv_episode_rewards[i] += rewards[i]

    def track_actions(self, states, actions, comm=False):

        def calc_distance(state):
            return round(
            math.sqrt((state[5] - state[7]) ** 2 + (state[6] - state[8]) ** 2), 2
        )
        distances = [calc_distance(state) for state in states]
        one_hot_arr = [1 if dist == min(distances) else 0 for dist in distances]

        for action, state, tracker, one_hot in zip(actions, states, self.trackers, one_hot_arr):
            tracker.track_actions(state, action)

            if action != 6 and ((state[8] == -1 and state[7] == -1) or one_hot == 0): 
                tracker.useless_steps += 1

    def track_epsilon(self, epsilon):
        for i, tracker in enumerate(self.trackers):
            tracker.track_epsilon(epsilon)

    def get_pick_ups(self):
        pick_ups = [tracker.get_pick_ups() for tracker in self.trackers]
        return pick_ups

    def get_rewards(self):
        rewards = [tracker.episode_reward for tracker in self.trackers]
        return rewards

    def get_do_nothing(self):
        do_nothing = [tracker.do_nothing for tracker in self.trackers]
        return do_nothing

    def plot(self, log_path):

        dfs = []
        for i, tracker in enumerate(self.trackers):
            df = pd.DataFrame()
            for value in total_values:
                df[value] = tracker.total_values_dict[value]
            dfs.append(df)
            file_name = os.path.join(log_path, "logs" + str(i + 1) + ".csv")
            df.to_csv(file_name)

        summed_df = dfs[0]
        values_to_add = ['illegal_pick_ups', 'illegal_moves', 'n_passengers', 'rewards', 'mean_pick_up_path',
                         'mean_drop_off_path', 'do_nothing_arr', 'do_nothing_opt_arr', 'do_nothing_sub_arr', 'useless_steps']

        plot_mult_agent(dfs, ["rewards"], log_path)
        plot_mult_agent(dfs, ["n_passengers"], log_path)
        plot_mult_agent(dfs, ["useless_steps"], log_path)
        plot_mult_agent(dfs, ["illegal_pick_ups", "illegal_moves"], log_path)
        plot_mult_agent(dfs, ["do_nothing_opt_arr", "do_nothing_sub_arr"], log_path)
                            
        for value in values_to_add: 
            summed_df[value] = dfs[0][value] + dfs[1][value]        
        file_name = os.path.join(log_path, "logs_summed.csv")
        summed_df.to_csv(file_name)

