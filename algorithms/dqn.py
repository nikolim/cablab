
import os
import gym
import torch

from collections import deque
import random

import copy
from torch.autograd import Variable

from tensorboard_tracker import track_reward
from features import feature_engineering
from dqn_model import DQN, gen_epsilon_greedy_policy

from pyvirtualdisplay import Display
disp = Display().start()

import gym_cabworld

env_name = "Assault-ram-v0"
env = gym.envs.make(env_name)

def normalize_array(ar): 
    divide_256 = lambda x: round((x/256),4)
    new_arr = map(divide_256, ar)
    return list(new_arr)

def q_learning(env, estimator, n_episode, replay_size, target_update=10, gamma=1.0, epsilon=0.1, epsilon_decay=.99):
    for episode in range(n_episode):

        global counter 
        counter += 1
        if episode % target_update == 0:
            estimator.copy_target()

        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        state = env.reset()
        state = normalize_array(state)
        is_done = False
        running_reward = 0

        while not is_done:
            action = policy(state)

            next_state, reward, is_done, _ = env.step(action)
            next_state = normalize_array(next_state)

            running_reward += reward            
            memory.append((state, action, next_state, reward, is_done))
            
            if is_done:
                print(f'Epsisode: {episode} Reward: {running_reward}')
                estimator.replay(memory, replay_size, gamma)
                break

            state = next_state

        epsilon = max(epsilon * epsilon_decay, 0.01)

counter = 0
n_state = 128
n_action = 7
n_hidden = 128
lr = 0.001

n_episode = 500
replay_size = 10000
target_update = 5

illegal_pick_ups = []
illegal_moves = []
do_nothing = []
episolons = []
n_passengers = []
rewards = []

dqn = DQN(n_state, n_action, n_hidden, lr)
memory = deque(maxlen=500000)

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


q_learning(env, dqn, n_episode, replay_size, target_update, gamma=.99, epsilon=1)
dqn.save_model(log_path)
